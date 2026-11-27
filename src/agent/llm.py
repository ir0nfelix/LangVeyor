from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from src.agent.state import LLMUsageRecord
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

RoleName = Literal["FAST", "INVESTIGATOR", "AUDITOR", "JUDGE"]
StructuredResponse = dict[str, Any]
ResponderFn = Callable[
    [list[BaseMessage], type[BaseModel] | None, list[str]],
    AIMessage | BaseModel | Awaitable[AIMessage | BaseModel],
]


def _ensure_ai_message_with_usage(
    res: AIMessage | BaseModel,
    messages: list[BaseMessage],
) -> AIMessage:
    """Wrap scripted responses into AIMessage and guarantee non-zero usage_metadata."""
    if not isinstance(res, AIMessage):
        ai_msg = AIMessage(
            content=res.model_dump_json() if isinstance(res, BaseModel) else str(res)
        )
    else:
        ai_msg = res

    existing = getattr(ai_msg, "usage_metadata", None) or {}
    if int(existing.get("total_tokens", 0) or 0) > 0:
        return ai_msg

    in_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    out_chars = len(str(ai_msg.content or "")) + (
        len(json.dumps(tool_calls, ensure_ascii=False)) if tool_calls else 0
    )
    in_tok = max(90, in_chars // 4)
    out_tok = max(35, out_chars // 4)
    return ai_msg.model_copy(
        update={
            "usage_metadata": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            }
        }
    )


def _clean_json_text(raw_content: Any) -> str:
    """Strip markdown code fences and outer noise around a JSON object."""
    if isinstance(raw_content, str):
        text = raw_content.strip()
    elif isinstance(raw_content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content
        ]
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            text = json.dumps(raw_content, ensure_ascii=False)
    else:
        text = json.dumps(raw_content, ensure_ascii=False)

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]
    return text


def _extract_parsed_and_usage(
    result: Any,
    schema: type[BaseModel],
) -> tuple[Any | None, dict[str, Any]]:
    """Extract parsed BaseModel from include_raw result with JSON-cleaning fallback."""
    if not isinstance(result, dict):
        return result, {}

    parsed = result.get("parsed")
    raw = result.get("raw")
    parsing_error = result.get("parsing_error")
    usage_meta = dict(getattr(raw, "usage_metadata", None) or {})

    if parsed is None:
        logger.warning(
            "Structured output parsing failed for %s: parsing_error=%r",
            schema.__name__,
            parsing_error,
        )
        if raw is not None and isinstance(raw, AIMessage):
            cleaned = _clean_json_text(raw.content)
            if cleaned:
                try:
                    parsed = schema.model_validate_json(cleaned)
                except Exception as exc:
                    logger.warning(
                        "Raw content JSON fallback failed for %s: %s",
                        schema.__name__,
                        exc,
                    )
            if parsed is None and getattr(raw, "tool_calls", None):
                try:
                    args = raw.tool_calls[0].get("args")
                    if isinstance(args, dict):
                        parsed = schema.model_validate(args)
                except Exception as exc:
                    logger.warning(
                        "Tool-call args fallback failed for %s: %s",
                        schema.__name__,
                        exc,
                    )

    return parsed, usage_meta


class ScriptedChatModel(BaseChatModel):
    """Deterministic BaseChatModel for offline tests and mock evals supporting bind_tools and structured output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role_name: str = "MOCK"
    model_id: str = "mock/scripted"
    responder: ResponderFn
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return f"scripted-{self.role_name.lower()}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self.responder(messages, None, self.bound_tool_names)
        if not isinstance(res, (AIMessage, BaseModel)):
            raise TypeError(f"Unexpected sync responder return type: {type(res)}")
        ai_msg = _ensure_ai_message_with_usage(res, messages)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self.responder(messages, None, self.bound_tool_names)
        if hasattr(res, "__await__"):
            res = await res
        ai_msg = _ensure_ai_message_with_usage(res, messages)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names: list[str] = []
        for t in tools:
            if isinstance(t, BaseTool):
                names.append(t.name)
            elif hasattr(t, "name"):
                names.append(str(t.name))
            elif callable(t):
                names.append(getattr(t, "__name__", "tool"))
        return ScriptedChatModel(
            role_name=self.role_name,
            model_id=self.model_id,
            responder=self.responder,
            bound_tool_names=names,
        )

    def with_structured_output(
        self,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        assert isinstance(schema, type) and issubclass(schema, BaseModel)
        target_schema = schema

        async def _invoke(inp: Any, _config: Any = None) -> Any:
            msgs = list(inp) if isinstance(inp, list) else []
            res = self.responder(msgs, target_schema, self.bound_tool_names)
            if hasattr(res, "__await__"):
                res = await res
            if isinstance(res, BaseModel):
                parsed = target_schema.model_validate(res.model_dump())
                raw = AIMessage(
                    content=parsed.model_dump_json(),
                    usage_metadata={"input_tokens": 150, "output_tokens": 80, "total_tokens": 230},
                )
            elif isinstance(res, AIMessage):
                raw = _ensure_ai_message_with_usage(res, msgs)
                cleaned = _clean_json_text(raw.content)
                parsed = target_schema.model_validate_json(cleaned)
            else:
                raise TypeError(f"Unexpected scripted structured output type: {type(res)}")

            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(_invoke)


class AgentModelBundle:
    """Container holding the 4 role-separated LLMs (FAST, INVESTIGATOR, AUDITOR, JUDGE)."""

    def __init__(
        self,
        *,
        fast_model: BaseChatModel,
        investigator_model: BaseChatModel,
        auditor_model: BaseChatModel,
        fast_model_name: str,
        investigator_model_name: str,
        auditor_model_name: str,
        judge_model: BaseChatModel | None = None,
        judge_model_name: str = "",
    ) -> None:
        self.fast_model = fast_model
        self.investigator_model = investigator_model
        self.auditor_model = auditor_model
        self.judge_model: BaseChatModel = judge_model if judge_model is not None else auditor_model
        self.fast_model_name = fast_model_name
        self.investigator_model_name = investigator_model_name
        self.auditor_model_name = auditor_model_name
        self.judge_model_name = judge_model_name or auditor_model_name


def build_openrouter_models(settings: Settings | None = None) -> AgentModelBundle:
    """Construct the 4 distinct-role OpenRouter ChatOpenAI models from Settings."""
    cfg = settings or get_settings()
    if not cfg.has_openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured. Set it in .env or use mock mode.")

    headers = {
        "HTTP-Referer": "https://github.com/langveyor/mes-rca",
        "X-Title": "LangVeyor MES RCA",
    }

    def _mk(model_slug: str, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_slug,
            api_key=SecretStr(cfg.openrouter_api_key),
            base_url=cfg.openrouter_base_url,
            temperature=temperature,
            default_headers=headers,
            timeout=60.0,
            max_retries=1,
        )

    return AgentModelBundle(
        fast_model=_mk(cfg.openrouter_model_fast, temperature=0.0),
        investigator_model=_mk(cfg.openrouter_model_investigator, temperature=0.0),
        auditor_model=_mk(cfg.openrouter_model_auditor, temperature=0.0),
        judge_model=_mk(cfg.openrouter_model_judge, temperature=0.0),
        fast_model_name=cfg.openrouter_model_fast,
        investigator_model_name=cfg.openrouter_model_investigator,
        auditor_model_name=cfg.openrouter_model_auditor,
        judge_model_name=cfg.openrouter_model_judge,
    )


async def invoke_structured_with_usage(
    model: BaseChatModel,
    schema: type[BaseModel],
    messages: list[BaseMessage],
    *,
    role: RoleName,
    node_name: str,
    model_name: str,
) -> tuple[Any, LLMUsageRecord]:
    """Invoke structured output with markdown-strip fallback and function_calling retry."""
    t0 = time.perf_counter()
    parsed: Any = None
    in_tokens = 0
    out_tokens = 0
    tot_tokens = 0

    def _accum_usage(meta: dict[str, Any]) -> None:
        nonlocal in_tokens, out_tokens, tot_tokens
        in_tokens += int(meta.get("input_tokens", 0) or 0)
        out_tokens += int(meta.get("output_tokens", 0) or 0)
        tot_tokens += int(meta.get("total_tokens", 0) or 0)

    prefer_fn_calling = "deepseek" in model_name.lower()
    methods: tuple[str | None, str | None] = (
        ("function_calling", None) if prefer_fn_calling else (None, "function_calling")
    )
    timeout_sec = 85.0

    def _build_runnable(method: str | None) -> Runnable[Any, Any]:
        if method is None:
            return model.with_structured_output(schema, include_raw=True)
        return model.with_structured_output(schema, method=method, include_raw=True)

    for attempt_idx, method in enumerate(methods):
        try:
            runnable = _build_runnable(method)
            result = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout_sec)
            parsed, usage_meta = _extract_parsed_and_usage(result, schema)
            _accum_usage(usage_meta)
        except Exception as exc:
            logger.warning(
                "Structured output attempt %d (method=%r) failed for %s (%s): %s",
                attempt_idx + 1,
                method or "default",
                schema.__name__,
                model_name,
                exc,
            )
            parsed = None

        if parsed is not None:
            break
        if attempt_idx == 0:
            logger.warning(
                "Retrying %s on %s with method=%r",
                schema.__name__,
                model_name,
                methods[1] or "default",
            )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if parsed is None:
        raise ValueError(f"Structured output parsing returned None for {schema.__name__}")

    record = LLMUsageRecord(
        role=role,
        node_name=node_name,
        model_name=model_name,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=tot_tokens,
        latency_ms=latency_ms,
    )
    return parsed, record


def extract_usage_from_ai_message(
    msg: AIMessage,
    *,
    role: RoleName,
    node_name: str,
    model_name: str,
    latency_ms: float,
) -> LLMUsageRecord:
    usage_meta = getattr(msg, "usage_metadata", None) or {}
    return LLMUsageRecord(
        role=role,
        node_name=node_name,
        model_name=model_name,
        input_tokens=int(usage_meta.get("input_tokens", 0)),
        output_tokens=int(usage_meta.get("output_tokens", 0)),
        total_tokens=int(usage_meta.get("total_tokens", 0)),
        latency_ms=round(latency_ms, 2),
    )
