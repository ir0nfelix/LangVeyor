from __future__ import annotations

import asyncio
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from ddgs import DDGS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.llm import invoke_structured_with_usage
from src.agent.state import (
    ComponentBomSummary,
    ExternalHypothesis,
    LLMUsageRecord,
    WebHypothesesOutput,
)

SearchCallable = Callable[[str, int], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]]


class WebResearchState(TypedDict):
    """Isolated subgraph state for context hygiene: raw web text never leaks to parent state."""

    symptom_category: str
    operating_conditions: str | None
    bom_summary: ComponentBomSummary | None
    queries: list[str]
    raw_snippets: list[str]
    search_status: str
    external_hypotheses: list[ExternalHypothesis]
    token_usage: Annotated[list[LLMUsageRecord], operator.add]


def default_ddg_search(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def build_web_research_subgraph(
    fast_model: BaseChatModel,
    fast_model_name: str,
    search_fn: SearchCallable | None = None,
) -> Any:
    """Create the isolated web_researcher_node subgraph."""
    runner: SearchCallable = search_fn or default_ddg_search

    async def plan_queries_node(state: WebResearchState) -> dict[str, Any]:
        bom = state.get("bom_summary")
        motor = bom.motor if bom and bom.motor else "RS550-24V"
        bms = bom.bms if bom and bom.bms else "JK-BMS-6S"
        chem = bom.chemistry if bom and bom.chemistry else "Li-Po 6S"
        cond = state.get("operating_conditions") or ""
        sym = state.get("symptom_category", "BATTERY_LOW_RANGE")

        if sym == "MOTOR_STALL_OR_NOISE":
            queries = [
                f"{motor} DC motor gearbox stall load current overheat {cond}".strip(),
                f"{bms} overcurrent protection peak start {motor}".strip(),
            ]
        else:
            queries = [
                f"{bms} {chem} low temperature discharge cutoff voltage sag {cond}".strip(),
                f"{motor} {chem} cold weather battery range drop voltage sag".strip(),
            ]
        return {"queries": queries[:3]}

    async def fetch_web_node(state: WebResearchState) -> dict[str, Any]:
        snippets: list[str] = []
        status = "ok"
        for q in state.get("queries", []):
            try:
                if asyncio.iscoroutinefunction(runner):
                    results = await runner(q, 4)
                else:
                    results = await asyncio.to_thread(runner, q, 4)
                if hasattr(results, "__await__"):
                    results = await results
                for r in results[:4]:
                    title = str(r.get("title", "")).strip()
                    href = str(r.get("href") or r.get("link") or "").strip()
                    body = str(r.get("body") or r.get("snippet") or "").strip()
                    if title or body:
                        snippets.append(f"[{q}] {title} | URL: {href}\n{body[:350]}")
            except Exception as exc:
                status = f"fallback:{type(exc).__name__}"

        if not snippets:
            status = "fallback_offline"
            snippets.append(
                "DuckDuckGo недоступен или вернул пустую выдачу. "
                "Используй инженерные физические характеристики BOM (Li-Po 6S, JK-BMS-6S, RS550-24V) "
                "и явно отметь в источниках, что применена резервная инженерная справка (offline fallback)."
            )
        return {"raw_snippets": snippets, "search_status": status}

    async def synthesize_hypotheses_node(state: WebResearchState) -> dict[str, Any]:
        bom = state.get("bom_summary")
        bom_txt = bom.model_dump_json() if bom else "{}"
        cond = state.get("operating_conditions") or "не указаны"
        snippets_txt = "\n---\n".join(state.get("raw_snippets", []))

        sys_prompt = (
            "Ты — инженер внешней технической диагностики (Context Hygiene Subgraph).\n"
            "Внутренняя БД производства 100% ЧИСТАЯ: все замеры и партии в норме.\n"
            "На основе BOM-компонентов, условий эксплуатации и результатов поиска сформируй "
            "2-3 компактные, ранжированные инженерные гипотезы (ExternalHypothesis). "
            "Не выдумывай URL-ы, которых нет в результатах поиска; если поиск вернул fallback, честно укажи это."
        )
        usr_prompt = (
            f"Категория симптома: {state.get('symptom_category')}\n"
            f"Условия эксплуатации: {cond}\n"
            f"BOM из БД: {bom_txt}\n"
            f"Статус поиска: {state.get('search_status')}\n\n"
            f"Сырые веб-фрагменты:\n{snippets_txt}"
        )
        parsed, usage = await invoke_structured_with_usage(
            fast_model,
            WebHypothesesOutput,
            [SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)],
            role="FAST",
            node_name="web_researcher_node",
            model_name=fast_model_name,
        )
        return {
            "external_hypotheses": parsed.hypotheses[:3],
            "token_usage": [usage],
        }

    sg = StateGraph(WebResearchState)
    sg.add_node("plan_queries", plan_queries_node)
    sg.add_node("fetch_web", fetch_web_node)
    sg.add_node("synthesize_hypotheses", synthesize_hypotheses_node)
    sg.add_edge(START, "plan_queries")
    sg.add_edge("plan_queries", "fetch_web")
    sg.add_edge("fetch_web", "synthesize_hypotheses")
    sg.add_edge("synthesize_hypotheses", END)
    return sg.compile()
