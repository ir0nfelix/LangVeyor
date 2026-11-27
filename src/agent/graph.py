from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.agent.llm import (
    AgentModelBundle,
    extract_usage_from_ai_message,
    invoke_structured_with_usage,
)
from src.agent.state import (
    ActionResult,
    AgentState,
    AuditVerdict,
    ComponentBomSummary,
    FinalVerdict,
    IntakeResult,
    InvestigationDraft,
    ProposedAction,
    RetrievedChunkModel,
    ToolEvidence,
)
from src.agent.tools import build_domain_tools
from src.agent.web_research import SearchCallable, build_web_research_subgraph
from src.config import Settings, get_settings
from src.db.enums import RecallNoticeStatus, RecallNoticeType, SupportTicketStatus
from src.db.models import Instrument, RecallNotice, RouteSheet, SupportTicket
from src.rag.bpmn_parser import BpmnWorkflowParser
from src.rag.hybrid_retriever import HybridRetriever

SERIAL_RE = re.compile(r"\b(CAR-\d{4}-\d{4})\b", re.IGNORECASE)

REQUIRED_CLEAN_DB_TOOLS: tuple[str, ...] = (
    "get_item_genealogy",
    "get_item_history",
    "get_lot_quality_stats",
    "get_instrument_channel_anomalies",
    "get_operator_anomalies",
)


def _is_item_missing_in_evidence(evidence: list[ToolEvidence]) -> bool:
    """Return True if the latest get_item_genealogy tool result explicitly reported found=False."""
    for ev in reversed(evidence):
        if ev.tool_name == "get_item_genealogy":
            try:
                data = json.loads(ev.result_json)
            except Exception:
                return False
            if isinstance(data, dict):
                return data.get("found") is False
            return False
    return False


ALLOWED_CHECKPOINT_MODULES = [
    ("src.agent.state", cls_name)
    for cls_name in (
        "ActionResult",
        "AuditVerdict",
        "ComponentBomSummary",
        "ExternalHypothesis",
        "FinalVerdict",
        "IntakeResult",
        "InvestigationDraft",
        "LLMUsageRecord",
        "ProposedAction",
        "RetrievedChunkModel",
        "ToolEvidence",
    )
]


@asynccontextmanager
async def open_postgres_checkpointer(
    checkpointer_dsn: str,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Initialize AsyncPostgresSaver with AsyncConnectionPool(autocommit=True) and run setup()."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=checkpointer_dsn,
        min_size=1,
        max_size=5,
        connection_class=AsyncConnection[DictRow],
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_CHECKPOINT_MODULES)
        saver = AsyncPostgresSaver(pool, serde=serde)
        await saver.setup()
        yield saver
    finally:
        await pool.close()


def _extract_bom_and_cars_from_evidence(
    evidence_log: list[ToolEvidence],
) -> tuple[ComponentBomSummary | None, list[str]]:
    bom: ComponentBomSummary | None = None
    affected_cars: list[str] = []
    for ev in evidence_log:
        try:
            data = json.loads(ev.result_json)
        except Exception:
            continue
        if ev.tool_name == "get_item_genealogy" and isinstance(data, dict) and data.get("found"):
            specs = data.get("bom_specs") or {}
            bat_comp = data.get("battery_component") or {}
            lots = data.get("root_lots") or []
            lot_code = lots[0].get("tracking_code") if lots else None
            bom = ComponentBomSummary(
                motor=specs.get("motor_compat") or specs.get("motor"),
                bms=specs.get("bms"),
                chemistry=specs.get("chemistry"),
                battery_tracking_code=bat_comp.get("tracking_code"),
                root_lot_code=lot_code,
            )
        elif (
            ev.tool_name == "find_affected_products_by_lot"
            and isinstance(data, dict)
            and data.get("found")
        ):
            cars = data.get("affected_car_codes") or []
            if isinstance(cars, list):
                affected_cars = [str(c) for c in cars]
    return bom, affected_cars


def build_rca_graph(
    *,
    models: AgentModelBundle,
    session_factory: async_sessionmaker[AsyncSession],
    bpmn_parser: BpmnWorkflowParser,
    retriever: HybridRetriever,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    settings: Settings | None = None,
    web_search_fn: SearchCallable | None = None,
    on_node_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Any:
    """Build and compile the 7-node LangGraph MES RCA StateGraph."""
    cfg = settings or get_settings()
    domain_tools = build_domain_tools(session_factory, retriever, cfg)
    tool_node = ToolNode(domain_tools, handle_tool_errors=True)
    web_subgraph = build_web_research_subgraph(
        models.fast_model, models.fast_model_name, search_fn=web_search_fn
    )

    def _emit(event: str, payload: dict[str, Any]) -> None:
        if on_node_event is not None:
            on_node_event(event, payload)

    def _effective_max_tool_steps(state: AgentState) -> int:
        revisions = max(0, int(state.get("revision_count", 0)))
        hard_cap = cfg.max_tool_steps + cfg.max_revisions * 3
        return min(cfg.max_tool_steps + revisions * 3, hard_cap)

    # -------------------------------------------------------------------------
    # Node 1: intake_node (FAST)
    # -------------------------------------------------------------------------
    async def intake_node(state: AgentState) -> dict[str, Any]:
        _emit("node_start", {"node": "intake_node", "model": models.fast_model_name})
        human_parts: list[str] = []
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                txt = (
                    msg.content
                    if isinstance(msg.content, str)
                    else json.dumps(msg.content, ensure_ascii=False)
                ).strip()
                if txt and txt not in human_parts:
                    human_parts.append(txt)

        raw_uq = state.get("user_query", "").strip()
        had_human_msgs = bool(human_parts)
        if raw_uq and raw_uq not in human_parts:
            human_parts.append(raw_uq)
        query = "\n".join(human_parts).strip()

        sys_prompt = (
            "Ты — оператор приёма обращений поддержки парка аттракционов (MES RCA Intake).\n"
            "Извлеки из сообщения:\n"
            "1. tracking_code: серийный номер машинки формата CAR-YYYY-XXXX (или None, если не указан).\n"
            "2. symptom_category: строго одно из BATTERY_LOW_RANGE, MOTOR_STALL_OR_NOISE, NO_POWER_DEAD, OTHER.\n"
            "3. operating_conditions: температура, горки, нагрузка (или None).\n"
            "Если серийный номер НЕ указан, поставь needs_clarification=True и напиши вежливый запрос уточнения."
        )
        parsed, usage = await invoke_structured_with_usage(
            models.fast_model,
            IntakeResult,
            [SystemMessage(content=sys_prompt), HumanMessage(content=query)],
            role="FAST",
            node_name="intake_node",
            model_name=models.fast_model_name,
        )
        # Regex safety check for serial number
        if not parsed.tracking_code:
            m = SERIAL_RE.search(query)
            if m:
                parsed.tracking_code = m.group(1).upper()
                parsed.needs_clarification = False
                parsed.clarification_message = None
        else:
            parsed.tracking_code = parsed.tracking_code.strip().upper()

        if not parsed.tracking_code:
            parsed.needs_clarification = True
            if not parsed.clarification_message:
                parsed.clarification_message = (
                    "Пожалуйста, уточните серийный номер электромашинки (например, CAR-2026-0412), "
                    "чтобы я смог проверить её родословную и замеры в MES."
                )

        _emit("node_end", {"node": "intake_node", "intake": parsed.model_dump()})
        out: dict[str, Any] = {
            "user_query": query,
            "intake": parsed,
            "token_usage": [usage],
        }
        if (not had_human_msgs) and raw_uq:
            out["messages"] = [HumanMessage(content=raw_uq)]
        return out

    # -------------------------------------------------------------------------
    # Node 2: triage_node (0 LLM calls: BPMN + Hybrid RAG)
    # -------------------------------------------------------------------------
    async def triage_node(state: AgentState) -> dict[str, Any]:
        _emit("node_start", {"node": "triage_node", "model": "deterministic"})
        intake = state.get("intake")
        cat = intake.symptom_category if intake else "OTHER"
        guidance = bpmn_parser.get_symptom_branch(cat)
        rag_query = f"{cat} {state.get('user_query', '')}"
        chunks = retriever.search(rag_query, top_k=cfg.rag_top_k)
        rag_models = [
            RetrievedChunkModel(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                doc_title=c.doc_title,
                section_title=c.section_title,
                content=c.content,
                rrf_score=c.rrf_score,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
            )
            for c in chunks
        ]
        _emit(
            "node_end",
            {
                "node": "triage_node",
                "required_tools": guidance.required_tools,
                "rag_chunks": len(rag_models),
            },
        )
        return {
            "bpmn_guidance": guidance.summary_text,
            "required_tools": guidance.required_tools,
            "rag_context": rag_models,
        }

    # -------------------------------------------------------------------------
    # Node 3: investigator_node (INVESTIGATOR + bind_tools loop)
    # -------------------------------------------------------------------------
    async def investigator_node(state: AgentState) -> dict[str, Any]:
        _emit(
            "node_start",
            {
                "node": "investigator_node",
                "model": models.investigator_model_name,
                "tool_steps": state.get("tool_steps", 0),
                "revision_count": state.get("revision_count", 0),
            },
        )
        intake = state.get("intake")
        serial = intake.tracking_code if intake else ""
        rag_txt = "\n\n".join(
            f"[{c.source_file} | {c.section_title}]\n{c.content}"
            for c in state.get("rag_context", [])
        )
        evidence = state.get("evidence_log", [])
        evidence_txt = (
            "\n".join(f"- {ev.tool_name}({ev.args_summary}) -> {ev.result_json}" for ev in evidence)
            if evidence
            else "Пока нет вызовов тулов."
        )
        audit = state.get("audit_result")
        feedback_txt = ""
        if audit and audit.decision == "REJECTED":
            feedback_txt = (
                f"\n\nЗАМЕЧАНИЯ РЕВИЗОРА (ОБЯЗАТЕЛЬНО ИСПРАВИТЬ):\n{audit.feedback}\n"
                f"Пропущенные проверки: {audit.missing_checks}\n"
                f"Галлюцинации/неточности: {audit.hallucinated_claims}"
            )

        sys_prompt = (
            "Ты — ведущий следователь первопричин (Investigator Agent) в MES.\n"
            "Расследуй обращение строго по ветке BPMN и инженерным регламентам SOP, используя типизированные SQL-тулы.\n"
            "НИКОГДА не выдумывай цифры, серийники или даты. Опирайся только на evidence_log.\n"
            "Обязательно проверь:\n"
            "1) get_item_genealogy(tracking_code) и get_item_history(route_sheet_id);\n"
            "2) get_lot_quality_stats(lot_code) для корневой партии АКБ; если exceeds_10pct_recall_threshold=true (брак > 10% или HOLD) -> ОБЯЗАТЕЛЬНО вызови find_affected_products_by_lot(lot_code); если exceeds_10pct_recall_threshold=false (брак партии <= 10% и hold_reason=null) -> НИКОГДА не вызывай find_affected_products_by_lot;\n"
            "3) если в истории АКБ или машинки есть REWORK, пограничные замеры или просрочка поверки стенда -> вызови get_instrument_channel_anomalies и/или get_operator_anomalies (дату бери из completed_at в YYYY-MM-DD).\n"
            "4) Регламенты SOP (RAG) уже загружены в контекст выше — вызывай search_engineering_kb только при явной нехватке данных.\n"
            "5) При возврате от Ревизора с missing_checks обязательно вызови перечисленные недостающие тулы, взяв параметры "
            "(instrument_code, channel_no, operator_code, completed_at в формате YYYY-MM-DD) из уже полученного get_item_history.\n"
            "Для ускорения вызывай независимые SQL-тулы параллельно за один шаг. Когда все нужные данные собраны, прекрати вызывать тулы и дай короткий текстовый итог."
        )

        # Build strictly valid message sequence for the current investigation pass:
        # Collect only the trailing (AIMessage(tool_calls), ToolMessage...) pairs if mid-tool-loop.
        state_msgs = state.get("messages", [])
        active_loop_msgs: list[Any] = []
        if state_msgs and isinstance(state_msgs[-1], ToolMessage):
            # Walk backwards until we hit a non-(AIMessage/ToolMessage) or an AIMessage without tool_calls
            collected: list[Any] = []
            for m in reversed(state_msgs):
                if isinstance(m, ToolMessage):
                    collected.append(m)
                elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    collected.append(m)
                else:
                    break
            active_loop_msgs = list(reversed(collected))

        # Avoid duplicating raw ToolMessage JSONs inside context_msg while active_loop_msgs already carry them
        active_tool_count = sum(1 for m in active_loop_msgs if isinstance(m, ToolMessage))
        prior_evidence = evidence[:-active_tool_count] if active_tool_count > 0 else evidence
        prior_evidence_txt = (
            "\n".join(
                f"- {ev.tool_name}({ev.args_summary}) -> {ev.result_json}" for ev in prior_evidence
            )
            if prior_evidence
            else (
                "Результаты тулов текущей итерации переданы в ToolMessage ниже."
                if active_tool_count > 0
                else "Пока нет вызовов тулов."
            )
        )

        context_msg = (
            f"Обращение: {state.get('user_query')}\n"
            f"Серийник: {serial} | Категория: {intake.symptom_category if intake else 'OTHER'} | "
            f"Условия: {intake.operating_conditions if intake else None}\n\n"
            f"{state.get('bpmn_guidance', '')}\n\n"
            f"Регламенты SOP (RAG):\n{rag_txt}\n\n"
            f"Уже собранные факты (evidence_log):\n{prior_evidence_txt}"
            f"{feedback_txt}"
        )

        loop_msgs: list[Any] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=context_msg),
            *active_loop_msgs,
        ]

        tool_steps = state.get("tool_steps", 0)
        usage_records = []

        if tool_steps < _effective_max_tool_steps(state):
            t0 = time.perf_counter()
            bound = models.investigator_model.bind_tools(domain_tools)
            ai_msg: Any = None
            last_timeout: TimeoutError | None = None
            for _attempt in range(2):
                try:
                    ai_msg = await asyncio.wait_for(bound.ainvoke(loop_msgs), timeout=85.0)
                    last_timeout = None
                    break
                except TimeoutError as exc:
                    last_timeout = exc
            if last_timeout is not None:
                raise last_timeout
            lat = (time.perf_counter() - t0) * 1000.0
            if isinstance(ai_msg, AIMessage):
                usage_records.append(
                    extract_usage_from_ai_message(
                        ai_msg,
                        role="INVESTIGATOR",
                        node_name="investigator_node_tools",
                        model_name=models.investigator_model_name,
                        latency_ms=lat,
                    )
                )
                if getattr(ai_msg, "tool_calls", None):
                    _emit(
                        "tool_calls_planned",
                        {"calls": [tc.get("name") for tc in ai_msg.tool_calls]},
                    )
                    return {
                        "messages": [ai_msg],
                        "token_usage": usage_records,
                    }

        # Synthesize structured InvestigationDraft from verified evidence_log
        draft_prompt = (
            "На основе ТОЛЬКО подтверждённого evidence_log ниже сформируй структурированный InvestigationDraft.\n"
            "Правила формирования:\n"
            "1) Строго перечисляй только точные цифры из evidence_log.\n"
            "2) В cited_db_facts и root_cause_hypothesis обязательно указывай ключевые сущности из evidence_log (если присутствуют): "
            "код партии и наименование поставщика (supplier_name), код стенда + номер канала + дату поверки (calibration_due_at), "
            "ФИО/код оператора (OP-XX) и модели компонентов BOM (RS550-24V, JK-BMS-6S).\n"
            "3) В recommended_actions: для любого первичного обращения всегда создавай CREATE_SUPPORT_TICKET на целевую машинку "
            "(или ESCALATE_REPEAT_TICKET при наличии прошлого тикета), а системные меры CREATE_LOT_RECALL (при браке партии > 10% или HOLD) "
            "и/или CREATE_CALIBRATION_HOLD (при дрейфе/просрочке стенда) добавляй ДОПОЛНИТЕЛЬНО, НЕ заменяя клиентский тикет.\n"
            "4) Если get_item_genealogy вернул found=false, ОБЯЗАТЕЛЬНО ставь root_cause_type='INCONCLUSIVE', db_anomalies_found=False и укажи, что серийный номер отсутствует в MES.\n"
            "5) Если изделие НАЙДЕНО (found=true), ВСЕ проверки БД выполнены и ни одной внутренней аномалии нет (брак партии <= 10%, "
            "нет REWORK, поверки в норме, оператор и каналы стендов в норме, нет прошлых тикетов) — "
            "поставь db_anomalies_found=False и root_cause_type='DB_CLEAN_EXTERNAL_SUSPECTED'.\n\n"
            f"Запрос: {state.get('user_query')}\n"
            f"BPMN:\n{state.get('bpmn_guidance', '')}\n\n"
            f"evidence_log:\n{evidence_txt}"
            f"{feedback_txt}"
        )
        draft, draft_usage = await invoke_structured_with_usage(
            models.investigator_model,
            InvestigationDraft,
            [
                SystemMessage(content="Формируй точный JSON-черновик расследования."),
                HumanMessage(content=draft_prompt),
            ],
            role="INVESTIGATOR",
            node_name="investigator_node_draft",
            model_name=models.investigator_model_name,
        )
        usage_records.append(draft_usage)

        if _is_item_missing_in_evidence(evidence):
            draft.root_cause_type = "INCONCLUSIVE"
            draft.db_anomalies_found = False

        bom, affected_cars = _extract_bom_and_cars_from_evidence(evidence)
        if affected_cars and not draft.affected_cars:
            draft.affected_cars = affected_cars

        _emit(
            "node_end",
            {
                "node": "investigator_node",
                "root_cause_type": draft.root_cause_type,
                "db_anomalies_found": draft.db_anomalies_found,
            },
        )
        return {
            "messages": [AIMessage(content=f"InvestigationDraft ready: {draft.root_cause_type}")],
            "investigator_draft": draft,
            "bom_summary": bom or state.get("bom_summary"),
            "token_usage": usage_records,
        }

    # -------------------------------------------------------------------------
    # Tool execution wrapper node (records raw ToolEvidence into evidence_log!)
    # -------------------------------------------------------------------------
    async def tools_node(state: AgentState) -> dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) if isinstance(last_msg, AIMessage) else []
        t0 = time.perf_counter()
        result = await tool_node.ainvoke({"messages": state["messages"]})
        total_ms = (time.perf_counter() - t0) * 1000.0

        out_messages: list[ToolMessage] = result.get("messages", [])
        evidences: list[ToolEvidence] = []
        per_call_ms = round(total_ms / max(1, len(out_messages)), 2)

        for idx, tmsg in enumerate(out_messages):
            tc = tool_calls[idx] if idx < len(tool_calls) else {}
            tname = str(getattr(tmsg, "name", "") or tc.get("name", "unknown"))
            args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
            content_str = (
                tmsg.content
                if isinstance(tmsg.content, str)
                else json.dumps(tmsg.content, ensure_ascii=False)
            )
            evidences.append(
                ToolEvidence(
                    tool_name=tname,
                    args_summary=args_str,
                    result_json=content_str,
                    latency_ms=per_call_ms,
                )
            )
            _emit("tool_result", {"tool": tname, "args": args_str, "latency_ms": per_call_ms})

        all_evidence = list(state.get("evidence_log", [])) + evidences
        bom, _ = _extract_bom_and_cars_from_evidence(all_evidence)

        return {
            "messages": out_messages,
            "evidence_log": evidences,
            "tool_steps": state.get("tool_steps", 0) + 1,
            "bom_summary": bom or state.get("bom_summary"),
        }

    # -------------------------------------------------------------------------
    # Node 4: auditor_node (AUDITOR — different model family! Reflection pattern)
    # -------------------------------------------------------------------------
    async def auditor_node(state: AgentState) -> dict[str, Any]:
        _emit(
            "node_start",
            {
                "node": "auditor_node",
                "model": models.auditor_model_name,
                "revision_count": state.get("revision_count", 0),
            },
        )
        draft = state.get("investigator_draft")
        evidence = state.get("evidence_log", [])

        # Guardrail #1: missing serial number in MES (found=false) must never route to web search
        if _is_item_missing_in_evidence(evidence):
            verdict = AuditVerdict(
                decision="APPROVED",
                feedback="Изделие отсутствует в MES (found=false): веб-поиск запрещён детерминированным guardrail.",
                db_is_clean=False,
            )
            _emit(
                "node_end",
                {
                    "node": "auditor_node",
                    "decision": verdict.decision,
                    "feedback": verdict.feedback,
                },
            )
            return {
                "audit_result": verdict,
                "revision_count": state.get("revision_count", 0),
                "token_usage": [],
            }

        evidence_txt = "\n".join(
            f"- {ev.tool_name}({ev.args_summary}) -> {ev.result_json}" for ev in evidence
        )
        draft_txt = draft.model_dump_json(indent=2) if draft else "{}"

        sys_prompt = (
            "Ты — независимый Ревизор качества расследования (Auditor Agent).\n"
            "Сверь черновик Следователя (investigator_draft) с сырыми ответами SQL-тулов (evidence_log) и BPMN.\n"
            "Проверь строго 4 вещи:\n"
            "1. Нет ли выдуманных чисел (hallucination check): все числа/коды в cited_db_facts должны присутствовать в evidence_log.\n"
            "2. Если get_item_genealogy вернул found=false -> НЕ отправляй в интернет: ставь decision='APPROVED', db_is_clean=False.\n"
            "3. Выполнены ли все обязательные проверки по BPMN:\n"
            "   - get_item_genealogy + get_item_history вызваны?\n"
            "   - Если брак в партии АКБ > 10% -> вызван ли find_affected_products_by_lot для списка отзывных машинок?\n"
            "   - Если на АКБ/машинке есть REWORK на стенде/канале или у оператора -> вызваны ли get_instrument_channel_anomalies / get_operator_anomalies?\n"
            "   - Если есть прошлый support_ticket -> отражена ли эскалация?\n"
            "4. decision='NEED_WEB_RESEARCH' (db_is_clean=True) допустим ТОЛЬКО если в evidence_log присутствуют ВСЕ 5 проверок: "
            "get_item_genealogy, get_item_history, get_lot_quality_stats, get_instrument_channel_anomalies, get_operator_anomalies и аномалий нет.\n"
            "Если есть пропуски или галлюцинации -> decision='REJECTED' с конкретным feedback.\n"
            "Если всё доказано фактами БД -> decision='APPROVED'."
        )
        usr_prompt = (
            f"BPMN-регламент:\n{state.get('bpmn_guidance', '')}\n\n"
            f"Сырой evidence_log (единственный источник правды о БД):\n{evidence_txt}\n\n"
            f"Черновик Следователя:\n{draft_txt}\n\n"
            f"Текущий revision_count: {state.get('revision_count', 0)} (максимум {cfg.max_revisions})."
        )
        verdict, usage = await invoke_structured_with_usage(
            models.auditor_model,
            AuditVerdict,
            [SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)],
            role="AUDITOR",
            node_name="auditor_node",
            model_name=models.auditor_model_name,
        )

        rev = state.get("revision_count", 0)
        called_tools = {ev.tool_name for ev in evidence}
        missing_clean_checks = [t for t in REQUIRED_CLEAN_DB_TOOLS if t not in called_tools]

        # Deterministic clean-DB invariant: never finish APPROVED without web research if both
        # Investigator and Auditor agree the MES DB has zero anomalies
        if (
            verdict.decision == "APPROVED"
            and verdict.db_is_clean
            and draft is not None
            and not draft.db_anomalies_found
        ):
            verdict.decision = "NEED_WEB_RESEARCH"

        # Guardrail #2: enforce full 5-tool DB sweep before allowing NEED_WEB_RESEARCH
        if (
            verdict.decision == "NEED_WEB_RESEARCH"
            and missing_clean_checks
            and rev < cfg.max_revisions
        ):
            verdict.decision = "REJECTED"
            verdict.db_is_clean = False
            verdict.missing_checks = missing_clean_checks
            verdict.feedback = (
                "Перед признанием БД чистой и выходом в веб-поиск необходимо выполнить все 5 проверок БД. "
                f"Недостающие тулы: {', '.join(missing_clean_checks)}. "
                "Вызови их, взяв параметры (instrument_code, channel_no, operator_code, "
                "completed_at в формате YYYY-MM-DD) из уже полученного get_item_history."
            )

        if verdict.decision == "REJECTED" and rev >= cfg.max_revisions:
            verdict.decision = "NEED_WEB_RESEARCH" if verdict.db_is_clean else "APPROVED"
            verdict.feedback += (
                " [Достигнут лимит ревизий max_revisions=2; передача на следующий этап.]"
            )
        elif verdict.decision == "REJECTED":
            rev += 1

        _emit(
            "node_end",
            {
                "node": "auditor_node",
                "decision": verdict.decision,
                "feedback": verdict.feedback,
            },
        )
        return {
            "audit_result": verdict,
            "revision_count": rev,
            "token_usage": [usage],
        }

    # -------------------------------------------------------------------------
    # Node 5: web_researcher_node (Isolated Context Hygiene Subgraph)
    # -------------------------------------------------------------------------
    async def web_researcher_node(state: AgentState) -> dict[str, Any]:
        _emit("node_start", {"node": "web_researcher_node", "model": models.fast_model_name})
        intake = state.get("intake")
        sub_out = await web_subgraph.ainvoke(
            {
                "symptom_category": intake.symptom_category if intake else "OTHER",
                "operating_conditions": intake.operating_conditions if intake else None,
                "bom_summary": state.get("bom_summary"),
                "queries": [],
                "raw_snippets": [],
                "search_status": "",
                "external_hypotheses": [],
                "token_usage": [],
            }
        )
        hyps = sub_out.get("external_hypotheses", [])
        usages = sub_out.get("token_usage", [])
        _emit("node_end", {"node": "web_researcher_node", "hypotheses_count": len(hyps)})
        return {
            "external_hypotheses": hyps,
            "token_usage": usages,
        }

    # -------------------------------------------------------------------------
    # Node 6: verdict_node (INVESTIGATOR — separates DB facts & external hypotheses)
    # -------------------------------------------------------------------------
    async def verdict_node(state: AgentState) -> dict[str, Any]:
        _emit("node_start", {"node": "verdict_node", "model": models.investigator_model_name})
        draft = state.get("investigator_draft")
        ext_hyps = state.get("external_hypotheses", [])
        evidence = state.get("evidence_log", [])
        _, verified_cars = _extract_bom_and_cars_from_evidence(evidence)

        evidence_txt = "\n".join(
            f"- {ev.tool_name}({ev.args_summary}) -> {ev.result_json}" for ev in evidence
        )
        ext_txt = (
            json.dumps([h.model_dump() for h in ext_hyps], ensure_ascii=False, indent=2)
            if ext_hyps
            else "[]"
        )

        sys_prompt = (
            "Ты формируешь финальный отчёт RCA (FinalVerdict) для оператора.\n"
            "КРИТИЧНО:\n"
            "1. В confirmed_db_facts и summary включай ТОЛЬКО факты, подтверждённые в evidence_log, и обязательно указывай "
            "ключевые сущности (если присутствуют в evidence_log): код партии и наименование поставщика (supplier_name), "
            "код стенда + номер канала + calibration_due_at, ФИО/код оператора и модели BOM (RS550-24V, JK-BMS-6S).\n"
            "2. В external_hypotheses перенеси ТОЛЬКО гипотезы из переданного списка external_hypotheses (не смешивай с фактами БД!).\n"
            "3. В proposed_actions для любого первичного обращения ВСЕГДА создавай CREATE_SUPPORT_TICKET на целевую машинку "
            "(или ESCALATE_REPEAT_TICKET при наличии прошлого тикета), а системные меры добавляй ТОЛЬКО при подтверждённом основании: "
            "CREATE_LOT_RECALL — ТОЛЬКО если по get_lot_quality_stats exceeds_10pct_recall_threshold=true (брак партии > 10% или HOLD; "
            "высокий rework_rate_pct при defect_rate_pct <= 10% и сбое стенда/оператора НЕ является поводом для CREATE_LOT_RECALL!), "
            "CREATE_CALIBRATION_HOLD — ТОЛЬКО если is_channel_anomalous=true или calibration_expired_in_window=true."
        )
        usr_prompt = (
            f"Обращение: {state.get('user_query')}\n"
            f"Черновик: {draft.model_dump_json() if draft else '{}'}\n"
            f"Верифицированные affected_cars из БД: {verified_cars}\n"
            f"External hypotheses из web_researcher_node:\n{ext_txt}\n\n"
            f"evidence_log:\n{evidence_txt}"
        )
        verdict, usage = await invoke_structured_with_usage(
            models.investigator_model,
            FinalVerdict,
            [SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)],
            role="INVESTIGATOR",
            node_name="verdict_node",
            model_name=models.investigator_model_name,
        )

        # Enforce deterministic separation & verified recall list
        verdict.external_hypotheses = list(ext_hyps)
        if verified_cars and not verdict.affected_cars:
            verdict.affected_cars = verified_cars
        if not verdict.confirmed_db_facts and draft:
            verdict.confirmed_db_facts = list(draft.cited_db_facts)

        _emit(
            "node_end",
            {
                "node": "verdict_node",
                "severity": verdict.severity,
                "actions_count": len(verdict.proposed_actions),
            },
        )
        return {
            "confirmed_db_facts": verdict.confirmed_db_facts,
            "final_verdict": verdict,
            "token_usage": [usage],
        }

    # -------------------------------------------------------------------------
    # Node 7: action_node (HITL interrupt() -> single PostgreSQL ACID transaction)
    # CRITICAL RULES:
    # - NEVER wrap interrupt() in try/except
    # - ZERO DB writes before interrupt()
    # - Pass pure JSON-serializable payload to interrupt()
    # -------------------------------------------------------------------------
    async def action_node(state: AgentState) -> dict[str, Any]:
        _emit("node_start", {"node": "action_node", "phase": "prepare_interrupt"})
        verdict = state.get("final_verdict")
        intake = state.get("intake")
        tracking_code = intake.tracking_code if intake and intake.tracking_code else "UNKNOWN"
        symptom_cat = intake.symptom_category if intake else "OTHER"

        actions: list[ProposedAction] = (
            list(verdict.proposed_actions) if verdict and verdict.proposed_actions else []
        )
        if not actions and verdict is not None:
            actions = [
                ProposedAction(
                    action_type="CREATE_SUPPORT_TICKET",
                    title=f"Обращение по {tracking_code}",
                    description=verdict.summary,
                    priority="HIGH" if verdict.severity in {"HIGH", "CRITICAL"} else "MEDIUM",
                    target_tracking_code=tracking_code,
                    affected_cars=list(verdict.affected_cars),
                )
            ]

        hitl_payload = {
            "tracking_code": tracking_code,
            "symptom_category": symptom_cat,
            "severity": verdict.severity if verdict else "MEDIUM",
            "summary": verdict.summary if verdict else "",
            "affected_cars": list(verdict.affected_cars) if verdict else [],
            "proposed_actions": [a.model_dump() for a in actions],
            "prompt": "Подтвердите выполнение предложенных записей в БД MES (support_tickets / recall_notices).",
        }

        # HITL pause — MUST NOT be wrapped in try/except, NO DB side effects before this line!
        human_decision = interrupt(hitl_payload)

        approved = bool(
            human_decision.get("approved", False)
            if isinstance(human_decision, dict)
            else human_decision
        )
        operator_comment = (
            str(human_decision.get("comment") or "").strip() or None
            if isinstance(human_decision, dict)
            else None
        )

        if not approved:
            res = ActionResult(
                approved=False,
                operator_comment=operator_comment,
                message="Оператор отклонил запись действий в БД MES.",
            )
            _emit("node_end", {"node": "action_node", "approved": False})
            return {"action_result": res}

        created_tickets: list[str] = []
        created_recalls: list[str] = []
        thread_id = state.get("thread_id", "")

        async with session_factory() as session:
            async with session.begin():
                rs = await session.scalar(
                    select(RouteSheet).where(RouteSheet.tracking_code == tracking_code)
                )
                rs_id: UUID | None = rs.id if rs else None

                ticket_actions = [
                    a
                    for a in actions
                    if a.action_type in {"CREATE_SUPPORT_TICKET", "ESCALATE_REPEAT_TICKET"}
                ]
                if not ticket_actions and rs_id is not None:
                    ticket_actions = [
                        ProposedAction(
                            action_type="CREATE_SUPPORT_TICKET",
                            title=f"RCA-тикет по {tracking_code}",
                            description=verdict.summary if verdict else "",
                            priority="HIGH"
                            if verdict and verdict.severity in {"HIGH", "CRITICAL"}
                            else "MEDIUM",
                            target_tracking_code=tracking_code,
                        )
                    ]

                for act in ticket_actions:
                    if rs_id is None:
                        continue
                    is_esc = act.action_type == "ESCALATE_REPEAT_TICKET"
                    ticket = SupportTicket(
                        route_sheet_id=rs_id,
                        tracking_code=act.target_tracking_code or tracking_code,
                        title=act.title[:255],
                        symptom_category=symptom_cat,
                        root_cause_summary=verdict.summary if verdict else act.description,
                        recommended_action=act.description,
                        status=SupportTicketStatus.ESCALATED.value
                        if is_esc
                        else SupportTicketStatus.OPEN.value,
                        priority="CRITICAL" if is_esc else act.priority,
                        operator_comment=operator_comment,
                        thread_id=thread_id,
                    )
                    session.add(ticket)
                    await session.flush()
                    created_tickets.append(str(ticket.id))

                recall_actions = [
                    a
                    for a in actions
                    if a.action_type in {"CREATE_LOT_RECALL", "CREATE_CALIBRATION_HOLD"}
                ]
                for act in recall_actions:
                    lot_id: UUID | None = None
                    instr_id: UUID | None = None
                    if act.target_lot_code:
                        lot_rs = await session.scalar(
                            select(RouteSheet).where(
                                RouteSheet.tracking_code == act.target_lot_code.strip().upper()
                            )
                        )
                        if lot_rs is not None:
                            lot_id = lot_rs.id
                    if act.target_instrument_code:
                        instr = await session.scalar(
                            select(Instrument).where(
                                Instrument.code == act.target_instrument_code.strip().upper()
                            )
                        )
                        if instr is not None:
                            instr_id = instr.id

                    cars_list = act.affected_cars or (
                        list(verdict.affected_cars) if verdict else []
                    )
                    ntype = (
                        RecallNoticeType.CALIBRATION_HOLD.value
                        if act.action_type == "CREATE_CALIBRATION_HOLD"
                        else RecallNoticeType.LOT_RECALL.value
                    )
                    notice = RecallNotice(
                        lot_rs_id=lot_id,
                        instrument_id=instr_id,
                        notice_type=ntype,
                        reason=f"{act.title}: {act.description}"
                        + (f" (Комментарий: {operator_comment})" if operator_comment else ""),
                        affected_tracking_codes=cars_list,
                        affected_count=len(cars_list),
                        status=RecallNoticeStatus.ACTIVE.value,
                        approved_by="operator",
                        thread_id=thread_id,
                    )
                    session.add(notice)
                    await session.flush()
                    created_recalls.append(str(notice.id))

        res = ActionResult(
            approved=True,
            operator_comment=operator_comment,
            created_ticket_ids=created_tickets,
            created_recall_ids=created_recalls,
            message=(
                f"ACID-транзакция выполнена: создано тикетов={len(created_tickets)}, "
                f"отзывов/блокировок={len(created_recalls)}."
            ),
        )
        _emit(
            "node_end",
            {
                "node": "action_node",
                "approved": True,
                "tickets": created_tickets,
                "recalls": created_recalls,
            },
        )
        return {"action_result": res}

    # -------------------------------------------------------------------------
    # Conditional Edge Routers
    # -------------------------------------------------------------------------
    def route_after_intake(state: AgentState) -> str:
        intake = state.get("intake")
        if intake is None or intake.needs_clarification:
            return "end_clarify"
        return "triage_node"

    def route_after_investigator(state: AgentState) -> str:
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
            if state.get("tool_steps", 0) < _effective_max_tool_steps(state):
                return "tools_node"
        return "auditor_node"

    def route_after_auditor(state: AgentState) -> str:
        audit = state.get("audit_result")
        if audit is None:
            return "verdict_node"
        if audit.decision == "REJECTED":
            return "investigator_node"
        if audit.decision == "NEED_WEB_RESEARCH":
            return "web_researcher_node"
        return "verdict_node"

    # -------------------------------------------------------------------------
    # Graph Assembly
    # -------------------------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("intake_node", intake_node)
    graph.add_node("triage_node", triage_node)
    graph.add_node("investigator_node", investigator_node)
    graph.add_node("tools_node", tools_node)
    graph.add_node("auditor_node", auditor_node)
    graph.add_node("web_researcher_node", web_researcher_node)
    graph.add_node("verdict_node", verdict_node)
    graph.add_node("action_node", action_node)

    graph.add_edge(START, "intake_node")
    graph.add_conditional_edges(
        "intake_node",
        route_after_intake,
        {"end_clarify": END, "triage_node": "triage_node"},
    )
    graph.add_edge("triage_node", "investigator_node")
    graph.add_conditional_edges(
        "investigator_node",
        route_after_investigator,
        {"tools_node": "tools_node", "auditor_node": "auditor_node"},
    )
    graph.add_edge("tools_node", "investigator_node")
    graph.add_conditional_edges(
        "auditor_node",
        route_after_auditor,
        {
            "investigator_node": "investigator_node",
            "web_researcher_node": "web_researcher_node",
            "verdict_node": "verdict_node",
        },
    )
    graph.add_edge("web_researcher_node", "verdict_node")
    graph.add_edge("verdict_node", "action_node")
    graph.add_edge("action_node", END)

    return graph.compile(checkpointer=checkpointer)
