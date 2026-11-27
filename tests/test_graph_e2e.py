from __future__ import annotations

import uuid
from typing import Any

from evals.metrics import (
    compute_action_match_score,
    compute_numeric_groundedness,
    compute_root_cause_accuracy,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from src.agent.graph import build_rca_graph, open_postgres_checkpointer
from src.agent.llm import AgentModelBundle, ScriptedChatModel, invoke_structured_with_usage
from src.agent.mock_models import build_mock_model_bundle
from src.agent.state import (
    AuditVerdict,
    FinalVerdict,
    IntakeResult,
    InvestigationDraft,
    ProposedAction,
)
from src.config import Settings
from src.db.models import RecallNotice, SupportTicket
from src.rag.bpmn_parser import BpmnWorkflowParser
from src.rag.hybrid_retriever import HybridRetriever


async def test_graph_e2e_reflection_recall_and_hitl_postgres_commit(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Case 1:
    - CAR-2026-0412 (LOT-BAT-2026-014 20% defect rate).
    - Auditor REJECTS pass 1 (missing find_affected_products_by_lot) -> Investigator loops back -> APPROVED.
    - web_researcher_node is NOT invoked (external_hypotheses == []).
    - Pauses at interrupt() with 0 premature DB writes.
    - Resumes with Command(resume={"approved": True, ...}) -> persists SupportTicket & RecallNotice (12 cars) in PostgreSQL.
    """
    thread_id = f"test-e2e-1-{uuid.uuid4().hex[:8]}"
    models = build_mock_model_bundle(force_auditor_reject_once=True)
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
            )
            config = {"configurable": {"thread_id": thread_id}}
            query = "В парке жалуются на CAR-2026-0412 — батарея не держит заезд."
            init_state = {
                "messages": [HumanMessage(content=query)],
                "thread_id": thread_id,
                "user_query": query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }

            visited_nodes: list[str] = []
            async for event in graph.astream(init_state, config=config, stream_mode="updates"):
                visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert "web_researcher_node" not in visited_nodes
            assert visited_nodes.count("auditor_node") == 2

            # Verify graph is paused at interrupt() and NO writes happened yet
            snap = await graph.aget_state(config)
            assert snap.tasks and any(getattr(t, "interrupts", None) for t in snap.tasks)
            assert snap.values["revision_count"] == 1
            assert snap.values["audit_result"].decision == "APPROVED"
            assert len(snap.values["final_verdict"].affected_cars) == 12
            assert snap.values["external_hypotheses"] == []

            async with session_factory() as s:
                pre_tickets = list(
                    (
                        await s.scalars(
                            select(SupportTicket).where(SupportTicket.thread_id == thread_id)
                        )
                    ).all()
                )
                assert pre_tickets == []

            # Resume HITL with approval
            resumed = await graph.ainvoke(
                Command(resume={"approved": True, "comment": "Запустить отзыв партии 014"}),
                config=config,
            )
            act_res = resumed["action_result"]
            assert act_res.approved is True
            assert len(act_res.created_ticket_ids) == 1
            assert len(act_res.created_recall_ids) == 1

            async with session_factory() as s:
                ticket = await s.scalar(
                    select(SupportTicket).where(SupportTicket.thread_id == thread_id)
                )
                notice = await s.scalar(
                    select(RecallNotice).where(RecallNotice.thread_id == thread_id)
                )
                assert ticket is not None
                assert ticket.tracking_code == "CAR-2026-0412"
                assert notice is not None
                assert notice.affected_count == 12
                assert "CAR-2026-0412" in notice.affected_tracking_codes
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))


async def test_graph_e2e_clean_db_routes_to_web_researcher_and_hitl_reject(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Case 2:
    - CAR-2026-0777 (100% clean DB) -> Auditor returns NEED_WEB_RESEARCH.
    - Isolated web_researcher_node runs (mocked DDG) and populates external_hypotheses separately from confirmed_db_facts.
    - Resuming HITL with approved=False leaves zero DB rows created.
    """
    thread_id = f"test-e2e-2-{uuid.uuid4().hex[:8]}"
    models = build_mock_model_bundle()
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    def fake_ddg(query: str, max_results: int = 4) -> list[dict[str, Any]]:
        _ = (query, max_results)
        return [
            {
                "title": "JK-BMS-6S Low Temperature Discharge Cutoff Guide",
                "href": "https://example.org/jk-bms-6s-cold-cutoff",
                "body": "JK-BMS triggers discharge protection below 0C and under peak RS550-24V voltage sag.",
            }
        ]

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
                web_search_fn=fake_ddg,
            )
            config = {"configurable": {"thread_id": thread_id}}
            query = "На машинке CAR-2026-0777 при -5°C в горку резко падает запас хода и отключается питание."
            init_state = {
                "messages": [HumanMessage(content=query)],
                "thread_id": thread_id,
                "user_query": query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }

            visited_nodes: list[str] = []
            async for event in graph.astream(init_state, config=config, stream_mode="updates"):
                visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert "web_researcher_node" in visited_nodes

            snap = await graph.aget_state(config)
            assert snap.values["audit_result"].decision == "NEED_WEB_RESEARCH"
            verdict = snap.values["final_verdict"]
            assert len(verdict.confirmed_db_facts) >= 1
            assert len(verdict.external_hypotheses) >= 1
            assert (
                "https://example.org/jk-bms-6s-cold-cutoff"
                in verdict.external_hypotheses[0].sources
            )

            resumed = await graph.ainvoke(
                Command(resume={"approved": False, "comment": "Только консультация"}),
                config=config,
            )
            assert resumed["action_result"].approved is False
            assert resumed["action_result"].created_ticket_ids == []

            async with session_factory() as s:
                tickets = list(
                    (
                        await s.scalars(
                            select(SupportTicket).where(SupportTicket.thread_id == thread_id)
                        )
                    ).all()
                )
                assert tickets == []
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))


async def test_graph_e2e_multiturn_clarification_preserves_symptom_and_conditions(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Turn 1 without serial triggers clarification; Turn 2 with only serial aggregates both turns cleanly."""
    thread_id = f"test-e2e-mt-{uuid.uuid4().hex[:8]}"
    models = build_mock_model_bundle()
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    def fake_ddg(query: str, max_results: int = 4) -> list[dict[str, Any]]:
        _ = (query, max_results)
        return [
            {
                "title": "JK-BMS-6S Cold Guide",
                "href": "https://example.org/jk-bms-cold",
                "body": "Voltage sag below 0C causes cutoff.",
            }
        ]

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
                web_search_fn=fake_ddg,
            )
            config = {"configurable": {"thread_id": thread_id}}

            # Turn 1: symptom + conditions, NO serial number
            turn1_query = "У нас на улице при -5°C батарея не держит нагрузку и садится."
            turn1_state = {
                "messages": [HumanMessage(content=turn1_query)],
                "thread_id": thread_id,
                "user_query": turn1_query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }
            turn1_nodes: list[str] = []
            async for event in graph.astream(turn1_state, config=config, stream_mode="updates"):
                turn1_nodes.extend(k for k in event.keys() if k != "__interrupt__")
            assert turn1_nodes == ["intake_node"]

            snap1 = await graph.aget_state(config)
            assert snap1.values["intake"].needs_clarification is True
            assert snap1.values["intake"].symptom_category == "BATTERY_LOW_RANGE"

            # Turn 2: operator replies with ONLY the serial number on the SAME thread_id
            turn2_state = {
                "messages": [HumanMessage(content="CAR-2026-0777")],
                "user_query": "CAR-2026-0777",
                "tool_steps": 0,
                "revision_count": 0,
            }
            turn2_nodes: list[str] = []
            async for event in graph.astream(turn2_state, config=config, stream_mode="updates"):
                turn2_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert "triage_node" in turn2_nodes
            assert "web_researcher_node" in turn2_nodes

            snap2 = await graph.aget_state(config)
            assert snap2.values["intake"].needs_clarification is False
            assert snap2.values["intake"].tracking_code == "CAR-2026-0777"
            assert snap2.values["intake"].symptom_category == "BATTERY_LOW_RANGE"
            assert snap2.values["intake"].operating_conditions == "-5°C"
            assert "-5°C" in snap2.values["user_query"]
            assert "CAR-2026-0777" in snap2.values["user_query"]
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))


async def test_llm_structured_fallback_and_scripted_tool_usage() -> None:
    """Verify markdown fence stripping fallback, function_calling retry, and tool_call token accounting."""
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable, RunnableLambda
    from pydantic import BaseModel

    # 1. Markdown-wrapped JSON fallback when primary returns parsed=None
    fenced_payload = (
        "Вот разбор обращения:\n```json\n"
        '{"tracking_code": "CAR-2026-0412", "symptom_category": "BATTERY_LOW_RANGE", '
        '"operating_conditions": null, "needs_clarification": false, "clarification_message": null}'
        "\n```\nГотово."
    )

    class FencedRawModel(ScriptedChatModel):
        def with_structured_output(
            self,
            schema: type[BaseModel] | dict[str, Any] | None = None,
            *,
            include_raw: bool = False,
            **kwargs: Any,
        ) -> Runnable[Any, Any]:
            async def _inv(_inp: Any, _cfg: Any = None) -> dict[str, Any]:
                raw = AIMessage(
                    content=fenced_payload,
                    usage_metadata={"input_tokens": 111, "output_tokens": 44, "total_tokens": 155},
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("strict fail")}

            return RunnableLambda(_inv)

    m_fenced = FencedRawModel(
        role_name="FAST",
        model_id="test/fenced",
        responder=lambda _m, _s, _t: AIMessage(content="ok"),
    )
    parsed1, usage1 = await invoke_structured_with_usage(
        m_fenced,
        IntakeResult,
        [HumanMessage(content="test")],
        role="FAST",
        node_name="test_fenced",
        model_name="test/fenced",
    )
    assert parsed1.tracking_code == "CAR-2026-0412"
    assert usage1.total_tokens == 155

    # 2. Exception on default with_structured_output triggers single retry with method='function_calling'
    seen_methods: list[str | None] = []

    class StrictFailsThenFnModel(ScriptedChatModel):
        def with_structured_output(
            self,
            schema: type[BaseModel] | dict[str, Any] | None = None,
            *,
            include_raw: bool = False,
            **kwargs: Any,
        ) -> Runnable[Any, Any]:
            method = kwargs.get("method")
            seen_methods.append(method)

            async def _inv(_inp: Any, _cfg: Any = None) -> dict[str, Any]:
                if method != "function_calling":
                    raise RuntimeError("400 json_schema unsupported")
                model_obj = IntakeResult(
                    tracking_code="CAR-2026-0388",
                    symptom_category="BATTERY_LOW_RANGE",
                    needs_clarification=False,
                )
                raw = AIMessage(
                    content=model_obj.model_dump_json(),
                    usage_metadata={"input_tokens": 90, "output_tokens": 50, "total_tokens": 140},
                )
                return {"raw": raw, "parsed": model_obj, "parsing_error": None}

            return RunnableLambda(_inv)

    m_retry = StrictFailsThenFnModel(
        role_name="INVESTIGATOR",
        model_id="test/retry",
        responder=lambda _m, _s, _t: AIMessage(content="ok"),
    )
    parsed2, usage2 = await invoke_structured_with_usage(
        m_retry,
        IntakeResult,
        [HumanMessage(content="test")],
        role="INVESTIGATOR",
        node_name="test_retry",
        model_name="test/retry",
    )
    assert seen_methods == [None, "function_calling"]
    assert parsed2.tracking_code == "CAR-2026-0388"
    assert usage2.total_tokens == 140

    # 3. ScriptedChatModel tool_calls AIMessage gets non-zero usage_metadata
    def tool_resp(_msgs: list[BaseMessage], _s: type[BaseModel] | None, _t: list[str]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_item_genealogy",
                    "args": {"tracking_code": "CAR-2026-0412"},
                    "id": "tc1",
                }
            ],
        )

    m_tool = ScriptedChatModel(role_name="INVESTIGATOR", model_id="mock/tool", responder=tool_resp)
    res_msg = await m_tool.ainvoke([HumanMessage(content="CAR-2026-0412")])
    assert isinstance(res_msg, AIMessage)
    assert res_msg.usage_metadata is not None
    assert res_msg.usage_metadata["total_tokens"] > 0


def test_eval_metrics_action_groundedness_and_entity_matching() -> None:
    """Verify Block 3 evals hardening: action F1, require_facts, UUID/ISO scrubbing, and entity normalization."""
    assert compute_action_match_score([], []) == 1.0
    assert (
        compute_action_match_score(
            ["CREATE_SUPPORT_TICKET", "CREATE_LOT_RECALL"],
            ["CREATE_LOT_RECALL", "CREATE_SUPPORT_TICKET"],
        )
        == 1.0
    )
    assert (
        compute_action_match_score(
            ["CREATE_SUPPORT_TICKET"], ["CREATE_SUPPORT_TICKET", "CREATE_LOT_RECALL"]
        )
        < 1.0
    )

    # require_facts=True catches empty confirmed_db_facts
    g_empty, un_empty = compute_numeric_groundedness([], ["{}"], require_facts=True)
    assert g_empty == 0.0 and un_empty == ["NO_DB_FACTS_CITED"]

    # UUIDs/ISO timestamps inside evidence do NOT falsely ground hallucinated '14'; tolerances DO ground '33.0'
    ev = [
        '{"id": "14000000-0014-4000-8000-000000000014", "completed_at": "2026-10-14T14:22:00+00:00", '
        '"defect_rate_pct": 20.0, "min_value": 33.0}'
    ]
    g_ok, un_ok = compute_numeric_groundedness(
        ["Брак партии 20%, минимум 33.0 Ач на 2026-10-14T14:22:00+00:00."], ev, require_facts=True
    )
    assert g_ok == 1.0 and un_ok == []

    g_hal, un_hal = compute_numeric_groundedness(["Брак партии 14%."], ev, require_facts=True)
    assert g_hal == 0.0 and un_hal == ["14"]

    # Entity matching: 20% <-> 20.0, закрыт <-> CLOSED, канал 7 <-> 7 (and CAR-2026-0777 does NOT satisfy '7')
    acc_syn = compute_root_cause_accuracy(
        actual_needs_clarification=False,
        expected_needs_clarification=False,
        actual_root_cause_type="LOT_DEFECT",
        expected_root_cause_type="LOT_DEFECT",
        verdict_text="Партия LOT-BAT-2026-014 от VoltCell: брак 20%, машинка CAR-2026-0412.",
        expected_entities=["LOT-BAT-2026-014", "VoltCell", "20.0", "CAR-2026-0412"],
    )
    assert acc_syn == 1.0

    acc_closed_ch7 = compute_root_cause_accuracy(
        actual_needs_clarification=False,
        expected_needs_clarification=False,
        actual_root_cause_type="INSTRUMENT_CHANNEL_DRIFT",
        expected_root_cause_type="INSTRUMENT_CHANNEL_DRIFT",
        verdict_text="Стенд STAND-03 канал 7, поверка 2026-09-30, прошлый тикет закрыт.",
        expected_entities=["STAND-03", "7", "2026-09-30", "CLOSED"],
    )
    assert acc_closed_ch7 == 1.0

    acc_no_false_7 = compute_root_cause_accuracy(
        actual_needs_clarification=False,
        expected_needs_clarification=False,
        actual_root_cause_type="INSTRUMENT_CHANNEL_DRIFT",
        expected_root_cause_type="INSTRUMENT_CHANNEL_DRIFT",
        verdict_text="Машинка CAR-2026-0777 проверена без упоминания стенда.",
        expected_entities=["7"],
    )
    assert acc_no_false_7 == 0.5


async def test_graph_e2e_unknown_serial_inconclusive_skips_web_research(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Guardrail #1: CAR-2026-9999 (found=False) must force INCONCLUSIVE and block web_researcher_node even if investigator misclassifies."""
    thread_id = f"test-e2e-unk-{uuid.uuid4().hex[:8]}"
    base_bundle = build_mock_model_bundle()

    def sneaky_investigator(
        messages: list[Any], schema: type[Any] | None, bound_tools: list[str]
    ) -> AIMessage | Any:
        if schema is InvestigationDraft:
            return InvestigationDraft(
                root_cause_hypothesis="БД чиста, отправим в веб.",
                root_cause_type="DB_CLEAN_EXTERNAL_SUSPECTED",
                db_anomalies_found=False,
                cited_db_facts=["CAR-2026-9999 не найдена в route_sheets."],
                recommended_actions=[
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title="Тикет CAR-2026-9999",
                        description="Проверка серийника.",
                        priority="MEDIUM",
                        target_tracking_code="CAR-2026-9999",
                    )
                ],
                confidence=0.6,
            )
        if schema is FinalVerdict:
            return FinalVerdict(
                summary="Серийный номер CAR-2026-9999 отсутствует в БД MES.",
                severity="LOW",
                confirmed_db_facts=["Изделие CAR-2026-9999 отсутствует в MES."],
                affected_cars=[],
                proposed_actions=[
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title="Тикет CAR-2026-9999",
                        description="Уточнить серийник на месте.",
                        priority="MEDIUM",
                        target_tracking_code="CAR-2026-9999",
                    )
                ],
            )
        return base_bundle.investigator_model.responder(messages, schema, bound_tools)  # type: ignore[attr-defined]

    def sneaky_auditor(
        _messages: list[Any], _schema: type[Any] | None, _bound_tools: list[str]
    ) -> AuditVerdict:
        return AuditVerdict(
            decision="NEED_WEB_RESEARCH", feedback="Should be blocked!", db_is_clean=True
        )

    models = AgentModelBundle(
        fast_model=base_bundle.fast_model,
        investigator_model=ScriptedChatModel(
            role_name="INVESTIGATOR",
            model_id="mock/sneaky-inv",
            responder=sneaky_investigator,
        ),
        auditor_model=ScriptedChatModel(
            role_name="AUDITOR",
            model_id="mock/sneaky-aud",
            responder=sneaky_auditor,
        ),
        judge_model=base_bundle.judge_model,
        fast_model_name=base_bundle.fast_model_name,
        investigator_model_name="mock/sneaky-inv",
        auditor_model_name="mock/sneaky-aud",
        judge_model_name=base_bundle.judge_model_name,
    )
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
            )
            config = {"configurable": {"thread_id": thread_id}}
            query = "Проверьте машинку CAR-2026-9999 — не включается."
            init_state = {
                "messages": [HumanMessage(content=query)],
                "thread_id": thread_id,
                "user_query": query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }
            visited_nodes: list[str] = []
            async for event in graph.astream(init_state, config=config, stream_mode="updates"):
                visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert "web_researcher_node" not in visited_nodes
            snap = await graph.aget_state(config)
            assert snap.values["investigator_draft"].root_cause_type == "INCONCLUSIVE"
            assert snap.values["audit_result"].decision == "APPROVED"
            assert snap.values["audit_result"].db_is_clean is False
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))


async def test_graph_e2e_auditor_guardrail_enforces_all_5_tools_before_web_research(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Guardrail #2: auditor_node overrides premature NEED_WEB_RESEARCH -> REJECTED until all 5 DB tools are called."""
    thread_id = f"test-e2e-g2-{uuid.uuid4().hex[:8]}"
    base_bundle = build_mock_model_bundle()

    def lazy_investigator(
        messages: list[Any], schema: type[Any] | None, bound_tools: list[str]
    ) -> AIMessage | Any:
        # On pass 1 (no feedback), stop tool loop early before calling get_operator_anomalies!
        full_text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        if schema is None and bound_tools and "ЗАМЕЧАНИЯ РЕВИЗОРА" not in full_text:
            # Check if 4 tools ran already -> stop without calling get_operator_anomalies
            from langchain_core.messages import ToolMessage

            tool_names = {str(m.name) for m in messages if isinstance(m, ToolMessage) and m.name}
            if "get_instrument_channel_anomalies" in tool_names:
                return AIMessage(
                    content="Поверхностный анализ завершён без get_operator_anomalies."
                )
        return base_bundle.investigator_model.responder(messages, schema, bound_tools)  # type: ignore[attr-defined]

    def permissive_auditor(
        _messages: list[Any], _schema: type[Any] | None, _bound_tools: list[str]
    ) -> AuditVerdict:
        # Even if LLM Auditor mistakenly says NEED_WEB_RESEARCH on pass 1, Guardrail #2 must reject!
        return AuditVerdict(
            decision="NEED_WEB_RESEARCH",
            feedback="Выход в веб.",
            db_is_clean=True,
        )

    def fake_ddg(query: str, max_results: int = 4) -> list[dict[str, Any]]:
        _ = (query, max_results)
        return [
            {"title": "JK-BMS-6S Guide", "href": "https://example.org/bms", "body": "UVP at -5C."}
        ]

    models = AgentModelBundle(
        fast_model=base_bundle.fast_model,
        investigator_model=ScriptedChatModel(
            role_name="INVESTIGATOR",
            model_id="mock/lazy-inv",
            responder=lazy_investigator,
        ),
        auditor_model=ScriptedChatModel(
            role_name="AUDITOR",
            model_id="mock/perm-aud",
            responder=permissive_auditor,
        ),
        judge_model=base_bundle.judge_model,
        fast_model_name=base_bundle.fast_model_name,
        investigator_model_name="mock/lazy-inv",
        auditor_model_name="mock/perm-aud",
        judge_model_name=base_bundle.judge_model_name,
    )
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
                web_search_fn=fake_ddg,
            )
            config = {"configurable": {"thread_id": thread_id}}
            query = "У CAR-2026-0777 при -5°C падает запас хода."
            init_state = {
                "messages": [HumanMessage(content=query)],
                "thread_id": thread_id,
                "user_query": query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }
            visited_nodes: list[str] = []
            async for event in graph.astream(init_state, config=config, stream_mode="updates"):
                visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert visited_nodes.count("auditor_node") == 2
            assert "web_researcher_node" in visited_nodes
            snap = await graph.aget_state(config)
            assert snap.values["revision_count"] == 1
            called = {ev.tool_name for ev in snap.values["evidence_log"]}
            assert "get_operator_anomalies" in called
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))


async def test_graph_e2e_auditor_clean_db_approved_forces_need_web_research(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Clean-DB invariant: APPROVED + db_is_clean=True + not db_anomalies_found forces NEED_WEB_RESEARCH."""
    thread_id = f"test-e2e-cwr-{uuid.uuid4().hex[:8]}"
    base_bundle = build_mock_model_bundle()

    def mistaken_approved_auditor(
        _messages: list[Any], _schema: type[Any] | None, _bound_tools: list[str]
    ) -> AuditVerdict:
        return AuditVerdict(
            decision="APPROVED",
            feedback="БД чиста, внешняя диагностика обоснована.",
            db_is_clean=True,
        )

    def fake_ddg(query: str, max_results: int = 4) -> list[dict[str, Any]]:
        _ = (query, max_results)
        return [
            {
                "title": "JK-BMS-6S Guide",
                "href": "https://example.org/bms-cold",
                "body": "0C cutoff.",
            }
        ]

    models = AgentModelBundle(
        fast_model=base_bundle.fast_model,
        investigator_model=base_bundle.investigator_model,
        auditor_model=ScriptedChatModel(
            role_name="AUDITOR",
            model_id="mock/mistaken-approved-aud",
            responder=mistaken_approved_auditor,
        ),
        judge_model=base_bundle.judge_model,
        fast_model_name=base_bundle.fast_model_name,
        investigator_model_name=base_bundle.investigator_model_name,
        auditor_model_name="mock/mistaken-approved-aud",
        judge_model_name=base_bundle.judge_model_name,
    )
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            graph = build_rca_graph(
                models=models,
                session_factory=session_factory,
                bpmn_parser=bpmn_parser,
                retriever=retriever,
                checkpointer=checkpointer,
                settings=settings,
                web_search_fn=fake_ddg,
            )
            config = {"configurable": {"thread_id": thread_id}}
            query = "На уличной площадке при -5°C у CAR-2026-0777 падает запас хода."
            init_state = {
                "messages": [HumanMessage(content=query)],
                "thread_id": thread_id,
                "user_query": query,
                "evidence_log": [],
                "tool_steps": 0,
                "revision_count": 0,
                "confirmed_db_facts": [],
                "external_hypotheses": [],
                "token_usage": [],
            }
            visited_nodes: list[str] = []
            async for event in graph.astream(init_state, config=config, stream_mode="updates"):
                visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

            assert "web_researcher_node" in visited_nodes
            snap = await graph.aget_state(config)
            assert snap.values["audit_result"].decision == "NEED_WEB_RESEARCH"
            assert snap.values["audit_result"].db_is_clean is True
            assert len(snap.values["external_hypotheses"]) >= 1
    finally:
        async with session_factory() as s:
            async with s.begin():
                await s.execute(delete(SupportTicket).where(SupportTicket.thread_id == thread_id))
                await s.execute(delete(RecallNotice).where(RecallNotice.thread_id == thread_id))
