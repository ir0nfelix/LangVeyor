from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from src.agent.graph import build_rca_graph, open_postgres_checkpointer
from src.agent.llm import AgentModelBundle, build_openrouter_models, invoke_structured_with_usage
from src.agent.mock_models import build_mock_model_bundle
from src.agent.state import AuditVerdict, LLMUsageRecord
from src.config import PROJECT_ROOT, Settings, get_settings
from src.db.models import RecallNotice, SupportTicket
from src.db.session import build_engine, build_session_factory
from src.rag.bpmn_parser import BpmnWorkflowParser
from src.rag.hybrid_retriever import HybridRetriever

from evals.metrics import (
    CaseEvalResult,
    compute_action_match_score,
    compute_numeric_groundedness,
    compute_recall_prf1,
    compute_root_cause_accuracy,
    compute_tool_trajectory_score,
    estimate_cost_usd,
)


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def _accumulate_role_usage(
    role_stats: dict[str, dict[str, Any]],
    usage: LLMUsageRecord,
) -> float:
    cost = estimate_cost_usd(usage.model_name, usage.input_tokens, usage.output_tokens)
    rs = role_stats.setdefault(
        usage.role,
        {
            "model": usage.model_name,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "cost_usd": 0.0,
        },
    )
    rs["calls"] += 1
    rs["input_tokens"] += usage.input_tokens
    rs["output_tokens"] += usage.output_tokens
    rs["total_tokens"] += usage.total_tokens
    rs["latency_ms"] = round(rs["latency_ms"] + usage.latency_ms, 2)
    rs["cost_usd"] = round(rs["cost_usd"] + cost, 6)
    return cost


def _mock_web_search(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    _ = (query, max_results)
    return [
        {
            "title": "JK-BMS-6S Cold Weather Cutoff & RS550-24V Voltage Sag Reference",
            "href": "https://example.org/jk-bms-6s-lipo-cold-guide",
            "body": (
                "Li-Po 6S cells exhibit high internal resistance below 0C (-5C), causing voltage sag "
                "under RS550-24V peak load and triggering JK-BMS-6S low-temperature / UVP cutoff."
            ),
        }
    ]


async def _run_llm_judge(
    models: AgentModelBundle,
    *,
    case: dict[str, Any],
    actual_root_cause_type: str | None,
    actual_action_types: list[str],
    actual_summary: str,
) -> tuple[float, str, LLMUsageRecord]:
    sys_msg = (
        "Ты — независимый судья (LLM-as-a-Judge) оценки RCA-отчётов. "
        "Оцени от 0.0 до 1.0, насколько диагноз, ключевые сущности и предложенные действия точны относительно эталона. "
        "Не снижай балл за использование русских формулировок вместо английских технических enum-статусов "
        "(например, «закрытый тикет» / «ранее закрытое обращение» полностью эквивалентно статусу CLOSED, а «повторный отказ» — REPEAT_FAILURE). "
        "Игнорируй длину текста (anti-verbosity bias); оценивай исключительно фактическую точность первопричины и полноту доказательств."
    )
    usr_msg = (
        f"Запрос: {case['query']}\n"
        f"Эталонное описание: {case['description']}\n"
        f"Ожидаемый root_cause_type: {case.get('expected_root_cause_type')}\n"
        f"Ожидаемые сущности: {case.get('expected_entities', [])}\n"
        f"Ожидаемые действия: {case.get('expected_action_types', [])}\n\n"
        f"Фактический root_cause_type: {actual_root_cause_type}\n"
        f"Фактические действия: {actual_action_types}\n"
        f"Отчёт агента:\n{actual_summary}"
    )
    verdict, usage = await invoke_structured_with_usage(
        models.judge_model,
        JudgeVerdict,
        [SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)],
        role="JUDGE",
        node_name="eval_llm_judge",
        model_name=models.judge_model_name,
    )
    return float(verdict.score), str(verdict.rationale), usage


async def _run_auditor_unit_check(
    models: AgentModelBundle,
    bpmn_parser: BpmnWorkflowParser,
    role_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify independently that the Auditor catches an incomplete LOT_DEFECT draft missing recall cars."""
    guidance = bpmn_parser.get_symptom_branch("BATTERY_LOW_RANGE")
    evidence_txt = (
        '- get_item_genealogy(CAR-2026-0412) -> {"found": true, "tracking_code": "CAR-2026-0412", '
        '"target_route_sheet_id": "rs-car-0412", "root_lots": [{"tracking_code": "LOT-BAT-2026-014"}]}\n'
        '- get_item_history(rs-car-0412) -> {"found": true, "rework_count": 0, '
        '"component_rework_count": 1, "prior_support_tickets": []}\n'
        '- get_lot_quality_stats(LOT-BAT-2026-014) -> {"found": true, "lot_tracking_code": "LOT-BAT-2026-014", '
        '"defect_rate_pct": 20.0, "defect_units": 6, "total_split_units": 30, "exceeds_10pct_recall_threshold": true}'
    )
    incomplete_draft = {
        "root_cause_hypothesis": "Одиночная просадка АКБ на CAR-2026-0412.",
        "root_cause_type": "LOT_DEFECT",
        "db_anomalies_found": True,
        "cited_db_facts": ["LOT-BAT-2026-014: defect_rate_pct 20.0% (6/30)."],
        "affected_cars": [],
        "target_lot_code": "LOT-BAT-2026-014",
        "recommended_actions": [
            {
                "action_type": "CREATE_SUPPORT_TICKET",
                "title": "Проверить CAR-2026-0412",
                "description": "Локальная замена АКБ.",
                "priority": "MEDIUM",
                "target_tracking_code": "CAR-2026-0412",
            }
        ],
    }
    sys_prompt = (
        "Ты — независимый Ревизор качества расследования (Auditor Agent).\n"
        "Сверь черновик Следователя с сырыми ответами SQL-тулов (evidence_log) и BPMN.\n"
        "Если брак партии > 10%, а find_affected_products_by_lot не вызван и список affected_cars пуст -> decision='REJECTED'."
    )
    usr_prompt = (
        f"BPMN-регламент:\n{guidance.summary_text}\n\n"
        f"Сырой evidence_log:\n{evidence_txt}\n\n"
        f"Черновик Следователя:\n{json.dumps(incomplete_draft, ensure_ascii=False, indent=2)}\n\n"
        "Текущий revision_count: 0 (максимум 2)."
    )
    verdict, usage = await invoke_structured_with_usage(
        models.auditor_model,
        AuditVerdict,
        [SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)],
        role="AUDITOR",
        node_name="auditor_unit_check",
        model_name=models.auditor_model_name,
    )
    _accumulate_role_usage(role_stats, usage)
    return {
        "passed": verdict.decision == "REJECTED",
        "decision": verdict.decision,
        "feedback": verdict.feedback,
        "missing_checks": verdict.missing_checks,
    }


async def run_eval_suite(
    *,
    mode: str,
    limit: int | None = None,
    use_judge: bool = False,
) -> dict[str, Any]:
    settings: Settings = get_settings()
    if mode == "live" and not settings.has_openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in .env — cannot run --mode live. "
            "Use --mode mock or add OPENROUTER_API_KEY to .env."
        )

    cases_path = Path(PROJECT_ROOT) / "evals" / "golden_cases.json"
    manifest_path = Path(PROJECT_ROOT) / "evals" / "seed_manifest.json"
    cases: list[dict[str, Any]] = json.loads(cases_path.read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_recall_cars: list[str] = manifest["anomalies"]["lot_defect_recall"]["affected_cars"]

    engine = build_engine(settings)
    session_factory: async_sessionmaker[AsyncSession] = build_session_factory(engine)
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    results: list[CaseEvalResult] = []
    role_stats: dict[str, dict[str, Any]] = {}

    try:
        async with open_postgres_checkpointer(settings.checkpointer_dsn) as checkpointer:
            for case in cases:
                cid = str(case["case_id"])
                thread_id = f"eval-{mode}-{cid}-{uuid.uuid4().hex[:6]}"
                catch_sim = bool(case.get("simulate_incomplete_draft_catch", False))

                models = (
                    build_mock_model_bundle(force_auditor_reject_once=catch_sim)
                    if mode == "mock"
                    else build_openrouter_models(settings)
                )
                graph = build_rca_graph(
                    models=models,
                    session_factory=session_factory,
                    bpmn_parser=bpmn_parser,
                    retriever=retriever,
                    checkpointer=checkpointer,
                    settings=settings,
                    web_search_fn=_mock_web_search if mode == "mock" else None,
                )

                config = {"configurable": {"thread_id": thread_id}}
                init_state = {
                    "messages": [HumanMessage(content=case["query"])],
                    "thread_id": thread_id,
                    "user_query": case["query"],
                    "evidence_log": [],
                    "tool_steps": 0,
                    "revision_count": 0,
                    "confirmed_db_facts": [],
                    "external_hypotheses": [],
                    "token_usage": [],
                }

                t0 = time.perf_counter()
                visited_nodes: list[str] = []
                state_vals: dict[str, Any] = {}
                case_error: str | None = None
                try:
                    async for event in graph.astream(
                        init_state, config=config, stream_mode="updates"
                    ):
                        visited_nodes.extend(k for k in event.keys() if k != "__interrupt__")

                    snap = await graph.aget_state(config)
                    state_vals = snap.values
                    # If paused at HITL interrupt, reject so we don't mutate eval DB state
                    if snap.tasks and any(getattr(t, "interrupts", None) for t in snap.tasks):
                        await graph.ainvoke(
                            Command(resume={"approved": False, "comment": "eval dry-run"}),
                            config=config,
                        )
                except Exception as exc:
                    case_error = f"{type(exc).__name__}: {exc}"
                    try:
                        snap = await graph.aget_state(config)
                        state_vals = snap.values or {}
                    except Exception:
                        state_vals = {}
                finally:
                    async with session_factory() as s:
                        async with s.begin():
                            await s.execute(
                                delete(SupportTicket).where(SupportTicket.thread_id == thread_id)
                            )
                            await s.execute(
                                delete(RecallNotice).where(RecallNotice.thread_id == thread_id)
                            )

                total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                intake = state_vals.get("intake")
                draft = state_vals.get("investigator_draft")
                verdict = state_vals.get("final_verdict")
                audit = state_vals.get("audit_result")
                evidence_log = state_vals.get("evidence_log", [])

                actual_needs_clar = bool(intake.needs_clarification) if intake else False
                actual_rc_type = draft.root_cause_type if draft else None
                verdict_blob = ""
                if verdict:
                    verdict_blob = (
                        verdict.summary
                        + "\n"
                        + "\n".join(verdict.confirmed_db_facts)
                        + "\n"
                        + "\n".join(
                            h.title + " " + h.explanation for h in verdict.external_hypotheses
                        )
                    )
                elif intake and intake.clarification_message:
                    verdict_blob = intake.clarification_message

                actual_action_types = (
                    [str(a.action_type) for a in verdict.proposed_actions] if verdict else []
                )
                action_score = compute_action_match_score(
                    actual_action_types,
                    list(case.get("expected_action_types", [])),
                )

                rc_acc = compute_root_cause_accuracy(
                    actual_needs_clarification=actual_needs_clar,
                    expected_needs_clarification=bool(case["expected_needs_clarification"]),
                    actual_root_cause_type=actual_rc_type,
                    expected_root_cause_type=case.get("expected_root_cause_type"),
                    verdict_text=verdict_blob,
                    expected_entities=list(case.get("expected_entities", [])),
                )
                extra_usage: list[LLMUsageRecord] = []
                judge_score: float | None = None
                judge_rationale: str | None = None
                if (
                    case_error is None
                    and use_judge
                    and verdict is not None
                    and (mode == "live" or use_judge)
                ):
                    judge_score, judge_rationale, judge_usage = await _run_llm_judge(
                        models,
                        case=case,
                        actual_root_cause_type=actual_rc_type,
                        actual_action_types=actual_action_types,
                        actual_summary=verdict_blob,
                    )
                    extra_usage.append(judge_usage)
                    if mode == "live":
                        rc_acc = round(0.6 * rc_acc + 0.4 * judge_score, 4)

                # Recall Precision/Recall/F1
                rec_p: float | None = None
                rec_r: float | None = None
                rec_f1: float | None = None
                if case.get("expected_recall_cars_from_manifest"):
                    pred_cars = verdict.affected_cars if verdict else []
                    rec_p, rec_r, rec_f1 = compute_recall_prf1(pred_cars, expected_recall_cars)

                # Tool Trajectory
                called_tools = [ev.tool_name for ev in evidence_log]
                traj_score = compute_tool_trajectory_score(
                    called_tools,
                    list(case.get("expected_required_tools", [])),
                )

                # Web routing
                actual_web = "web_researcher_node" in visited_nodes
                exp_web = bool(case["expected_web_research"])
                web_ok = actual_web == exp_web

                # Auditor catch & recovery
                revs = int(state_vals.get("revision_count", 0))
                auditor_caught = revs > 0
                auditor_recovered = (
                    auditor_caught
                    and audit is not None
                    and audit.decision in {"APPROVED", "NEED_WEB_RESEARCH"}
                    and rc_acc >= 0.8
                )

                # Faithfulness / Groundedness
                db_facts = verdict.confirmed_db_facts if verdict else []
                ev_jsons = [ev.result_json for ev in evidence_log]
                require_facts = (
                    (not bool(case["expected_needs_clarification"]))
                    and (not exp_web)
                    and (case.get("expected_root_cause_type") != "INCONCLUSIVE")
                )
                groundedness, ungrounded = compute_numeric_groundedness(
                    db_facts,
                    ev_jsons,
                    require_facts=require_facts,
                )

                # Token & cost accounting
                case_tokens = 0
                case_cost = 0.0
                for u in [*state_vals.get("token_usage", []), *extra_usage]:
                    case_tokens += u.total_tokens
                    case_cost += _accumulate_role_usage(role_stats, u)

                passed = (
                    case_error is None
                    and rc_acc >= 0.75
                    and action_score >= 1.0
                    and traj_score >= 0.99
                    and web_ok
                    and groundedness >= 0.99
                    and (rec_f1 is None or rec_f1 == 1.0)
                    and (mode != "mock" or not catch_sim or auditor_recovered)
                )
                print(
                    f"[{mode}] {cid}: {'PASS' if passed else 'FAIL'} | "
                    f"rc={rc_acc:.2f} act={action_score:.2f} traj={traj_score:.2f} "
                    f"ground={groundedness:.2f} tok={case_tokens} {total_ms}ms"
                    + (f" err={case_error}" if case_error else ""),
                    flush=True,
                )

                results.append(
                    CaseEvalResult(
                        case_id=cid,
                        passed=passed,
                        root_cause_accuracy=rc_acc,
                        action_match_score=action_score,
                        recall_precision=rec_p,
                        recall_recall=rec_r,
                        recall_f1=rec_f1,
                        tool_trajectory_score=traj_score,
                        web_routing_correct=web_ok,
                        expected_web_research=exp_web,
                        actual_web_research=actual_web,
                        auditor_caught=auditor_caught,
                        auditor_recovered=auditor_recovered,
                        groundedness_score=groundedness,
                        ungrounded_numbers=ungrounded,
                        total_latency_ms=total_ms,
                        total_tokens=case_tokens,
                        estimated_cost_usd=round(case_cost, 6),
                        details={
                            "called_tools": called_tools,
                            "revision_count": revs,
                            "audit_decision": audit.decision if audit else None,
                            "actual_root_cause_type": actual_rc_type,
                            "actual_action_types": actual_action_types,
                            "judge_score": judge_score,
                            "judge_rationale": judge_rationale,
                            "error": case_error,
                        },
                    )
                )

            check_models = (
                build_mock_model_bundle() if mode == "mock" else build_openrouter_models(settings)
            )
            auditor_unit_check = await _run_auditor_unit_check(
                check_models, bpmn_parser, role_stats
            )
    finally:
        await engine.dispose()

    # Aggregate metrics
    n = len(results)
    avg_rc = round(sum(r.root_cause_accuracy for r in results) / max(1, n), 4)
    avg_act = round(sum(r.action_match_score for r in results) / max(1, n), 4)
    avg_traj = round(sum(r.tool_trajectory_score for r in results) / max(1, n), 4)
    avg_ground = round(sum(r.groundedness_score for r in results) / max(1, n), 4)
    recall_cases = [r for r in results if r.recall_f1 is not None]
    avg_recall_f1 = (
        round(sum(r.recall_f1 or 0.0 for r in recall_cases) / len(recall_cases), 4)
        if recall_cases
        else None
    )

    internal_cases = [r for r in results if not r.expected_web_research]
    web_fpr = (
        round(
            sum(1 for r in internal_cases if r.actual_web_research) / len(internal_cases),
            4,
        )
        if internal_cases
        else 0.0
    )
    web_cases = [r for r in results if r.expected_web_research]
    web_recall = (
        round(sum(1 for r in web_cases if r.actual_web_research) / len(web_cases), 4)
        if web_cases
        else 1.0
    )
    web_routing_acc = round(sum(1 for r in results if r.web_routing_correct) / max(1, n), 4)

    caught_cases = [r for r in results if r.auditor_caught]
    auditor_catch_rate = round(len(caught_cases) / max(1, n), 4)
    auditor_recovery_rate = (
        round(sum(1 for r in caught_cases if r.auditor_recovered) / len(caught_cases), 4)
        if caught_cases
        else 1.0
    )

    summary = {
        "mode": mode,
        "total_cases": n,
        "passed_cases": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / max(1, n), 4),
        "auditor_unit_check": auditor_unit_check,
        "metrics": {
            "root_cause_accuracy": avg_rc,
            "action_match_score": avg_act,
            "recall_completeness_f1": avg_recall_f1,
            "tool_trajectory_score": avg_traj,
            "web_routing_precision_acc": web_routing_acc,
            "web_routing_fpr": web_fpr,
            "web_routing_recall": web_recall,
            "auditor_catch_count": len(caught_cases),
            "auditor_catch_rate": auditor_catch_rate,
            "auditor_recovery_rate": auditor_recovery_rate,
            "auditor_unit_check_passed": auditor_unit_check["passed"],
            "faithfulness_groundedness": avg_ground,
            "avg_latency_ms": round(sum(r.total_latency_ms for r in results) / max(1, n), 2),
            "total_tokens": sum(r.total_tokens for r in results),
            "total_estimated_cost_usd": round(sum(r.estimated_cost_usd for r in results), 6),
        },
        "role_breakdown": role_stats,
        "cases": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "root_cause_accuracy": r.root_cause_accuracy,
                "action_match_score": r.action_match_score,
                "recall_precision": r.recall_precision,
                "recall_recall": r.recall_recall,
                "recall_f1": r.recall_f1,
                "tool_trajectory_score": r.tool_trajectory_score,
                "web_routing_correct": r.web_routing_correct,
                "auditor_caught": r.auditor_caught,
                "auditor_recovered": r.auditor_recovered,
                "groundedness_score": r.groundedness_score,
                "ungrounded_numbers": r.ungrounded_numbers,
                "total_latency_ms": r.total_latency_ms,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": r.estimated_cost_usd,
                "details": r.details,
            }
            for r in results
        ],
    }

    _write_reports(summary)
    return summary


def _write_reports(summary: dict[str, Any]) -> None:
    eval_dir = Path(PROJECT_ROOT) / "evals"
    json_path = eval_dir / "last_report.json"
    md_path = eval_dir / "last_report.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    m = summary["metrics"]
    lines = [
        f"# Отчёт Evals MES RCA (`mode={summary['mode']}`)",
        "",
        f"- **Итого кейсов пройдено:** {summary['passed_cases']} / {summary['total_cases']} ({summary['pass_rate']:.0%})",
        f"- **Auditor Unit Check (`missing recall catch`):** `{'PASS' if m['auditor_unit_check_passed'] else 'FAIL'}` ({summary['auditor_unit_check']['decision']})",
        "",
        "## Сводная таблица метрик",
        "",
        "| Метрика | Значение |",
        "| :--- | :---: |",
        f"| 1. Root Cause Accuracy | `{m['root_cause_accuracy']:.2%}` |",
        f"| 2. Action Match Score (`expected_action_types`) | `{m['action_match_score']:.2%}` |",
        f"| 3. Recall Completeness (F1 по `LOT-BAT-2026-014`) | `{m['recall_completeness_f1']}` |",
        f"| 4. Tool Trajectory Score | `{m['tool_trajectory_score']:.2%}` |",
        f"| 5a. Web-Routing Accuracy | `{m['web_routing_precision_acc']:.2%}` |",
        f"| 5b. Web-Routing FPR (ложные походы в интернет на аномалиях БД) | `{m['web_routing_fpr']:.2%}` |",
        f"| 5c. Web-Routing Recall (чистая БД `CAR-2026-0777` и др.) | `{m['web_routing_recall']:.2%}` |",
        f"| 6. Auditor Catch & Recovery Rate | `{m['auditor_catch_count']}` перехватов / recovery `{m['auditor_recovery_rate']:.2%}` (unit check: `{'PASS' if m['auditor_unit_check_passed'] else 'FAIL'}`) |",
        f"| 7. Faithfulness / Groundedness (цифры в `evidence_log`) | `{m['faithfulness_groundedness']:.2%}` |",
        f"| 8. Средняя задержка / Всего токенов / Стоимость | `{m['avg_latency_ms']} ms` / `{m['total_tokens']}` / `${m['total_estimated_cost_usd']}` |",
        "",
        "## Разбивка по ролям моделей",
        "",
        "| Роль | Модель | Вызовы | Input tokens | Output tokens | Latency (ms) | Cost (USD) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for role, rs in summary["role_breakdown"].items():
        lines.append(
            f"| `{role}` | `{rs['model']}` | {rs['calls']} | {rs['input_tokens']} | "
            f"{rs['output_tokens']} | {rs['latency_ms']} | ${rs['cost_usd']} |"
        )

    lines.extend(
        [
            "",
            "## Детализация по кейсам Golden Dataset",
            "",
            "| Case ID | Статус | RCA Acc | Actions | Recall F1 | Trajectory | Web OK | Auditor Catch/Rec | Groundedness | Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for c in summary["cases"]:
        st = "PASS" if c["passed"] else "FAIL"
        rf1 = f"{c['recall_f1']:.2f}" if c["recall_f1"] is not None else "—"
        aud = f"{c['auditor_caught']}/{c['auditor_recovered']}"
        lines.append(
            f"| `{c['case_id']}` | {st} | {c['root_cause_accuracy']:.2f} | {c['action_match_score']:.2f} | {rf1} | "
            f"{c['tool_trajectory_score']:.2f} | {c['web_routing_correct']} | {aud} | "
            f"{c['groundedness_score']:.2f} | {c['total_latency_ms']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangVeyor MES RCA Evals")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of golden cases")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-a-Judge evaluation")
    args = parser.parse_args()

    summary = asyncio.run(run_eval_suite(mode=args.mode, limit=args.limit, use_judge=args.judge))

    console = Console()
    tbl = Table(title=f"LangVeyor Evals Summary ({summary['mode']})")
    tbl.add_column("Case ID", style="cyan")
    tbl.add_column("Status")
    tbl.add_column("RC Acc")
    tbl.add_column("Actions")
    tbl.add_column("Recall F1")
    tbl.add_column("Trajectory")
    tbl.add_column("Web OK")
    tbl.add_column("Grounded")
    tbl.add_column("Latency ms")

    for c in summary["cases"]:
        st = "[green]PASS[/green]" if c["passed"] else "[red]FAIL[/red]"
        rf1 = f"{c['recall_f1']:.2f}" if c["recall_f1"] is not None else "—"
        tbl.add_row(
            c["case_id"],
            st,
            f"{c['root_cause_accuracy']:.2f}",
            f"{c['action_match_score']:.2f}",
            rf1,
            f"{c['tool_trajectory_score']:.2f}",
            str(c["web_routing_correct"]),
            f"{c['groundedness_score']:.2f}",
            str(c["total_latency_ms"]),
        )
    console.print(tbl)
    console.print(
        f"Passed: {summary['passed_cases']}/{summary['total_cases']} | "
        f"Actions: {summary['metrics']['action_match_score']:.2%} | "
        f"AuditorUnit: {summary['metrics']['auditor_unit_check_passed']} | "
        f"Web FPR: {summary['metrics']['web_routing_fpr']:.2%} | "
        f"Groundedness: {summary['metrics']['faithfulness_groundedness']:.2%} | "
        f"Report written to evals/last_report.md"
    )
    if (
        summary["passed_cases"] != summary["total_cases"]
        or not summary["auditor_unit_check"]["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
