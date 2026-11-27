from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agent.graph import build_rca_graph, open_postgres_checkpointer
from src.agent.llm import build_openrouter_models
from src.agent.mock_models import build_mock_model_bundle
from src.agent.state import ActionResult, FinalVerdict
from src.config import get_settings
from src.db.session import build_engine, build_session_factory
from src.rag.bpmn_parser import BpmnWorkflowParser
from src.rag.hybrid_retriever import HybridRetriever

DEMO_CASES: list[tuple[str, str]] = [
    (
        "1. Дефект партии АКБ + Recall 12 машинок (CAR-2026-0412)",
        "В парке жалуются на машинку CAR-2026-0412 — батарея совсем не держит заезд, садится через 10 минут.",
    ),
    (
        "2. Дрейф канала 7 стенда STAND-03 + просрочка поверки (CAR-2026-0388)",
        "Машинка CAR-2026-0388 быстро теряет заряд на трассе, батарея не вытягивает смену.",
    ),
    (
        "3. Просадка качества оператора Петрова А.В. (CAR-2026-0295)",
        "У машинки CAR-2026-0295 заклинил моторчик, слышен скрежет редуктора.",
    ),
    (
        "4. Повторное обращение / Эскалация (CAR-2026-0199)",
        "Машинка CAR-2026-0199 снова дёргается при старте и глохнет под нагрузкой.",
    ),
    (
        "5. Чистая БД -> Web Research по BOM (CAR-2026-0777)",
        "У машинки CAR-2026-0777 на уличной площадке при -5°C резко падает запас хода и отключается питание в горку.",
    ),
    (
        "6. Обращение без серийника (уточнение у оператора)",
        "В парке одна из машинок перестала заряжаться и не держит смену.",
    ),
]


def _render_verdict(console: Console, verdict: FinalVerdict) -> None:
    db_text = (
        "\n".join(f"• {f}" for f in verdict.confirmed_db_facts)
        or "Нет зафиксированных аномалий в БД."
    )
    console.print(
        Panel(
            db_text,
            title=f"[bold green]Факты из БД MES (Severity: {verdict.severity})[/bold green]",
            border_style="green",
        )
    )

    if verdict.external_hypotheses:
        ext_lines: list[str] = []
        for idx, h in enumerate(verdict.external_hypotheses, start=1):
            src = ", ".join(h.sources) if h.sources else "offline fallback"
            steps = "\n   ".join(f"-> {s}" for s in h.mechanic_check_steps)
            ext_lines.append(
                f"[bold]{idx}. {h.title}[/bold] (уверенность: {h.confidence:.0%})\n"
                f"   {h.explanation}\n"
                f"   Источники: {src}\n"
                f"   Шаги проверки механиком:\n   {steps}"
            )
        console.print(
            Panel(
                "\n\n".join(ext_lines),
                title="[bold cyan]Гипотезы из внешних источников (Web Research — НЕ факты БД)[/bold cyan]",
                border_style="cyan",
            )
        )

    if verdict.affected_cars:
        console.print(
            Panel(
                ", ".join(verdict.affected_cars),
                title=f"[bold red]Затронутые машинки под отзыв / Recall ({len(verdict.affected_cars)} шт.)[/bold red]",
                border_style="red",
            )
        )

    tbl = Table(title="Предлагаемые действия (HITL)")
    tbl.add_column("Тип", style="yellow")
    tbl.add_column("Приоритет", style="red")
    tbl.add_column("Заголовок", style="bold")
    tbl.add_column("Описание")
    for act in verdict.proposed_actions:
        tbl.add_row(act.action_type, act.priority, act.title, act.description)
    console.print(tbl)


async def run_investigation_turn(
    graph: Any,
    console: Console,
    *,
    user_query: str,
    thread_id: str,
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    init_state = {
        "messages": [HumanMessage(content=user_query)],
        "thread_id": thread_id,
        "user_query": user_query,
        "evidence_log": [],
        "tool_steps": 0,
        "revision_count": 0,
        "confirmed_db_facts": [],
        "external_hypotheses": [],
        "token_usage": [],
    }

    async for event in graph.astream(init_state, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "__interrupt__":
                continue
            if node_name == "intake_node" and update.get("intake"):
                it = update["intake"]
                if it.needs_clarification:
                    console.print(
                        Panel(
                            it.clarification_message or "Уточните серийник.",
                            title="[yellow]Требуется уточнение[/yellow]",
                        )
                    )
                    return
                console.print(
                    f"[bold blue]→ intake_node:[/bold blue] серийник=[cyan]{it.tracking_code}[/cyan], "
                    f"категория=[magenta]{it.symptom_category}[/magenta], условия={it.operating_conditions}"
                )
            elif node_name == "triage_node":
                console.print(
                    f"[bold blue]→ triage_node:[/bold blue] BPMN обязательные тулы={update.get('required_tools')}, "
                    f"SOP чанков={len(update.get('rag_context', []))}"
                )
            elif node_name == "tools_node":
                for ev in update.get("evidence_log", []):
                    console.print(
                        f"  [green]⚙ SQL/RAG Tool:[/green] [bold]{ev.tool_name}[/bold]({ev.args_summary}) [{ev.latency_ms} ms]"
                    )
            elif node_name == "auditor_node" and update.get("audit_result"):
                aud = update["audit_result"]
                color = (
                    "green"
                    if aud.decision == "APPROVED"
                    else ("cyan" if aud.decision == "NEED_WEB_RESEARCH" else "red")
                )
                console.print(
                    f"[bold {color}]→ auditor_node: {aud.decision}[/bold {color}] — {aud.feedback}"
                )
            elif node_name == "web_researcher_node":
                hyps = update.get("external_hypotheses", [])
                console.print(
                    f"[bold cyan]→ web_researcher_node:[/bold cyan] получено внешних гипотез: {len(hyps)}"
                )
            elif node_name == "verdict_node" and update.get("final_verdict"):
                _render_verdict(console, update["final_verdict"])

    # Check for HITL interrupt pause
    snapshot = await graph.aget_state(config)
    if snapshot.tasks and any(getattr(t, "interrupts", None) for t in snapshot.tasks):
        interrupt_val: dict[str, Any] = {}
        for t in snapshot.tasks:
            if getattr(t, "interrupts", None):
                interrupt_val = t.interrupts[0].value
                break

        console.print(
            Panel(
                f"Серийник: [bold]{interrupt_val.get('tracking_code')}[/bold] | "
                f"Действий к записи: [bold]{len(interrupt_val.get('proposed_actions', []))}[/bold]\n"
                f"{interrupt_val.get('prompt', '')}",
                title="[bold yellow]HITL Пауза (interrupt): Требуется подтверждение оператора[/bold yellow]",
                border_style="yellow",
            )
        )
        choice = Prompt.ask(
            "Решение оператора: [y] Подтвердить / [n] Отклонить / [e] Подтвердить с комментарием",
            choices=["y", "n", "e"],
            default="y",
        )
        comment: str | None = None
        approved = choice in {"y", "e"}
        if choice == "e":
            comment = Prompt.ask("Введите комментарий оператора")

        resumed = await graph.ainvoke(
            Command(resume={"approved": approved, "comment": comment}),
            config=config,
        )
        act_res: ActionResult | None = resumed.get("action_result")
        if act_res:
            style = "green" if act_res.approved else "yellow"
            console.print(
                Panel(
                    f"{act_res.message}\n"
                    f"Тикеты (PostgreSQL): {act_res.created_ticket_ids}\n"
                    f"Отзывы/блокировки (PostgreSQL): {act_res.created_recall_ids}",
                    title="[bold]Результат записи в PostgreSQL[/bold]",
                    border_style=style,
                )
            )


async def async_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LangVeyor MES RCA Interactive CLI")
    parser.add_argument(
        "--demo", action="store_true", help="Показать меню демо-сценариев по посаженным аномалиям"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Принудительно использовать оффлайн ScriptedChatModel вместо OpenRouter",
    )
    parser.add_argument("--thread", default="", help="Идентификатор thread_id сеанса")
    parser.add_argument("--query", default="", help="Однократный запрос в RCA-агент")
    args = parser.parse_args(argv)

    console = Console()
    settings = get_settings()
    use_mock = args.mock or (not settings.has_openrouter_key)

    if use_mock and not args.mock:
        console.print(
            "[yellow]OPENROUTER_API_KEY не задан в .env — CLI переключён в детерминированный режим --mock "
            "(реальные PostgreSQL 17, Qdrant Hybrid RAG и AsyncPostgresSaver, симулированные LLM).[/yellow]"
        )

    models = build_mock_model_bundle() if use_mock else build_openrouter_models(settings)
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    bpmn_parser = BpmnWorkflowParser(settings.bpmn_path)
    retriever = HybridRetriever(settings)

    console.print(
        Panel.fit(
            f"[bold cyan]LangVeyor MES Root Cause Analysis CLI[/bold cyan]\n"
            f"FAST={models.fast_model_name} | INVESTIGATOR={models.investigator_model_name} | AUDITOR={models.auditor_model_name}",
            border_style="cyan",
        )
    )

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

            thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
            if args.query:
                await run_investigation_turn(
                    graph, console, user_query=args.query, thread_id=thread_id
                )
                return

            if args.demo:
                for label, q in DEMO_CASES:
                    console.print(f"[bold]{label}[/bold]\n  [dim]{q}[/dim]")
                idx_str = Prompt.ask(
                    "Выберите номер демо-кейса (1-6)",
                    choices=["1", "2", "3", "4", "5", "6"],
                    default="1",
                )
                selected_q = DEMO_CASES[int(idx_str) - 1][1]
                await run_investigation_turn(
                    graph, console, user_query=selected_q, thread_id=thread_id
                )
                return

            console.print(
                "Введите обращение из парка аттракционов, `/demo` для примеров или `/exit` для выхода."
            )
            while True:
                q = Prompt.ask(f"\n[bold green][{thread_id}]>[/bold green]").strip()
                if not q:
                    continue
                if q in {"/exit", "/quit", "exit", "quit"}:
                    break
                if q == "/demo":
                    for label, dq in DEMO_CASES:
                        console.print(f"[bold]{label}[/bold]\n  [dim]{dq}[/dim]")
                    idx_str = Prompt.ask(
                        "Выберите номер демо-кейса (1-6)",
                        choices=["1", "2", "3", "4", "5", "6"],
                        default="1",
                    )
                    q = DEMO_CASES[int(idx_str) - 1][1]
                elif q.startswith("/new"):
                    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                    console.print(f"[dim]Новый thread_id: {thread_id}[/dim]")
                    continue

                await run_investigation_turn(graph, console, user_query=q, thread_id=thread_id)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(async_main())
