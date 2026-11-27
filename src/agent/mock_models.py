from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from src.agent.llm import AgentModelBundle, ScriptedChatModel
from src.agent.state import (
    AuditVerdict,
    ExternalHypothesis,
    FinalVerdict,
    IntakeResult,
    InvestigationDraft,
    ProposedAction,
    SymptomCategory,
    WebHypothesesOutput,
)

SERIAL_RE = re.compile(r"\b(CAR-\d{4}-\d{4})\b", re.IGNORECASE)
LOT_RE = re.compile(r"LOT-BAT-\d{4}-\d{3}")


def _extract_tool_results_from_text_and_msgs(
    messages: list[BaseMessage],
) -> dict[str, list[dict[str, Any]]]:
    """Parse tool outputs from ToolMessages and serialized evidence_log in prompt text."""
    results: dict[str, list[dict[str, Any]]] = {}

    # 1. From evidence_log lines in HumanMessage context
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else ""
        for line in content.splitlines():
            m = re.match(r"^- ([a-zA-Z0-9_]+)\(.*?\) -> (\{.*\})$", line.strip())
            if m:
                tname, raw_json = m.group(1), m.group(2)
                try:
                    results.setdefault(tname, []).append(json.loads(raw_json))
                except Exception:
                    pass

    # 2. Also from active ToolMessages (which overwrite/append latest)
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tname = str(msg.name or "")
            raw = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    existing = results.setdefault(tname, [])
                    if parsed not in existing:
                        existing.append(parsed)
            except Exception:
                pass
    return results


def _classify_symptom(text: str) -> SymptomCategory:
    low = text.lower()
    if any(w in low for w in ["мотор", "заклинил", "редуктор", "скрежет", "шум", "рывк", "дёрг"]):
        return "MOTOR_STALL_OR_NOISE"
    if any(w in low for w in ["не включается", "не работает", "мёртв", "мертв", "нет питания"]):
        return "NO_POWER_DEAD"
    if any(
        w in low for w in ["батаре", "акб", "заряд", "запас хода", "разряд", "заезд", "-5", "холод"]
    ):
        return "BATTERY_LOW_RANGE"
    return "OTHER"


def _extract_conditions(text: str) -> str | None:
    m = re.search(
        r"(-?\d+\s*°?C|-?\d+\s*град[^\s,.]*|горк[а-я]*|нагрузк[а-я]*|уличн[а-я]*|прохлад[а-я]*)",
        text,
        re.IGNORECASE,
    )
    return m.group(0) if m else None


def build_mock_model_bundle(*, force_auditor_reject_once: bool = False) -> AgentModelBundle:
    """Build deterministic ScriptedChatModels that drive the real SQL tools & BPMN/RAG pipeline."""

    def fast_responder(
        messages: list[BaseMessage], schema: type[BaseModel] | None, _tools: list[str]
    ) -> BaseModel:
        full_text = "\n".join(str(m.content) for m in messages)
        if schema is IntakeResult:
            user_text = str(messages[-1].content) if messages else full_text
            m_serial = SERIAL_RE.search(user_text)
            serial = m_serial.group(1).upper() if m_serial else None
            cat = _classify_symptom(user_text)
            cond = _extract_conditions(user_text)
            if not serial:
                return IntakeResult(
                    tracking_code=None,
                    symptom_category=cat,
                    operating_conditions=cond,
                    needs_clarification=True,
                    clarification_message="Укажите, пожалуйста, серийный номер машинки (например CAR-2026-0412) для диагностики в MES.",
                )
            return IntakeResult(
                tracking_code=serial,
                symptom_category=cat,
                operating_conditions=cond,
                needs_clarification=False,
                clarification_message=None,
            )

        if schema is WebHypothesesOutput:
            # Parse snippets from raw_snippets in prompt
            urls = re.findall(r"URL:\s*(https?://\S+)", full_text)
            clean_urls = [u for u in urls if u] or ["offline-engineering-fallback://SOP-01"]
            return WebHypothesesOutput(
                hypotheses=[
                    ExternalHypothesis(
                        title="Температурная отсечка и просадка напряжения Li-Po 6S на JK-BMS-6S при -5°C",
                        confidence=0.88,
                        explanation=(
                            "При отрицательных температурах (-5°C) внутреннее сопротивление Li-Po ячеек возрастает, "
                            "что под пиковым стартовым током мотора RS550-24V вызывает мгновенную просадку напряжения (voltage sag) "
                            "и срабатывание низкотемпературной / UVP отсечки JK-BMS-6S."
                        ),
                        sources=clean_urls[:2],
                        mechanic_check_steps=[
                            "Провести контрольный заезд в тёплом помещении (+20°C) для подтверждения восстановления запаса хода.",
                            "Проверить пороги низкотемпературной защиты и UVP в конфигурации JK-BMS-6S.",
                            "Установить термоизоляцию батарейного отсека для зимней уличной эксплуатации.",
                        ],
                    ),
                    ExternalHypothesis(
                        title="Токовая перегрузка двигателя RS550-24V при движении в горку на холодном редукторе",
                        confidence=0.72,
                        explanation=(
                            "Загустевание смазки редуктора на холоде увеличивает пусковой момент и ток мотора RS550-24V, "
                            "усиливая просадку на клеммах батареи."
                        ),
                        sources=clean_urls[:1],
                        mechanic_check_steps=[
                            "Проверить лёгкость вращения редуктора при -5°C и измерить пусковой ток клещами.",
                        ],
                    ),
                ]
            )

        raise ValueError(f"Unsupported FAST schema: {schema}")

    def investigator_responder(
        messages: list[BaseMessage], schema: type[BaseModel] | None, bound_tools: list[str]
    ) -> AIMessage | BaseModel:
        full_text = "\n".join(str(m.content) for m in messages)
        ev = _extract_tool_results_from_text_and_msgs(messages)

        m_serial = SERIAL_RE.search(full_text)
        serial = m_serial.group(1).upper() if m_serial else "CAR-2026-0412"
        has_feedback = "ЗАМЕЧАНИЯ РЕВИЗОРА" in full_text

        # Tool-calling phase
        if schema is None and bound_tools:
            if "get_item_genealogy" not in ev:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_item_genealogy",
                            "args": {"tracking_code": serial},
                            "id": "tc_gen_1",
                        }
                    ],
                )

            gen = ev["get_item_genealogy"][-1]
            target_rs_id = gen.get("target_route_sheet_id")
            lots = gen.get("root_lots") or []
            lot_code = lots[0].get("tracking_code") if lots else None

            if "get_item_history" not in ev and target_rs_id:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_item_history",
                            "args": {"route_sheet_id": target_rs_id},
                            "id": "tc_hist_1",
                        }
                    ],
                )

            if lot_code and "get_lot_quality_stats" not in ev:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_lot_quality_stats",
                            "args": {"lot_tracking_code_or_id": lot_code},
                            "id": "tc_lot_1",
                        }
                    ],
                )

            lot_stats = ev.get("get_lot_quality_stats", [{}])[-1]
            if (
                lot_code
                and lot_stats.get("exceeds_10pct_recall_threshold")
                and "find_affected_products_by_lot" not in ev
            ):
                # Optional reflection test: skip recall on first pass if requested!
                if not (force_auditor_reject_once and not has_feedback):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "find_affected_products_by_lot",
                                "args": {"lot_tracking_code_or_id": lot_code},
                                "id": "tc_rec_1",
                            }
                        ],
                    )

            # Inspect history for instrument/channel and operator checks
            hist = ev.get("get_item_history", [{}])[-1]
            car_execs = hist.get("executions") or []
            comp_hists = hist.get("component_histories") or {}
            bat_execs: list[dict[str, Any]] = []
            for hlist in comp_hists.values():
                if isinstance(hlist, list):
                    bat_execs.extend(hlist)

            # Find charging stand execution
            chg_exec = next(
                (
                    e
                    for e in bat_execs
                    if e.get("operation_code") == "OP-BAT-CHG" and e.get("instrument_code")
                ),
                None,
            )
            if chg_exec and "get_instrument_channel_anomalies" not in ev:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_instrument_channel_anomalies",
                            "args": {
                                "instrument_code": chg_exec["instrument_code"],
                                "channel_no": chg_exec.get("channel_no"),
                                "around_date": str(chg_exec["completed_at"])[:10],
                            },
                            "id": "tc_instr_1",
                        }
                    ],
                )

            # Find car drive/assembly operator execution
            drv_exec = next(
                (e for e in car_execs if e.get("operation_code") == "OP-CAR-DRIVE"),
                car_execs[0] if car_execs else None,
            )
            if drv_exec and "get_operator_anomalies" not in ev:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_operator_anomalies",
                            "args": {
                                "operator_id_or_name": drv_exec["operator_code"],
                                "around_date": str(drv_exec["completed_at"])[:10],
                            },
                            "id": "tc_op_1",
                        }
                    ],
                )

            return AIMessage(
                content="Все проверки по ветке BPMN выполнены, перехожу к формированию черновика."
            )

        # Structured InvestigationDraft synthesis strictly from evidence_log
        if schema is InvestigationDraft:
            gen = ev.get("get_item_genealogy", [{}])[-1]
            hist = ev.get("get_item_history", [{}])[-1]
            lot_stats = ev.get("get_lot_quality_stats", [{}])[-1]
            recall = ev.get("find_affected_products_by_lot", [{}])[-1]
            instr = ev.get("get_instrument_channel_anomalies", [{}])[-1]
            op_anom = ev.get("get_operator_anomalies", [{}])[-1]

            facts: list[str] = []
            actions: list[ProposedAction] = []
            lot_code = lot_stats.get("lot_tracking_code")
            prior_tickets = hist.get("prior_support_tickets") or []
            affected_cars = list(recall.get("affected_car_codes") or [])

            if not gen.get("found"):
                return InvestigationDraft(
                    root_cause_hypothesis=f"Изделие {serial} не найдено в MES.",
                    root_cause_type="INCONCLUSIVE",
                    db_anomalies_found=False,
                    cited_db_facts=[f"Изделие {serial} отсутствует в route_sheets."],
                    confidence=0.95,
                )

            facts.append(
                f"Машинка {serial}: корневая партия АКБ {lot_code}, поставщик {lot_stats.get('supplier_name')}, "
                f"дефектность партии {lot_stats.get('defect_rate_pct')}% ({lot_stats.get('defect_units')}/{lot_stats.get('total_split_units')}), "
                f"REWORK в партии: {lot_stats.get('rework_units')}."
            )
            if instr.get("found"):
                tstat = instr.get("target_channel_stats") or {}
                bstat = instr.get("other_channels_baseline") or {}
                facts.append(
                    f"Зарядный стенд {instr.get('instrument_code')} канал {instr.get('channel_no')}: "
                    f"поверка до {str(instr.get('calibration_due_at'))[:10]} (просрочена={instr.get('calibration_expired_in_window')}), "
                    f"аномалии канала {tstat.get('anomaly_rate_pct')}% (среднее {tstat.get('avg_numeric_value')} В) vs "
                    f"база других каналов {bstat.get('anomaly_rate_pct')}% ({bstat.get('avg_numeric_value')} В)."
                )
            if op_anom.get("found"):
                os_t = op_anom.get("operator_stats") or {}
                os_b = op_anom.get("shop_other_operators_baseline") or {}
                facts.append(
                    f"Оператор {op_anom.get('operator_name')} ({op_anom.get('operator_code')}): "
                    f"REWORK/брак {os_t.get('rework_rate_pct')}% (средний момент {os_t.get('avg_torque_nm')} Н·м) vs "
                    f"среднее по цеху {os_b.get('rework_rate_pct')}%."
                )

            # Priority 1: Repeat complaint
            if prior_tickets:
                t0 = prior_tickets[0]
                facts.append(
                    f"Повторное обращение: прошлый тикет {t0.get('ticket_id')} ({t0.get('status')}) от {str(t0.get('created_at'))[:10]}, "
                    f"REWORK сборки={hist.get('rework_count')}."
                )
                actions.append(
                    ProposedAction(
                        action_type="ESCALATE_REPEAT_TICKET",
                        title=f"Эскалация повторного инцидента {serial}",
                        description="Повторное обращение по тому же серийнику — эскалация ведущему инженеру сервиса.",
                        priority="CRITICAL",
                        target_tracking_code=serial,
                    )
                )
                return InvestigationDraft(
                    root_cause_hypothesis=f"Повторный отказ привода на {serial} после предыдущего REWORK на тестовом заезде и закрытого тикета.",
                    root_cause_type="REPEAT_FAILURE",
                    db_anomalies_found=True,
                    cited_db_facts=facts,
                    is_repeat_complaint=True,
                    recommended_actions=actions,
                    confidence=0.96,
                )

            # Priority 2: Lot defect > 10%
            if lot_stats.get("exceeds_10pct_recall_threshold"):
                actions.append(
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title=f"Сервисный тикет по деградации АКБ {serial}",
                        description=f"Деградация ёмкости партии {lot_code} ({lot_stats.get('defect_rate_pct')}% брака).",
                        priority="CRITICAL",
                        target_tracking_code=serial,
                    )
                )
                actions.append(
                    ProposedAction(
                        action_type="CREATE_LOT_RECALL",
                        title=f"Отзыв машинок партии {lot_code}",
                        description=f"Брак партии {lot_code} превысил 10% ({lot_stats.get('defect_rate_pct')}%). Проверить {len(affected_cars)} машинок.",
                        priority="CRITICAL",
                        target_lot_code=lot_code,
                        affected_cars=affected_cars,
                    )
                )
                return InvestigationDraft(
                    root_cause_hypothesis=f"Системный брак партии АКБ {lot_code} ({lot_stats.get('defect_rate_pct')}% брака по разрядной ёмкости от поставщика {lot_stats.get('supplier_name')}).",
                    root_cause_type="LOT_DEFECT",
                    db_anomalies_found=True,
                    cited_db_facts=facts,
                    affected_cars=affected_cars,
                    target_lot_code=lot_code,
                    recommended_actions=actions,
                    confidence=0.97,
                )

            # Priority 3: Instrument channel drift / expired calibration
            if instr.get("is_channel_anomalous") or instr.get("calibration_expired_in_window"):
                icode = instr.get("instrument_code")
                ch = instr.get("channel_no")
                actions.append(
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title=f"Недозаряд АКБ {serial} на {icode} канал {ch}",
                        description="Замена/перезаряд АКБ после дрейфа канала зарядного стенда.",
                        priority="HIGH",
                        target_tracking_code=serial,
                    )
                )
                actions.append(
                    ProposedAction(
                        action_type="CREATE_CALIBRATION_HOLD",
                        title=f"Блокировка и поверка {icode} (канал {ch})",
                        description=f"Просрочена поверка ({str(instr.get('calibration_due_at'))[:10]}) и подтверждён дрейф канала {ch}.",
                        priority="HIGH",
                        target_instrument_code=icode,
                    )
                )
                return InvestigationDraft(
                    root_cause_hypothesis=f"Дрейф канала {ch} зарядного стенда {icode} и работа после истечения срока поверки.",
                    root_cause_type="INSTRUMENT_CHANNEL_DRIFT",
                    db_anomalies_found=True,
                    cited_db_facts=facts,
                    target_instrument_code=icode,
                    target_channel_no=ch,
                    recommended_actions=actions,
                    confidence=0.95,
                )

            # Priority 4: Operator quality drop
            if op_anom.get("is_operator_anomalous"):
                op_name = op_anom.get("operator_name")
                actions.append(
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title=f"Дефект сборки привода {serial} (оператор {op_name})",
                        description="Переборка и проверка зацепления редуктора RS550-24V после перетяжки и REWORK на смене.",
                        priority="HIGH",
                        target_tracking_code=serial,
                    )
                )
                return InvestigationDraft(
                    root_cause_hypothesis=f"Просадка качества сборки у оператора {op_name} ({op_anom.get('operator_stats', {}).get('rework_rate_pct')}% REWORK в окне ±14 дней, перетяжка редуктора RS550-24V).",
                    root_cause_type="OPERATOR_QUALITY_DROP",
                    db_anomalies_found=True,
                    cited_db_facts=facts,
                    recommended_actions=actions,
                    confidence=0.94,
                )

            # Clean DB -> external factors suspected
            return InvestigationDraft(
                root_cause_hypothesis="Внутренняя БД производства 100% чистая (партия, стенды, поверки и оператор в норме). Требуется внешний поиск по BOM (RS550-24V, JK-BMS-6S).",
                root_cause_type="DB_CLEAN_EXTERNAL_SUSPECTED",
                db_anomalies_found=False,
                cited_db_facts=facts,
                recommended_actions=[],
                confidence=0.85,
            )

        if schema is FinalVerdict:
            facts_match = re.search(r"Черновик:\s*(\{.*\})", full_text)
            draft_dict: dict[str, Any] = {}
            if facts_match:
                try:
                    draft_dict = json.loads(facts_match.group(1))
                except Exception:
                    pass
            cited = list(draft_dict.get("cited_db_facts") or [])
            aff = list(draft_dict.get("affected_cars") or [])
            acts = [
                ProposedAction.model_validate(a)
                for a in (draft_dict.get("recommended_actions") or [])
            ]
            hyp = str(draft_dict.get("root_cause_hypothesis") or "Расследование завершено.")
            rc_type = str(draft_dict.get("root_cause_type") or "")
            sev: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = (
                "CRITICAL"
                if rc_type in {"LOT_DEFECT", "REPEAT_FAILURE"}
                else ("HIGH" if draft_dict.get("db_anomalies_found") else "MEDIUM")
            )
            if not acts:
                acts = [
                    ProposedAction(
                        action_type="CREATE_SUPPORT_TICKET",
                        title=f"Инженерная проверка {serial}",
                        description=hyp,
                        priority=sev,
                        target_tracking_code=serial,
                    )
                ]
            return FinalVerdict(
                summary=hyp,
                severity=sev,
                confirmed_db_facts=cited,
                external_hypotheses=[],
                affected_cars=aff,
                proposed_actions=acts,
            )

        raise ValueError(f"Unsupported INVESTIGATOR call: schema={schema}")

    def auditor_responder(
        messages: list[BaseMessage], schema: type[BaseModel] | None, _tools: list[str]
    ) -> BaseModel:
        assert schema is AuditVerdict
        ev = _extract_tool_results_from_text_and_msgs(messages)
        full_text = "\n".join(str(m.content) for m in messages)

        missing: list[str] = []
        if "get_item_genealogy" not in ev:
            missing.append("get_item_genealogy")
        gen = ev.get("get_item_genealogy", [{}])[-1]
        if gen and not gen.get("found"):
            return AuditVerdict(
                decision="APPROVED",
                feedback="Изделие отсутствует в БД, оснований для веб-поиска нет.",
                db_is_clean=False,
            )
        if "get_item_history" not in ev:
            missing.append("get_item_history")

        lot_stats = ev.get("get_lot_quality_stats", [{}])[-1]
        if (
            lot_stats.get("exceeds_10pct_recall_threshold")
            and "find_affected_products_by_lot" not in ev
        ):
            missing.append("find_affected_products_by_lot (обязателен при браке партии > 10%!)")

        if missing:
            return AuditVerdict(
                decision="REJECTED",
                feedback="Не выполнены обязательные шаги по регламенту BPMN: " + "; ".join(missing),
                missing_checks=missing,
                db_is_clean=False,
            )

        hist = ev.get("get_item_history", [{}])[-1]
        instr = ev.get("get_instrument_channel_anomalies", [{}])[-1]
        op_anom = ev.get("get_operator_anomalies", [{}])[-1]

        has_db_anomaly = any(
            [
                bool(lot_stats.get("exceeds_10pct_recall_threshold")),
                int(hist.get("rework_count") or 0) > 0,
                int(hist.get("component_rework_count") or 0) > 0,
                bool(hist.get("prior_support_tickets")),
                bool(instr.get("is_channel_anomalous")),
                bool(instr.get("calibration_expired_in_window")),
                bool(op_anom.get("is_operator_anomalous")),
            ]
        )

        # Verify clean DB checks were run before routing to web_researcher
        if not has_db_anomaly:
            if "get_instrument_channel_anomalies" not in ev or "get_operator_anomalies" not in ev:
                return AuditVerdict(
                    decision="REJECTED",
                    feedback="Перед признанием БД чистой необходимо проверить канал стенда и статистику оператора.",
                    missing_checks=["get_instrument_channel_anomalies", "get_operator_anomalies"],
                    db_is_clean=False,
                )
            return AuditVerdict(
                decision="NEED_WEB_RESEARCH",
                feedback="Все проверки БД выполнены, аномалий не выявлено (БД 100% чистая). Запустить внешний поиск по BOM.",
                db_is_clean=True,
            )

        # Ensure draft isn't claiming clean when DB has anomaly
        if '"db_anomalies_found":false' in full_text.replace(" ", "").lower():
            return AuditVerdict(
                decision="REJECTED",
                feedback="В evidence_log присутствуют подтверждённые аномалии БД, черновик ошибочно признал БД чистой.",
                db_is_clean=False,
            )

        return AuditVerdict(
            decision="APPROVED",
            feedback="Первопричина полностью подтверждена фактами из evidence_log и соответствует BPMN.",
            db_is_clean=False,
        )

    def judge_responder(
        _messages: list[BaseMessage], schema: type[BaseModel] | None, _tools: list[str]
    ) -> BaseModel:
        if schema is not None:
            return schema.model_validate(
                {
                    "score": 0.92,
                    "rationale": "Оффлайн-судья (mock): диагноз и предложенные действия соответствуют фактам и эталону.",
                }
            )
        raise ValueError("Unsupported JUDGE call without schema")

    return AgentModelBundle(
        fast_model=ScriptedChatModel(
            role_name="FAST", model_id="mock/gemini-2.5-flash", responder=fast_responder
        ),
        investigator_model=ScriptedChatModel(
            role_name="INVESTIGATOR",
            model_id="mock/deepseek-v3.2",
            responder=investigator_responder,
        ),
        auditor_model=ScriptedChatModel(
            role_name="AUDITOR", model_id="mock/gpt-4.1-mini", responder=auditor_responder
        ),
        judge_model=ScriptedChatModel(
            role_name="JUDGE", model_id="mock/claude-haiku-4.5", responder=judge_responder
        ),
        fast_model_name="mock/gemini-2.5-flash",
        investigator_model_name="mock/deepseek-v3.2",
        auditor_model_name="mock/gpt-4.1-mini",
        judge_model_name="mock/claude-haiku-4.5",
    )
