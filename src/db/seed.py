from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import PROJECT_ROOT, get_settings
from src.db.enums import (
    GuardValidationType,
    InstrumentType,
    NodeExecutionScope,
    RouteSheetLinkType,
    RouteSheetStatus,
    SamplingAggregationType,
    SamplingNonSampledPolicy,
    SamplingResolutionScope,
    StepOutcome,
    SupportTicketPriority,
    SupportTicketStatus,
    TransitionConditionLabel,
    TransitionType,
)
from src.db.models import (
    Instrument,
    LotMetadata,
    Operation,
    Role,
    RouteSheet,
    RouteSheetLink,
    SamplingNodeState,
    StepExecution,
    StepMeasurement,
    SupportTicket,
    TechMap,
    TechMapNode,
    TechMapNodeGuard,
    TechMapSamplingRule,
    TechMapTransition,
    User,
    Workstation,
)
from src.db.session import build_engine, build_session_factory
from src.mes.sampling import (
    SamplingDecision,
    SamplingState,
    check_stop_crane,
    compute_hmac_uniform,
    init_sampling_state,
    step_sampling,
)

SEED_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def det_uuid(key: str) -> UUID:
    return uuid5(SEED_NAMESPACE, key)


async def seed_database(session: AsyncSession, *, write_manifest: bool = True) -> dict[str, Any]:
    """Deterministically populate 12 months of MES data (~451 cars + 5 planted anomalies)."""
    rng = random.Random(20260101)
    settings = get_settings()

    # Clean domain tables in a single fast TRUNCATE (does not touch LangGraph checkpoint tables)
    await session.execute(
        text(
            "TRUNCATE TABLE "
            "step_measurements, step_executions, sampling_node_states, "
            "support_tickets, recall_notices, route_sheet_links, lot_metadata, route_sheets, "
            "tech_map_transitions, tech_map_sampling_rules, tech_map_node_guards, tech_map_nodes, tech_maps, "
            "instruments, operations, workstations, users, roles "
            "RESTART IDENTITY CASCADE"
        )
    )

    now_ref = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)

    # -------------------------------------------------------------------------
    # 1. Roles & Users (6 operators + 4 engineering/management)
    # -------------------------------------------------------------------------
    role_defs = [
        ("Оператор", "Оператор сборочной линии", True),
        ("Технолог", "Инженер-технолог производства", True),
        ("Начальник участка", "Руководитель смены и участка", True),
        ("Инженер ОТК", "Контроль качества продукции", True),
        ("Кладовщик", "Комплектовщик и кладовщик", True),
        ("Администратор", "Администратор MES", True),
    ]
    role_ids = {name: det_uuid(f"role:{name}") for name, _, _ in role_defs}
    await session.execute(
        insert(Role),
        [
            {
                "id": role_ids[name],
                "name": name,
                "description": desc,
                "is_system": is_sys,
                "created_at": now_ref,
                "updated_at": now_ref,
            }
            for name, desc, is_sys in role_defs
        ],
    )

    user_defs = [
        ("TECH-01", "Соколов", "Иван", "Николаевич", "sokolov@langveyor.local", "Технолог"),
        ("SUP-01", "Волков", "Дмитрий", "Сергеевич", "volkov@langveyor.local", "Начальник участка"),
        ("OTK-01", "Морозова", "Елена", "Алексеевна", "morozova@langveyor.local", "Инженер ОТК"),
        ("ADM-01", "Лебедев", "Артём", "Олегович", "admin@langveyor.local", "Администратор"),
        ("OP-01", "Иванов", "Сергей", "Петрович", "ivanov@langveyor.local", "Оператор"),
        ("OP-02", "Петров", "Андрей", "Викторович", "petrov@langveyor.local", "Оператор"),
        ("OP-03", "Сидорова", "Мария", "Константиновна", "sidorova@langveyor.local", "Оператор"),
        ("OP-04", "Козлов", "Денис", "Игоревич", "kozlov@langveyor.local", "Оператор"),
        ("OP-05", "Смирнова", "Ольга", "Николаевна", "smirnova@langveyor.local", "Оператор"),
        ("OP-06", "Кузнецов", "Виктор", "Андреевич", "kuznetsov@langveyor.local", "Оператор"),
    ]
    user_ids = {code: det_uuid(f"user:{code}") for code, *_ in user_defs}
    await session.execute(
        insert(User),
        [
            {
                "id": user_ids[code],
                "role_id": role_ids[role_name],
                "employee_code": code,
                "last_name": ln,
                "first_name": fn,
                "middle_name": mn,
                "email": email,
                "is_active": True,
                "created_at": now_ref,
                "updated_at": now_ref,
            }
            for code, ln, fn, mn, email, role_name in user_defs
        ],
    )
    operators = ["OP-01", "OP-03", "OP-04", "OP-05", "OP-06"]
    petrov_id = user_ids["OP-02"]

    # -------------------------------------------------------------------------
    # 2. Workstations, Operations, Instruments
    # -------------------------------------------------------------------------
    ws_defs = [
        ("WS-BAT-IN", "Пост приёмки партий АКБ"),
        ("WS-BAT-SPLIT", "Пост распаковки и маркировки АКБ"),
        ("WS-BAT-VIS", "Пост визуального осмотра АКБ"),
        ("WS-BAT-CHG", "Пост многоканального заряда АКБ"),
        ("WS-BAT-TST", "Пост электрических испытаний АКБ"),
        ("WS-BAT-LAB", "Лабораторный стенд глубокого циклирования АКБ"),
        ("WS-BAT-DONE", "Склад годных АКБ"),
        ("WS-BAT-RWK", "Пост доработки АКБ"),
        ("WS-BAT-DEF", "Изолятор брака АКБ"),
        ("WS-BODY-PAINT", "Пост контроля пластика и покраски"),
        ("WS-BODY-STEER", "Пост сборки рулевого узла и сиденья"),
        ("WS-BODY-CHK", "Пост контроля крепежа кузова"),
        ("WS-BODY-RWK", "Пост доработки кузова"),
        ("WS-CAR-DRIVE", "Пост монтажа редуктора и мотора RS550-24V"),
        ("WS-CAR-ASM", "Пост узловой комплектации машинки"),
        ("WS-CAR-TEST", "Стенд тестового заезда"),
        ("WS-CAR-QC", "Финальный пост ОТК готовой машинки"),
        ("WS-CAR-RWK", "Пост доработки готовой машинки"),
    ]
    ws_ids = {code: det_uuid(f"ws:{code}") for code, _ in ws_defs}
    await session.execute(
        insert(Workstation),
        [
            {
                "id": ws_ids[code],
                "code": code,
                "name": name,
                "is_active": True,
                "created_at": now_ref,
            }
            for code, name in ws_defs
        ],
    )

    op_defs = [
        ("OP-BAT-IN", "Входной контроль тары и пломб партии"),
        ("OP-BAT-SPLIT", "Распаковка и сериализация АКБ"),
        ("OP-BAT-VIS", "Визуальный осмотр корпуса и клемм АКБ"),
        ("OP-BAT-CHG", "Зарядка и балансировка АКБ на стенде"),
        ("OP-BAT-TST", "Электрические испытания АКБ (ёмкость и R_ac)"),
        ("OP-BAT-LAB", "Углублённое тестирование выборки АКБ"),
        ("OP-BAT-DONE", "Приёмка АКБ на комплектацию"),
        ("OP-BAT-RWK", "Перезаряд и восстановление АКБ"),
        ("OP-BAT-DEF", "Отбраковка АКБ в карантин"),
        ("OP-BODY-PAINT", "Контроль покраски и пластика кузова"),
        ("OP-BODY-STEER", "Сборка рулевой колонки и сиденья"),
        ("OP-BODY-CHK", "Проверка крепежа и жгутов кузова"),
        ("OP-BODY-RWK", "Устранение замечаний по кузову"),
        ("OP-CAR-DRIVE", "Монтаж редуктора и мотора RS550-24V"),
        ("OP-CAR-ASM", "Сборка машинки (кузов + АКБ + JK-BMS-6S)"),
        ("OP-CAR-TEST", "Тестовый заезд под нагрузкой"),
        ("OP-CAR-QC", "Финальный приёмочный контроль ОТК"),
        ("OP-CAR-RWK", "Доработка ходовой и электроузлов машинки"),
    ]
    op_ids = {code: det_uuid(f"op:{code}") for code, _ in op_defs}
    await session.execute(
        insert(Operation),
        [
            {
                "id": op_ids[code],
                "code": code,
                "name": name,
                "is_active": True,
                "created_at": now_ref,
            }
            for code, name in op_defs
        ],
    )

    # Planted Anomaly #2: STAND-03 calibration expired 2026-09-30 before October 2026 issues!
    instr_defs = [
        (
            "STAND-01",
            "Зарядный стенд №1 (16 каналов)",
            InstrumentType.CHARGING_STAND.value,
            "WS-BAT-CHG",
            16,
            datetime(2025, 12, 15, tzinfo=UTC),
            datetime(2027, 6, 15, tzinfo=UTC),
        ),
        (
            "STAND-02",
            "Зарядный стенд №2 (16 каналов)",
            InstrumentType.CHARGING_STAND.value,
            "WS-BAT-CHG",
            16,
            datetime(2025, 11, 1, tzinfo=UTC),
            datetime(2027, 5, 1, tzinfo=UTC),
        ),
        (
            "STAND-03",
            "Зарядный стенд №3 (16 каналов)",
            InstrumentType.CHARGING_STAND.value,
            "WS-BAT-CHG",
            16,
            datetime(2025, 9, 30, tzinfo=UTC),
            datetime(2026, 9, 30, tzinfo=UTC),
        ),
        (
            "TESTER-IR-01",
            "Анализатор АКБ и R_ac Hioki BT3554",
            InstrumentType.IR_TESTER.value,
            "WS-BAT-TST",
            1,
            datetime(2026, 2, 1, tzinfo=UTC),
            datetime(2027, 8, 1, tzinfo=UTC),
        ),
        (
            "TORQUE-01",
            "Динамометрический ключ Tohnichi CEM3",
            InstrumentType.TORQUE_WRENCH.value,
            "WS-CAR-DRIVE",
            1,
            datetime(2026, 1, 20, tzinfo=UTC),
            datetime(2027, 4, 20, tzinfo=UTC),
        ),
        (
            "TEST-RIG-01",
            "Испытательный беговой стенд",
            InstrumentType.TEST_RIG.value,
            "WS-CAR-TEST",
            1,
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2027, 7, 10, tzinfo=UTC),
        ),
    ]
    instr_ids = {code: det_uuid(f"instr:{code}") for code, *_ in instr_defs}
    await session.execute(
        insert(Instrument),
        [
            {
                "id": instr_ids[code],
                "code": code,
                "name": name,
                "instrument_type": itype,
                "workstation_id": ws_ids[ws_code],
                "channel_count": ch,
                "last_calibrated_at": last_cal,
                "calibration_due_at": due_cal,
                "is_active": True,
                "created_at": now_ref,
            }
            for code, name, itype, ws_code, ch, last_cal, due_cal in instr_defs
        ],
    )

    # -------------------------------------------------------------------------
    # 3. TechMaps, Nodes, Guards, SamplingRules, Transitions
    # -------------------------------------------------------------------------
    tm_bat_id = det_uuid("tm:TM-BAT-6S")
    tm_body_id = det_uuid("tm:TM-BODY")
    tm_car_id = det_uuid("tm:TM-CAR-FINAL")

    await session.execute(
        insert(TechMap),
        [
            {
                "id": tm_bat_id,
                "item_code": "TM-BAT-6S",
                "name": "Приёмка и зарядка АКБ 6S 35Ah",
                "version": 1,
                "description": "Гибридный маршрут LOT -> SPLIT -> ITEM с многоканальной зарядкой и выборкой ALG-001",
                "base_cycle_time_seconds": 3600,
                "is_approved": True,
                "approved_by_id": user_ids["TECH-01"],
                "approved_at": datetime(2025, 12, 20, tzinfo=UTC),
                "is_active": True,
                "activated_by_id": user_ids["SUP-01"],
                "activated_at": datetime(2025, 12, 21, tzinfo=UTC),
                "created_by_id": user_ids["TECH-01"],
                "stop_min_defects": 3,
                "stop_defect_slope": Decimal("0.3500"),
                "created_at": datetime(2025, 12, 20, tzinfo=UTC),
                "updated_at": datetime(2025, 12, 21, tzinfo=UTC),
            },
            {
                "id": tm_body_id,
                "item_code": "TM-BODY",
                "name": "Сборка кузова машинки из запчастей",
                "version": 1,
                "description": "Поштучная сборка кузова электромашинки",
                "base_cycle_time_seconds": 1200,
                "is_approved": True,
                "approved_by_id": user_ids["TECH-01"],
                "approved_at": datetime(2025, 12, 20, tzinfo=UTC),
                "is_active": True,
                "activated_by_id": user_ids["SUP-01"],
                "activated_at": datetime(2025, 12, 21, tzinfo=UTC),
                "created_by_id": user_ids["TECH-01"],
                "stop_min_defects": 3,
                "stop_defect_slope": Decimal("0.0800"),
                "created_at": datetime(2025, 12, 20, tzinfo=UTC),
                "updated_at": datetime(2025, 12, 21, tzinfo=UTC),
            },
            {
                "id": tm_car_id,
                "item_code": "TM-CAR-FINAL",
                "name": "Монтаж ходовой части с электромотором и финальная сборка",
                "version": 1,
                "description": "Монтаж RS550-24V, комплектация ASSEMBLED_INTO (BODY+BAT+JK-BMS-6S), тестовый заезд и ОТК",
                "base_cycle_time_seconds": 1800,
                "is_approved": True,
                "approved_by_id": user_ids["TECH-01"],
                "approved_at": datetime(2025, 12, 20, tzinfo=UTC),
                "is_active": True,
                "activated_by_id": user_ids["SUP-01"],
                "activated_at": datetime(2025, 12, 21, tzinfo=UTC),
                "created_by_id": user_ids["TECH-01"],
                "stop_min_defects": 3,
                "stop_defect_slope": Decimal("0.0800"),
                "created_at": datetime(2025, 12, 20, tzinfo=UTC),
                "updated_at": datetime(2025, 12, 21, tzinfo=UTC),
            },
        ],
    )

    # Nodes
    node_specs = [
        # TM-BAT-6S
        (
            "BAT-IN",
            tm_bat_id,
            "OP-BAT-IN",
            "WS-BAT-IN",
            1,
            True,
            False,
            NodeExecutionScope.LOT.value,
        ),
        (
            "BAT-SPLIT",
            tm_bat_id,
            "OP-BAT-SPLIT",
            "WS-BAT-SPLIT",
            2,
            False,
            False,
            NodeExecutionScope.LOT.value,
        ),
        (
            "BAT-VIS",
            tm_bat_id,
            "OP-BAT-VIS",
            "WS-BAT-VIS",
            3,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-CHG",
            tm_bat_id,
            "OP-BAT-CHG",
            "WS-BAT-CHG",
            4,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-TST",
            tm_bat_id,
            "OP-BAT-TST",
            "WS-BAT-TST",
            5,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-LAB",
            tm_bat_id,
            "OP-BAT-LAB",
            "WS-BAT-LAB",
            6,
            False,
            True,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-DONE",
            tm_bat_id,
            "OP-BAT-DONE",
            "WS-BAT-DONE",
            6,
            False,
            True,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-RWK",
            tm_bat_id,
            "OP-BAT-RWK",
            "WS-BAT-RWK",
            7,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BAT-DEF",
            tm_bat_id,
            "OP-BAT-DEF",
            "WS-BAT-DEF",
            8,
            False,
            True,
            NodeExecutionScope.ITEM.value,
        ),
        # TM-BODY
        (
            "BODY-PAINT",
            tm_body_id,
            "OP-BODY-PAINT",
            "WS-BODY-PAINT",
            1,
            True,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BODY-STEER",
            tm_body_id,
            "OP-BODY-STEER",
            "WS-BODY-STEER",
            2,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BODY-CHK",
            tm_body_id,
            "OP-BODY-CHK",
            "WS-BODY-CHK",
            3,
            False,
            True,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "BODY-RWK",
            tm_body_id,
            "OP-BODY-RWK",
            "WS-BODY-RWK",
            4,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        # TM-CAR-FINAL
        (
            "CAR-DRIVE",
            tm_car_id,
            "OP-CAR-DRIVE",
            "WS-CAR-DRIVE",
            1,
            True,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "CAR-ASM",
            tm_car_id,
            "OP-CAR-ASM",
            "WS-CAR-ASM",
            2,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "CAR-TEST",
            tm_car_id,
            "OP-CAR-TEST",
            "WS-CAR-TEST",
            3,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "CAR-QC",
            tm_car_id,
            "OP-CAR-QC",
            "WS-CAR-QC",
            4,
            False,
            True,
            NodeExecutionScope.ITEM.value,
        ),
        (
            "CAR-RWK",
            tm_car_id,
            "OP-CAR-RWK",
            "WS-CAR-RWK",
            5,
            False,
            False,
            NodeExecutionScope.ITEM.value,
        ),
    ]
    node_ids = {code: det_uuid(f"node:{code}") for code, *_ in node_specs}
    await session.execute(
        insert(TechMapNode),
        [
            {
                "id": node_ids[code],
                "tech_map_id": tm_id,
                "operation_id": op_ids[op_c],
                "workstation_id": ws_ids[ws_c],
                "step_order": step_ord,
                "cycle_time_seconds": 300,
                "is_start_node": is_s,
                "is_end_node": is_e,
                "execution_scope": scope,
                "created_at": now_ref,
            }
            for code, tm_id, op_c, ws_c, step_ord, is_s, is_e, scope in node_specs
        ],
    )

    # Guards
    guard_specs = [
        (
            "G-BAT-SEAL",
            "BAT-IN",
            "Целостность тары и заводской пломбы",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Повреждена тара или пломба партии",
        ),
        (
            "G-BAT-BARCODE",
            "BAT-SPLIT",
            "Валидация серийного штрихкода АКБ",
            GuardValidationType.BARCODE_REGEX.value,
            None,
            None,
            None,
            None,
            None,
            r"^BAT-2026-\d{3}-\d{3}$",
            "Неверный формат серийного номера АКБ",
        ),
        (
            "G-BAT-VIS",
            "BAT-VIS",
            "Отсутствие механических повреждений и следов электролита",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Выявлен дефект корпуса/клемм АКБ",
        ),
        (
            "G-BAT-VOLT",
            "BAT-CHG",
            "Напряжение после зарядки 6S",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("24.6000"),
            Decimal("25.4000"),
            Decimal("25.2000"),
            "V",
            None,
            None,
            "Напряжение АКБ 6S вне допуска 24.6-25.4 В",
        ),
        (
            "G-BAT-CAP",
            "BAT-TST",
            "Разрядная ёмкость АКБ 6S",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("33.0000"),
            Decimal("37.5000"),
            Decimal("35.0000"),
            "Ah",
            None,
            None,
            "Разрядная ёмкость ниже 33.0 Ач",
        ),
        (
            "G-BAT-IR",
            "BAT-TST",
            "Внутреннее сопротивление R_ac",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("8.0000"),
            Decimal("18.0000"),
            Decimal("12.0000"),
            "mOhm",
            None,
            None,
            "Внутреннее сопротивление вне допуска 8-18 мОм",
        ),
        (
            "G-BODY-PAINT",
            "BODY-PAINT",
            "Качество окраски и целостность пластика",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Сколы или дефект литья кузова",
        ),
        (
            "G-BODY-STEER",
            "BODY-STEER",
            "Люфт рулевой колонки",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("0.0000"),
            Decimal("1.5000"),
            Decimal("0.5000"),
            "deg",
            None,
            None,
            "Люфт рулевой колонки превышает 1.5 град",
        ),
        (
            "G-BODY-CHK",
            "BODY-CHK",
            "Надёжность крепежа сиденья и жгутов",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Незафиксирован крепёж или жгут",
        ),
        (
            "G-CAR-TORQUE",
            "CAR-DRIVE",
            "Момент затяжки редуктора и мотора RS550-24V",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("8.0000"),
            Decimal("12.0000"),
            Decimal("10.0000"),
            "Nm",
            None,
            None,
            "Момент затяжки мотора вне допуска 8.0-12.0 Н*м",
        ),
        (
            "G-CAR-ASM",
            "CAR-ASM",
            "Комплектность кузов + АКБ + JK-BMS-6S",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Неполный комплект сборки машинки",
        ),
        (
            "G-CAR-CURR",
            "CAR-TEST",
            "Ток двигателя под нагрузкой на тестовом заезде",
            GuardValidationType.NUMERIC_RANGE.value,
            Decimal("12.0000"),
            Decimal("22.0000"),
            Decimal("16.5000"),
            "A",
            None,
            None,
            "Ток нагрузки мотора вне допуска 12-22 А",
        ),
        (
            "G-CAR-QC",
            "CAR-QC",
            "Финальный приёмочный контроль ОТК",
            GuardValidationType.BOOLEAN.value,
            None,
            None,
            None,
            None,
            True,
            None,
            "Замечания финального ОТК",
        ),
    ]
    guard_ids = {code: det_uuid(f"guard:{code}") for code, *_ in guard_specs}
    await session.execute(
        insert(TechMapNodeGuard),
        [
            {
                "id": guard_ids[code],
                "node_id": node_ids[nc],
                "name": name,
                "guard_type": gtype,
                "is_mandatory": True,
                "min_value": mn,
                "max_value": mx,
                "nominal_value": nom,
                "unit": unit,
                "expected_boolean": exp_b,
                "validation_pattern": pat,
                "error_message": err,
            }
            for code, nc, name, gtype, mn, mx, nom, unit, exp_b, pat, err in guard_specs
        ],
    )

    # Sampling Rule on BAT-TST (ALG-001)
    sampling_rule_id = det_uuid("sampling:BAT-TST")
    await session.execute(
        insert(TechMapSamplingRule),
        [
            {
                "id": sampling_rule_id,
                "node_id": node_ids["BAT-TST"],
                "measurement_guard_id": guard_ids["G-BAT-IR"],
                "name": "Стратифицированный контроль R_ac АКБ 6S (ALG-001)",
                "resolution_scope": SamplingResolutionScope.ITEM.value,
                "sample_size": 5,
                "sample_aggregation": SamplingAggregationType.PER_ITEM.value,
                "sample_threshold": None,
                "non_sampled_policy": SamplingNonSampledPolicy.PASS_BY_DEFAULT.value,
                "escalation_sample_size": 3,
                "max_escalations": 2,
                "created_at": now_ref,
            }
        ],
    )

    # Transitions
    trans_specs = [
        # TM-BAT-6S
        ("TR-BAT-1", tm_bat_id, "BAT-IN", "BAT-SPLIT", TransitionType.NORMAL.value, None, None),
        ("TR-BAT-2", tm_bat_id, "BAT-SPLIT", "BAT-VIS", TransitionType.NORMAL.value, None, None),
        ("TR-BAT-3", tm_bat_id, "BAT-VIS", "BAT-CHG", TransitionType.NORMAL.value, None, None),
        ("TR-BAT-4", tm_bat_id, "BAT-CHG", "BAT-TST", TransitionType.NORMAL.value, None, None),
        (
            "TR-BAT-CHG-RWK",
            tm_bat_id,
            "BAT-CHG",
            "BAT-RWK",
            TransitionType.REWORK.value,
            None,
            None,
        ),
        (
            "TR-BAT-5-SAMPLE",
            tm_bat_id,
            "BAT-TST",
            "BAT-LAB",
            TransitionType.NORMAL.value,
            TransitionConditionLabel.IN_SAMPLE.value,
            5,
        ),
        (
            "TR-BAT-5-MAIN",
            tm_bat_id,
            "BAT-TST",
            "BAT-DONE",
            TransitionType.NORMAL.value,
            TransitionConditionLabel.NOT_IN_SAMPLE.value,
            None,
        ),
        ("TR-BAT-5-RWK", tm_bat_id, "BAT-TST", "BAT-RWK", TransitionType.REWORK.value, None, None),
        ("TR-BAT-5-DEF", tm_bat_id, "BAT-TST", "BAT-DEF", TransitionType.DEFECT.value, None, None),
        (
            "TR-BAT-RWK-RET",
            tm_bat_id,
            "BAT-RWK",
            "BAT-TST",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
        # TM-BODY
        (
            "TR-BODY-1",
            tm_body_id,
            "BODY-PAINT",
            "BODY-STEER",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
        (
            "TR-BODY-2",
            tm_body_id,
            "BODY-STEER",
            "BODY-CHK",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
        (
            "TR-BODY-2-RWK",
            tm_body_id,
            "BODY-STEER",
            "BODY-RWK",
            TransitionType.REWORK.value,
            None,
            None,
        ),
        (
            "TR-BODY-RWK-RET",
            tm_body_id,
            "BODY-RWK",
            "BODY-CHK",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
        # TM-CAR-FINAL
        ("TR-CAR-1", tm_car_id, "CAR-DRIVE", "CAR-ASM", TransitionType.NORMAL.value, None, None),
        (
            "TR-CAR-1-RWK",
            tm_car_id,
            "CAR-DRIVE",
            "CAR-RWK",
            TransitionType.REWORK.value,
            None,
            None,
        ),
        ("TR-CAR-2", tm_car_id, "CAR-ASM", "CAR-TEST", TransitionType.NORMAL.value, None, None),
        ("TR-CAR-3", tm_car_id, "CAR-TEST", "CAR-QC", TransitionType.NORMAL.value, None, None),
        ("TR-CAR-3-RWK", tm_car_id, "CAR-TEST", "CAR-RWK", TransitionType.REWORK.value, None, None),
        (
            "TR-CAR-RWK-RET",
            tm_car_id,
            "CAR-RWK",
            "CAR-TEST",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
        (
            "TR-CAR-RWK-DRIVE",
            tm_car_id,
            "CAR-RWK",
            "CAR-DRIVE",
            TransitionType.NORMAL.value,
            None,
            None,
        ),
    ]
    tr_ids = {code: det_uuid(f"tr:{code}") for code, *_ in trans_specs}
    await session.execute(
        insert(TechMapTransition),
        [
            {
                "id": tr_ids[code],
                "tech_map_id": tm_id,
                "from_node_id": node_ids[fn],
                "to_node_id": node_ids[tn],
                "transition_type": ttype,
                "condition_label": clab,
                "sample_size": ssize,
            }
            for code, tm_id, fn, tn, ttype, clab, ssize in trans_specs
        ],
    )

    # -------------------------------------------------------------------------
    # 4. Generate 12-month Lots, Batteries, Bodies, Cars & Step Executions
    # -------------------------------------------------------------------------
    rs_rows: list[dict[str, Any]] = []
    lot_meta_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    exec_rows: list[dict[str, Any]] = []
    meas_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    ticket_rows: list[dict[str, Any]] = []

    def add_exec(
        rs_id: UUID,
        node_code: str,
        operator_id: UUID,
        outcome: str,
        tr_code: str | None,
        ts: datetime,
        instrument_code: str | None = None,
        channel_no: int | None = None,
        group_batch_code: str | None = None,
        measurements: list[tuple[str, Decimal | None, bool | None, str | None, bool]] | None = None,
    ) -> None:
        exec_id = det_uuid(f"exec:{rs_id}:{node_code}:{ts.isoformat()}:{outcome}:{len(exec_rows)}")
        exec_rows.append(
            {
                "id": exec_id,
                "route_sheet_id": rs_id,
                "node_id": node_ids[node_code],
                "operator_id": operator_id,
                "instrument_id": instr_ids[instrument_code] if instrument_code else None,
                "channel_no": channel_no,
                "group_batch_code": group_batch_code,
                "outcome": outcome,
                "applied_transition_id": tr_ids[tr_code] if tr_code else None,
                "started_at": ts - timedelta(minutes=8),
                "completed_at": ts,
            }
        )
        for gcode, nval, bval, tval, passed in measurements or []:
            meas_rows.append(
                {
                    "execution_id": exec_id,
                    "guard_id": guard_ids[gcode],
                    "numeric_value": nval,
                    "boolean_value": bval,
                    "text_value": tval,
                    "is_passed": passed,
                    "measured_at": ts,
                }
            )

    # 18 Battery Lots from Jan 2026 to Nov 2026 (30 units per lot = 540 batteries)
    lot_months = [
        (
            1,
            "LOT-BAT-2026-001",
            datetime(2026, 1, 8, 9, 0, tzinfo=UTC),
            "CATL Industrial Packs",
            "INV-2026-0101",
        ),
        (
            2,
            "LOT-BAT-2026-002",
            datetime(2026, 1, 22, 9, 0, tzinfo=UTC),
            "CATL Industrial Packs",
            "INV-2026-0122",
        ),
        (
            3,
            "LOT-BAT-2026-003",
            datetime(2026, 2, 10, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0210",
        ),
        (
            4,
            "LOT-BAT-2026-004",
            datetime(2026, 2, 24, 9, 0, tzinfo=UTC),
            "CATL Industrial Packs",
            "INV-2026-0224",
        ),
        (
            5,
            "LOT-BAT-2026-005",
            datetime(2026, 3, 9, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0309",
        ),
        (
            6,
            "LOT-BAT-2026-006",
            datetime(2026, 3, 25, 9, 0, tzinfo=UTC),
            "EVE Energy Co.",
            "INV-2026-0325",
        ),
        (
            7,
            "LOT-BAT-2026-007",
            datetime(2026, 4, 7, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0407",
        ),
        (
            8,
            "LOT-BAT-2026-008",
            datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
            "EVE Energy Co.",
            "INV-2026-0421",
        ),
        # Lot 9 (June): Planted HOLD batch (DEFECT_RATE_EXCEEDED) - not used in finished cars
        (
            9,
            "LOT-BAT-2026-009",
            datetime(2026, 6, 4, 9, 0, tzinfo=UTC),
            "Shenzhen LowCost Cells",
            "INV-2026-0604",
        ),
        (
            10,
            "LOT-BAT-2026-010",
            datetime(2026, 6, 19, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0619",
        ),
        (
            11,
            "LOT-BAT-2026-011",
            datetime(2026, 7, 8, 9, 0, tzinfo=UTC),
            "CATL Industrial Packs",
            "INV-2026-0708",
        ),
        (
            12,
            "LOT-BAT-2026-012",
            datetime(2026, 7, 25, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0725",
        ),
        (
            13,
            "LOT-BAT-2026-013",
            datetime(2026, 8, 12, 9, 0, tzinfo=UTC),
            "EVE Energy Co.",
            "INV-2026-0812",
        ),
        # Lot 14 (May 2026): Planted Anomaly #1 (20% capacity defects, 12 batteries assembled into cars including CAR-2026-0412)
        (
            14,
            "LOT-BAT-2026-014",
            datetime(2026, 5, 11, 9, 0, tzinfo=UTC),
            "VoltCell Ningbo Ltd.",
            "INV-2026-0514",
        ),
        # Lot 15 (Sep 2026): Ultra-clean lot for CAR-2026-0777
        (
            15,
            "LOT-BAT-2026-015",
            datetime(2026, 9, 5, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-0905",
        ),
        # Lot 16 (Oct 2026): Used for STAND-03 ch 7 drift including CAR-2026-0388
        (
            16,
            "LOT-BAT-2026-016",
            datetime(2026, 10, 6, 9, 0, tzinfo=UTC),
            "CATL Industrial Packs",
            "INV-2026-1006",
        ),
        (
            17,
            "LOT-BAT-2026-017",
            datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
            "EVE Energy Co.",
            "INV-2026-1020",
        ),
        (
            18,
            "LOT-BAT-2026-018",
            datetime(2026, 11, 4, 9, 0, tzinfo=UTC),
            "BYD Cell Direct",
            "INV-2026-1104",
        ),
    ]

    usable_batteries: list[tuple[UUID, str, datetime]] = []
    lot14_usable: list[tuple[UUID, str, datetime]] = []
    lot15_clean_bat: tuple[UUID, str, datetime] | None = None
    lot16_stand03_ch7_bat: tuple[UUID, str, datetime] | None = None
    lot12_aug_bat: tuple[UUID, str, datetime] | None = None
    lot07_apr_bat: tuple[UUID, str, datetime] | None = None

    for lot_num, lot_code, arrival_ts, supplier, invoice in lot_months:
        lot_id = det_uuid(f"rs:{lot_code}")
        planned_qty = 30
        is_hold_lot = lot_code == "LOT-BAT-2026-009"
        is_bad_lot14 = lot_code == "LOT-BAT-2026-014"
        is_oct_lot = lot_code in {"LOT-BAT-2026-016", "LOT-BAT-2026-017"}

        s_state = init_sampling_state(planned_qty=planned_qty, sample_size=5)
        lot_defects = 0
        lot_processed = 0
        hold_reason: str | None = None

        rs_rows.append(
            {
                "id": lot_id,
                "tech_map_id": tm_bat_id,
                "tracking_code": lot_code,
                "qty": planned_qty,
                "planned_qty": planned_qty,
                "current_node_id": None,
                "status": RouteSheetStatus.QUARANTINE.value
                if is_hold_lot
                else RouteSheetStatus.SPLIT_CLOSED.value,
                "hold_reason": None,
                "root_lot_id": None,
                "defect_count": 0,
                "processed_count": 0,
                "created_at": arrival_ts,
                "updated_at": arrival_ts + timedelta(hours=4),
            }
        )
        lot_meta_rows.append(
            {
                "route_sheet_id": lot_id,
                "supplier_name": supplier,
                "invoice_number": invoice,
                "arrival_date": arrival_ts,
                "component_specs": {
                    "bms": "JK-BMS-6S",
                    "motor_compat": "RS550-24V",
                    "chemistry": "Li-Po 6S 35Ah",
                    "nominal_voltage_v": 22.2,
                    "max_charge_v": 25.2,
                    "low_temp_discharge_limit_c": 0,
                    "cell_batch": f"CELL-{lot_code}",
                },
            }
        )

        # Lot-level intake + split execution
        add_exec(
            lot_id,
            "BAT-IN",
            user_ids["OP-06"],
            StepOutcome.PASS.value,
            "TR-BAT-1",
            arrival_ts + timedelta(minutes=20),
            measurements=[("G-BAT-SEAL", None, True, None, True)],
        )
        add_exec(
            lot_id,
            "BAT-SPLIT",
            user_ids["OP-06"],
            StepOutcome.PASS.value,
            "TR-BAT-2",
            arrival_ts + timedelta(minutes=45),
        )

        for idx in range(1, planned_qty + 1):
            bat_code = f"BAT-2026-{lot_num:03d}-{idx:03d}"
            bat_id = det_uuid(f"rs:{bat_code}")
            bat_ts = arrival_ts + timedelta(hours=2, minutes=idx * 12)
            op_bat = user_ids[operators[idx % len(operators)]]
            if lot_code == "LOT-BAT-2026-016" and idx == 7:
                op_bat = user_ids["OP-03"]

            # Charging stand assignment
            stand_code = ["STAND-01", "STAND-02", "STAND-03"][(lot_num + idx) % 3]
            channel_no = ((idx - 1) % 16) + 1
            if is_oct_lot and idx in {7, 14, 21, 23, 28}:
                stand_code = "STAND-03"
                channel_no = 7
            if lot_code == "LOT-BAT-2026-015":
                stand_code = "STAND-01"
                channel_no = ((idx - 1) % 8) + 1

            # Determine battery behavior
            is_defect = False
            needs_cap_rework = False

            if is_hold_lot and idx <= 6:
                is_defect = True
            elif is_bad_lot14:
                if idx in {4, 9, 14, 19, 24, 29}:
                    is_defect = True  # Spaced 20% capacity defect below linear stop-crane threshold
                elif idx in {7, 8, 10, 12}:
                    needs_cap_rework = True  # Extra rework loop (idx 12 -> CAR-2026-0412)

            # Stand-03 channel 7 drift in October 2026
            is_stand03_ch7_oct = is_oct_lot and stand_code == "STAND-03" and channel_no == 7

            is_halted_by_hold = hold_reason is not None
            s_dec: SamplingDecision | None = None
            if is_halted_by_hold:
                bat_status = RouteSheetStatus.QUARANTINE.value
                bat_hold_reason: str | None = hold_reason
                bat_defect_cnt = 0
                bat_proc_cnt = 0
                new_n = max(0, s_state.n_rem - 1)
                s_state = SamplingState(
                    m_rem=s_state.m_rem,
                    n_rem=new_n,
                    w_rem=new_n
                    if (s_state.m_rem > 0 and new_n <= s_state.m_rem)
                    else min(s_state.w_rem, new_n),
                    taken_in_window=s_state.taken_in_window,
                    escalations_used=s_state.escalations_used,
                )
            else:
                lot_processed += 1
                if is_defect:
                    lot_defects += 1
                    new_n = max(0, s_state.n_rem - 1)
                    s_state = SamplingState(
                        m_rem=s_state.m_rem,
                        n_rem=new_n,
                        w_rem=new_n
                        if (s_state.m_rem > 0 and new_n <= s_state.m_rem)
                        else min(s_state.w_rem, new_n),
                        taken_in_window=s_state.taken_in_window,
                        escalations_used=s_state.escalations_used,
                    )
                else:
                    u = compute_hmac_uniform(
                        settings.sampling_hmac_secret,
                        str(lot_id),
                        str(node_ids["BAT-TST"]),
                        bat_code,
                    )
                    s_state, s_dec = step_sampling(s_state, u)

                stop_res = check_stop_crane(lot_defects, lot_processed, 3, Decimal("0.3500"))
                if stop_res.triggered and hold_reason is None:
                    hold_reason = stop_res.hold_reason

                bat_status = (
                    RouteSheetStatus.DEFECT.value if is_defect else RouteSheetStatus.COMPLETED.value
                )
                bat_hold_reason = None
                bat_defect_cnt = 1 if is_defect else 0
                bat_proc_cnt = 1

            rs_rows.append(
                {
                    "id": bat_id,
                    "tech_map_id": tm_bat_id,
                    "tracking_code": bat_code,
                    "qty": 1,
                    "planned_qty": 1,
                    "current_node_id": None,
                    "status": bat_status,
                    "hold_reason": bat_hold_reason,
                    "root_lot_id": lot_id,
                    "defect_count": bat_defect_cnt,
                    "processed_count": bat_proc_cnt,
                    "created_at": arrival_ts + timedelta(minutes=45),
                    "updated_at": bat_ts + timedelta(hours=2),
                }
            )
            link_rows.append(
                {
                    "id": det_uuid(f"link:split:{lot_code}:{bat_code}"),
                    "source_rs_id": lot_id,
                    "target_rs_id": bat_id,
                    "link_type": RouteSheetLinkType.SPLIT.value,
                    "created_at": arrival_ts + timedelta(minutes=45),
                }
            )

            # Skip executions for units halted in QUARANTINE after stop-crane HOLD
            if is_halted_by_hold:
                continue

            # Step 3: Visual
            add_exec(
                bat_id,
                "BAT-VIS",
                op_bat,
                StepOutcome.PASS.value,
                "TR-BAT-3",
                bat_ts,
                measurements=[("G-BAT-VIS", None, True, None, True)],
            )

            # Step 4: Charging
            if is_stand03_ch7_oct:
                # Initial undercharge on drifting channel 7 -> REWORK, then borderline pass
                add_exec(
                    bat_id,
                    "BAT-CHG",
                    op_bat,
                    StepOutcome.REWORK.value,
                    "TR-BAT-CHG-RWK",
                    bat_ts + timedelta(minutes=40),
                    instrument_code=stand_code,
                    channel_no=channel_no,
                    group_batch_code=f"CHG-{lot_code}",
                    measurements=[("G-BAT-VOLT", Decimal("24.0800"), None, None, False)],
                )
                add_exec(
                    bat_id,
                    "BAT-CHG",
                    op_bat,
                    StepOutcome.PASS.value,
                    "TR-BAT-4",
                    bat_ts + timedelta(minutes=70),
                    instrument_code=stand_code,
                    channel_no=channel_no,
                    group_batch_code=f"CHG-{lot_code}-R",
                    measurements=[("G-BAT-VOLT", Decimal("24.6200"), None, None, True)],
                )
            else:
                volt = Decimal(str(round(25.05 + rng.uniform(-0.18, 0.25), 4)))
                add_exec(
                    bat_id,
                    "BAT-CHG",
                    op_bat,
                    StepOutcome.PASS.value,
                    "TR-BAT-4",
                    bat_ts + timedelta(minutes=40),
                    instrument_code=stand_code,
                    channel_no=channel_no,
                    group_batch_code=f"CHG-{lot_code}",
                    measurements=[("G-BAT-VOLT", volt, None, None, True)],
                )

            # Step 5: Electrical test (capacity + R_ac + ALG-001 sample branch)
            tst_ts = bat_ts + timedelta(minutes=95)
            ir_val = Decimal(str(round(11.5 + rng.uniform(-1.5, 2.5), 4)))
            if is_defect:
                cap_bad = Decimal(str(round(rng.uniform(26.8, 29.8), 4)))
                ir_bad = Decimal(str(round(rng.uniform(19.5, 24.5), 4)))
                add_exec(
                    bat_id,
                    "BAT-TST",
                    op_bat,
                    StepOutcome.DEFECT.value,
                    "TR-BAT-5-DEF",
                    tst_ts,
                    instrument_code="TESTER-IR-01",
                    measurements=[
                        ("G-BAT-CAP", cap_bad, None, None, False),
                        ("G-BAT-IR", ir_bad, None, None, False),
                    ],
                )
            else:
                if needs_cap_rework:
                    cap_rwk = Decimal(str(round(rng.uniform(31.2, 32.6), 4)))
                    add_exec(
                        bat_id,
                        "BAT-TST",
                        op_bat,
                        StepOutcome.REWORK.value,
                        "TR-BAT-5-RWK",
                        tst_ts,
                        instrument_code="TESTER-IR-01",
                        measurements=[
                            ("G-BAT-CAP", cap_rwk, None, None, False),
                            ("G-BAT-IR", ir_val, None, None, True),
                        ],
                    )
                    add_exec(
                        bat_id,
                        "BAT-RWK",
                        op_bat,
                        StepOutcome.PASS.value,
                        "TR-BAT-RWK-RET",
                        tst_ts + timedelta(minutes=25),
                    )
                    tst_ts = tst_ts + timedelta(minutes=45)
                    cap_final = Decimal(str(round(rng.uniform(33.1, 33.5), 4)))
                elif is_bad_lot14:
                    cap_final = Decimal(str(round(rng.uniform(33.2, 34.1), 4)))
                else:
                    cap_final = Decimal(str(round(35.1 + rng.uniform(-0.8, 1.1), 4)))

                assert s_dec is not None
                out_label = (
                    StepOutcome.IN_SAMPLE.value
                    if s_dec.selected
                    else StepOutcome.NOT_IN_SAMPLE.value
                )
                tr_label = "TR-BAT-5-SAMPLE" if s_dec.selected else "TR-BAT-5-MAIN"
                add_exec(
                    bat_id,
                    "BAT-TST",
                    op_bat,
                    out_label,
                    tr_label,
                    tst_ts,
                    instrument_code="TESTER-IR-01",
                    measurements=[
                        ("G-BAT-CAP", cap_final, None, None, True),
                        ("G-BAT-IR", ir_val, None, None, True),
                    ],
                )
                if not is_hold_lot:
                    if is_bad_lot14:
                        lot14_usable.append((bat_id, bat_code, tst_ts + timedelta(hours=1)))
                    elif lot_code == "LOT-BAT-2026-015" and idx == 15:
                        lot15_clean_bat = (bat_id, bat_code, tst_ts + timedelta(hours=1))
                    elif lot_code == "LOT-BAT-2026-016" and idx == 7:
                        lot16_stand03_ch7_bat = (bat_id, bat_code, tst_ts + timedelta(hours=1))
                    elif lot_code == "LOT-BAT-2026-012" and idx == 10:
                        lot12_aug_bat = (bat_id, bat_code, tst_ts + timedelta(hours=1))
                    elif lot_code == "LOT-BAT-2026-007" and idx == 5:
                        lot07_apr_bat = (bat_id, bat_code, tst_ts + timedelta(hours=1))
                    else:
                        usable_batteries.append((bat_id, bat_code, tst_ts + timedelta(hours=1)))

        # Update root lot counters & ALG-001 state
        for r in rs_rows:
            if r["id"] == lot_id:
                r["defect_count"] = lot_defects
                r["processed_count"] = lot_processed
                r["hold_reason"] = hold_reason
                break

        sampling_rows.append(
            {
                "lot_rs_id": lot_id,
                "node_id": node_ids["BAT-TST"],
                "m_rem": s_state.m_rem,
                "n_rem": s_state.n_rem,
                "w_rem": s_state.w_rem,
                "taken_in_window": s_state.taken_in_window,
                "escalations_used": s_state.escalations_used,
                "updated_at": arrival_ts + timedelta(hours=8),
            }
        )

    # -------------------------------------------------------------------------
    # 5. Assemble 451 Finished Cars (CAR-2026-0001..0450 + CAR-2026-0777)
    # -------------------------------------------------------------------------
    # Ensure exact 12 cars receive batteries from LOT-BAT-2026-014 (including CAR-2026-0412)
    lot14_target_cars = [
        "CAR-2026-0401",
        "CAR-2026-0402",
        "CAR-2026-0403",
        "CAR-2026-0404",
        "CAR-2026-0405",
        "CAR-2026-0406",
        "CAR-2026-0407",
        "CAR-2026-0408",
        "CAR-2026-0409",
        "CAR-2026-0410",
        "CAR-2026-0411",
        "CAR-2026-0412",
    ]
    lot14_car_map: dict[str, tuple[UUID, str, datetime]] = {}
    # Put BAT-2026-014-012 specifically into CAR-2026-0412
    bat14_012 = next(b for b in lot14_usable if b[1] == "BAT-2026-014-012")
    other_14 = [b for b in lot14_usable if b[1] != "BAT-2026-014-012"]
    for idx_c, ccode in enumerate(lot14_target_cars):
        if ccode == "CAR-2026-0412":
            lot14_car_map[ccode] = bat14_012
        else:
            lot14_car_map[ccode] = other_14[idx_c]

    all_car_codes = [f"CAR-2026-{i:04d}" for i in range(1, 451)] + ["CAR-2026-0777"]
    usable_idx = 0

    for c_num, car_code in enumerate(all_car_codes, start=1):
        body_code = car_code.replace("CAR-", "BODY-")
        body_id = det_uuid(f"rs:{body_code}")
        car_id = det_uuid(f"rs:{car_code}")

        # Pick battery & date
        if car_code in lot14_car_map:
            bat_id, _bat_code, bat_ready_ts = lot14_car_map[car_code]
            asm_ts = datetime(2026, 5, 22, 10, 0, tzinfo=UTC) + timedelta(hours=c_num % 10)
        elif car_code == "CAR-2026-0777":
            assert lot15_clean_bat is not None
            bat_id, _bat_code, bat_ready_ts = lot15_clean_bat
            asm_ts = datetime(2026, 9, 14, 11, 30, tzinfo=UTC)
        elif car_code == "CAR-2026-0388":
            assert lot16_stand03_ch7_bat is not None
            bat_id, _bat_code, bat_ready_ts = lot16_stand03_ch7_bat
            asm_ts = datetime(2026, 10, 16, 14, 0, tzinfo=UTC)
        elif car_code == "CAR-2026-0295":
            assert lot12_aug_bat is not None
            bat_id, _bat_code, bat_ready_ts = lot12_aug_bat
            asm_ts = datetime(2026, 8, 18, 15, 0, tzinfo=UTC)
        elif car_code == "CAR-2026-0199":
            assert lot07_apr_bat is not None
            bat_id, _bat_code, bat_ready_ts = lot07_apr_bat
            asm_ts = datetime(2026, 4, 22, 13, 0, tzinfo=UTC)
        else:
            bat_id, _bat_code, bat_ready_ts = usable_batteries[usable_idx % len(usable_batteries)]
            usable_idx += 1
            # Spread across 2026 roughly after battery ready
            asm_ts = max(
                bat_ready_ts + timedelta(days=2),
                datetime(2026, 1, 15, 10, 0, tzinfo=UTC) + timedelta(hours=c_num * 16),
            )
            if asm_ts > now_ref - timedelta(days=2):
                asm_ts = now_ref - timedelta(days=3) - timedelta(hours=c_num % 48)

        # Pick operators
        body_op = user_ids[operators[c_num % len(operators)]]
        if car_code == "CAR-2026-0777":
            car_op = user_ids["OP-01"]  # Ivanov S.P. (top clean operator)
        elif car_code == "CAR-2026-0295":
            car_op = petrov_id
        elif asm_ts.month == 8 and c_num % 3 == 0:
            car_op = petrov_id  # Petrov worked heavily in August 2026
        else:
            car_op = user_ids[operators[(c_num + 1) % len(operators)]]

        # Body RouteSheet & Executions
        body_ts = asm_ts - timedelta(hours=5)
        rs_rows.append(
            {
                "id": body_id,
                "tech_map_id": tm_body_id,
                "tracking_code": body_code,
                "qty": 1,
                "planned_qty": 1,
                "current_node_id": None,
                "status": RouteSheetStatus.COMPLETED.value,
                "hold_reason": None,
                "root_lot_id": None,
                "defect_count": 0,
                "processed_count": 1,
                "created_at": body_ts - timedelta(hours=2),
                "updated_at": body_ts + timedelta(hours=1),
            }
        )
        add_exec(
            body_id,
            "BODY-PAINT",
            body_op,
            StepOutcome.PASS.value,
            "TR-BODY-1",
            body_ts,
            measurements=[("G-BODY-PAINT", None, True, None, True)],
        )
        add_exec(
            body_id,
            "BODY-STEER",
            body_op,
            StepOutcome.PASS.value,
            "TR-BODY-2",
            body_ts + timedelta(minutes=25),
            measurements=[
                (
                    "G-BODY-STEER",
                    Decimal(str(round(0.4 + rng.uniform(0.0, 0.5), 4))),
                    None,
                    None,
                    True,
                )
            ],
        )
        add_exec(
            body_id,
            "BODY-CHK",
            body_op,
            StepOutcome.PASS.value,
            None,
            body_ts + timedelta(minutes=50),
            measurements=[("G-BODY-CHK", None, True, None, True)],
        )

        # Car RouteSheet & Executions
        rs_rows.append(
            {
                "id": car_id,
                "tech_map_id": tm_car_id,
                "tracking_code": car_code,
                "qty": 1,
                "planned_qty": 1,
                "current_node_id": None,
                "status": RouteSheetStatus.COMPLETED.value,
                "hold_reason": None,
                "root_lot_id": None,
                "defect_count": 0,
                "processed_count": 1,
                "created_at": asm_ts - timedelta(hours=1),
                "updated_at": asm_ts + timedelta(hours=2),
            }
        )
        # ASSEMBLED_INTO links (BODY + BAT -> CAR)
        link_rows.append(
            {
                "id": det_uuid(f"link:asm:body:{car_code}"),
                "source_rs_id": body_id,
                "target_rs_id": car_id,
                "link_type": RouteSheetLinkType.ASSEMBLED_INTO.value,
                "created_at": asm_ts + timedelta(minutes=25),
            }
        )
        link_rows.append(
            {
                "id": det_uuid(f"link:asm:bat:{car_code}"),
                "source_rs_id": bat_id,
                "target_rs_id": car_id,
                "link_type": RouteSheetLinkType.ASSEMBLED_INTO.value,
                "created_at": asm_ts + timedelta(minutes=25),
            }
        )

        # Step 1: CAR-DRIVE (RS550-24V torque)
        is_petrov_aug_anomaly = car_code == "CAR-2026-0295" or (
            car_op == petrov_id and asm_ts.month == 8 and c_num % 5 != 0
        )
        if is_petrov_aug_anomaly:
            bad_torque = Decimal(str(round(rng.uniform(12.35, 13.45), 4)))
            add_exec(
                car_id,
                "CAR-DRIVE",
                car_op,
                StepOutcome.REWORK.value,
                "TR-CAR-1-RWK",
                asm_ts,
                instrument_code="TORQUE-01",
                measurements=[("G-CAR-TORQUE", bad_torque, None, None, False)],
            )
            add_exec(
                car_id,
                "CAR-RWK",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-RWK-DRIVE",
                asm_ts + timedelta(minutes=15),
            )
            border_torque = (
                Decimal("11.9500") if car_code == "CAR-2026-0295" else Decimal("11.6500")
            )
            add_exec(
                car_id,
                "CAR-DRIVE",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-1",
                asm_ts + timedelta(minutes=20),
                instrument_code="TORQUE-01",
                measurements=[("G-CAR-TORQUE", border_torque, None, None, True)],
            )
            drive_offset = 30
        else:
            ok_torque = (
                Decimal("10.0200")
                if car_code == "CAR-2026-0777"
                else Decimal(str(round(10.0 + rng.uniform(-0.7, 0.7), 4)))
            )
            add_exec(
                car_id,
                "CAR-DRIVE",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-1",
                asm_ts,
                instrument_code="TORQUE-01",
                measurements=[("G-CAR-TORQUE", ok_torque, None, None, True)],
            )
            drive_offset = 0

        # Step 2: CAR-ASM
        add_exec(
            car_id,
            "CAR-ASM",
            car_op,
            StepOutcome.PASS.value,
            "TR-CAR-2",
            asm_ts + timedelta(minutes=25 + drive_offset),
            measurements=[("G-CAR-ASM", None, True, None, True)],
        )

        # Step 3: CAR-TEST (Anomaly #4 CAR-2026-0199 had rework on test ride!)
        if car_code == "CAR-2026-0199":
            add_exec(
                car_id,
                "CAR-TEST",
                car_op,
                StepOutcome.REWORK.value,
                "TR-CAR-3-RWK",
                asm_ts + timedelta(minutes=50 + drive_offset),
                instrument_code="TEST-RIG-01",
                measurements=[("G-CAR-CURR", Decimal("22.8000"), None, None, False)],
            )
            add_exec(
                car_id,
                "CAR-RWK",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-RWK-RET",
                asm_ts + timedelta(minutes=70 + drive_offset),
            )
            add_exec(
                car_id,
                "CAR-TEST",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-3",
                asm_ts + timedelta(minutes=85 + drive_offset),
                instrument_code="TEST-RIG-01",
                measurements=[("G-CAR-CURR", Decimal("18.4000"), None, None, True)],
            )
            test_offset = 40
        else:
            curr = (
                Decimal("16.2000")
                if car_code == "CAR-2026-0777"
                else Decimal(str(round(16.5 + rng.uniform(-1.5, 1.5), 4)))
            )
            add_exec(
                car_id,
                "CAR-TEST",
                car_op,
                StepOutcome.PASS.value,
                "TR-CAR-3",
                asm_ts + timedelta(minutes=50 + drive_offset),
                instrument_code="TEST-RIG-01",
                measurements=[("G-CAR-CURR", curr, None, None, True)],
            )
            test_offset = 0

        # Step 4: CAR-QC
        add_exec(
            car_id,
            "CAR-QC",
            user_ids["OTK-01"],
            StepOutcome.PASS.value,
            None,
            asm_ts + timedelta(minutes=75 + drive_offset + test_offset),
            measurements=[("G-CAR-QC", None, True, None, True)],
        )

        # Anomaly #4: Prior closed ticket on CAR-2026-0199
        if car_code == "CAR-2026-0199":
            ticket_rows.append(
                {
                    "id": det_uuid("ticket:CAR-2026-0199-prior"),
                    "route_sheet_id": car_id,
                    "tracking_code": "CAR-2026-0199",
                    "title": "Повторное обращение: рывки и остановка мотора при старте",
                    "symptom_category": "MOTOR_STALL_OR_NOISE",
                    "root_cause_summary": "Выявлен повышенный ток нагрузки (22.8 А) на стенде и доработка контактной группы редуктора.",
                    "recommended_action": "Выполнена зачистка клемм и регулировка зацепления редуктора.",
                    "status": SupportTicketStatus.CLOSED.value,
                    "priority": SupportTicketPriority.HIGH.value,
                    "operator_comment": "Закрыто по первичному обращению в июне 2026",
                    "thread_id": "seed-prior-0199",
                    "created_at": datetime(2026, 6, 15, 10, 0, tzinfo=UTC),
                    "closed_at": datetime(2026, 6, 18, 16, 30, tzinfo=UTC),
                }
            )

    # Batch inserts (chunked for asyncpg 32767 parameter limit)
    async def chunked_insert(
        model: Any, rows: list[dict[str, Any]], chunk_size: int = 1000
    ) -> None:
        for i in range(0, len(rows), chunk_size):
            await session.execute(insert(model), rows[i : i + chunk_size])

    await chunked_insert(RouteSheet, rs_rows, 800)
    await chunked_insert(LotMetadata, lot_meta_rows, 500)
    await chunked_insert(RouteSheetLink, link_rows, 1000)
    await chunked_insert(SamplingNodeState, sampling_rows, 500)
    await chunked_insert(StepExecution, exec_rows, 1000)
    await chunked_insert(StepMeasurement, meas_rows, 1500)
    if ticket_rows:
        await chunked_insert(SupportTicket, ticket_rows, 100)

    await session.commit()

    manifest = {
        "seed": 20260101,
        "summary": {
            "total_cars": len(all_car_codes),
            "total_battery_lots": len(lot_months),
            "total_route_sheets": len(rs_rows),
            "total_links": len(link_rows),
            "total_executions": len(exec_rows),
            "total_measurements": len(meas_rows),
            "hold_lot": "LOT-BAT-2026-009",
        },
        "anomalies": {
            "lot_defect_recall": {
                "target_car": "CAR-2026-0412",
                "root_lot": "LOT-BAT-2026-014",
                "supplier": "VoltCell Ningbo Ltd.",
                "defect_rate_pct": 20.0,
                "affected_cars": sorted(lot14_target_cars),
                "affected_count": len(lot14_target_cars),
            },
            "charging_stand_channel_drift": {
                "target_car": "CAR-2026-0388",
                "instrument_code": "STAND-03",
                "channel_no": 7,
                "calibration_due_at": "2026-09-30T00:00:00+00:00",
                "date_window": "2026-10",
            },
            "operator_quality_drop": {
                "target_car": "CAR-2026-0295",
                "operator_name": "Петров А.В.",
                "operator_code": "OP-02",
                "tech_map": "TM-CAR-FINAL",
                "date_window": "2026-08",
            },
            "repeat_complaint": {
                "target_car": "CAR-2026-0199",
                "prior_ticket_status": "CLOSED",
                "prior_rework_node": "CAR-TEST",
            },
            "clean_db_web_research": {
                "target_car": "CAR-2026-0777",
                "root_lot": "LOT-BAT-2026-015",
                "component_specs": {
                    "motor": "RS550-24V",
                    "bms": "JK-BMS-6S",
                    "chemistry": "Li-Po 6S 35Ah",
                },
            },
        },
    }

    if write_manifest:
        manifest_path = Path(PROJECT_ROOT) / "evals" / "seed_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return manifest


async def _main() -> None:
    t0 = time.perf_counter()
    engine = build_engine()
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            manifest = await seed_database(session, write_manifest=True)
        elapsed = time.perf_counter() - t0
        summary = manifest["summary"]
        print(
            f"Seed completed in {elapsed:.2f}s | "
            f"cars={summary['total_cars']} | lots={summary['total_battery_lots']} | "
            f"route_sheets={summary['total_route_sheets']} | links={summary['total_links']} | "
            f"executions={summary['total_executions']} | measurements={summary['total_measurements']}"
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
