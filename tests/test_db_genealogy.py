from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from src.agent.tools import (
    query_affected_products_by_lot,
    query_instrument_channel_anomalies,
    query_item_genealogy,
    query_item_history,
    query_lot_quality_stats,
    query_operator_anomalies,
)
from src.config import PROJECT_ROOT
from src.db.enums import RouteSheetLinkType, RouteSheetStatus
from src.db.models import Instrument, RouteSheet, RouteSheetLink, StepExecution, TechMap, User


async def test_seeded_volumes_and_hold_lot(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        car_count = await session.scalar(
            select(func.count(RouteSheet.id))
            .join(TechMap, TechMap.id == RouteSheet.tech_map_id)
            .where(TechMap.item_code == "TM-CAR-FINAL")
        )
        assert car_count == 451

        split_count = await session.scalar(
            select(func.count(RouteSheetLink.id)).where(
                RouteSheetLink.link_type == RouteSheetLinkType.SPLIT.value
            )
        )
        asm_count = await session.scalar(
            select(func.count(RouteSheetLink.id)).where(
                RouteSheetLink.link_type == RouteSheetLinkType.ASSEMBLED_INTO.value
            )
        )
        assert split_count == 18 * 30
        assert asm_count == 451 * 2

        hold_lot = await session.scalar(
            select(RouteSheet).where(RouteSheet.tracking_code == "LOT-BAT-2026-009")
        )
        assert hold_lot is not None
        assert hold_lot.hold_reason == "DEFECT_RATE_EXCEEDED"


async def test_anomaly_1_lot_defect_genealogy_and_recall(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    manifest = json.loads(
        (Path(PROJECT_ROOT) / "evals" / "seed_manifest.json").read_text(encoding="utf-8")
    )
    expected_cars = manifest["anomalies"]["lot_defect_recall"]["affected_cars"]
    assert len(expected_cars) == 12

    async with session_factory() as session:
        gen = await query_item_genealogy(session, "CAR-2026-0412")
        assert gen["found"] is True
        assert gen["battery_component"]["tracking_code"] == "BAT-2026-014-012"
        assert gen["root_lots"][0]["tracking_code"] == "LOT-BAT-2026-014"
        assert gen["bom_specs"]["motor_compat"] == "RS550-24V"
        assert gen["bom_specs"]["bms"] == "JK-BMS-6S"

        lot_stats = await query_lot_quality_stats(session, "LOT-BAT-2026-014")
        assert lot_stats["found"] is True
        assert lot_stats["defect_rate_pct"] == 20.0
        assert lot_stats["exceeds_10pct_recall_threshold"] is True

        recall = await query_affected_products_by_lot(session, "LOT-BAT-2026-014")
        assert recall["found"] is True
        assert recall["affected_count"] == 12
        assert recall["affected_car_codes"] == expected_cars


async def test_anomaly_2_stand03_channel_7_drift_and_expired_calibration(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        gen = await query_item_genealogy(session, "CAR-2026-0388")
        assert gen["found"] is True
        hist = await query_item_history(session, gen["target_route_sheet_id"])
        bat_code = gen["battery_component"]["tracking_code"]
        bat_execs = hist["component_histories"][bat_code]
        chg_exec = next(e for e in bat_execs if e["operation_code"] == "OP-BAT-CHG")
        assert chg_exec["instrument_code"] == "STAND-03"
        assert chg_exec["channel_no"] == 7
        assert chg_exec["calibration_expired_at_execution"] is True

        anom = await query_instrument_channel_anomalies(
            session, "STAND-03", 7, str(chg_exec["completed_at"])[:10]
        )
        assert anom["found"] is True
        assert anom["calibration_expired_in_window"] is True
        assert anom["is_channel_anomalous"] is True
        assert anom["total_channels_observed"] > 5
        assert len(anom["per_channel_breakdown"]) <= 4
        assert (
            anom["target_channel_stats"]["anomaly_rate_pct"]
            > anom["other_channels_baseline"]["anomaly_rate_pct"]
        )


async def test_anomaly_3_operator_petrov_august_quality_drop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        gen = await query_item_genealogy(session, "CAR-2026-0295")
        hist = await query_item_history(session, gen["target_route_sheet_id"])
        assert hist["rework_count"] >= 1
        drive_exec = next(e for e in hist["executions"] if e["operation_code"] == "OP-CAR-DRIVE")
        assert "Петров" in drive_exec["operator_name"]

        op_stats = await query_operator_anomalies(
            session, "Петров А.В.", str(drive_exec["completed_at"])[:10]
        )
        assert op_stats["found"] is True
        assert op_stats["is_operator_anomalous"] is True
        assert (
            op_stats["operator_stats"]["rework_rate_pct"]
            > op_stats["shop_other_operators_baseline"]["rework_rate_pct"]
        )


async def test_anomaly_4_repeat_complaint_and_anomaly_5_clean_db(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        # Anomaly #4: CAR-2026-0199 repeat ticket + rework
        h_199 = await query_item_history(session, "CAR-2026-0199")
        assert h_199["found"] is True
        assert h_199["rework_count"] >= 1
        assert len(h_199["prior_support_tickets"]) >= 1
        assert h_199["prior_support_tickets"][0]["status"] == "CLOSED"

        # Anomaly #5: CAR-2026-0777 100% clean DB
        gen_777 = await query_item_genealogy(session, "CAR-2026-0777")
        assert gen_777["found"] is True
        assert gen_777["bom_specs"]["bms"] == "JK-BMS-6S"
        assert gen_777["bom_specs"]["motor_compat"] == "RS550-24V"

        h_777 = await query_item_history(session, gen_777["target_route_sheet_id"])
        assert h_777["rework_count"] == 0
        assert h_777["component_rework_count"] == 0
        assert h_777["prior_support_tickets"] == []

        lot_777 = await query_lot_quality_stats(session, gen_777["root_lots"][0]["tracking_code"])
        assert lot_777["defect_rate_pct"] == 0.0
        assert lot_777["exceeds_10pct_recall_threshold"] is False


async def test_hold_lot_stops_honestly_at_5_and_lot14_has_no_hold(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """LOT-BAT-2026-009 stops honestly at item 5 (processed=5, QUARANTINE on idx>5); LOT-BAT-2026-014 has hold_reason=None."""
    async with session_factory() as session:
        lot09 = await session.scalar(
            select(RouteSheet).where(RouteSheet.tracking_code == "LOT-BAT-2026-009")
        )
        assert lot09 is not None
        assert lot09.hold_reason == "DEFECT_RATE_EXCEEDED"
        assert lot09.processed_count == 5
        assert lot09.defect_count == 5

        lot09_children = list(
            (
                await session.scalars(select(RouteSheet).where(RouteSheet.root_lot_id == lot09.id))
            ).all()
        )
        assert len(lot09_children) == 30
        quarantine_ids = [
            c.id for c in lot09_children if c.status == RouteSheetStatus.QUARANTINE.value
        ]
        assert len(quarantine_ids) == 25
        q_execs = await session.scalar(
            select(func.count(StepExecution.id)).where(
                StepExecution.route_sheet_id.in_(quarantine_ids)
            )
        )
        assert q_execs == 0

        lot14 = await session.scalar(
            select(RouteSheet).where(RouteSheet.tracking_code == "LOT-BAT-2026-014")
        )
        assert lot14 is not None
        assert lot14.hold_reason is None
        assert lot14.processed_count == 30
        assert lot14.defect_count == 6


async def test_tester_ir01_no_unit_mixing_and_no_join_fanout(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """query_instrument_channel_anomalies must count DISTINCT StepExecution ids and separate Ah vs mOhm."""
    async with session_factory() as session:
        instr = await session.scalar(select(Instrument).where(Instrument.code == "TESTER-IR-01"))
        assert instr is not None

        res = await query_instrument_channel_anomalies(session, "TESTER-IR-01", None, "2026-05-15")
        assert res["found"] is True
        assert res["primary_unit"] in {"Ah", "mOhm"}
        breakdown = res["per_channel_breakdown"]
        assert len(breakdown) == 1
        ch_entry = breakdown[0]

        exact_exec_cnt = await session.scalar(
            select(func.count(func.distinct(StepExecution.id))).where(
                StepExecution.instrument_id == instr.id,
                StepExecution.completed_at >= datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
                StepExecution.completed_at <= datetime(2026, 5, 29, 0, 0, tzinfo=UTC),
            )
        )
        assert ch_entry["total_execs"] == exact_exec_cnt

        per_guard = {g["unit"]: g for g in ch_entry["per_guard_stats"]}
        assert "Ah" in per_guard and "mOhm" in per_guard
        assert per_guard["Ah"]["avg_numeric_value"] > 25.0
        assert per_guard["mOhm"]["avg_numeric_value"] < 20.0


async def test_operator_anomalies_same_node_baseline_and_nm_unit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """query_operator_anomalies must compare OP-02 strictly against same-node (OP-CAR-DRIVE) peers using unit='Nm'."""
    async with session_factory() as session:
        op_petrov = await session.scalar(select(User).where(User.employee_code == "OP-02"))
        assert op_petrov is not None

        op_res = await query_operator_anomalies(session, "OP-02", "2026-08-18")
        assert op_res["found"] is True
        assert "OP-CAR-DRIVE" in op_res["compared_operations"]
        assert all(op.startswith("OP-CAR-") for op in op_res["compared_operations"])
        assert "OP-BAT-CHG" not in op_res["compared_operations"]
        assert op_res["operator_stats"]["avg_torque_nm"] is not None
        assert 10.0 <= op_res["operator_stats"]["avg_torque_nm"] <= 14.0
        assert op_res["shop_other_operators_baseline"]["avg_torque_nm"] is not None

        # Verify total_execs uses DISTINCT step_executions (matching raw table count, no join inflation)
        raw_petrov_execs = await session.scalar(
            select(func.count(func.distinct(StepExecution.id))).where(
                StepExecution.operator_id == op_petrov.id,
                StepExecution.completed_at >= datetime(2026, 8, 4, 0, 0, tzinfo=UTC),
                StepExecution.completed_at <= datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
            )
        )
        assert op_res["operator_stats"]["total_execs"] == raw_petrov_execs
