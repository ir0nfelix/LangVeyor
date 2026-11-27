from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings, get_settings
from src.db.enums import StepOutcome
from src.db.models import (
    Instrument,
    LotMetadata,
    Operation,
    RouteSheet,
    SamplingNodeState,
    StepExecution,
    StepMeasurement,
    SupportTicket,
    TechMapNode,
    TechMapNodeGuard,
    User,
    Workstation,
)
from src.rag.hybrid_retriever import HybridRetriever


def _parse_iso_date(date_str: str) -> datetime:
    cleaned = date_str.strip()
    if len(cleaned) == 10:
        return datetime.fromisoformat(cleaned).replace(tzinfo=UTC)
    dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _try_uuid(val: str) -> UUID | None:
    try:
        return UUID(val.strip())
    except ValueError:
        return None


# =============================================================================
# Pure Async Domain SQL Queries (Zero raw Text-to-SQL!)
# =============================================================================


async def query_item_genealogy(session: AsyncSession, tracking_code: str) -> dict[str, Any]:
    """Backward recursive CTE over route_sheet_links (target -> source) for full product lineage."""
    code = tracking_code.strip().upper()
    root_rs = await session.scalar(select(RouteSheet).where(RouteSheet.tracking_code == code))
    if root_rs is None:
        return {"found": False, "tracking_code": code, "error": f"Изделие {code} не найдено в БД."}

    cte_sql = text(
        """
        WITH RECURSIVE lineage AS (
            SELECT
                rs.id,
                rs.tracking_code,
                rs.status,
                rs.hold_reason,
                rs.tech_map_id,
                rs.root_lot_id,
                rs.created_at,
                0 AS depth,
                CAST(NULL AS VARCHAR(30)) AS via_link_type,
                CAST(NULL AS UUID) AS child_rs_id
            FROM route_sheets rs
            WHERE rs.tracking_code = :code
            UNION ALL
            SELECT
                src.id,
                src.tracking_code,
                src.status,
                src.hold_reason,
                src.tech_map_id,
                src.root_lot_id,
                src.created_at,
                l.depth + 1 AS depth,
                rsl.link_type AS via_link_type,
                l.id AS child_rs_id
            FROM route_sheet_links rsl
            JOIN lineage l ON rsl.target_rs_id = l.id
            JOIN route_sheets src ON src.id = rsl.source_rs_id
            WHERE l.depth < 10
        )
        SELECT
            l.id,
            l.tracking_code,
            l.status,
            l.hold_reason,
            l.depth,
            l.via_link_type,
            l.created_at,
            tm.item_code AS tech_map_code,
            tm.name AS tech_map_name,
            lm.supplier_name,
            lm.invoice_number,
            lm.arrival_date,
            lm.component_specs
        FROM lineage l
        JOIN tech_maps tm ON tm.id = l.tech_map_id
        LEFT JOIN lot_metadata lm ON lm.route_sheet_id = l.id
        ORDER BY l.depth ASC, l.tracking_code ASC
        """
    )
    rows = (await session.execute(cte_sql, {"code": code})).mappings().all()

    nodes: list[dict[str, Any]] = []
    root_lots: list[dict[str, Any]] = []
    bom_specs: dict[str, Any] = {}
    battery_rs: dict[str, Any] | None = None

    for r in rows:
        item = {
            "route_sheet_id": str(r["id"]),
            "tracking_code": r["tracking_code"],
            "status": r["status"],
            "hold_reason": r["hold_reason"],
            "depth": int(r["depth"]),
            "via_link_type": r["via_link_type"],
            "tech_map_code": r["tech_map_code"],
            "tech_map_name": r["tech_map_name"],
        }
        if r["supplier_name"] is not None:
            item["lot_metadata"] = {
                "supplier_name": r["supplier_name"],
                "invoice_number": r["invoice_number"],
                "arrival_date": r["arrival_date"].isoformat() if r["arrival_date"] else None,
                "component_specs": r["component_specs"] or {},
            }
            root_lots.append(item)
            bom_specs.update(r["component_specs"] or {})
        if r["tech_map_code"] == "TM-BAT-6S" and r["via_link_type"] == "ASSEMBLED_INTO":
            battery_rs = item
        nodes.append(item)

    return {
        "found": True,
        "target_tracking_code": code,
        "target_route_sheet_id": str(root_rs.id),
        "lineage_nodes": nodes,
        "battery_component": battery_rs,
        "root_lots": root_lots,
        "bom_specs": bom_specs,
    }


async def _fetch_executions_for_rs(session: AsyncSession, rs_id: UUID) -> list[dict[str, Any]]:
    exec_stmt = (
        select(
            StepExecution,
            TechMapNode,
            Operation,
            Workstation,
            User,
            Instrument,
        )
        .join(TechMapNode, TechMapNode.id == StepExecution.node_id)
        .join(Operation, Operation.id == TechMapNode.operation_id)
        .join(Workstation, Workstation.id == TechMapNode.workstation_id)
        .join(User, User.id == StepExecution.operator_id)
        .outerjoin(Instrument, Instrument.id == StepExecution.instrument_id)
        .where(StepExecution.route_sheet_id == rs_id)
        .order_by(StepExecution.completed_at.asc())
    )
    exec_rows = (await session.execute(exec_stmt)).all()
    exec_ids = [row[0].id for row in exec_rows]

    meas_by_exec: dict[UUID, list[dict[str, Any]]] = {}
    if exec_ids:
        meas_stmt = (
            select(StepMeasurement, TechMapNodeGuard)
            .join(TechMapNodeGuard, TechMapNodeGuard.id == StepMeasurement.guard_id)
            .where(StepMeasurement.execution_id.in_(exec_ids))
            .order_by(StepMeasurement.id.asc())
        )
        for m, g in (await session.execute(meas_stmt)).all():
            meas_by_exec.setdefault(m.execution_id, []).append(
                {
                    "guard_name": g.name,
                    "guard_type": g.guard_type,
                    "numeric_value": float(m.numeric_value)
                    if m.numeric_value is not None
                    else None,
                    "boolean_value": m.boolean_value,
                    "text_value": m.text_value,
                    "min_value": float(g.min_value) if g.min_value is not None else None,
                    "max_value": float(g.max_value) if g.max_value is not None else None,
                    "nominal_value": float(g.nominal_value)
                    if g.nominal_value is not None
                    else None,
                    "unit": g.unit,
                    "is_passed": m.is_passed,
                }
            )

    out: list[dict[str, Any]] = []
    for se, node, op, ws, usr, instr in exec_rows:
        cal_expired = False
        if instr is not None and instr.calibration_due_at is not None:
            cal_expired = se.completed_at > instr.calibration_due_at
        out.append(
            {
                "execution_id": str(se.id),
                "step_order": node.step_order,
                "execution_scope": node.execution_scope,
                "operation_code": op.code,
                "operation_name": op.name,
                "workstation_code": ws.code,
                "operator_id": str(usr.id),
                "operator_code": usr.employee_code,
                "operator_name": usr.short_name,
                "instrument_code": instr.code if instr else None,
                "channel_no": se.channel_no,
                "calibration_due_at": instr.calibration_due_at.isoformat() if instr else None,
                "calibration_expired_at_execution": cal_expired,
                "outcome": se.outcome,
                "completed_at": se.completed_at.isoformat(),
                "measurements": meas_by_exec.get(se.id, []),
            }
        )
    return out


async def query_item_history(session: AsyncSession, route_sheet_id: str) -> dict[str, Any]:
    """Fetch complete execution, measurement, REWORK, calibration, and ticket history."""
    parsed_id = _try_uuid(route_sheet_id)
    if parsed_id is not None:
        rs = await session.scalar(select(RouteSheet).where(RouteSheet.id == parsed_id))
    else:
        rs = await session.scalar(
            select(RouteSheet).where(RouteSheet.tracking_code == route_sheet_id.strip().upper())
        )
    if rs is None:
        return {
            "found": False,
            "route_sheet_id": route_sheet_id,
            "error": "Маршрутный лист не найден.",
        }

    executions = await _fetch_executions_for_rs(session, rs.id)
    tickets = list(
        (
            await session.scalars(
                select(SupportTicket)
                .where(SupportTicket.route_sheet_id == rs.id)
                .order_by(SupportTicket.created_at.desc())
            )
        ).all()
    )
    ticket_list = [
        {
            "ticket_id": str(t.id),
            "title": t.title,
            "symptom_category": t.symptom_category,
            "status": t.status,
            "priority": t.priority,
            "root_cause_summary": t.root_cause_summary,
            "created_at": t.created_at.isoformat(),
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in tickets
    ]

    # Also summarize direct ASSEMBLED_INTO component histories (e.g. battery stand/channel & rework)
    comp_stmt = text(
        """
        SELECT src.id, src.tracking_code
        FROM route_sheet_links rsl
        JOIN route_sheets src ON src.id = rsl.source_rs_id
        WHERE rsl.target_rs_id = :rs_id AND rsl.link_type = 'ASSEMBLED_INTO'
        ORDER BY src.tracking_code
        """
    )
    comp_rows = (await session.execute(comp_stmt, {"rs_id": rs.id})).all()
    component_histories: dict[str, Any] = {}
    for cid, ccode in comp_rows:
        component_histories[str(ccode)] = await _fetch_executions_for_rs(session, cid)

    rework_count = sum(1 for e in executions if e["outcome"] == StepOutcome.REWORK.value)
    comp_rework_count = sum(
        1
        for hist in component_histories.values()
        for e in hist
        if e["outcome"] == StepOutcome.REWORK.value
    )

    return {
        "found": True,
        "route_sheet_id": str(rs.id),
        "tracking_code": rs.tracking_code,
        "status": rs.status,
        "hold_reason": rs.hold_reason,
        "rework_count": rework_count,
        "component_rework_count": comp_rework_count,
        "executions": executions,
        "component_histories": component_histories,
        "prior_support_tickets": ticket_list,
    }


async def query_lot_quality_stats(
    session: AsyncSession, lot_tracking_code_or_id: str
) -> dict[str, Any]:
    """Return lot size, HOLD status, defect/rework rates, failed guards, and ALG-001 state."""
    parsed_id = _try_uuid(lot_tracking_code_or_id)
    if parsed_id is not None:
        lot = await session.scalar(select(RouteSheet).where(RouteSheet.id == parsed_id))
    else:
        lot = await session.scalar(
            select(RouteSheet).where(
                RouteSheet.tracking_code == lot_tracking_code_or_id.strip().upper()
            )
        )
    if lot is None:
        return {"found": False, "lot": lot_tracking_code_or_id, "error": "Партия не найдена."}

    meta = await session.scalar(select(LotMetadata).where(LotMetadata.route_sheet_id == lot.id))
    children = list(
        (await session.scalars(select(RouteSheet).where(RouteSheet.root_lot_id == lot.id))).all()
    )
    child_ids = [c.id for c in children]
    total_units = len(children) if children else lot.planned_qty
    defect_units = (
        sum(1 for c in children if c.status == "DEFECT") if children else lot.defect_count
    )

    rework_exec_count = 0
    rework_unit_count = 0
    failed_guards: list[dict[str, Any]] = []

    if child_ids:
        rework_stmt = select(
            func.count(StepExecution.id),
            func.count(func.distinct(StepExecution.route_sheet_id)),
        ).where(
            StepExecution.route_sheet_id.in_(child_ids),
            StepExecution.outcome == StepOutcome.REWORK.value,
        )
        rw_total, rw_units = (await session.execute(rework_stmt)).one()
        rework_exec_count = int(rw_total or 0)
        rework_unit_count = int(rw_units or 0)

        fg_stmt = (
            select(TechMapNodeGuard.name, func.count(StepMeasurement.id).label("fails"))
            .join(StepMeasurement, StepMeasurement.guard_id == TechMapNodeGuard.id)
            .join(StepExecution, StepExecution.id == StepMeasurement.execution_id)
            .where(
                StepExecution.route_sheet_id.in_(child_ids),
                StepMeasurement.is_passed.is_(False),
            )
            .group_by(TechMapNodeGuard.name)
            .order_by(func.count(StepMeasurement.id).desc())
        )
        for gname, cnt in (await session.execute(fg_stmt)).all():
            failed_guards.append({"guard_name": gname, "fail_count": int(cnt)})

    sampling_rows = list(
        (
            await session.scalars(
                select(SamplingNodeState).where(SamplingNodeState.lot_rs_id == lot.id)
            )
        ).all()
    )
    defect_rate_pct = round((defect_units / total_units) * 100.0, 2) if total_units else 0.0
    rework_rate_pct = round((rework_unit_count / total_units) * 100.0, 2) if total_units else 0.0

    return {
        "found": True,
        "lot_route_sheet_id": str(lot.id),
        "lot_tracking_code": lot.tracking_code,
        "status": lot.status,
        "hold_reason": lot.hold_reason,
        "supplier_name": meta.supplier_name if meta else None,
        "arrival_date": meta.arrival_date.isoformat() if meta else None,
        "planned_qty": lot.planned_qty,
        "total_split_units": total_units,
        "defect_units": defect_units,
        "defect_rate_pct": defect_rate_pct,
        "rework_units": rework_unit_count,
        "rework_executions": rework_exec_count,
        "rework_rate_pct": rework_rate_pct,
        "exceeds_10pct_recall_threshold": defect_rate_pct > 10.0 or lot.hold_reason is not None,
        "top_failed_guards": failed_guards,
        "alg001_sampling_states": [
            {
                "node_id": str(s.node_id),
                "m_rem": s.m_rem,
                "n_rem": s.n_rem,
                "w_rem": s.w_rem,
                "taken_in_window": s.taken_in_window,
                "escalations_used": s.escalations_used,
            }
            for s in sampling_rows
        ],
    }


async def query_affected_products_by_lot(
    session: AsyncSession, lot_tracking_code_or_id: str
) -> dict[str, Any]:
    """Forward recursive CTE over route_sheet_links (source -> target) for Recall Analysis."""
    parsed_id = _try_uuid(lot_tracking_code_or_id)
    if parsed_id is not None:
        lot = await session.scalar(select(RouteSheet).where(RouteSheet.id == parsed_id))
    else:
        lot = await session.scalar(
            select(RouteSheet).where(
                RouteSheet.tracking_code == lot_tracking_code_or_id.strip().upper()
            )
        )
    if lot is None:
        return {
            "found": False,
            "lot": lot_tracking_code_or_id,
            "affected_cars": [],
            "affected_count": 0,
        }

    cte_sql = text(
        """
        WITH RECURSIVE downstream AS (
            SELECT
                rs.id,
                rs.tracking_code,
                rs.status,
                rs.tech_map_id,
                0 AS depth,
                CAST(NULL AS VARCHAR(100)) AS source_component_code
            FROM route_sheets rs
            WHERE rs.id = :lot_id
            UNION ALL
            SELECT
                tgt.id,
                tgt.tracking_code,
                tgt.status,
                tgt.tech_map_id,
                d.depth + 1 AS depth,
                d.tracking_code AS source_component_code
            FROM route_sheet_links rsl
            JOIN downstream d ON rsl.source_rs_id = d.id
            JOIN route_sheets tgt ON tgt.id = rsl.target_rs_id
            WHERE d.depth < 10
        )
        SELECT DISTINCT
            d.id,
            d.tracking_code,
            d.status,
            d.source_component_code,
            tm.item_code AS tech_map_code
        FROM downstream d
        JOIN tech_maps tm ON tm.id = d.tech_map_id
        WHERE tm.item_code = 'TM-CAR-FINAL'
        ORDER BY d.tracking_code ASC
        """
    )
    rows = (await session.execute(cte_sql, {"lot_id": lot.id})).mappings().all()
    affected = [
        {
            "route_sheet_id": str(r["id"]),
            "tracking_code": r["tracking_code"],
            "status": r["status"],
            "battery_tracking_code": r["source_component_code"],
        }
        for r in rows
    ]
    return {
        "found": True,
        "lot_tracking_code": lot.tracking_code,
        "affected_count": len(affected),
        "affected_car_codes": [a["tracking_code"] for a in affected],
        "affected_cars": affected,
    }


async def query_instrument_channel_anomalies(
    session: AsyncSession,
    instrument_code: str,
    channel_no: int | None,
    around_date: str,
) -> dict[str, Any]:
    """Check instrument calibration status and compare channel vs sibling channels in +-14d window."""
    code = instrument_code.strip().upper()
    instr = await session.scalar(select(Instrument).where(Instrument.code == code))
    if instr is None:
        return {"found": False, "instrument_code": code, "error": f"Прибор {code} не найден."}

    ref_dt = _parse_iso_date(around_date)
    win_start = ref_dt - timedelta(days=14)
    win_end = ref_dt + timedelta(days=14)

    cal_expired = ref_dt > instr.calibration_due_at
    exec_stats_sql = text(
        """
        SELECT
            se.channel_no,
            COUNT(DISTINCT se.id) AS total_execs,
            COUNT(DISTINCT se.id) FILTER (WHERE se.outcome IN ('REWORK', 'DEFECT')) AS fail_or_rework_execs
        FROM step_executions se
        WHERE se.instrument_id = :instr_id
          AND se.completed_at >= :win_start
          AND se.completed_at <= :win_end
        GROUP BY se.channel_no
        ORDER BY se.channel_no NULLS LAST
        """
    )
    exec_rows = (
        (
            await session.execute(
                exec_stats_sql,
                {"instr_id": instr.id, "win_start": win_start, "win_end": win_end},
            )
        )
        .mappings()
        .all()
    )

    guard_stats_sql = text(
        """
        SELECT
            se.channel_no,
            g.name AS guard_name,
            g.unit AS unit,
            COUNT(sm.id) AS measurement_count,
            AVG(sm.numeric_value) AS avg_value,
            MIN(sm.numeric_value) AS min_value,
            MAX(sm.numeric_value) AS max_value
        FROM step_executions se
        JOIN step_measurements sm ON sm.execution_id = se.id AND sm.numeric_value IS NOT NULL
        JOIN tech_map_node_guards g ON g.id = sm.guard_id
        WHERE se.instrument_id = :instr_id
          AND se.completed_at >= :win_start
          AND se.completed_at <= :win_end
        GROUP BY se.channel_no, g.name, g.unit
        ORDER BY se.channel_no NULLS LAST, g.name
        """
    )
    guard_rows = (
        (
            await session.execute(
                guard_stats_sql,
                {"instr_id": instr.id, "win_start": win_start, "win_end": win_end},
            )
        )
        .mappings()
        .all()
    )

    # Determine a single consistent primary physical measurement (prefer 'V' if present,
    # otherwise the most frequent guard+unit) so heterogeneous units are never averaged together.
    guard_totals: dict[tuple[str, str | None], int] = {}
    per_channel_guards: dict[int | None, list[dict[str, Any]]] = {}
    for gr in guard_rows:
        gname = str(gr["guard_name"])
        unit = str(gr["unit"]) if gr["unit"] is not None else None
        cnt = int(gr["measurement_count"])
        guard_totals[(gname, unit)] = guard_totals.get((gname, unit), 0) + cnt
        per_channel_guards.setdefault(gr["channel_no"], []).append(
            {
                "guard_name": gname,
                "unit": unit,
                "measurement_count": cnt,
                "avg_numeric_value": round(float(gr["avg_value"]), 4)
                if gr["avg_value"] is not None
                else None,
                "min_numeric_value": round(float(gr["min_value"]), 4)
                if gr["min_value"] is not None
                else None,
                "max_numeric_value": round(float(gr["max_value"]), 4)
                if gr["max_value"] is not None
                else None,
            }
        )

    primary_key: tuple[str, str | None] | None = None
    if guard_totals:
        volt_keys = [k for k in guard_totals if k[1] == "V"]
        if volt_keys:
            primary_key = max(volt_keys, key=lambda k: (guard_totals[k], k[0]))
        else:
            primary_key = max(guard_totals.keys(), key=lambda k: (guard_totals[k], k[0]))

    primary_guard = primary_key[0] if primary_key else None
    primary_unit = primary_key[1] if primary_key else None

    channel_stats: list[dict[str, Any]] = []
    target_stat: dict[str, Any] | None = None
    other_total = 0
    other_fails = 0
    other_vals: list[float] = []

    for r in exec_rows:
        ch_no = r["channel_no"]
        tot = int(r["total_execs"])
        fails = int(r["fail_or_rework_execs"])
        rate = round((fails / tot) * 100.0, 2) if tot else 0.0
        guards_for_ch = per_channel_guards.get(ch_no, [])
        primary_stat = (
            next(
                (
                    gs
                    for gs in guards_for_ch
                    if gs["guard_name"] == primary_guard and gs["unit"] == primary_unit
                ),
                None,
            )
            if primary_key is not None
            else None
        )
        avg_v = primary_stat["avg_numeric_value"] if primary_stat else None
        entry = {
            "channel_no": ch_no,
            "total_execs": tot,
            "fail_or_rework_execs": fails,
            "anomaly_rate_pct": rate,
            "primary_guard": primary_guard,
            "primary_unit": primary_unit,
            "avg_numeric_value": avg_v,
            "min_numeric_value": primary_stat["min_numeric_value"] if primary_stat else None,
            "max_numeric_value": primary_stat["max_numeric_value"] if primary_stat else None,
            "per_guard_stats": guards_for_ch,
        }
        channel_stats.append(entry)
        if channel_no is not None and ch_no == channel_no:
            target_stat = entry
        else:
            other_total += tot
            other_fails += fails
            if avg_v is not None:
                other_vals.append(avg_v)

    other_rate = round((other_fails / other_total) * 100.0, 2) if other_total else 0.0
    other_avg = round(sum(other_vals) / len(other_vals), 4) if other_vals else None

    is_channel_anomalous = False
    if target_stat is not None:
        if target_stat["anomaly_rate_pct"] >= max(25.0, other_rate * 3.0):
            is_channel_anomalous = True
        if (
            other_avg is not None
            and target_stat["avg_numeric_value"] is not None
            and primary_unit == "V"
            and abs(target_stat["avg_numeric_value"] - other_avg) >= 0.4
        ):
            is_channel_anomalous = True

    total_channels_observed = len(channel_stats)
    breakdown_out = channel_stats
    if total_channels_observed > 5:
        top_limit = 3 if target_stat is not None else 4
        others_ranked = sorted(
            (c for c in channel_stats if channel_no is None or c["channel_no"] != channel_no),
            key=lambda c: (
                -float(c["anomaly_rate_pct"]),
                -int(c["total_execs"]),
                str(c["channel_no"]),
            ),
        )[:top_limit]
        breakdown_out = ([target_stat] if target_stat is not None else []) + others_ranked

    return {
        "found": True,
        "instrument_code": instr.code,
        "instrument_name": instr.name,
        "channel_no": channel_no,
        "primary_guard": primary_guard,
        "primary_unit": primary_unit,
        "window": f"{win_start.date().isoformat()}..{win_end.date().isoformat()}",
        "calibration_due_at": instr.calibration_due_at.isoformat(),
        "calibration_expired_in_window": cal_expired,
        "target_channel_stats": target_stat,
        "other_channels_baseline": {
            "total_execs": other_total,
            "fail_or_rework_execs": other_fails,
            "anomaly_rate_pct": other_rate,
            "primary_unit": primary_unit,
            "avg_numeric_value": other_avg,
        },
        "is_channel_anomalous": is_channel_anomalous,
        "total_channels_observed": total_channels_observed,
        "per_channel_breakdown": breakdown_out,
    }


async def query_operator_anomalies(
    session: AsyncSession, operator_id_or_name: str, around_date: str
) -> dict[str, Any]:
    """Compare operator REWORK/DEFECT rate and measurement deviations vs shop average in +-14d window."""
    raw = operator_id_or_name.strip()
    parsed_id = _try_uuid(raw)
    usr: User | None = None
    if parsed_id is not None:
        usr = await session.scalar(select(User).where(User.id == parsed_id))
    else:
        usr = await session.scalar(select(User).where(User.employee_code == raw.upper()))
        if usr is None:
            last_token = raw.split()[0]
            usr = await session.scalar(
                select(User).where(func.lower(User.last_name).contains(last_token.lower()))
            )

    if usr is None:
        return {"found": False, "operator": raw, "error": f"Оператор '{raw}' не найден."}

    ref_dt = _parse_iso_date(around_date)
    win_start = ref_dt - timedelta(days=14)
    win_end = ref_dt + timedelta(days=14)

    op_sql = text(
        """
        WITH target_nodes AS (
            SELECT DISTINCT se.node_id
            FROM step_executions se
            WHERE se.operator_id = :op_id
              AND se.completed_at >= :win_start
              AND se.completed_at <= :win_end
        )
        SELECT
            se.operator_id = :op_id AS is_target,
            COUNT(DISTINCT se.id) AS total_execs,
            COUNT(DISTINCT se.id) FILTER (WHERE se.outcome IN ('REWORK', 'DEFECT')) AS fail_or_rework_execs,
            AVG(sm.numeric_value) FILTER (WHERE g.unit = 'Nm') AS avg_torque_nm
        FROM step_executions se
        JOIN target_nodes tn ON tn.node_id = se.node_id
        LEFT JOIN step_measurements sm ON sm.execution_id = se.id AND sm.numeric_value IS NOT NULL
        LEFT JOIN tech_map_node_guards g ON g.id = sm.guard_id
        WHERE se.completed_at >= :win_start
          AND se.completed_at <= :win_end
        GROUP BY (se.operator_id = :op_id)
        """
    )
    rows = (
        (
            await session.execute(
                op_sql,
                {"op_id": usr.id, "win_start": win_start, "win_end": win_end},
            )
        )
        .mappings()
        .all()
    )

    ops_sql = text(
        """
        SELECT DISTINCT o.code
        FROM step_executions se
        JOIN tech_map_nodes tmn ON tmn.id = se.node_id
        JOIN operations o ON o.id = tmn.operation_id
        WHERE se.operator_id = :op_id
          AND se.completed_at >= :win_start
          AND se.completed_at <= :win_end
        ORDER BY o.code
        """
    )
    compared_ops = list(
        (
            await session.scalars(
                ops_sql,
                {"op_id": usr.id, "win_start": win_start, "win_end": win_end},
            )
        ).all()
    )

    target_data: dict[str, Any] = {
        "total_execs": 0,
        "fail_or_rework_execs": 0,
        "rework_rate_pct": 0.0,
        "avg_torque_nm": None,
    }
    shop_data: dict[str, Any] = {
        "total_execs": 0,
        "fail_or_rework_execs": 0,
        "rework_rate_pct": 0.0,
        "avg_torque_nm": None,
    }

    for r in rows:
        tot = int(r["total_execs"])
        fails = int(r["fail_or_rework_execs"])
        rate = round((fails / tot) * 100.0, 2) if tot else 0.0
        avg_t = round(float(r["avg_torque_nm"]), 4) if r["avg_torque_nm"] is not None else None
        bucket = target_data if r["is_target"] else shop_data
        bucket["total_execs"] = tot
        bucket["fail_or_rework_execs"] = fails
        bucket["rework_rate_pct"] = rate
        bucket["avg_torque_nm"] = avg_t

    is_anomalous = target_data["rework_rate_pct"] >= max(15.0, shop_data["rework_rate_pct"] * 3.0)

    return {
        "found": True,
        "operator_id": str(usr.id),
        "operator_code": usr.employee_code,
        "operator_name": usr.short_name,
        "window": f"{win_start.date().isoformat()}..{win_end.date().isoformat()}",
        "compared_operations": compared_ops,
        "operator_stats": target_data,
        "shop_other_operators_baseline": shop_data,
        "is_operator_anomalous": is_anomalous,
    }


# =============================================================================
# LangChain Typed Domain Tools Factory (Strictly Typed, No Text-to-SQL!)
# =============================================================================


def build_domain_tools(
    session_factory: async_sessionmaker[AsyncSession],
    retriever: HybridRetriever,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """Create the 7 strictly typed domain tools bound to async PostgreSQL and Hybrid RAG."""
    _ = settings or get_settings()

    @tool
    async def get_item_genealogy(tracking_code: str) -> str:
        """Раскрутить родословную изделия (ASSEMBLED_INTO -> SPLIT): машинка -> кузов + АКБ + корневая партия АКБ, поставщик и BOM."""
        async with session_factory() as session:
            data = await query_item_genealogy(session, tracking_code)
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def get_item_history(route_sheet_id: str) -> str:
        """Получить полную историю операций, замеров, петель REWORK, использованных стендов/каналов, поверок и прошлых тикетов по route_sheet_id или tracking_code."""
        async with session_factory() as session:
            data = await query_item_history(session, route_sheet_id)
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def get_lot_quality_stats(lot_tracking_code_or_id: str) -> str:
        """Получить статистику качества партии: объём, hold_reason, процент брака/REWORK, топ проваленных гардов и состояние ALG-001."""
        async with session_factory() as session:
            data = await query_lot_quality_stats(session, lot_tracking_code_or_id)
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def find_affected_products_by_lot(lot_tracking_code_or_id: str) -> str:
        """Прямой Recall-поиск (WITH RECURSIVE): список всех готовых машинок, в которые попали комплектующие из указанной партии."""
        async with session_factory() as session:
            data = await query_affected_products_by_lot(session, lot_tracking_code_or_id)
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def get_instrument_channel_anomalies(
        instrument_code: str, channel_no: int | None, around_date: str
    ) -> str:
        """Проверить статус поверки стенда (calibration_due_at) и сравнить аномалии/средние замеры канала vs остальные каналы в окне +-14 дней."""
        async with session_factory() as session:
            data = await query_instrument_channel_anomalies(
                session, instrument_code, channel_no, around_date
            )
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def get_operator_anomalies(operator_id_or_name: str, around_date: str) -> str:
        """Сравнить долю REWORK/брака и средние замеры оператора vs среднее по цеху в окне +-14 дней вокруг указанной даты (YYYY-MM-DD)."""
        async with session_factory() as session:
            data = await query_operator_anomalies(session, operator_id_or_name, around_date)
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def search_engineering_kb(query: str) -> str:
        """Гибридный поиск (Qdrant LaBSE-en-ru + BM25 RRF) по инженерным регламентам SOP (допуски АКБ, RS550-24V, стенды, гарантия/отзыв)."""
        hits = retriever.search(query)
        payload = [
            {
                "source_file": h.source_file,
                "section_title": h.section_title,
                "rrf_score": h.rrf_score,
                "content": h.content,
            }
            for h in hits
        ]
        return json.dumps(payload, ensure_ascii=False)

    return [
        get_item_genealogy,
        get_item_history,
        get_lot_quality_stats,
        find_affected_products_by_lot,
        get_instrument_channel_anomalies,
        get_operator_anomalies,
        search_engineering_kb,
    ]
