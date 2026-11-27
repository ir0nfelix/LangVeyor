from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from src.db.enums import RouteSheetStatus
from src.db.models import (
    RouteSheet,
    SamplingNodeState,
    StepExecution,
    TechMap,
    TechMapNode,
    TechMapNodeGuard,
    User,
)
from src.mes.engine import GuardInput, MesEngine
from src.mes.sampling import (
    SamplingService,
    SamplingState,
    TriadState,
    apply_escalation,
    apply_underdelivery_delta,
    check_stop_crane,
    compute_hmac_uniform,
    evaluate_triad_state,
    init_sampling_state,
    step_sampling,
)


def test_alg001_scenario_1_hmac_determinism_and_uniformity() -> None:
    """Scenario 1: HMAC-SHA256 draws are strictly deterministic and reproducible."""
    secret = "test-secret-2026"
    u1 = compute_hmac_uniform(secret, "lot-1", "node-5", "BAT-001")
    u2 = compute_hmac_uniform(secret, "lot-1", "node-5", "BAT-001")
    u3 = compute_hmac_uniform(secret, "lot-1", "node-5", "BAT-002")

    assert u1 == u2
    assert 0.0 <= u1 < 1.0
    assert 0.0 <= u3 < 1.0
    assert u1 != u3


def test_alg001_scenario_2_stratified_window_exact_quota() -> None:
    """Scenario 2: Stratified Fan-Muller-Rezucha guarantees exactly 1 sample per stratum."""
    state = init_sampling_state(planned_qty=30, sample_size=5)
    assert state.n_rem == 30
    assert state.m_rem == 5
    assert state.w_rem == 6

    selected_indices: list[int] = []
    for idx in range(30):
        u = compute_hmac_uniform("seed", "lot-30", "node-tst", f"ITEM-{idx:03d}")
        state, dec = step_sampling(state, u)
        if dec.selected:
            selected_indices.append(idx)

    assert len(selected_indices) == 5
    assert state.n_rem == 0
    assert state.m_rem == 0
    # Each 6-item stratum [0..5], [6..11], [12..17], [18..23], [24..29] must have exactly 1 sample
    strata = [idx // 6 for idx in selected_indices]
    assert strata == [0, 1, 2, 3, 4]


def test_alg001_scenario_3_linear_stop_crane() -> None:
    """Scenario 3: Linear stop-crane d_min + R_slope * processed_count prevents small-N false stops."""
    # d_min = 3, R_slope = 0.08
    # Small-N early noise: 2 defects out of 5 items (40% rate!) must NOT stop the line (< 3 + 0.4 = 3.4)
    early = check_stop_crane(
        defect_count=2, processed_count=5, stop_min_defects=3, stop_defect_slope=0.08
    )
    assert early.triggered is False
    assert early.hold_reason is None

    # Sustained rate: 5 defects out of 25 items -> threshold = 3 + 0.08*25 = 5.0 -> TRIGGERED!
    sustained = check_stop_crane(
        defect_count=5, processed_count=25, stop_min_defects=3, stop_defect_slope=0.08
    )
    assert sustained.triggered is True
    assert sustained.hold_reason == "DEFECT_RATE_EXCEEDED"


def test_alg001_scenario_4_escalation_triad_surplus_exact_deficit() -> None:
    """Scenario 4: Escalation triad N_rem > m_rem, N_rem == m_rem, and N_rem < m_rem."""
    # Case A: N_rem > m_rem (surplus)
    s_surplus = SamplingState(m_rem=2, n_rem=12, w_rem=6, taken_in_window=0, escalations_used=0)
    nxt_a, res_a = apply_escalation(s_surplus, escalation_sample_size=3, max_escalations=2)
    assert res_a.escalated is True
    assert res_a.triad_state == TriadState.SURPLUS
    assert nxt_a.m_rem == 5
    assert nxt_a.w_rem == 12 // 5

    # Case B: N_rem == m_rem (exact match -> 100% remaining)
    s_exact = SamplingState(m_rem=2, n_rem=5, w_rem=2, taken_in_window=0, escalations_used=0)
    nxt_b, res_b = apply_escalation(s_exact, escalation_sample_size=3, max_escalations=2)
    assert res_b.triad_state == TriadState.EXACT
    assert res_b.shortfall == 0
    _, dec_b = step_sampling(nxt_b, 0.9999)
    assert dec_b.selected is True
    assert dec_b.probability == 1.0

    # Case C: N_rem < m_rem (deficit -> 100% remaining + shortfall reported)
    s_def = SamplingState(m_rem=3, n_rem=4, w_rem=1, taken_in_window=0, escalations_used=1)
    nxt_c, res_c = apply_escalation(s_def, escalation_sample_size=3, max_escalations=2)
    assert res_c.triad_state == TriadState.DEFICIT
    assert res_c.shortfall == 2
    _, dec_c = step_sampling(nxt_c, 0.9999)
    assert dec_c.selected is True
    assert dec_c.probability == 1.0
    assert dec_c.shortfall == 2


def test_alg001_scenario_5_max_escalations_cap() -> None:
    """Scenario 5: Escalations cannot exceed max_escalations."""
    s = SamplingState(m_rem=4, n_rem=20, w_rem=5, taken_in_window=0, escalations_used=2)
    nxt, res = apply_escalation(s, escalation_sample_size=3, max_escalations=2)
    assert res.escalated is False
    assert res.reason == "MAX_ESCALATIONS_REACHED"
    assert nxt == s


def test_alg001_scenario_6_full_lot_with_escalation_and_quota_enforcement() -> None:
    """Scenario 6: Full lot run with mid-lot escalation still draws exact updated quota."""
    state = init_sampling_state(planned_qty=40, sample_size=4)
    drawn = 0
    for idx in range(40):
        if idx == 10:
            state, esc = apply_escalation(state, escalation_sample_size=4, max_escalations=2)
            assert esc.escalated is True
        u = compute_hmac_uniform("lot-40-secret", "lot-40", "node-5", f"UNIT-{idx:03d}")
        state, dec = step_sampling(state, u)
        if dec.selected:
            drawn += 1

    # Before idx 10 (items 0..9, 1st stratum of 10), 1 sample drawn, 3 remaining + 4 escalated = 8 total
    assert state.n_rem == 0
    assert state.m_rem == 0
    assert drawn == 8


def test_alg001_underdelivery_delta_adjustment() -> None:
    """Underdelivery delta adjusts n_rem and preserves feasibility or enters 100% mode."""
    # Case 1: Moderate underdelivery (still N_rem > m_rem)
    s1 = SamplingState(m_rem=4, n_rem=20, w_rem=3, taken_in_window=1, escalations_used=0)
    adj1 = apply_underdelivery_delta(s1, delta=14)  # new_n = 6, m_rem = 4 -> max_skip = 2
    assert adj1.n_rem == 6
    assert adj1.w_rem <= 2
    assert evaluate_triad_state(adj1.n_rem, adj1.m_rem) == TriadState.SURPLUS

    # Simulate remaining 6 items -> must still draw all 4 remaining samples!
    drawn = 0
    cur = adj1
    for _i in range(6):
        cur, dec = step_sampling(cur, 0.5)
        if dec.selected:
            drawn += 1
    assert drawn == 4

    # Case 2: Heavy underdelivery dropping N_rem < m_rem -> switches to 100% control
    s2 = SamplingState(m_rem=5, n_rem=20, w_rem=4, taken_in_window=0, escalations_used=0)
    adj2 = apply_underdelivery_delta(s2, delta=17)  # new_n = 3 < m_rem = 5
    assert adj2.n_rem == 3
    assert evaluate_triad_state(adj2.n_rem, adj2.m_rem) == TriadState.DEFICIT
    cur2 = adj2
    drawn2 = 0
    for _ in range(3):
        cur2, dec2 = step_sampling(cur2, 0.99)
        if dec2.selected:
            drawn2 += 1
    assert drawn2 == 3


def test_alg001_deficit_sets_hold_insufficient_sample() -> None:
    """Triad DEFICIT (n_rem < m_rem) must flag trigger_hold_reason='HOLD_INSUFFICIENT_SAMPLE'."""
    s_def = SamplingState(m_rem=4, n_rem=2, w_rem=2, taken_in_window=0, escalations_used=0)
    nxt, dec = step_sampling(s_def, 0.5)
    assert dec.triad_state == TriadState.DEFICIT
    assert dec.selected is True
    assert dec.shortfall == 2
    assert dec.trigger_hold_reason == "HOLD_INSUFFICIENT_SAMPLE"
    assert nxt.n_rem == 1
    assert nxt.m_rem == 3

    _, esc = apply_escalation(
        SamplingState(m_rem=2, n_rem=4, w_rem=2, taken_in_window=0, escalations_used=0),
        escalation_sample_size=3,
        max_escalations=2,
    )
    assert esc.triad_state == TriadState.DEFICIT
    assert esc.trigger_hold_reason == "HOLD_INSUFFICIENT_SAMPLE"


async def test_sampling_service_deficit_persists_lot_hold_reason(
    db_engine: AsyncEngine,
) -> None:
    """SamplingService.decide_and_advance writes HOLD_INSUFFICIENT_SAMPLE on root lot on DEFICIT."""
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        try:
            async with AsyncSession(bind=conn, expire_on_commit=False) as session:
                tm_bat = await session.scalar(
                    select(TechMap).where(TechMap.item_code == "TM-BAT-6S")
                )
                assert tm_bat is not None
                bat_tst = await session.scalar(
                    select(TechMapNode).where(
                        TechMapNode.tech_map_id == tm_bat.id, TechMapNode.step_order == 5
                    )
                )
                assert bat_tst is not None

                lot = RouteSheet(
                    tech_map_id=tm_bat.id,
                    tracking_code=f"LOT-DEF-{uuid.uuid4().hex[:6].upper()}",
                    qty=5,
                    planned_qty=5,
                    status=RouteSheetStatus.SPLIT_CLOSED.value,
                )
                session.add(lot)
                await session.flush()

                session.add(
                    SamplingNodeState(
                        lot_rs_id=lot.id,
                        node_id=bat_tst.id,
                        m_rem=4,
                        n_rem=2,
                        w_rem=2,
                        taken_in_window=0,
                        escalations_used=0,
                    )
                )
                await session.flush()

                svc = SamplingService("test-secret")
                dec = await svc.decide_and_advance(session, lot.id, bat_tst.id, "BAT-DEF-001")
                assert dec.triad_state == TriadState.DEFICIT
                assert dec.trigger_hold_reason == "HOLD_INSUFFICIENT_SAMPLE"

                refreshed = await session.scalar(select(RouteSheet).where(RouteSheet.id == lot.id))
                assert refreshed is not None
                assert refreshed.hold_reason == "HOLD_INSUFFICIENT_SAMPLE"
        finally:
            await trans.rollback()


async def test_mes_engine_split_underdelivery_rework_dedup_and_early_defect_stop_crane(
    db_engine: AsyncEngine,
) -> None:
    """
    Verify transactional MesEngine guarantees inside rollback isolation:
    1) split_lot updates lot.qty and applies apply_underdelivery_delta to SamplingNodeState.
    2) Early DEFECT on a node WITHOUT sampling_rule (BAT-VIS) decrements SamplingNodeState.n_rem and triggers DEFECT_RATE_EXCEEDED.
    3) REWORK passes on the same child RouteSheet do NOT double-increment lot.processed_count.
    """
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        try:
            async with AsyncSession(bind=conn, expire_on_commit=False) as session:
                tm_bat = await session.scalar(
                    select(TechMap).where(TechMap.item_code == "TM-BAT-6S")
                )
                assert tm_bat is not None
                nodes = list(
                    (
                        await session.scalars(
                            select(TechMapNode).where(TechMapNode.tech_map_id == tm_bat.id)
                        )
                    ).all()
                )
                node_by_step = {n.step_order: n for n in nodes}
                bat_split = node_by_step[2]
                bat_vis = node_by_step[3]
                bat_tst = node_by_step[5]

                op_user = await session.scalar(select(User).where(User.employee_code == "OP-01"))
                assert op_user is not None

                g_vis = await session.scalar(
                    select(TechMapNodeGuard).where(TechMapNodeGuard.node_id == bat_vis.id)
                )
                g_chg = await session.scalar(
                    select(TechMapNodeGuard).where(TechMapNodeGuard.node_id == node_by_step[4].id)
                )
                assert g_vis is not None and g_chg is not None

                lot_code = f"LOT-ENG-{uuid.uuid4().hex[:6].upper()}"
                lot = RouteSheet(
                    tech_map_id=tm_bat.id,
                    tracking_code=lot_code,
                    qty=10,
                    planned_qty=10,
                    current_node_id=bat_split.id,
                    status=RouteSheetStatus.IN_PROGRESS.value,
                )
                session.add(lot)
                await session.flush()

                session.add(
                    SamplingNodeState(
                        lot_rs_id=lot.id,
                        node_id=bat_tst.id,
                        m_rem=4,
                        n_rem=10,
                        w_rem=2,
                        taken_in_window=0,
                        escalations_used=0,
                    )
                )
                await session.flush()

                engine = MesEngine(SamplingService("test-secret"))
                child_codes = [
                    f"{lot_code}-{i:02d}" for i in range(1, 9)
                ]  # 8 delivered vs 10 planned
                children = await engine.split_lot(
                    session,
                    lot_rs_id=lot.id,
                    child_tracking_codes=child_codes,
                    next_node_id=bat_vis.id,
                )
                assert lot.qty == 8
                assert lot.status == RouteSheetStatus.SPLIT_CLOSED.value
                samp_row = await session.scalar(
                    select(SamplingNodeState).where(
                        SamplingNodeState.lot_rs_id == lot.id,
                        SamplingNodeState.node_id == bat_tst.id,
                    )
                )
                assert samp_row is not None
                assert samp_row.n_rem == 8

                ts = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
                # Child 0 fails on BAT-VIS (no sampling rule!) -> processed=1, defect=1, n_rem=7
                await engine.execute_step(
                    session,
                    route_sheet_id=children[0].id,
                    operator_id=op_user.id,
                    guard_inputs=[GuardInput(guard_id=g_vis.id, boolean_value=False)],
                    prefer_rework_on_fail=False,
                    completed_at=ts,
                )
                assert lot.processed_count == 1
                assert lot.defect_count == 1
                assert samp_row.n_rem == 7

                # Child 1 passes BAT-VIS -> fails BAT-CHG (REWORK -> BAT-RWK) -> passes BAT-RWK:
                # 3 executions on SAME child RouteSheet -> processed_count must increase only once (to 2)!
                await engine.execute_step(
                    session,
                    route_sheet_id=children[1].id,
                    operator_id=op_user.id,
                    guard_inputs=[GuardInput(guard_id=g_vis.id, boolean_value=True)],
                    completed_at=ts,
                )
                await engine.execute_step(
                    session,
                    route_sheet_id=children[1].id,
                    operator_id=op_user.id,
                    guard_inputs=[GuardInput(guard_id=g_chg.id, numeric_value=23.9)],
                    prefer_rework_on_fail=True,
                    completed_at=ts,
                )
                await engine.execute_step(
                    session,
                    route_sheet_id=children[1].id,
                    operator_id=op_user.id,
                    guard_inputs=[],
                    completed_at=ts,
                )
                exec_cnt = await session.scalar(
                    select(func.count(StepExecution.id)).where(
                        StepExecution.route_sheet_id == children[1].id
                    )
                )
                assert exec_cnt == 3
                assert lot.processed_count == 2
                assert lot.defect_count == 1

                # Children 2..6 fail on BAT-VIS -> reach 6 defects out of 7 processed (>= 3 + 0.35*7 = 5.45) -> HOLD!
                for ch in children[2:7]:
                    await engine.execute_step(
                        session,
                        route_sheet_id=ch.id,
                        operator_id=op_user.id,
                        guard_inputs=[GuardInput(guard_id=g_vis.id, boolean_value=False)],
                        prefer_rework_on_fail=False,
                        completed_at=ts,
                    )
                assert lot.defect_count == 6
                assert lot.processed_count == 7
                assert lot.hold_reason == "DEFECT_RATE_EXCEEDED"
        finally:
            await trans.rollback()
