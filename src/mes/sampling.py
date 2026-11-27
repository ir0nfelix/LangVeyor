from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import RouteSheet, SamplingNodeState, TechMap, TechMapSamplingRule


class TriadState(StrEnum):
    DEFICIT = "N_REM_LT_M_REM"
    EXACT = "N_REM_EQ_M_REM"
    SURPLUS = "N_REM_GT_M_REM"


@dataclass(frozen=True, slots=True)
class SamplingState:
    m_rem: int
    n_rem: int
    w_rem: int
    taken_in_window: int = 0
    escalations_used: int = 0


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    selected: bool
    probability: float
    uniform_draw: float
    triad_state: TriadState
    reason: str
    shortfall: int = 0
    trigger_hold_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EscalationResult:
    escalated: bool
    added_samples: int
    triad_state: TriadState
    shortfall: int
    reason: str
    trigger_hold_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StopCraneResult:
    triggered: bool
    defect_count: int
    processed_count: int
    threshold: float
    hold_reason: str | None = None


def compute_hmac_uniform(secret: str, lot_id: str, node_id: str, item_code: str) -> float:
    """Compute a deterministic uniform U[0, 1) draw using HMAC-SHA256."""
    msg = f"{lot_id}:{node_id}:{item_code}".encode()
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / float(1 << 64)


def evaluate_triad_state(n_rem: int, m_rem: int) -> TriadState:
    if n_rem < m_rem:
        return TriadState.DEFICIT
    if n_rem == m_rem:
        return TriadState.EXACT
    return TriadState.SURPLUS


def init_sampling_state(planned_qty: int, sample_size: int) -> SamplingState:
    """Initialize stratified Fan-Muller-Rezucha state for a lot at a sampling node."""
    n_rem = max(0, planned_qty)
    m_rem = max(0, sample_size)
    if n_rem == 0 or m_rem == 0:
        w_rem = 0
    elif n_rem <= m_rem:
        w_rem = n_rem
    else:
        w_rem = max(1, n_rem // m_rem)
    return SamplingState(
        m_rem=m_rem, n_rem=n_rem, w_rem=w_rem, taken_in_window=0, escalations_used=0
    )


def step_sampling(
    state: SamplingState, uniform_draw: float
) -> tuple[SamplingState, SamplingDecision]:
    """Advance ALG-001 stratified Fan-Muller-Rezucha selection for a single arriving item."""
    if not (0.0 <= uniform_draw < 1.0):
        raise ValueError(f"uniform_draw must be in [0.0, 1.0), got {uniform_draw}")

    triad = evaluate_triad_state(state.n_rem, state.m_rem)

    if state.n_rem <= 0:
        return state, SamplingDecision(
            selected=False,
            probability=0.0,
            uniform_draw=uniform_draw,
            triad_state=triad,
            reason="LOT_EXHAUSTED",
        )

    if state.m_rem <= 0:
        next_state = SamplingState(
            m_rem=0,
            n_rem=state.n_rem - 1,
            w_rem=max(0, state.w_rem - 1),
            taken_in_window=state.taken_in_window,
            escalations_used=state.escalations_used,
        )
        return next_state, SamplingDecision(
            selected=False,
            probability=0.0,
            uniform_draw=uniform_draw,
            triad_state=triad,
            reason="SAMPLE_QUOTA_FULFILLED",
        )

    # Triad branches 1 & 2: N_rem < m_rem or N_rem == m_rem -> mandatory 100% sampling
    if state.n_rem <= state.m_rem:
        shortfall = max(0, state.m_rem - state.n_rem)
        new_n = state.n_rem - 1
        new_m = state.m_rem - 1
        next_state = SamplingState(
            m_rem=new_m,
            n_rem=new_n,
            w_rem=new_n,
            taken_in_window=0,
            escalations_used=state.escalations_used,
        )
        return next_state, SamplingDecision(
            selected=True,
            probability=1.0,
            uniform_draw=uniform_draw,
            triad_state=triad,
            reason="MANDATORY_100PCT_SHORTFALL"
            if triad == TriadState.DEFICIT
            else "MANDATORY_EXACT_MATCH",
            shortfall=shortfall,
            trigger_hold_reason="HOLD_INSUFFICIENT_SAMPLE" if triad == TriadState.DEFICIT else None,
        )

    # Triad branch 3: N_rem > m_rem -> Stratified Fan-Muller-Rezucha within current window
    cur_w = state.w_rem if state.w_rem > 0 else max(1, state.n_rem // state.m_rem)
    cur_taken = 0 if state.w_rem <= 0 else state.taken_in_window

    if cur_taken >= 1:
        prob = 0.0
        selected = False
        reason = "STRATUM_ALREADY_SAMPLED"
    else:
        prob = 1.0 / float(cur_w)
        selected = uniform_draw < prob
        reason = "FMR_STRATUM_SELECTED" if selected else "FMR_STRATUM_SKIPPED"

    new_n = state.n_rem - 1
    new_m = state.m_rem - 1 if selected else state.m_rem
    rem_w = cur_w - 1
    new_taken = cur_taken + (1 if selected else 0)

    if new_m <= 0:
        next_w = 0
        next_taken = 0
    elif rem_w <= 0:
        next_w = new_n if new_n <= new_m else max(1, new_n // new_m)
        next_taken = 0
    else:
        next_w = rem_w
        next_taken = new_taken

    next_state = SamplingState(
        m_rem=new_m,
        n_rem=new_n,
        w_rem=next_w,
        taken_in_window=next_taken,
        escalations_used=state.escalations_used,
    )
    return next_state, SamplingDecision(
        selected=selected,
        probability=prob,
        uniform_draw=uniform_draw,
        triad_state=triad,
        reason=reason,
    )


def apply_escalation(
    state: SamplingState,
    escalation_sample_size: int,
    max_escalations: int,
) -> tuple[SamplingState, EscalationResult]:
    """Increase remaining sample target upon defect detection up to max_escalations."""
    if escalation_sample_size <= 0 or state.escalations_used >= max_escalations:
        triad = evaluate_triad_state(state.n_rem, state.m_rem)
        return state, EscalationResult(
            escalated=False,
            added_samples=0,
            triad_state=triad,
            shortfall=max(0, state.m_rem - state.n_rem),
            reason="MAX_ESCALATIONS_REACHED"
            if state.escalations_used >= max_escalations
            else "ESCALATION_DISABLED",
        )

    new_m = state.m_rem + escalation_sample_size
    new_escalations = state.escalations_used + 1
    triad = evaluate_triad_state(state.n_rem, new_m)
    shortfall = max(0, new_m - state.n_rem)

    if state.n_rem <= new_m:
        new_w = state.n_rem
        new_taken = 0
    else:
        new_w = max(1, state.n_rem // new_m)
        new_taken = 0

    next_state = SamplingState(
        m_rem=new_m,
        n_rem=state.n_rem,
        w_rem=new_w,
        taken_in_window=new_taken,
        escalations_used=new_escalations,
    )
    return next_state, EscalationResult(
        escalated=True,
        added_samples=escalation_sample_size,
        triad_state=triad,
        shortfall=shortfall,
        reason=f"ESCALATED_{triad.value}",
        trigger_hold_reason="HOLD_INSUFFICIENT_SAMPLE" if triad == TriadState.DEFICIT else None,
    )


def apply_underdelivery_delta(state: SamplingState, delta: int) -> SamplingState:
    """Adjust remaining lot size n_rem and stratified window when physical delivery < planned_qty."""
    if delta <= 0:
        return state

    new_n = max(0, state.n_rem - delta)
    if state.m_rem <= 0 or new_n == 0:
        return SamplingState(
            m_rem=state.m_rem,
            n_rem=new_n,
            w_rem=0,
            taken_in_window=0,
            escalations_used=state.escalations_used,
        )

    if new_n <= state.m_rem:
        return SamplingState(
            m_rem=state.m_rem,
            n_rem=new_n,
            w_rem=new_n,
            taken_in_window=0,
            escalations_used=state.escalations_used,
        )

    # new_n > m_rem: preserve feasibility (after current window, at least m_rem items must remain)
    if state.taken_in_window >= 1 and state.w_rem > 0:
        max_skip = new_n - state.m_rem
        new_w = min(state.w_rem, max_skip)
        if new_w <= 0:
            new_w = max(1, new_n // state.m_rem)
            new_taken = 0
        else:
            new_taken = state.taken_in_window
    else:
        new_w = max(1, min(state.w_rem or new_n, new_n // state.m_rem))
        new_taken = 0

    return SamplingState(
        m_rem=state.m_rem,
        n_rem=new_n,
        w_rem=new_w,
        taken_in_window=new_taken,
        escalations_used=state.escalations_used,
    )


def check_stop_crane(
    defect_count: int,
    processed_count: int,
    stop_min_defects: int,
    stop_defect_slope: float | Decimal,
) -> StopCraneResult:
    """Evaluate linear stop-crane threshold: d_min + R_slope * processed_count."""
    threshold = float(stop_min_defects) + float(stop_defect_slope) * float(processed_count)
    triggered = defect_count > 0 and float(defect_count) >= threshold
    return StopCraneResult(
        triggered=triggered,
        defect_count=defect_count,
        processed_count=processed_count,
        threshold=round(threshold, 4),
        hold_reason="DEFECT_RATE_EXCEEDED" if triggered else None,
    )


class SamplingService:
    """Transactional database adapter for ALG-001 with SELECT ... FOR UPDATE locking."""

    def __init__(self, hmac_secret: str) -> None:
        self._secret = hmac_secret

    async def get_or_init_state(
        self,
        session: AsyncSession,
        lot_rs_id: UUID,
        node_id: UUID,
        for_update: bool = True,
    ) -> SamplingNodeState:
        stmt = select(SamplingNodeState).where(
            SamplingNodeState.lot_rs_id == lot_rs_id,
            SamplingNodeState.node_id == node_id,
        )
        if for_update:
            stmt = stmt.with_for_update()

        row = await session.scalar(stmt)
        if row is not None:
            return row

        lot = await session.scalar(select(RouteSheet).where(RouteSheet.id == lot_rs_id))
        if lot is None:
            raise ValueError(f"Lot route sheet {lot_rs_id} not found")

        rule = await session.scalar(
            select(TechMapSamplingRule).where(TechMapSamplingRule.node_id == node_id)
        )
        sample_size = rule.sample_size if rule is not None else 1
        init = init_sampling_state(planned_qty=lot.planned_qty, sample_size=sample_size)
        underdelivery = max(0, lot.planned_qty - lot.qty) if lot.status == "SPLIT_CLOSED" else 0
        if underdelivery > 0:
            init = apply_underdelivery_delta(init, underdelivery)
        if lot.defect_count > 0:
            adjusted_n = max(0, init.n_rem - lot.defect_count)
            init = SamplingState(
                m_rem=init.m_rem,
                n_rem=adjusted_n,
                w_rem=adjusted_n
                if (init.m_rem > 0 and adjusted_n <= init.m_rem)
                else min(init.w_rem, adjusted_n),
                taken_in_window=init.taken_in_window,
                escalations_used=init.escalations_used,
            )

        row = SamplingNodeState(
            lot_rs_id=lot_rs_id,
            node_id=node_id,
            m_rem=init.m_rem,
            n_rem=init.n_rem,
            w_rem=init.w_rem,
            taken_in_window=init.taken_in_window,
            escalations_used=init.escalations_used,
        )
        session.add(row)
        await session.flush()
        return row

    async def apply_underdelivery_to_lot(
        self,
        session: AsyncSession,
        lot_rs_id: UUID,
        delta: int,
    ) -> list[SamplingNodeState]:
        """Apply apply_underdelivery_delta to all active sampling states of a lot."""
        if delta <= 0:
            return []
        rows = list(
            (
                await session.scalars(
                    select(SamplingNodeState)
                    .where(SamplingNodeState.lot_rs_id == lot_rs_id)
                    .with_for_update()
                )
            ).all()
        )
        for row in rows:
            cur = SamplingState(
                m_rem=row.m_rem,
                n_rem=row.n_rem,
                w_rem=row.w_rem,
                taken_in_window=row.taken_in_window,
                escalations_used=row.escalations_used,
            )
            nxt = apply_underdelivery_delta(cur, delta)
            row.n_rem = nxt.n_rem
            row.w_rem = nxt.w_rem
            row.taken_in_window = nxt.taken_in_window
        await session.flush()
        return rows

    async def record_item_defect_for_sampling(
        self,
        session: AsyncSession,
        lot_rs_id: UUID,
    ) -> list[SamplingNodeState]:
        """Decrement remaining N_rem (and clamp W_rem) on active lot sampling states when an item defects."""
        rows = list(
            (
                await session.scalars(
                    select(SamplingNodeState)
                    .where(
                        SamplingNodeState.lot_rs_id == lot_rs_id,
                        SamplingNodeState.n_rem > 0,
                    )
                    .with_for_update()
                )
            ).all()
        )
        for row in rows:
            new_n = max(0, row.n_rem - 1)
            row.n_rem = new_n
            if row.m_rem > 0 and new_n <= row.m_rem:
                row.w_rem = new_n
            else:
                row.w_rem = min(row.w_rem, new_n)
        await session.flush()
        return rows

    async def decide_and_advance(
        self,
        session: AsyncSession,
        lot_rs_id: UUID,
        node_id: UUID,
        item_tracking_code: str,
    ) -> SamplingDecision:
        row = await self.get_or_init_state(session, lot_rs_id, node_id, for_update=True)
        state = SamplingState(
            m_rem=row.m_rem,
            n_rem=row.n_rem,
            w_rem=row.w_rem,
            taken_in_window=row.taken_in_window,
            escalations_used=row.escalations_used,
        )
        draw = compute_hmac_uniform(self._secret, str(lot_rs_id), str(node_id), item_tracking_code)
        next_state, decision = step_sampling(state, draw)
        row.m_rem = next_state.m_rem
        row.n_rem = next_state.n_rem
        row.w_rem = next_state.w_rem
        row.taken_in_window = next_state.taken_in_window
        row.escalations_used = next_state.escalations_used

        if decision.trigger_hold_reason is not None:
            lot = await session.scalar(
                select(RouteSheet).where(RouteSheet.id == lot_rs_id).with_for_update()
            )
            if lot is not None and lot.hold_reason is None:
                lot.hold_reason = decision.trigger_hold_reason

        await session.flush()
        return decision

    async def register_defect_and_check_hold(
        self,
        session: AsyncSession,
        lot_rs_id: UUID,
        node_id: UUID | None = None,
        *,
        is_defect: bool,
        increment_processed: bool = True,
    ) -> tuple[StopCraneResult, EscalationResult | None]:
        lot = await session.scalar(
            select(RouteSheet).where(RouteSheet.id == lot_rs_id).with_for_update()
        )
        if lot is None:
            raise ValueError(f"Lot route sheet {lot_rs_id} not found")

        tech_map = await session.scalar(select(TechMap).where(TechMap.id == lot.tech_map_id))
        if tech_map is None:
            raise ValueError(f"TechMap {lot.tech_map_id} not found")

        if increment_processed:
            lot.processed_count += 1
        if is_defect:
            lot.defect_count += 1

        stop_res = check_stop_crane(
            defect_count=lot.defect_count,
            processed_count=lot.processed_count,
            stop_min_defects=tech_map.stop_min_defects,
            stop_defect_slope=tech_map.stop_defect_slope,
        )
        if stop_res.triggered and lot.hold_reason is None:
            lot.hold_reason = stop_res.hold_reason

        esc_res: EscalationResult | None = None
        if is_defect and node_id is not None:
            rule = await session.scalar(
                select(TechMapSamplingRule).where(TechMapSamplingRule.node_id == node_id)
            )
            if rule is not None and rule.escalation_sample_size > 0:
                row = await self.get_or_init_state(session, lot_rs_id, node_id, for_update=True)
                cur = SamplingState(
                    m_rem=row.m_rem,
                    n_rem=row.n_rem,
                    w_rem=row.w_rem,
                    taken_in_window=row.taken_in_window,
                    escalations_used=row.escalations_used,
                )
                nxt, esc_res = apply_escalation(
                    cur, rule.escalation_sample_size, rule.max_escalations
                )
                row.m_rem = nxt.m_rem
                row.w_rem = nxt.w_rem
                row.taken_in_window = nxt.taken_in_window
                row.escalations_used = nxt.escalations_used

        await session.flush()
        return stop_res, esc_res
