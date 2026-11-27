from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.enums import (
    GuardValidationType,
    RouteSheetLinkType,
    RouteSheetStatus,
    StepOutcome,
    TransitionConditionLabel,
    TransitionType,
)
from src.db.models import (
    RouteSheet,
    RouteSheetLink,
    StepExecution,
    StepMeasurement,
    TechMap,
    TechMapNode,
    TechMapNodeGuard,
    TechMapSamplingRule,
    TechMapTransition,
)
from src.mes.sampling import SamplingService


@dataclass(frozen=True, slots=True)
class GuardInput:
    guard_id: UUID
    numeric_value: float | Decimal | None = None
    boolean_value: bool | None = None
    text_value: str | None = None


@dataclass(frozen=True, slots=True)
class GuardEvaluation:
    guard_id: UUID
    is_passed: bool
    is_sampling_guard: bool
    numeric_value: Decimal | None = None
    boolean_value: bool | None = None
    text_value: str | None = None


def evaluate_guards(
    guards: list[TechMapNodeGuard],
    inputs: dict[UUID, GuardInput],
    sampling_guard_ids: set[UUID] | None = None,
) -> list[GuardEvaluation]:
    """Validate guard inputs against node guard definitions."""
    sampling_ids = sampling_guard_ids or set()
    results: list[GuardEvaluation] = []
    for g in guards:
        inp = inputs.get(g.id)
        is_sampling = g.id in sampling_ids
        if inp is None:
            if g.is_mandatory and not is_sampling:
                results.append(
                    GuardEvaluation(guard_id=g.id, is_passed=False, is_sampling_guard=is_sampling)
                )
            continue

        passed = False
        num_val: Decimal | None = None
        if g.guard_type == GuardValidationType.NUMERIC_RANGE.value:
            if (
                inp.numeric_value is not None
                and g.min_value is not None
                and g.max_value is not None
            ):
                num_val = Decimal(str(inp.numeric_value))
                passed = g.min_value <= num_val <= g.max_value
        elif g.guard_type == GuardValidationType.BOOLEAN.value:
            if inp.boolean_value is not None and g.expected_boolean is not None:
                passed = inp.boolean_value == g.expected_boolean
        elif g.guard_type == GuardValidationType.BARCODE_REGEX.value:
            if inp.text_value is not None and g.validation_pattern is not None:
                passed = re.fullmatch(g.validation_pattern, inp.text_value) is not None

        results.append(
            GuardEvaluation(
                guard_id=g.id,
                is_passed=passed,
                is_sampling_guard=is_sampling,
                numeric_value=num_val,
                boolean_value=inp.boolean_value,
                text_value=inp.text_value,
            )
        )
    return results


class MesEngine:
    """State-machine engine executing route sheet steps, splits, assemblies, and ALG-001 checks."""

    def __init__(self, sampling_service: SamplingService) -> None:
        self._sampling = sampling_service

    async def execute_step(
        self,
        session: AsyncSession,
        *,
        route_sheet_id: UUID,
        operator_id: UUID,
        guard_inputs: list[GuardInput],
        instrument_id: UUID | None = None,
        channel_no: int | None = None,
        group_batch_code: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        prefer_rework_on_fail: bool = True,
    ) -> StepExecution:
        rs = await session.scalar(
            select(RouteSheet).where(RouteSheet.id == route_sheet_id).with_for_update()
        )
        if rs is None:
            raise ValueError(f"RouteSheet {route_sheet_id} not found")
        if rs.hold_reason is not None:
            raise RuntimeError(f"RouteSheet {rs.tracking_code} is on HOLD: {rs.hold_reason}")
        if rs.root_lot_id is not None:
            root_lot = await session.scalar(
                select(RouteSheet).where(RouteSheet.id == rs.root_lot_id)
            )
            if root_lot is not None and root_lot.hold_reason is not None:
                raise RuntimeError(
                    f"Root lot {root_lot.tracking_code} is on HOLD: {root_lot.hold_reason}"
                )
        if rs.current_node_id is None:
            raise ValueError(f"RouteSheet {rs.tracking_code} has no active current_node_id")

        tech_map = await session.scalar(select(TechMap).where(TechMap.id == rs.tech_map_id))
        if tech_map is None or not (tech_map.is_approved and tech_map.is_active):
            raise RuntimeError("Two-Key Authorization failed: TechMap must be approved and active")

        node = await session.scalar(select(TechMapNode).where(TechMapNode.id == rs.current_node_id))
        if node is None:
            raise ValueError(f"Node {rs.current_node_id} not found")

        guards = list(
            (
                await session.scalars(
                    select(TechMapNodeGuard).where(TechMapNodeGuard.node_id == node.id)
                )
            ).all()
        )
        sampling_rule = await session.scalar(
            select(TechMapSamplingRule).where(TechMapSamplingRule.node_id == node.id)
        )
        transitions = list(
            (
                await session.scalars(
                    select(TechMapTransition).where(TechMapTransition.from_node_id == node.id)
                )
            ).all()
        )

        sampling_ids = {sampling_rule.measurement_guard_id} if sampling_rule else set()
        input_map = {gi.guard_id: gi for gi in guard_inputs}
        evals = evaluate_guards(guards, input_map, sampling_ids)

        solid_failed = any((not ev.is_passed) and (not ev.is_sampling_guard) for ev in evals)
        sampling_failed = any((not ev.is_passed) and ev.is_sampling_guard for ev in evals)

        chosen_transition: TechMapTransition | None = None
        outcome = StepOutcome.PASS.value

        if solid_failed or sampling_failed:
            rework_tr = next(
                (t for t in transitions if t.transition_type == TransitionType.REWORK.value),
                None,
            )
            defect_tr = next(
                (
                    t
                    for t in transitions
                    if t.transition_type == TransitionType.DEFECT.value
                    and t.condition_label is None
                ),
                None,
            )
            if prefer_rework_on_fail and rework_tr is not None:
                chosen_transition = rework_tr
                outcome = StepOutcome.REWORK.value
            elif defect_tr is not None:
                chosen_transition = defect_tr
                outcome = StepOutcome.DEFECT.value
            else:
                outcome = StepOutcome.DEFECT.value
        else:
            in_sample_tr = next(
                (
                    t
                    for t in transitions
                    if t.condition_label == TransitionConditionLabel.IN_SAMPLE.value
                ),
                None,
            )
            not_in_sample_tr = next(
                (
                    t
                    for t in transitions
                    if t.condition_label == TransitionConditionLabel.NOT_IN_SAMPLE.value
                ),
                None,
            )
            if in_sample_tr is not None and not_in_sample_tr is not None:
                lot_id = rs.root_lot_id or rs.id
                decision = await self._sampling.decide_and_advance(
                    session, lot_id, node.id, rs.tracking_code
                )
                if decision.selected:
                    chosen_transition = in_sample_tr
                    outcome = StepOutcome.IN_SAMPLE.value
                else:
                    chosen_transition = not_in_sample_tr
                    outcome = StepOutcome.NOT_IN_SAMPLE.value
            else:
                chosen_transition = next(
                    (
                        t
                        for t in transitions
                        if t.transition_type == TransitionType.NORMAL.value
                        and t.condition_label is None
                    ),
                    None,
                )
                outcome = StepOutcome.PASS.value

        prior_exec_exists = False
        if rs.root_lot_id is not None:
            prior_exec_exists = (
                await session.scalar(
                    select(StepExecution.id).where(StepExecution.route_sheet_id == rs.id).limit(1)
                )
            ) is not None

        now = completed_at or datetime.now(UTC)
        start = started_at or now
        execution = StepExecution(
            route_sheet_id=rs.id,
            node_id=node.id,
            operator_id=operator_id,
            instrument_id=instrument_id,
            channel_no=channel_no,
            group_batch_code=group_batch_code,
            outcome=outcome,
            applied_transition_id=chosen_transition.id if chosen_transition else None,
            started_at=start,
            completed_at=now,
        )
        session.add(execution)
        await session.flush()

        for ev in evals:
            session.add(
                StepMeasurement(
                    execution_id=execution.id,
                    guard_id=ev.guard_id,
                    numeric_value=ev.numeric_value,
                    boolean_value=ev.boolean_value,
                    text_value=ev.text_value,
                    is_passed=ev.is_passed,
                    measured_at=now,
                )
            )

        if rs.root_lot_id is not None:
            is_defect = outcome == StepOutcome.DEFECT.value
            if is_defect:
                await self._sampling.record_item_defect_for_sampling(session, rs.root_lot_id)
            await self._sampling.register_defect_and_check_hold(
                session,
                rs.root_lot_id,
                node.id,
                is_defect=is_defect,
                increment_processed=not prior_exec_exists,
            )

        if outcome == StepOutcome.DEFECT.value and chosen_transition is None:
            rs.status = RouteSheetStatus.DEFECT.value
            rs.current_node_id = None
        elif chosen_transition is not None:
            rs.current_node_id = chosen_transition.to_node_id
            if chosen_transition.transition_type == TransitionType.DEFECT.value:
                rs.status = RouteSheetStatus.DEFECT.value
        elif node.is_end_node:
            rs.status = RouteSheetStatus.COMPLETED.value
            rs.current_node_id = None

        await session.flush()
        return execution

    async def split_lot(
        self,
        session: AsyncSession,
        *,
        lot_rs_id: UUID,
        child_tracking_codes: list[str],
        next_node_id: UUID,
    ) -> list[RouteSheet]:
        """Split a LOT route sheet into N individual ITEM route sheets with SPLIT links."""
        lot = await session.scalar(
            select(RouteSheet).where(RouteSheet.id == lot_rs_id).with_for_update()
        )
        if lot is None:
            raise ValueError(f"Lot {lot_rs_id} not found")

        children: list[RouteSheet] = []
        for code in child_tracking_codes:
            child = RouteSheet(
                tech_map_id=lot.tech_map_id,
                tracking_code=code,
                qty=1,
                planned_qty=1,
                current_node_id=next_node_id,
                status=RouteSheetStatus.IN_PROGRESS.value,
                root_lot_id=lot.id,
            )
            session.add(child)
            children.append(child)
        await session.flush()

        for child in children:
            session.add(
                RouteSheetLink(
                    source_rs_id=lot.id,
                    target_rs_id=child.id,
                    link_type=RouteSheetLinkType.SPLIT.value,
                )
            )
        actual_qty = len(child_tracking_codes)
        lot.qty = actual_qty
        lot.status = RouteSheetStatus.SPLIT_CLOSED.value
        lot.current_node_id = None
        if actual_qty < lot.planned_qty:
            await self._sampling.apply_underdelivery_to_lot(
                session, lot.id, lot.planned_qty - actual_qty
            )
        await session.flush()
        return children

    async def assemble_product(
        self,
        session: AsyncSession,
        *,
        component_rs_ids: list[UUID],
        target_rs_id: UUID,
    ) -> list[RouteSheetLink]:
        """Link N component route sheets into a final assembled product via ASSEMBLED_INTO."""
        links: list[RouteSheetLink] = []
        for comp_id in component_rs_ids:
            link = RouteSheetLink(
                source_rs_id=comp_id,
                target_rs_id=target_rs_id,
                link_type=RouteSheetLinkType.ASSEMBLED_INTO.value,
            )
            session.add(link)
            links.append(link)
        await session.flush()
        return links


def build_link_metadata(specs: dict[str, Any]) -> dict[str, Any]:
    return dict(specs)
