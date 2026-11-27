from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import ActivatableMixin, Base, CreatedMixin, DatedMixin, PkUUIDMixin
from src.db.enums import (
    GuardValidationType,
    InstrumentType,
    NodeExecutionScope,
    RecallNoticeStatus,
    RecallNoticeType,
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


def _enum_values(enum_cls: type[str]) -> str:
    return ", ".join(f"'{item}'" for item in enum_cls)  # type: ignore[attr-defined]


# =============================================================================
# Domain 1 (DB-01): Roles and Users / Operators
# =============================================================================


class Role(Base, PkUUIDMixin, DatedMixin):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(sa.String(100), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    is_system: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )

    users: Mapped[list["User"]] = relationship(back_populates="role")


class User(Base, PkUUIDMixin, DatedMixin, ActivatableMixin):
    __tablename__ = "users"

    role_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False
    )
    employee_code: Mapped[str] = mapped_column(sa.String(50), nullable=False, unique=True)
    last_name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    first_name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    middle_name: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    email: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True)

    role: Mapped["Role"] = relationship(back_populates="users")

    __table_args__ = (
        sa.Index("idx_users_role_id", "role_id"),
        sa.Index("idx_users_names", "last_name", "first_name"),
    )

    @property
    def short_name(self) -> str:
        initials = f"{self.first_name[0]}." if self.first_name else ""
        if self.middle_name:
            initials += f"{self.middle_name[0]}."
        return f"{self.last_name} {initials}".strip()


# =============================================================================
# Domain 3 (DB-03 + Calibration + ALG-001 Static Extensions)
# =============================================================================


class Workstation(Base, PkUUIDMixin, CreatedMixin, ActivatableMixin):
    __tablename__ = "workstations"

    code: Mapped[str] = mapped_column(sa.String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class Instrument(Base, PkUUIDMixin, CreatedMixin, ActivatableMixin):
    __tablename__ = "instruments"

    code: Mapped[str] = mapped_column(sa.String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(sa.String(150), nullable=False)
    instrument_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    workstation_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("workstations.id", ondelete="SET NULL"), nullable=True
    )
    channel_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=1, server_default=sa.text("1")
    )
    last_calibrated_at: Mapped[datetime | None] = mapped_column(nullable=True)
    calibration_due_at: Mapped[datetime] = mapped_column(nullable=False)

    workstation: Mapped["Workstation | None"] = relationship()

    __table_args__ = (
        sa.CheckConstraint(
            f"instrument_type IN ({_enum_values(InstrumentType)})",
            name="instrument_type",
        ),
        sa.CheckConstraint("channel_count >= 1", name="instrument_channels"),
    )


class Operation(Base, PkUUIDMixin, CreatedMixin, ActivatableMixin):
    __tablename__ = "operations"

    code: Mapped[str] = mapped_column(sa.String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class TechMap(Base, PkUUIDMixin, DatedMixin):
    __tablename__ = "tech_maps"

    item_code: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    version: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=1, server_default=sa.text("1")
    )
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    base_cycle_time_seconds: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    is_approved: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    approved_by_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    activated_by_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    activated_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_by_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    # ALG-001 Linear Stop-Crane parameters (d_min and R_slope)
    stop_min_defects: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=3, server_default=sa.text("3")
    )
    stop_defect_slope: Mapped[Decimal] = mapped_column(
        sa.Numeric(6, 4),
        nullable=False,
        default=Decimal("0.0800"),
        server_default=sa.text("0.0800"),
    )

    nodes: Mapped[list["TechMapNode"]] = relationship(
        back_populates="tech_map", cascade="all, delete-orphan"
    )
    transitions: Mapped[list["TechMapTransition"]] = relationship(
        back_populates="tech_map", cascade="all, delete-orphan"
    )

    __table_args__ = (
        sa.UniqueConstraint("item_code", "version", name="uq_tech_maps_code_version"),
        sa.Index(
            "uq_tech_maps_single_active",
            "item_code",
            unique=True,
            postgresql_where=sa.text("is_active = true"),
        ),
        sa.Index("idx_tech_maps_status", "is_active", "is_approved"),
        sa.CheckConstraint(
            "(NOT is_active OR is_approved) AND "
            "(NOT is_approved OR (approved_by_id IS NOT NULL AND approved_at IS NOT NULL)) AND "
            "(NOT is_active OR (activated_by_id IS NOT NULL AND activated_at IS NOT NULL))",
            name="two_keys",
        ),
        sa.CheckConstraint(
            "approved_by_id IS NULL OR activated_by_id IS NULL OR approved_by_id != activated_by_id",
            name="separation_of_duties",
        ),
        sa.CheckConstraint("stop_min_defects >= 1", name="stop_min_defects"),
        sa.CheckConstraint(
            "stop_defect_slope >= 0 AND stop_defect_slope <= 1", name="stop_defect_slope"
        ),
    )


class TechMapNode(Base, PkUUIDMixin, CreatedMixin):
    __tablename__ = "tech_map_nodes"

    tech_map_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_maps.id", ondelete="CASCADE"), nullable=False
    )
    operation_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("operations.id", ondelete="RESTRICT"), nullable=False
    )
    workstation_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("workstations.id", ondelete="RESTRICT"), nullable=False
    )
    step_order: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=1, server_default=sa.text("1")
    )
    cycle_time_seconds: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    is_start_node: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    is_end_node: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    # Domain 5 extension: execution_scope lives on the node ('LOT' | 'ITEM')
    execution_scope: Mapped[str] = mapped_column(
        sa.String(20),
        nullable=False,
        default=NodeExecutionScope.ITEM.value,
        server_default=sa.text("'ITEM'"),
    )

    tech_map: Mapped["TechMap"] = relationship(back_populates="nodes")
    operation: Mapped["Operation"] = relationship()
    workstation: Mapped["Workstation"] = relationship()
    guards: Mapped[list["TechMapNodeGuard"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )
    sampling_rules: Mapped[list["TechMapSamplingRule"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )

    __table_args__ = (
        sa.UniqueConstraint("tech_map_id", "id", name="uq_tech_map_nodes_map_node"),
        sa.UniqueConstraint(
            "tech_map_id",
            "step_order",
            "workstation_id",
            name="uq_tech_map_nodes_step_ws",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.Index(
            "uq_tech_map_start_node",
            "tech_map_id",
            unique=True,
            postgresql_where=sa.text("is_start_node = true"),
        ),
        sa.Index("idx_tech_map_nodes_map", "tech_map_id", "step_order"),
        sa.CheckConstraint("NOT (is_start_node AND is_end_node)", name="node_start_end"),
        sa.CheckConstraint(
            f"execution_scope IN ({_enum_values(NodeExecutionScope)})",
            name="node_execution_scope",
        ),
    )


class TechMapNodeGuard(Base, PkUUIDMixin):
    __tablename__ = "tech_map_node_guards"

    node_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_nodes.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    guard_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    is_mandatory: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )
    min_value: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4), nullable=True)
    max_value: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4), nullable=True)
    nominal_value: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4), nullable=True)
    unit: Mapped[str | None] = mapped_column(sa.String(30), nullable=True)
    expected_boolean: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    validation_pattern: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    error_message: Mapped[str] = mapped_column(sa.String(255), nullable=False)

    node: Mapped["TechMapNode"] = relationship(back_populates="guards")

    __table_args__ = (
        sa.UniqueConstraint("node_id", "id", name="uq_tech_map_node_guards_node_guard"),
        sa.Index("idx_tech_map_node_guards_node", "node_id"),
        sa.CheckConstraint(
            f"guard_type IN ({_enum_values(GuardValidationType)})", name="guard_type"
        ),
        sa.CheckConstraint(
            "(guard_type = 'NUMERIC_RANGE' AND min_value IS NOT NULL AND max_value IS NOT NULL AND unit IS NOT NULL) OR "
            "(guard_type = 'BOOLEAN' AND expected_boolean IS NOT NULL) OR "
            "(guard_type = 'BARCODE_REGEX' AND validation_pattern IS NOT NULL)",
            name="guard_fields",
        ),
        sa.CheckConstraint(
            "guard_type != 'NUMERIC_RANGE' OR "
            "(min_value <= max_value AND (nominal_value IS NULL OR (nominal_value >= min_value AND nominal_value <= max_value)))",
            name="guard_numeric_bounds",
        ),
    )


class TechMapSamplingRule(Base, PkUUIDMixin, CreatedMixin):
    __tablename__ = "tech_map_sampling_rules"

    node_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_nodes.id", ondelete="CASCADE"), nullable=False
    )
    measurement_guard_id: Mapped[UUID] = mapped_column(sa.Uuid, nullable=False)
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    resolution_scope: Mapped[str] = mapped_column(sa.String(30), nullable=False)
    sample_size: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    sample_aggregation: Mapped[str] = mapped_column(sa.String(30), nullable=False)
    sample_threshold: Mapped[Decimal | None] = mapped_column(sa.Numeric(5, 4), nullable=True)
    non_sampled_policy: Mapped[str | None] = mapped_column(sa.String(30), nullable=True)

    # ALG-001 escalation fields
    escalation_sample_size: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )
    max_escalations: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )

    node: Mapped["TechMapNode"] = relationship(back_populates="sampling_rules")

    __table_args__ = (
        sa.UniqueConstraint("node_id", "name", name="uq_tech_map_sampling_node_name"),
        sa.ForeignKeyConstraint(
            ["node_id", "measurement_guard_id"],
            ["tech_map_node_guards.node_id", "tech_map_node_guards.id"],
            name="fk_sampling_guard_composite",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            f"resolution_scope IN ({_enum_values(SamplingResolutionScope)})",
            name="sampling_resolution_scope",
        ),
        sa.CheckConstraint(
            f"sample_aggregation IN ({_enum_values(SamplingAggregationType)})",
            name="sampling_aggregation",
        ),
        sa.CheckConstraint("sample_size > 0", name="sampling_sample_size"),
        sa.CheckConstraint("escalation_sample_size >= 0", name="sampling_escalation_size"),
        sa.CheckConstraint("max_escalations >= 0", name="sampling_max_escalations"),
        sa.CheckConstraint(
            f"non_sampled_policy IS NULL OR non_sampled_policy IN ({_enum_values(SamplingNonSampledPolicy)})",
            name="sampling_non_sampled_policy",
        ),
        sa.CheckConstraint(
            "(resolution_scope = 'LOT' AND sample_aggregation IN ('ALL_PASS', 'ANY_FAIL') AND sample_threshold IS NULL AND non_sampled_policy IS NULL) OR "
            "(resolution_scope = 'LOT' AND sample_aggregation = 'FRACTION_FAIL_LE' AND sample_threshold IS NOT NULL AND sample_threshold >= 0 AND sample_threshold <= 1 AND non_sampled_policy IS NULL) OR "
            "(resolution_scope = 'ITEM' AND sample_aggregation = 'PER_ITEM' AND sample_threshold IS NULL AND non_sampled_policy IS NOT NULL)",
            name="sampling_scope",
        ),
    )


class TechMapTransition(Base, PkUUIDMixin):
    __tablename__ = "tech_map_transitions"

    tech_map_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_maps.id", ondelete="CASCADE"), nullable=False
    )
    from_node_id: Mapped[UUID] = mapped_column(sa.Uuid, nullable=False)
    to_node_id: Mapped[UUID] = mapped_column(sa.Uuid, nullable=False)
    transition_type: Mapped[str] = mapped_column(
        sa.String(50),
        nullable=False,
        default=TransitionType.NORMAL.value,
        server_default=sa.text("'NORMAL'"),
    )
    condition_label: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    sample_size: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

    tech_map: Mapped["TechMap"] = relationship(back_populates="transitions")

    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["tech_map_id", "from_node_id"],
            ["tech_map_nodes.tech_map_id", "tech_map_nodes.id"],
            name="fk_trans_from_composite",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tech_map_id", "to_node_id"],
            ["tech_map_nodes.tech_map_id", "tech_map_nodes.id"],
            name="fk_trans_to_composite",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "from_node_id",
            "to_node_id",
            "transition_type",
            "condition_label",
            name="uq_tech_map_transitions_dedup",
            postgresql_nulls_not_distinct=True,
        ),
        sa.Index("idx_tech_map_transitions_map", "tech_map_id"),
        sa.CheckConstraint(
            f"transition_type IN ({_enum_values(TransitionType)})",
            name="transition_type",
        ),
        sa.CheckConstraint(
            f"condition_label IS NULL OR condition_label IN ({_enum_values(TransitionConditionLabel)})",
            name="transition_condition_label",
        ),
        sa.CheckConstraint("from_node_id != to_node_id", name="transition_no_self_loop"),
        sa.CheckConstraint(
            "(condition_label = 'IN_SAMPLE' AND sample_size IS NOT NULL AND sample_size > 0) OR "
            "(COALESCE(condition_label, '') != 'IN_SAMPLE' AND sample_size IS NULL)",
            name="transition_sample_size",
        ),
    )


# =============================================================================
# Domain 5 (DB-05): Execution Core, Genealogy, ALG-001 State, Support & Recall
# =============================================================================


class RouteSheet(Base, PkUUIDMixin, DatedMixin):
    __tablename__ = "route_sheets"

    tech_map_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_maps.id", ondelete="RESTRICT"), nullable=False
    )
    tracking_code: Mapped[str] = mapped_column(sa.String(100), nullable=False, unique=True)
    qty: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=1, server_default=sa.text("1")
    )
    planned_qty: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=1, server_default=sa.text("1")
    )
    current_node_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_nodes.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        sa.String(30),
        nullable=False,
        default=RouteSheetStatus.IN_PROGRESS.value,
        server_default=sa.text("'IN_PROGRESS'"),
    )
    hold_reason: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    root_lot_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=True
    )
    defect_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )
    processed_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )

    tech_map: Mapped["TechMap"] = relationship()
    current_node: Mapped["TechMapNode | None"] = relationship()
    lot_meta: Mapped["LotMetadata | None"] = relationship(
        back_populates="route_sheet", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        sa.Index("idx_route_sheets_root_lot_id", "root_lot_id"),
        sa.Index("idx_route_sheets_status", "status"),
        sa.Index("idx_route_sheets_tech_map", "tech_map_id"),
        sa.CheckConstraint(
            f"status IN ({_enum_values(RouteSheetStatus)})",
            name="route_sheet_status",
        ),
        sa.CheckConstraint("qty >= 0 AND planned_qty > 0", name="route_sheet_qty"),
        sa.CheckConstraint(
            "defect_count >= 0 AND processed_count >= 0", name="route_sheet_counters"
        ),
    )


class LotMetadata(Base):
    __tablename__ = "lot_metadata"

    route_sheet_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="CASCADE"), primary_key=True
    )
    supplier_name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    invoice_number: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    arrival_date: Mapped[datetime] = mapped_column(nullable=False)
    component_specs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sa.text("'{}'::jsonb")
    )

    route_sheet: Mapped["RouteSheet"] = relationship(back_populates="lot_meta")


class RouteSheetLink(Base, PkUUIDMixin, CreatedMixin):
    __tablename__ = "route_sheet_links"

    source_rs_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=False
    )
    target_rs_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=False
    )
    link_type: Mapped[str] = mapped_column(sa.String(30), nullable=False)

    __table_args__ = (
        sa.UniqueConstraint(
            "source_rs_id", "target_rs_id", "link_type", name="uq_route_sheet_links_edge"
        ),
        sa.Index("idx_route_sheet_links_source_type", "source_rs_id", "link_type"),
        sa.Index("idx_route_sheet_links_target_type", "target_rs_id", "link_type"),
        sa.CheckConstraint("source_rs_id != target_rs_id", name="link_no_self"),
        sa.CheckConstraint(
            f"link_type IN ({_enum_values(RouteSheetLinkType)})",
            name="link_type",
        ),
    )


class StepExecution(Base, PkUUIDMixin):
    __tablename__ = "step_executions"

    route_sheet_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=False
    )
    node_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_nodes.id", ondelete="RESTRICT"), nullable=False
    )
    operator_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    instrument_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("instruments.id", ondelete="RESTRICT"), nullable=True
    )
    channel_no: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    group_batch_code: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    outcome: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    applied_transition_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_transitions.id", ondelete="RESTRICT"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    completed_at: Mapped[datetime] = mapped_column(nullable=False)

    route_sheet: Mapped["RouteSheet"] = relationship()
    node: Mapped["TechMapNode"] = relationship()
    operator: Mapped["User"] = relationship()
    instrument: Mapped["Instrument | None"] = relationship()
    measurements: Mapped[list["StepMeasurement"]] = relationship(
        back_populates="execution", cascade="all, delete-orphan"
    )

    __table_args__ = (
        sa.Index("idx_step_executions_rs_completed", "route_sheet_id", "completed_at"),
        sa.Index(
            "idx_step_executions_instr_channel", "instrument_id", "channel_no", "completed_at"
        ),
        sa.Index("idx_step_executions_operator_time", "operator_id", "completed_at"),
        sa.CheckConstraint("channel_no IS NULL OR channel_no >= 1", name="execution_channel_no"),
        sa.CheckConstraint(
            f"outcome IN ({_enum_values(StepOutcome)})",
            name="execution_outcome",
        ),
    )


class StepMeasurement(Base):
    __tablename__ = "step_measurements"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    execution_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("step_executions.id", ondelete="CASCADE"), nullable=False
    )
    guard_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_node_guards.id", ondelete="RESTRICT"), nullable=False
    )
    numeric_value: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4), nullable=True)
    boolean_value: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    text_value: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    is_passed: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    measured_at: Mapped[datetime] = mapped_column(
        server_default=sa.text("clock_timestamp()"), nullable=False
    )

    execution: Mapped["StepExecution"] = relationship(back_populates="measurements")
    guard: Mapped["TechMapNodeGuard"] = relationship()

    __table_args__ = (
        sa.Index("idx_step_measurements_execution", "execution_id"),
        sa.Index("idx_step_measurements_guard_time", "guard_id", "measured_at"),
    )


class SamplingNodeState(Base):
    __tablename__ = "sampling_node_states"

    lot_rs_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="CASCADE"), primary_key=True
    )
    node_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("tech_map_nodes.id", ondelete="CASCADE"), primary_key=True
    )
    m_rem: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_rem: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    w_rem: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    taken_in_window: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )
    escalations_used: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sa.text("clock_timestamp()"),
        onupdate=sa.text("clock_timestamp()"),
        nullable=False,
    )

    __table_args__ = (
        sa.CheckConstraint(
            "m_rem >= 0 AND n_rem >= 0 AND w_rem >= 0 AND taken_in_window >= 0 AND escalations_used >= 0",
            name="sampling_state_non_negative",
        ),
    )


class SupportTicket(Base, PkUUIDMixin, CreatedMixin):
    __tablename__ = "support_tickets"

    route_sheet_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=False
    )
    tracking_code: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    title: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    symptom_category: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    root_cause_summary: Mapped[str] = mapped_column(sa.Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[str] = mapped_column(
        sa.String(30),
        nullable=False,
        default=SupportTicketStatus.OPEN.value,
        server_default=sa.text("'OPEN'"),
    )
    priority: Mapped[str] = mapped_column(
        sa.String(20),
        nullable=False,
        default=SupportTicketPriority.MEDIUM.value,
        server_default=sa.text("'MEDIUM'"),
    )
    operator_comment: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    __table_args__ = (
        sa.Index("idx_support_tickets_rs", "route_sheet_id"),
        sa.Index("idx_support_tickets_tracking", "tracking_code"),
        sa.CheckConstraint(
            f"status IN ({_enum_values(SupportTicketStatus)})",
            name="support_ticket_status",
        ),
        sa.CheckConstraint(
            f"priority IN ({_enum_values(SupportTicketPriority)})",
            name="support_ticket_priority",
        ),
    )


class RecallNotice(Base, PkUUIDMixin, CreatedMixin):
    __tablename__ = "recall_notices"

    lot_rs_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("route_sheets.id", ondelete="RESTRICT"), nullable=True
    )
    instrument_id: Mapped[UUID | None] = mapped_column(
        sa.Uuid, sa.ForeignKey("instruments.id", ondelete="RESTRICT"), nullable=True
    )
    notice_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    affected_tracking_codes: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=sa.text("'[]'::jsonb")
    )
    affected_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0")
    )
    status: Mapped[str] = mapped_column(
        sa.String(30),
        nullable=False,
        default=RecallNoticeStatus.ACTIVE.value,
        server_default=sa.text("'ACTIVE'"),
    )
    approved_by: Mapped[str] = mapped_column(
        sa.String(100), nullable=False, default="operator", server_default=sa.text("'operator'")
    )
    thread_id: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)

    __table_args__ = (
        sa.Index("idx_recall_notices_lot", "lot_rs_id"),
        sa.Index("idx_recall_notices_instrument", "instrument_id"),
        sa.CheckConstraint(
            f"notice_type IN ({_enum_values(RecallNoticeType)})",
            name="recall_notice_type",
        ),
        sa.CheckConstraint(
            f"status IN ({_enum_values(RecallNoticeStatus)})",
            name="recall_notice_status",
        ),
        sa.CheckConstraint("affected_count >= 0", name="recall_affected_count"),
    )
