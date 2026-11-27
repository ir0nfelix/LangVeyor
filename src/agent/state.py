from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

SymptomCategory = Literal[
    "BATTERY_LOW_RANGE",
    "MOTOR_STALL_OR_NOISE",
    "NO_POWER_DEAD",
    "OTHER",
]


class IntakeResult(BaseModel):
    tracking_code: str | None = Field(
        default=None,
        description="Серийный номер машинки (например CAR-2026-0412) или None, если не указан.",
    )
    symptom_category: SymptomCategory = Field(
        default="OTHER",
        description="Категория симптома из BPMN.",
    )
    operating_conditions: str | None = Field(
        default=None,
        description="Внешние условия эксплуатации (температура, рельеф, нагрузка), если указаны.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True, если серийный номер не указан и требуется уточнение у оператора.",
    )
    clarification_message: str | None = Field(
        default=None,
        description="Вежливый уточняющий запрос оператору при отсутствии серийника.",
    )


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    source_file: str
    doc_title: str
    section_title: str
    content: str
    rrf_score: float
    dense_score: float
    bm25_score: float


class ToolEvidence(BaseModel):
    tool_name: str
    args_summary: str
    result_json: str
    latency_ms: float


class ProposedAction(BaseModel):
    action_type: Literal[
        "CREATE_SUPPORT_TICKET",
        "ESCALATE_REPEAT_TICKET",
        "CREATE_LOT_RECALL",
        "CREATE_CALIBRATION_HOLD",
    ]
    title: str
    description: str
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    target_tracking_code: str | None = None
    target_lot_code: str | None = None
    target_instrument_code: str | None = None
    affected_cars: list[str] = Field(default_factory=list)


class InvestigationDraft(BaseModel):
    root_cause_hypothesis: str
    root_cause_type: Literal[
        "LOT_DEFECT",
        "INSTRUMENT_CHANNEL_DRIFT",
        "OPERATOR_QUALITY_DROP",
        "REPEAT_FAILURE",
        "DB_CLEAN_EXTERNAL_SUSPECTED",
        "INCONCLUSIVE",
    ]
    db_anomalies_found: bool
    cited_db_facts: list[str] = Field(default_factory=list)
    affected_cars: list[str] = Field(default_factory=list)
    target_lot_code: str | None = None
    target_instrument_code: str | None = None
    target_channel_no: int | None = None
    is_repeat_complaint: bool = False
    recommended_actions: list[ProposedAction] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class AuditVerdict(BaseModel):
    decision: Literal["APPROVED", "REJECTED", "NEED_WEB_RESEARCH"]
    feedback: str
    hallucinated_claims: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    db_is_clean: bool = False


class ExternalHypothesis(BaseModel):
    title: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    sources: list[str] = Field(default_factory=list)
    mechanic_check_steps: list[str] = Field(default_factory=list)


class WebHypothesesOutput(BaseModel):
    hypotheses: list[ExternalHypothesis] = Field(default_factory=list)


class FinalVerdict(BaseModel):
    summary: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confirmed_db_facts: list[str] = Field(default_factory=list)
    external_hypotheses: list[ExternalHypothesis] = Field(default_factory=list)
    affected_cars: list[str] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)


class ActionResult(BaseModel):
    approved: bool
    operator_comment: str | None = None
    created_ticket_ids: list[str] = Field(default_factory=list)
    created_recall_ids: list[str] = Field(default_factory=list)
    message: str


class LLMUsageRecord(BaseModel):
    role: Literal["FAST", "INVESTIGATOR", "AUDITOR", "JUDGE"]
    node_name: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class ComponentBomSummary(BaseModel):
    motor: str | None = None
    bms: str | None = None
    chemistry: str | None = None
    battery_tracking_code: str | None = None
    root_lot_code: str | None = None


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    user_query: str
    intake: IntakeResult | None
    bpmn_guidance: str
    required_tools: list[str]
    rag_context: list[RetrievedChunkModel]
    evidence_log: Annotated[list[ToolEvidence], operator.add]
    tool_steps: int
    bom_summary: ComponentBomSummary | None
    investigator_draft: InvestigationDraft | None
    audit_result: AuditVerdict | None
    revision_count: int
    confirmed_db_facts: list[str]
    external_hypotheses: list[ExternalHypothesis]
    final_verdict: FinalVerdict | None
    action_result: ActionResult | None
    token_usage: Annotated[list[LLMUsageRecord], operator.add]
