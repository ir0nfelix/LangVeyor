from enum import StrEnum


class GuardValidationType(StrEnum):
    NUMERIC_RANGE = "NUMERIC_RANGE"
    BOOLEAN = "BOOLEAN"
    BARCODE_REGEX = "BARCODE_REGEX"


class SamplingResolutionScope(StrEnum):
    LOT = "LOT"
    ITEM = "ITEM"


class SamplingAggregationType(StrEnum):
    ALL_PASS = "ALL_PASS"
    ANY_FAIL = "ANY_FAIL"
    FRACTION_FAIL_LE = "FRACTION_FAIL_LE"
    PER_ITEM = "PER_ITEM"


class SamplingNonSampledPolicy(StrEnum):
    PASS_BY_DEFAULT = "PASS_BY_DEFAULT"
    HOLD = "HOLD"
    MEASURE_ALL = "MEASURE_ALL"


class TransitionType(StrEnum):
    NORMAL = "NORMAL"
    REWORK = "REWORK"
    DEFECT = "DEFECT"


class TransitionConditionLabel(StrEnum):
    IN_SAMPLE = "IN_SAMPLE"
    NOT_IN_SAMPLE = "NOT_IN_SAMPLE"
    SAMPLING_PASS = "SAMPLING_PASS"
    SAMPLING_FAIL = "SAMPLING_FAIL"


class NodeExecutionScope(StrEnum):
    LOT = "LOT"
    ITEM = "ITEM"


class RouteSheetStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    SPLIT_OPEN = "SPLIT_OPEN"
    SPLIT_CLOSED = "SPLIT_CLOSED"
    COMPLETED = "COMPLETED"
    DEFECT = "DEFECT"
    QUARANTINE = "QUARANTINE"


class RouteSheetLinkType(StrEnum):
    SPLIT = "SPLIT"
    ASSEMBLED_INTO = "ASSEMBLED_INTO"


class StepOutcome(StrEnum):
    PASS = "PASS"
    REWORK = "REWORK"
    DEFECT = "DEFECT"
    IN_SAMPLE = "IN_SAMPLE"
    NOT_IN_SAMPLE = "NOT_IN_SAMPLE"
    SAMPLING_PASS = "SAMPLING_PASS"
    SAMPLING_FAIL = "SAMPLING_FAIL"


class InstrumentType(StrEnum):
    CHARGING_STAND = "CHARGING_STAND"
    IR_TESTER = "IR_TESTER"
    TORQUE_WRENCH = "TORQUE_WRENCH"
    MULTIMETER = "MULTIMETER"
    TEST_RIG = "TEST_RIG"


class SupportTicketStatus(StrEnum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class SupportTicketPriority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecallNoticeType(StrEnum):
    LOT_RECALL = "LOT_RECALL"
    PREVENTIVE_INSPECTION = "PREVENTIVE_INSPECTION"
    QUARANTINE_HOLD = "QUARANTINE_HOLD"
    CALIBRATION_HOLD = "CALIBRATION_HOLD"


class RecallNoticeStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
