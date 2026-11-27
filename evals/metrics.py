from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

NUMBER_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_.])\d+(?:\.\d+)?(?=$|[^A-Za-zА-Яа-яЁё0-9_.]|(?:V|A|Ah|mOhm|Nm|C|В|А|Ач|мОм|Нм|Н\*м|Н·м)(?![A-Za-zА-Яа-яЁё0-9_]))"
)
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
CODE_TOKEN_RE = re.compile(
    r"\b(?:CAR|LOT|BAT|BODY|STAND|OP|TESTER-IR|TORQUE|TEST-RIG)-\d+(?:-\d+)*\b",
    re.IGNORECASE,
)
HARDWARE_MODEL_RE = re.compile(
    r"\b(?:RS550-24V|JK-BMS-6S|TM-BAT-6S|TM-CAR-FINAL|TM-BODY|\d+[Ss])\b",
    re.IGNORECASE,
)

DEFAULT_IGNORED_NUMBERS: frozenset[str] = frozenset({"2026"})

# Approximate OpenRouter pricing ($ per 1M tokens) for cost estimation
ROLE_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash": (0.30, 2.50),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "mock/gemini-2.5-flash": (0.0, 0.0),
    "mock/deepseek-v3.2": (0.0, 0.0),
    "mock/gpt-4.1-mini": (0.0, 0.0),
    "mock/claude-haiku-4.5": (0.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class CaseEvalResult:
    case_id: str
    passed: bool
    root_cause_accuracy: float
    action_match_score: float
    recall_precision: float | None
    recall_recall: float | None
    recall_f1: float | None
    tool_trajectory_score: float
    web_routing_correct: bool
    expected_web_research: bool
    actual_web_research: bool
    auditor_caught: bool
    auditor_recovered: bool
    groundedness_score: float
    ungrounded_numbers: list[str]
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    details: dict[str, Any] = field(default_factory=dict)


def compute_recall_prf1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    pred_set = {x.strip().upper() for x in predicted if x.strip()}
    exp_set = {x.strip().upper() for x in expected if x.strip()}
    if not exp_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not exp_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & exp_set)
    prec = tp / len(pred_set)
    rec = tp / len(exp_set)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def compute_action_match_score(actual_actions: list[str], expected_actions: list[str]) -> float:
    """Evaluate exact set F1 between proposed action types and golden expected_action_types."""
    act_set = {x.strip().upper() for x in actual_actions if x.strip()}
    exp_set = {x.strip().upper() for x in expected_actions if x.strip()}
    if not exp_set and not act_set:
        return 1.0
    if not act_set or not exp_set:
        return 0.0
    tp = len(act_set & exp_set)
    prec = tp / len(act_set)
    rec = tp / len(exp_set)
    if (prec + rec) == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def compute_tool_trajectory_score(
    called_tools: list[str],
    required_tools: list[str],
    *,
    max_allowed_calls: int = 8,
) -> float:
    if not required_tools:
        return 1.0 if not called_tools else 0.0
    called_set = set(called_tools)
    req_set = set(required_tools)
    coverage = len(called_set & req_set) / len(req_set)
    # Penalize excessive redundant looping
    excess_penalty = max(0.0, (len(called_tools) - max_allowed_calls) * 0.1)
    return round(max(0.0, coverage - excess_penalty), 4)


def _scrub_ids_and_timestamps(text: str) -> str:
    cleaned = UUID_RE.sub(" ", text)
    cleaned = ISO_TS_RE.sub(" ", cleaned)
    cleaned = HARDWARE_MODEL_RE.sub(" ", cleaned)
    return CODE_TOKEN_RE.sub(" ", cleaned)


def _collect_evidence_numbers(evidence_jsons: list[str]) -> tuple[set[str], set[float]]:
    raw_nums: set[str] = set()
    float_vals: set[float] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            fval = round(float(obj), 4)
            float_vals.add(fval)
            raw_nums.add(str(obj))
            if float(obj).is_integer():
                raw_nums.add(str(int(obj)))
            return
        if isinstance(obj, str):
            for num in NUMBER_RE.findall(_scrub_ids_and_timestamps(obj)):
                raw_nums.add(num)
                float_vals.add(round(float(num), 4))
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for raw_json in evidence_jsons:
        try:
            parsed = json.loads(raw_json)
            _walk(parsed)
        except Exception:
            for num in NUMBER_RE.findall(_scrub_ids_and_timestamps(raw_json)):
                raw_nums.add(num)
                float_vals.add(round(float(num), 4))

    return raw_nums, float_vals


def _is_number_grounded(
    num: str,
    evidence_nums: set[str],
    evidence_floats: set[float],
) -> bool:
    if num in evidence_nums:
        return True
    val = float(num)
    if round(val, 4) in evidence_floats:
        return True

    decimals = len(num.split(".")[1]) if "." in num else 0
    if decimals >= 1:
        tol = 0.5 * (10.0**-decimals) + 1e-9
        rounded_target = round(val, decimals)
        return any(
            round(ev_val, decimals) == rounded_target or abs(ev_val - val) <= tol
            for ev_val in evidence_floats
        )
    return any(abs(ev_val - val) <= 0.05 for ev_val in evidence_floats)


def compute_numeric_groundedness(
    db_facts: list[str],
    evidence_jsons: list[str],
    *,
    ignored_numbers: set[str] | None = None,
    require_facts: bool = False,
) -> tuple[float, list[str]]:
    """Deterministic faithfulness check: every numeric metric cited in DB facts must appear in evidence_log."""
    if not db_facts:
        if require_facts:
            return 0.0, ["NO_DB_FACTS_CITED"]
        return 1.0, []

    evidence_nums, evidence_floats = _collect_evidence_numbers(evidence_jsons)
    ignore = DEFAULT_IGNORED_NUMBERS if ignored_numbers is None else ignored_numbers
    cited_nums: list[str] = []
    ungrounded: list[str] = []

    for fact in db_facts:
        cleaned = _scrub_ids_and_timestamps(fact)
        for num in NUMBER_RE.findall(cleaned):
            if num in ignore:
                continue
            cited_nums.append(num)
            if not _is_number_grounded(num, evidence_nums, evidence_floats):
                ungrounded.append(num)

    if not cited_nums:
        return 1.0, []
    score = (len(cited_nums) - len(ungrounded)) / len(cited_nums)
    return round(score, 4), ungrounded


def _entity_present(entity: str, verdict_text: str) -> bool:
    ent = entity.strip()
    if not ent:
        return True
    low = verdict_text.lower()
    ent_low = ent.lower()

    if ent_low == "closed":
        return bool(re.search(r"(?:\bclosed\b|закрыт[а-я]*)", low))

    if re.fullmatch(r"\d+(?:\.\d+)?%?", ent):
        num_str = ent.rstrip("%")
        val = float(num_str)
        if val.is_integer():
            core = str(int(val))
            pat = rf"(?<![\d.-]){re.escape(core)}(?:\.0+)?\s*%?(?![\d.])"
        else:
            pat = rf"(?<![\d.-]){re.escape(num_str)}\s*%?(?![\d.])"
        if re.search(pat, low):
            return True
        if ent == "7":
            return bool(re.search(r"(?:канал[аеу]?\s*№?\s*7\b|\bch(?:annel)?\s*#?7\b)", low))
        return False

    return ent_low in low


def compute_root_cause_accuracy(
    *,
    actual_needs_clarification: bool,
    expected_needs_clarification: bool,
    actual_root_cause_type: str | None,
    expected_root_cause_type: str | None,
    verdict_text: str,
    expected_entities: list[str],
) -> float:
    if expected_needs_clarification:
        return 1.0 if actual_needs_clarification else 0.0
    if actual_needs_clarification:
        return 0.0

    type_match = 1.0 if actual_root_cause_type == expected_root_cause_type else 0.0
    if not expected_entities:
        return type_match

    hits = sum(1 for ent in expected_entities if _entity_present(ent, verdict_text))
    entity_score = hits / len(expected_entities)
    return round(0.5 * type_match + 0.5 * entity_score, 4)


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = ROLE_PRICING_PER_M.get(model_name, (0.30, 1.20))
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000.0, 6)
