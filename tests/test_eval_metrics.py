from __future__ import annotations

import json

from evals.metrics import (
    compute_action_match_score,
    compute_numeric_groundedness,
    compute_root_cause_accuracy,
)


def test_numeric_groundedness_gc02_attached_units_and_tolerances() -> None:
    """Verify attached physical units (24.08V, 24.6-25.4V, 25.064V) and hallucination detection."""
    evidence_hist = json.dumps(
        {
            "found": True,
            "tracking_code": "CAR-2026-0388",
            "component_histories": {
                "BAT-2026-016-007": [
                    {
                        "operation_code": "OP-BAT-CHG",
                        "instrument_code": "STAND-03",
                        "channel_no": 7,
                        "measurements": [
                            {
                                "guard_name": "Напряжение после зарядки 6S",
                                "numeric_value": 24.08,
                                "min_value": 24.6,
                                "max_value": 25.4,
                                "nominal_value": 25.2,
                                "unit": "V",
                                "is_passed": False,
                            }
                        ],
                    },
                    {
                        "operation_code": "OP-BAT-CHG",
                        "instrument_code": "STAND-03",
                        "channel_no": 7,
                        "measurements": [
                            {
                                "guard_name": "Напряжение после зарядки 6S",
                                "numeric_value": 24.62,
                                "min_value": 24.6,
                                "max_value": 25.4,
                                "nominal_value": 25.2,
                                "unit": "V",
                                "is_passed": True,
                            }
                        ],
                    },
                ]
            },
        },
        ensure_ascii=False,
    )
    evidence_stand = json.dumps(
        {
            "found": True,
            "instrument_code": "STAND-03",
            "channel_no": 7,
            "calibration_due_at": "2026-09-30T00:00:00+00:00",
            "target_channel_stats": {
                "channel_no": 7,
                "total_execs": 10,
                "fail_or_rework_execs": 5,
                "anomaly_rate_pct": 50.0,
                "avg_numeric_value": 24.35,
            },
            "other_channels_baseline": {
                "anomaly_rate_pct": 0.0,
                "avg_numeric_value": 25.064,
            },
        },
        ensure_ascii=False,
    )
    evidence_lot = json.dumps(
        {
            "found": True,
            "defect_rate_pct": 0.0,
            "rework_rate_pct": 16.67,
            "rework_units": 5,
            "total_split_units": 30,
        },
        ensure_ascii=False,
    )

    real_gc02_facts = [
        "При зарядке АКБ BAT-2026-016-007 на стенде STAND-03 канал 7 зафиксировано напряжение 24.08V (ниже диапазона 24.6-25.4V), затем повторный PASS 24.62V",
        "Среднее напряжение на канале 7 составляет 24.35V против 25.064V на других каналах стенда",
        "Партия LOT-BAT-2026-016: брак 0%, REWORK 16.67% (5 из 30), компоненты RS550-24V и АКБ 6S с JK-BMS-6S",
        "Стенд STAND-03 канал 7: калибровка до 2026-09-30, аномалия 50% (5 из 10 операций)",
    ]
    score_ok, ungrounded_ok = compute_numeric_groundedness(
        real_gc02_facts,
        [evidence_hist, evidence_stand, evidence_lot],
        require_facts=True,
    )
    assert score_ok == 1.0
    assert ungrounded_ok == []

    # Hallucinated reading with attached unit 'V' must be caught cleanly
    score_bad, ungrounded_bad = compute_numeric_groundedness(
        ["Зафиксировано пиковое напряжение 27.9V на канале 7."],
        [evidence_hist, evidence_stand],
        require_facts=True,
    )
    assert score_bad < 1.0
    assert ungrounded_bad == ["27.9"]


def test_numeric_groundedness_diverse_attached_units() -> None:
    """Verify Ah, Nm, mOhm, %, and C units parse full numbers without truncating decimals."""
    evidence = json.dumps(
        {
            "min_cap": 33.0,
            "max_torque": 12.0,
            "ir_max": 18,
            "defect_rate_pct": 20.0,
            "temp_c": 5,
        }
    )
    facts = [
        "Пороги: 33.0Ah, 12.0Nm, 18mOhm, брак 20%, температура -5°C.",
    ]
    score, ungrounded = compute_numeric_groundedness(facts, [evidence], require_facts=True)
    assert score == 1.0
    assert ungrounded == []


def test_action_match_and_root_cause_helpers() -> None:
    """Verify action F1 and entity synonym matching."""
    assert (
        compute_action_match_score(
            ["CREATE_SUPPORT_TICKET", "CREATE_LOT_RECALL"],
            ["CREATE_LOT_RECALL", "CREATE_SUPPORT_TICKET"],
        )
        == 1.0
    )
    assert (
        compute_root_cause_accuracy(
            actual_needs_clarification=False,
            expected_needs_clarification=False,
            actual_root_cause_type="LOT_DEFECT",
            expected_root_cause_type="LOT_DEFECT",
            verdict_text="Партия LOT-BAT-2026-014 от VoltCell Ningbo Ltd.: брак 20%, машинка CAR-2026-0412.",
            expected_entities=["LOT-BAT-2026-014", "VoltCell", "20.0", "CAR-2026-0412"],
        )
        == 1.0
    )


def test_numeric_groundedness_rounding_and_integer_guard() -> None:
    """Rounding 12.0418->12.04 and 10.0067->10.01 must pass while hallucinated 12.50 or truncated 24 must fail."""
    evidence = json.dumps(
        {
            "operator_stats": {"avg_torque_nm": 12.0418, "rework_rate_pct": 18.31},
            "shop_other_operators_baseline": {"avg_torque_nm": 10.0067, "rework_rate_pct": 0.0},
            "voltage_check": 24.42,
        }
    )
    ok_score, ok_ungrounded = compute_numeric_groundedness(
        ["Средний момент затяжки 12.04 Нм против 10.01 Нм, переделок 18.31%."],
        [evidence],
        require_facts=True,
    )
    assert ok_score == 1.0
    assert ok_ungrounded == []

    bad_score, bad_ungrounded = compute_numeric_groundedness(
        ["Средний момент 12.50 Нм, напряжение 24 В."],
        [evidence],
        require_facts=True,
    )
    assert bad_score == 0.0
    assert bad_ungrounded == ["12.50", "24"]
