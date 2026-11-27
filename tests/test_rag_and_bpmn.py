from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config import Settings
from src.rag.bpmn_parser import BpmnWorkflowParser
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.indexer import SopChunk


def test_bpmn_parser_structure_and_branches(settings: Settings) -> None:
    parser = BpmnWorkflowParser(settings.bpmn_path)
    errors = parser.validate_graph()
    assert errors == []
    assert parser.symptom_flow_by_category == {
        "BATTERY_LOW_RANGE": "Flow_Sym_Battery",
        "MOTOR_STALL_OR_NOISE": "Flow_Sym_Motor",
        "NO_POWER_DEAD": "Flow_Sym_NoPower",
    }

    repeat_gw_name = str(parser.graph.nodes["Gateway_RepeatTicket"]["name"])
    assert "REWORK" not in repeat_gw_name
    assert "support_ticket" in repeat_gw_name

    bat_branch = parser.get_symptom_branch("BATTERY_LOW_RANGE")
    assert "Task_BatteryLotAndStandCheck" in bat_branch.branch_nodes
    assert bat_branch.required_tools == [
        "get_item_genealogy",
        "get_item_history",
        "get_lot_quality_stats",
        "get_instrument_channel_anomalies",
    ]
    assert "10%" in bat_branch.summary_text
    assert any("Gateway_BatchDefectOver10" in r for r in bat_branch.escalation_rules)
    assert any("has_prior_ticket" in r for r in bat_branch.escalation_rules)

    motor_branch = parser.get_symptom_branch("MOTOR_STALL_OR_NOISE")
    assert "Task_MotorOperatorCheck" in motor_branch.branch_nodes
    assert motor_branch.required_tools == [
        "get_item_genealogy",
        "get_item_history",
        "get_operator_anomalies",
    ]

    dead_branch = parser.get_symptom_branch("NO_POWER_DEAD")
    assert "Task_FullPowertrainCheck" in dead_branch.branch_nodes
    assert len(dead_branch.required_tools) >= 4


def test_bpmn_parser_extracts_tools_and_rules_dynamically_from_xml(
    settings: Settings, tmp_path: Path
) -> None:
    """Changing tool names and gateway conditions in BPMN XML must dynamically update parser output."""
    xml_text = settings.bpmn_path.read_text(encoding="utf-8")
    modified = xml_text.replace("get_operator_anomalies", "get_custom_gearbox_telemetry").replace(
        "has_prior_ticket == true",
        "has_prior_ticket == true and warranty_active == true",
    )
    custom_bpmn = tmp_path / "custom_process.bpmn"
    custom_bpmn.write_text(modified, encoding="utf-8")

    custom_parser = BpmnWorkflowParser(custom_bpmn)
    motor_branch = custom_parser.get_symptom_branch("MOTOR_STALL_OR_NOISE")
    assert "get_custom_gearbox_telemetry" in motor_branch.required_tools
    assert "get_operator_anomalies" not in motor_branch.required_tools
    assert any("warranty_active == true" in r for r in motor_branch.escalation_rules)


def test_hybrid_retriever_qdrant_labse_and_bm25_rrf(settings: Settings) -> None:
    retriever = HybridRetriever(settings)

    hits_bat = retriever.search("разрядная ёмкость АКБ 6S ниже 33 Ач отзыв партии", top_k=3)
    assert len(hits_bat) == 3
    assert any(h.source_file == "01_battery_6s_diagnostics.md" for h in hits_bat)
    assert all(h.rrf_score > 0 and h.dense_score > 0 for h in hits_bat)

    hits_motor = retriever.search("момент затяжки редуктора RS550-24V 12 Нм перетяжка", top_k=3)
    assert any(h.source_file == "02_motor_rs550_gearbox.md" for h in hits_motor)

    hits_stand = retriever.search("дрейф канала зарядного стенда STAND-03 поверка", top_k=3)
    assert any(h.source_file == "03_charging_stand_calibration.md" for h in hits_stand)


def test_hybrid_retriever_bounds_candidate_k(settings: Settings) -> None:
    """candidate_k must stay bounded by min(len(chunks), max(k * 3, 20)) on large corpora."""
    synthetic_chunks = [
        SopChunk(
            chunk_id=f"chunk-{i}",
            point_id=f"00000000-0000-0000-0000-{i:012d}",
            source_file="sop.md",
            doc_title="SOP",
            section_title=f"Sec {i}",
            content=f"Аккумулятор 6S проверка канал {i}",
        )
        for i in range(50)
    ]

    recorded_limits: list[int] = []

    class SpyQdrant:
        def query_points(self, *, limit: int, **_: Any) -> SimpleNamespace:
            recorded_limits.append(limit)
            return SimpleNamespace(points=[])

    retriever = HybridRetriever(
        settings,
        chunks=synthetic_chunks,
        qdrant_client=SpyQdrant(),  # type: ignore[arg-type]
        use_dense=True,
    )
    retriever.search("аккумулятор", top_k=3)
    assert recorded_limits[-1] == 20

    retriever.search("аккумулятор", top_k=10)
    assert recorded_limits[-1] == 30
