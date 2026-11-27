from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
from lxml import etree

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS_MAP = {"bpmn": BPMN_NS}
TOOL_NAME_RE = re.compile(r"\b((?:get|find)_[a-z0-9_]+)\b")
SYMPTOM_COND_RE = re.compile(r"symptom_category\s*==\s*['\"]([A-Z0-9_]+)['\"]")


@dataclass(frozen=True, slots=True)
class BpmnGuidance:
    symptom_category: str
    branch_nodes: list[str]
    required_tools: list[str]
    escalation_rules: list[str]
    summary_text: str
    graph_stats: dict[str, int] = field(default_factory=dict)


class BpmnWorkflowParser:
    """Deterministic parser for BPMN 2.0 XML into a NetworkX directed workflow graph."""

    def __init__(self, bpmn_path: Path) -> None:
        self.bpmn_path = bpmn_path
        self.graph: nx.DiGraph[str] = nx.DiGraph()
        self.lanes: dict[str, str] = {}
        self.symptom_flow_by_category: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.bpmn_path.exists():
            raise FileNotFoundError(f"BPMN file not found: {self.bpmn_path}")

        tree = etree.parse(str(self.bpmn_path))
        root = tree.getroot()
        process = root.find("bpmn:process", namespaces=NS_MAP)
        if process is None:
            raise ValueError("Invalid BPMN 2.0 XML: missing <bpmn:process>")

        for lane in process.findall(".//bpmn:lane", namespaces=NS_MAP):
            lane_name = str(lane.get("name") or lane.get("id") or "")
            for ref in lane.findall("bpmn:flowNodeRef", namespaces=NS_MAP):
                if ref.text:
                    self.lanes[ref.text.strip()] = lane_name

        node_tags = [
            "startEvent",
            "endEvent",
            "userTask",
            "serviceTask",
            "task",
            "exclusiveGateway",
        ]
        for tag in node_tags:
            for elem in process.findall(f"bpmn:{tag}", namespaces=NS_MAP):
                node_id = str(elem.get("id") or "")
                node_name = str(elem.get("name") or node_id)
                tools_in_name = TOOL_NAME_RE.findall(node_name) if tag == "serviceTask" else []
                if node_id:
                    self.graph.add_node(
                        node_id,
                        name=node_name,
                        element_type=tag,
                        lane=self.lanes.get(node_id, ""),
                        tools=list(dict.fromkeys(tools_in_name)),
                    )

        for flow in process.findall("bpmn:sequenceFlow", namespaces=NS_MAP):
            flow_id = str(flow.get("id") or "")
            src = str(flow.get("sourceRef") or "")
            dst = str(flow.get("targetRef") or "")
            name = str(flow.get("name") or "")
            cond_el = flow.find("bpmn:conditionExpression", namespaces=NS_MAP)
            condition = cond_el.text.strip() if cond_el is not None and cond_el.text else ""
            if src in self.graph and dst in self.graph:
                self.graph.add_edge(
                    src,
                    dst,
                    id=flow_id,
                    name=name,
                    condition=condition,
                )
                m_sym = SYMPTOM_COND_RE.search(condition)
                if m_sym and flow_id:
                    self.symptom_flow_by_category[m_sym.group(1).upper()] = flow_id

    def validate_graph(self) -> list[str]:
        """Validate BPMN process connectivity from startEvent to endEvents."""
        errors: list[str] = []
        starts = [
            n for n, d in self.graph.nodes(data=True) if d.get("element_type") == "startEvent"
        ]
        ends = [n for n, d in self.graph.nodes(data=True) if d.get("element_type") == "endEvent"]
        if len(starts) != 1:
            errors.append(f"Expected 1 startEvent, found {len(starts)}")
            return errors
        if not ends:
            errors.append("Expected at least 1 endEvent")
            return errors

        reachable = nx.descendants(self.graph, starts[0]) | {starts[0]}
        unreachable = set(self.graph.nodes) - reachable
        if unreachable:
            errors.append(f"Unreachable BPMN nodes: {sorted(unreachable)}")

        for n, d in self.graph.nodes(data=True):
            if d.get("element_type") != "endEvent" and self.graph.out_degree(n) == 0:
                errors.append(f"Dead-end non-terminal node: {n}")

        return errors

    def get_symptom_branch(self, symptom_category: str) -> BpmnGuidance:
        """Extract deterministic branch steps, mandatory tools, and escalation rules directly from XML."""
        cat = symptom_category.upper().strip()
        target_flow = self.symptom_flow_by_category.get(cat)

        ordered_nodes: list[str] = []
        visited: set[str] = set()
        required_tools: list[str] = []

        serial_gate = "Gateway_HasSerial"
        symptom_gate = "Gateway_SymptomBranch"

        def dfs(node_id: str, *, in_conditional_escalation: bool) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            ordered_nodes.append(node_id)

            ndata = self.graph.nodes[node_id]
            if not in_conditional_escalation and ndata.get("element_type") == "serviceTask":
                for tname in ndata.get("tools", []):
                    if tname not in required_tools:
                        required_tools.append(str(tname))

            out_edges = list(self.graph.out_edges(node_id, data=True))
            if node_id == serial_gate:
                positive_edges = [
                    e for e in out_edges if "true" in str(e[2].get("condition", "")).lower()
                ]
                out_edges = positive_edges or out_edges
            elif node_id == symptom_gate and target_flow is not None:
                out_edges = [e for e in out_edges if e[2].get("id") == target_flow]

            for _, dst, edata in out_edges:
                edge_has_cond = bool(edata.get("condition"))
                next_cond = in_conditional_escalation or (
                    edge_has_cond and node_id not in {serial_gate, symptom_gate}
                )
                dfs(dst, in_conditional_escalation=next_cond)

        start_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("element_type") == "startEvent"
        ]
        if start_nodes:
            dfs(start_nodes[0], in_conditional_escalation=False)

        step_lines: list[str] = []
        escalation_rules: list[str] = []
        for nid in ordered_nodes:
            ndata: dict[str, Any] = self.graph.nodes[nid]
            etype = str(ndata.get("element_type", ""))
            name = str(ndata.get("name", nid))
            if etype in {"userTask", "serviceTask", "exclusiveGateway"}:
                step_lines.append(f"- [{etype}] {nid}: {name}")
            if etype == "exclusiveGateway" and nid not in {serial_gate, symptom_gate}:
                cond_branches: list[str] = []
                for _, dst, edata in self.graph.out_edges(nid, data=True):
                    cond = str(edata.get("condition") or "").strip()
                    ename = str(edata.get("name") or "").strip()
                    dst_name = str(self.graph.nodes[dst].get("name", dst))
                    if cond:
                        label = f"{cond} ({ename})" if ename else cond
                        cond_branches.append(f"если `{label}` -> {dst_name}")
                if cond_branches:
                    escalation_rules.append(f"{nid} ({name}): " + "; ".join(cond_branches))

        summary_text = (
            f"BPMN-регламент для категории '{cat}':\n"
            + "Обязательные шаги ветки:\n"
            + "\n".join(step_lines)
            + "\n\nПороги и правила эскалации:\n"
            + "\n".join(f"- {r}" for r in escalation_rules)
            + "\n\nОбязательные SQL-тулы ветки: "
            + ", ".join(required_tools)
        )

        return BpmnGuidance(
            symptom_category=cat,
            branch_nodes=ordered_nodes,
            required_tools=required_tools,
            escalation_rules=escalation_rules,
            summary_text=summary_text,
            graph_stats={
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "branch_nodes": len(ordered_nodes),
            },
        )
