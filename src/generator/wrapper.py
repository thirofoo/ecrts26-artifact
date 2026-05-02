# src/generator/wrapper.py
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import networkx as nx

from src.common.dag_models import DAG, Node


class RDGenWrapper:
    def __init__(self, rdgen_path="external/RD-Gen", verbose=True):
        # Resolve paths relative to the project root.
        self.rdgen_path = os.path.abspath(rdgen_path)
        self.script_path = os.path.join(self.rdgen_path, "run_generator.py")
        self.verbose = verbose

    def run(self, config_path: str, output_dir: str) -> str:
        """Run RD-Gen and return the generated XML file path."""

        # Convert paths to absolute paths before invoking the subprocess.
        abs_config_path = os.path.abspath(config_path)
        abs_output_dir = os.path.abspath(output_dir)

        if os.path.exists(abs_output_dir):
            if self.verbose:
                print(f"Cleaning up existing directory: {abs_output_dir}")
            shutil.rmtree(abs_output_dir)

        # RD-Gen command.
        cmd = [
            "python3",
            self.script_path,
            "-c",
            abs_config_path,
            "-d",
            abs_output_dir,
        ]

        if self.verbose:
            print(f"Running RD-Gen: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
        else:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            error_text = getattr(result, "stderr", None) or getattr(result, "stdout", "")
            raise RuntimeError(f"RD-Gen failed:\n{error_text}")

        # Search for the generated XML, usually under output_dir/DAGs/.
        for root, dirs, files in os.walk(abs_output_dir):
            for file in files:
                if file.endswith(".xml"):
                    return os.path.join(root, file)

        raise FileNotFoundError("Generated XML file not found.")

    def parse_xml(self, xml_path: str) -> list[DAG]:
        """Read a generated XML file and return DAG objects."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.tag.lower().endswith("graphml"):
            return self._parse_graphml(xml_path)
        return self._parse_legacy_xml(root)

    def _parse_graphml(self, xml_path: str) -> list[DAG]:
        """Parse GraphML through NetworkX."""
        graph = nx.read_graphml(xml_path)
        if isinstance(graph, nx.MultiDiGraph):
            base_graph = nx.DiGraph()
            base_graph.add_nodes_from(graph.nodes(data=True))
            base_graph.add_edges_from(graph.edges())
        else:
            base_graph = nx.DiGraph(graph)

        dag_id = os.path.splitext(os.path.basename(xml_path))[0]
        dag = DAG(dag_id)
        deadline_candidates: dict[int, int] = {}

        for raw_node_id, attrs in base_graph.nodes(data=True):
            node_id = self._parse_node_id(raw_node_id)
            exec_time = self._extract_execution_time(attrs)
            dag.add_node(Node(node_id, exec_time))
            deadline = self._extract_deadline(attrs)
            if deadline is not None:
                deadline_candidates[node_id] = deadline

        for src, dst in base_graph.edges():
            dag.add_edge(self._parse_node_id(src), self._parse_node_id(dst))

        if deadline_candidates:
            sink_ids = [
                node.id for node in dag.nodes.values() if not node.successors
            ]
            deadlines = [
                deadline_candidates[node_id]
                for node_id in sink_ids
                if node_id in deadline_candidates
            ]
            if not deadlines:
                deadlines = list(deadline_candidates.values())
            dag.deadline = max(deadlines)
            dag.period = dag.deadline

        return [dag]

    def _parse_legacy_xml(self, root: ET.Element) -> list[DAG]:
        """Fallback parser for the legacy RD-Gen XML structure."""
        dags: list[DAG] = []

        for dag_elem in root.findall(".//dag"):
            dag_id = dag_elem.get("id", "unknown")
            dag = DAG(dag_id)

            for node_elem in dag_elem.findall(".//node"):
                nid = int(node_elem.get("id"))
                exec_time = int(node_elem.get("execution_time", 0))
                dag.add_node(Node(nid, exec_time))

            for edge_elem in dag_elem.findall(".//edge"):
                src = int(edge_elem.get("source"))
                dst = int(edge_elem.get("destination"))
                dag.add_edge(src, dst)

            dags.append(dag)

        return dags

    @staticmethod
    def _parse_node_id(raw_id) -> int:
        """Convert a GraphML node identifier to an integer."""
        if isinstance(raw_id, int):
            return raw_id
        try:
            return int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported node identifier: {raw_id!r}") from exc

    @staticmethod
    def _extract_execution_time(attrs: dict) -> int:
        """Extract execution time from node attributes."""
        for key, value in attrs.items():
            if key and key.lower() in {"execution_time", "wcet", "c", "c_lo"}:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue
        return 0

    @staticmethod
    def _extract_deadline(attrs: dict) -> int | None:
        """Extract a deadline from GraphML node attributes."""
        for key, value in attrs.items():
            if not key:
                continue
            normalized = key.lower().replace("-", "_")
            if normalized in {"end_to_end_deadline", "deadline"}:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue
        return None
