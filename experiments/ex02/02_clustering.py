from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analyzer.multipath import MultipathAnalyzer
from src.clustering.proposed_clustering import ProposedClusteringEngine
from src.common.data_manager import DataManager
from src.common.system_models import SystemModel
from src.utils.visualizer import draw_dag

FIG_SIZE = (20, 12)
NODE_SIZE = 6500
RESULT_PATH = Path("data/results/02_proposed_clustering.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run proposed clustering with SA and visualize the reduced DAG.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dag_dataset.pkl"),
        help="Path to pickled DAG list.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="Time limit for simulated annealing (seconds).",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=32,
        help="Maximum core count to test in the multipath bound.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the optimizer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_PATH,
        help="Output path for the visualization.",
    )
    return parser


def load_target_dag(path: Path):
    dags = DataManager.load(str(path))
    hi_dags = [dag for dag in dags if dag.criticality == "HI"]
    return hi_dags[0] if hi_dags else dags[0]


def build_node_label(node, violation_prob: float) -> str:
    node_ids = node.merged_ids or [node.id]
    id_line = ",".join(str(node_id) for node_id in node_ids)
    return "\n".join(
        [
            id_line,
            f"{node.c_lo} -> {node.c_hi}",
            f"g_m={violation_prob:.2e}",
        ]
    )


def build_graph(dag) -> tuple[nx.DiGraph, dict[int, str], dict[int, str]]:
    graph = nx.DiGraph()
    node_labels = {}
    node_colors = {}
    for node in dag.nodes.values():
        graph.add_node(node.id)
        for succ in node.successors:
            graph.add_edge(node.id, succ)
        probs = node.constituent_probs or []
        if len(probs) <= 1:
            violation_prob = 0.0
        else:
            violation_prob = ProposedClusteringEngine.calculate_g_m(probs)
        node_labels[node.id] = build_node_label(node, violation_prob)
        node_colors[node.id] = "salmon"
    return graph, node_labels, node_colors


def main() -> None:
    args = build_parser().parse_args()
    try:
        target_dag = load_target_dag(args.dataset)
    except FileNotFoundError:
        print("Error: Data file not found. Run experiments/ex01/01_generate_data.py first.")
        return

    print(f"Target DAG: {target_dag.id} (Nodes: {len(target_dag.nodes)})")

    system = SystemModel(
        num_cores=4,
        allowable_failure_prob=1e-11,
        failure_rate=1e-9,
    )
    engine = ProposedClusteringEngine(system)
    temp_start = float(args.max_cores)
    temp_end = 0.0
    result = engine.optimize_dag(
        target_dag,
        time_limit=args.time_limit,
        max_cores=args.max_cores,
        seed=args.seed,
        temp_start=temp_start,
        temp_end=temp_end,
    )

    reduced_dag = result.reduced_dags[0] if result.reduced_dags else target_dag
    print(f"Clustering Complete. Nodes: {len(target_dag.nodes)} -> {len(reduced_dag.nodes)}")
    print(
        f"Best score={result.best_score:.6f} cores_sum={result.core_sum} clusters={result.total_clusters}"
    )

    w_lo = MultipathAnalyzer.get_workload(reduced_dag, "LO")
    w_hi = MultipathAnalyzer.get_workload(reduced_dag, "HI")
    l_lo = MultipathAnalyzer.get_critical_path(reduced_dag, "LO")
    l_hi = MultipathAnalyzer.get_critical_path(reduced_dag, "HI")

    cluster_count = len(reduced_dag.nodes)
    limit = system.failure_budget_per_cluster(cluster_count)
    title = "\n".join(
        [
            f"Proposed Clustering (Clusters={cluster_count})",
            f"W_lo={w_lo} W_hi={w_hi} | L_lo={l_lo} L_hi={l_hi}",
            f"P_allow={system.allowable_failure_prob:.1e} | P/M={limit:.1e} (M={cluster_count})",
        ]
    )

    graph, node_labels, node_colors = build_graph(reduced_dag)
    draw_dag(graph, reduced_dag, title, str(args.output), node_colors, node_labels)


if __name__ == "__main__":
    main()
