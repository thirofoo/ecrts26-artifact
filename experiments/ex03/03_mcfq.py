from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import networkx as nx

from src.analyzer.multipath import MultipathAnalyzer
from src.generator.converter import MCConfig, MCDAGConverter
from src.generator.wrapper import RDGenWrapper
from src.utils.visualizer import draw_dag

RED = "\x1b[31m"
RESET = "\x1b[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random DAGs and compare Graham vs Multipath bounds."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of random DAGs to generate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/generator/config_templates/test_config.yaml"),
        help="RD-Gen config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_dags/ex03_runs"),
        help="Working directory for generated DAGs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("data/results/ex03"),
        help="Directory to store DAG visualizations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for MC parameter generation.",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=None,
        help="Target total utilization (workload / period) for the generated DAG.",
    )
    parser.add_argument(
        "--util-mode",
        choices=["LO", "HI"],
        default="HI",
        help="Workload mode used when applying --utilization.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Override number of nodes in each generated DAG.",
    )
    parser.add_argument(
        "--edges",
        type=int,
        default=None,
        help="Approximate total edge count by fixing in/out degree (requires nodes).",
    )
    return parser


def build_graph(dag):
    graph = nx.DiGraph()
    colors = {}
    labels = {}
    for node in dag.nodes.values():
        graph.add_node(node.id)
        for succ in node.successors:
            graph.add_edge(node.id, succ)
        colors[node.id] = "lightblue" if dag.criticality == "LO" else "salmon"
        labels[node.id] = f"{node.id}\n{node.c_lo}->{node.c_hi}"
    return graph, colors, labels


def min_cores_graham(dag) -> int | None:
    deadline = dag.deadline or dag.period
    if deadline <= 0:
        return None
    mode = "HI" if dag.criticality == "HI" else "LO"
    workload = MultipathAnalyzer.get_workload(dag, mode)
    critical = MultipathAnalyzer.get_critical_path(dag, mode)
    if workload <= 0:
        return 0
    if deadline < critical:
        return None
    if workload == critical:
        return 1
    denom = deadline - critical
    if denom <= 0:
        return None
    m = math.ceil((workload - critical) / denom)
    return max(1, m)


def min_cores_multipath(dag) -> int | None:
    deadline = dag.deadline or dag.period
    if deadline <= 0:
        return None
    mode = "HI" if dag.criticality == "HI" else "LO"
    critical = MultipathAnalyzer.get_critical_path(dag, mode)
    if critical > deadline:
        return None
    m = 1
    while True:
        bound = MultipathAnalyzer.calculate_multipath_bounds(dag, m)[mode]
        if bound <= deadline:
            lo = 1 if m == 1 else m // 2 + 1
            hi = m
            while lo < hi:
                mid = (lo + hi) // 2
                if MultipathAnalyzer.calculate_multipath_bounds(dag, mid)[mode] <= deadline:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        m *= 2


def _replace_block(lines: list[str], key: str, values: list[str]) -> list[str]:
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(f"{key}:"):
            indent = len(line) - len(line.lstrip(" "))
            output.append(line)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= indent:
                    break
                i += 1
            for value in values:
                output.append(" " * (indent + 2) + value)
            continue
        output.append(line)
        i += 1
    return output


def _insert_block_after_key(lines: list[str], key: str, block: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}:"):
            indent = len(line) - len(line.lstrip(" "))
            child_indent = indent + 2
            inserted = lines[: i + 1]
            for block_line in block:
                inserted.append(" " * child_indent + block_line)
            inserted.extend(lines[i + 1 :])
            return inserted
    raise ValueError(f"{key} not found in config.")


def _extract_fixed_value(lines: list[str], key: str) -> int | None:
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}:"):
            indent = len(line) - len(line.lstrip(" "))
            for j in range(i + 1, len(lines)):
                candidate = lines[j]
                if not candidate.strip():
                    continue
                next_indent = len(candidate) - len(candidate.lstrip(" "))
                if next_indent <= indent:
                    break
                stripped = candidate.strip()
                if stripped.startswith("Fixed:"):
                    value = stripped.split(":", 1)[1].strip()
                    try:
                        return int(value)
                    except ValueError:
                        return None
            break
    return None


def apply_utilization(dag, target_util: float, mode: str) -> tuple[bool, float | None]:
    if target_util <= 0:
        raise ValueError("--utilization must be > 0.")
    workload = MultipathAnalyzer.get_workload(dag, mode)
    critical = MultipathAnalyzer.get_critical_path(dag, mode)
    if workload <= 0 or critical <= 0:
        return True, None
    max_util = workload / critical
    feasible = target_util <= max_util + 1e-9
    period = int(math.ceil(workload / target_util))
    if workload > critical and period <= critical:
        period = critical + 1
    else:
        period = max(period, critical)
    dag.period = period
    dag.deadline = period
    return feasible, max_util


def write_seeded_config(
    base_config: Path,
    run_seed: int,
    output_path: Path,
    nodes: int | None,
    edges: int | None,
) -> None:
    lines = base_config.read_text().splitlines()
    replaced = False
    updated = []
    for line in lines:
        if line.strip().startswith("Seed:"):
            updated.append(f"Seed: {run_seed}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"Seed: {run_seed}")

    if nodes is None and edges is not None:
        nodes = _extract_fixed_value(updated, "Number of nodes")
    if edges is not None and nodes is None:
        raise ValueError("--edges requires --nodes or a fixed Number of nodes in the config.")

    if nodes is not None:
        updated = _replace_block(updated, "Number of nodes", [f"Fixed: {nodes}"])

    if edges is not None:
        if nodes < 2:
            raise ValueError("--nodes must be >= 2 when using --edges.")
        target_degree = max(1, min(nodes - 1, int(round(edges / nodes))))
        updated = _replace_block(updated, "In-degree", [f"Fixed: {target_degree}"])
        updated = _replace_block(updated, "Out-degree", [f"Fixed: {target_degree}"])

    output_path.write_text("\n".join(updated) + "\n")


def colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def main() -> None:
    args = build_parser().parse_args()
    wrapper = RDGenWrapper()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx - 1
        seeded_config = args.plot_dir / f"rdgen_config_{run_idx:02d}.yaml"
        write_seeded_config(
            args.config,
            run_seed,
            seeded_config,
            args.nodes,
            args.edges,
        )

        xml_path = wrapper.run(str(seeded_config), str(args.output_dir))
        dags = wrapper.parse_xml(xml_path)
        if not dags:
            print(f"[run {run_idx}] No DAGs generated.")
            continue
        converter = MCDAGConverter(
            MCConfig(
                force_criticality="HI",
                seed=run_seed,
                deadline_type="implicit",
                deadline_slack_min=1.2,
                deadline_slack_max=1.5,
            )
        )
        dag = converter.convert([dags[0]])[0]
        infeasible = False
        max_util = None
        if args.utilization is not None:
            feasible, max_util = apply_utilization(dag, args.utilization, args.util_mode)
            if not feasible and max_util is not None:
                infeasible = True
                print(
                    f"[run {run_idx}] utilization target infeasible "
                    f"(target={args.utilization:.3g}, max={max_util:.3g})"
                )

        graph, colors, labels = build_graph(dag)
        plot_path = args.plot_dir / f"ex03_dag_{run_idx:02d}.png"
        title = f"DAG {dag.id} | N={len(dag.nodes)} | T={dag.period} D={dag.deadline}"
        if infeasible and max_util is not None:
            title += f" | infeasible Umax={max_util:.3g}"
        draw_dag(graph, dag, title, str(plot_path), colors, labels)

        if infeasible:
            graham_text = "None"
            multipath_text = "None"
        else:
            graham_cores = min_cores_graham(dag)
            multipath_cores = min_cores_multipath(dag)
            graham_text = "None" if graham_cores is None else str(graham_cores)
            multipath_text = "None" if multipath_cores is None else str(multipath_cores)
        line = f"[run {run_idx}] graham={graham_text} multipath={multipath_text}"
        print(colorize(line, RED))


if __name__ == "__main__":
    main()
