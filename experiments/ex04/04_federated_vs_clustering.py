from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import networkx as nx

from src.analyzer.mcfq import MCFQAnalyzer
from src.analyzer.multipath import MultipathAnalyzer
from src.clustering.proposed_clustering import ProposedClusteringEngine
from src.common.dag_models import DAG
from src.common.system_models import SystemModel
from src.generator.converter import MCConfig, MCDAGConverter
from src.generator.wrapper import RDGenWrapper
from src.utils.visualizer import draw_dag


def write_csv(rows: Sequence[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate high-utilization DAG task sets and compare "
            "federated (MCFQ) vs clustering+Multipath core counts."
        )
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of task sets to evaluate.")
    parser.add_argument("--tasks", type=int, default=5, help="DAGs per task set.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/generator/config_templates/test_config.yaml"),
        help="RD-Gen config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_dags/ex04_runs"),
        help="RD-Gen output directory.",
    )
    parser.add_argument(
        "--config-out-dir",
        type=Path,
        default=Path("data/results/ex04"),
        help="Directory to store generated RD-Gen configs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for generation.")
    parser.add_argument("--nodes", type=int, default=None, help="Override number of nodes.")
    parser.add_argument(
        "--edges",
        type=int,
        default=None,
        help="Approximate edge count by fixing in/out degree (requires nodes).",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=None,
        help="Target utilization per DAG (workload / period).",
    )
    parser.add_argument(
        "--util-mode",
        choices=["LO", "HI"],
        default="HI",
        help="Workload mode used for utilization targeting.",
    )
    parser.add_argument(
        "--allow-infeasible",
        action="store_true",
        help="Keep DAGs even if utilization target is infeasible.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of DAGs generated per RD-Gen batch (default: tasks).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Maximum number of RD-Gen batches per run.",
    )
    parser.add_argument(
        "--rdgen-verbose",
        action="store_true",
        help="Show RD-Gen command/output.",
    )
    parser.add_argument(
        "--criticality",
        choices=["HI", "LO", "MIX"],
        default="HI",
        help="Criticality assignment for DAGs.",
    )
    parser.add_argument(
        "--p-hi",
        type=float,
        default=0.5,
        help="Fraction of HI tasks when --criticality MIX is used.",
    )
    parser.add_argument("--factor-min", type=float, default=1.5, help="HI factor min.")
    parser.add_argument("--factor-max", type=float, default=3.5, help="HI factor max.")
    parser.add_argument(
        "--deadline-type",
        choices=["implicit", "constrained", "arbitrary"],
        default="implicit",
        help="Deadline/period generation policy.",
    )
    parser.add_argument("--deadline-slack-min", type=float, default=1.2)
    parser.add_argument("--deadline-slack-max", type=float, default=1.5)
    parser.add_argument(
        "--federated-max-cores",
        type=int,
        default=128,
        help="Upper bound for MCFQ core search.",
    )
    parser.add_argument(
        "--cluster-max-cores",
        type=int,
        default=128,
        help="Upper bound for multipath bound inside clustering.",
    )
    parser.add_argument(
        "--cluster-time-limit",
        type=float,
        default=5.0,
        help="Time limit for clustering optimization (seconds).",
    )
    parser.add_argument(
        "--cluster-score",
        choices=["cores", "workload", "cores+critical", "cores+workload", "critical"],
        default="cores",
        help="Objective for clustering optimization.",
    )
    parser.add_argument(
        "--cluster-quiet",
        action="store_true",
        help="Suppress clustering optimizer logs.",
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.0,
        help="Epsilon term for tie-breaker objectives.",
    )
    parser.add_argument(
        "--cluster-penalty-start",
        type=float,
        default=0.5,
        help="Initial penalty weight for clustering.",
    )
    parser.add_argument(
        "--cluster-penalty-end",
        type=float,
        default=5.0,
        help="Final penalty weight for clustering.",
    )
    parser.add_argument(
        "--cluster-plot-dir",
        type=Path,
        default=Path("data/results/ex04/clustered"),
        help="Directory to store clustered DAG visualizations.",
    )
    parser.add_argument(
        "--cluster-log-dir",
        type=Path,
        default=Path("data/results/ex04/anneal_logs"),
        help="Directory to store annealing best-update logs.",
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=None,
        help="Random seed for clustering optimizer.",
    )
    parser.add_argument(
        "--cluster-low-util",
        action="store_true",
        help="Apply clustering to low-utilization tasks as well.",
    )
    parser.add_argument(
        "--allowable-failure-prob",
        type=float,
        default=1e-9,
        help="System-wide allowable failure probability.",
    )
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=1e-9,
        help="Failure rate lambda for clustering.",
    )
    return parser


def color_red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def format_core(value: int | None) -> str:
    if value is None:
        return "None"
    return color_red(str(value))


def build_cluster_label(node) -> str:
    merged_ids = node.merged_ids or [node.id]
    id_line = ",".join(str(node_id) for node_id in merged_ids)
    parts = [id_line, f"{node.c_lo}->{node.c_hi}"]
    probs = node.constituent_probs or []
    if len(probs) > 1:
        g_m = ProposedClusteringEngine.calculate_g_m(probs)
        parts.append(f"g_m={g_m:.2e}")
    return "\n".join(parts)


def build_original_label(node, failure_rate: float) -> str:
    p = node.get_failure_prob(failure_rate)
    return "\n".join([f"{node.id}", f"{node.c_lo}->{node.c_hi}", f"p={p:.2e}"])


def build_cluster_graph(dag: DAG) -> tuple[nx.DiGraph, dict[int, str], dict[int, str]]:
    graph = nx.DiGraph()
    node_labels = {}
    node_colors = {}
    color = "salmon" if dag.criticality == "HI" else "lightblue"
    for node in dag.nodes.values():
        graph.add_node(node.id)
        for succ in node.successors:
            graph.add_edge(node.id, succ)
        node_labels[node.id] = build_cluster_label(node)
        node_colors[node.id] = color
    return graph, node_labels, node_colors


def build_original_graph(dag: DAG, failure_rate: float) -> tuple[nx.DiGraph, dict[int, str], dict[int, str]]:
    graph = nx.DiGraph()
    node_labels = {}
    node_colors = {}
    color = "salmon" if dag.criticality == "HI" else "lightblue"
    for node in dag.nodes.values():
        graph.add_node(node.id)
        for succ in node.successors:
            graph.add_edge(node.id, succ)
        node_labels[node.id] = build_original_label(node, failure_rate)
        node_colors[node.id] = color
    return graph, node_labels, node_colors


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


def _replace_top_level(lines: list[str], key: str, value: str) -> list[str]:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}:") and (len(line) - len(line.lstrip(" ")) == 0):
            lines[i] = f"{key}: {value}"
            return lines
    return [f"{key}: {value}", *lines]


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


def write_seeded_config(
    base_config: Path,
    run_seed: int,
    output_path: Path,
    num_dags: int,
    nodes: int | None,
    edges: int | None,
) -> None:
    lines = base_config.read_text().splitlines()
    updated = _replace_top_level(lines, "Seed", str(run_seed))
    updated = _replace_top_level(updated, "Number of DAGs", str(num_dags))

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


def _iter_xml_paths(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".xml"):
                yield Path(dirpath) / filename


def load_raw_dags(wrapper: RDGenWrapper, output_dir: Path) -> List[DAG]:
    dags: List[DAG] = []
    for xml_path in sorted(_iter_xml_paths(output_dir)):
        dags.extend(wrapper.parse_xml(str(xml_path)))
    return dags


def apply_utilization(dag: DAG, target_util: float, mode: str) -> tuple[bool, float | None]:
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


def generate_taskset(
    wrapper: RDGenWrapper,
    mc_config: MCConfig,
    config_path: Path,
    output_dir: Path,
    config_out_dir: Path,
    run_seed: int,
    tasks: int,
    batch_size: int,
    max_batches: int,
    nodes: int | None,
    edges: int | None,
    utilization: float | None,
    util_mode: str,
    allow_infeasible: bool,
) -> tuple[List[DAG], dict]:
    collected: List[DAG] = []
    raw_total = 0
    infeasible = 0
    batches = 0
    max_utils: List[float] = []

    while len(collected) < tasks and batches < max_batches:
        batch_seed = run_seed + batches
        config_out_dir.mkdir(parents=True, exist_ok=True)
        batch_count = max(1, batch_size)
        config_out = config_out_dir / f"rdgen_config_{run_seed}_{batches:02d}.yaml"
        write_seeded_config(config_path, batch_seed, config_out, batch_count, nodes, edges)

        wrapper.run(str(config_out), str(output_dir))
        raw_dags = load_raw_dags(wrapper, output_dir)
        if not raw_dags:
            batches += 1
            continue

        converter = MCDAGConverter(mc_config)
        mc_dags = converter.convert(raw_dags)
        for dag in mc_dags:
            raw_total += 1
            if utilization is not None:
                feasible, max_util = apply_utilization(dag, utilization, util_mode)
                if max_util is not None:
                    max_utils.append(max_util)
                if not feasible and not allow_infeasible:
                    infeasible += 1
                    continue
            collected.append(dag)
            if len(collected) >= tasks:
                break

        batches += 1

    stats = {
        "requested": tasks,
        "collected": len(collected),
        "raw_total": raw_total,
        "infeasible": infeasible,
        "batches": batches,
        "max_util_min": min(max_utils) if max_utils else None,
        "max_util_avg": (sum(max_utils) / len(max_utils)) if max_utils else None,
        "max_util_max": max(max_utils) if max_utils else None,
    }
    return collected, stats


def summarize_utilization(dags: Sequence[DAG], mode: str) -> tuple[float, float, float]:
    utils: List[float] = []
    for dag in dags:
        period = dag.period or dag.deadline
        if period <= 0:
            continue
        workload = MultipathAnalyzer.get_workload(dag, mode)
        utils.append(workload / period)
    if not utils:
        return 0.0, 0.0, 0.0
    return min(utils), sum(utils) / len(utils), max(utils)


def main() -> None:
    args = build_parser().parse_args()
    batch_size = args.batch_size or args.tasks

    if args.criticality == "MIX":
        force_criticality = None
        p_hi = args.p_hi
    else:
        force_criticality = args.criticality
        p_hi = 1.0 if args.criticality == "HI" else 0.0

    mc_config = MCConfig(
        p_hi=p_hi,
        force_criticality=force_criticality,
        factor_min=args.factor_min,
        factor_max=args.factor_max,
        seed=args.seed,
        deadline_type=args.deadline_type,
        deadline_slack_min=args.deadline_slack_min,
        deadline_slack_max=args.deadline_slack_max,
    )

    wrapper = RDGenWrapper(verbose=args.rdgen_verbose)
    mcfq = MCFQAnalyzer()
    system = SystemModel(
        num_cores=args.cluster_max_cores,
        allowable_failure_prob=args.allowable_failure_prob,
        failure_rate=args.failure_rate,
    )
    clustering_engine = ProposedClusteringEngine(system)
    temp_start = 1e-2
    temp_end = 1e-5

    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx - 1
        taskset, stats = generate_taskset(
            wrapper=wrapper,
            mc_config=mc_config,
            config_path=args.config,
            output_dir=args.output_dir,
            config_out_dir=args.config_out_dir,
            run_seed=run_seed,
            tasks=args.tasks,
            batch_size=batch_size,
            max_batches=args.max_batches,
            nodes=args.nodes,
            edges=args.edges,
            utilization=args.utilization,
            util_mode=args.util_mode,
            allow_infeasible=args.allow_infeasible,
        )

        if not taskset:
            if args.utilization is not None:
                max_util = stats.get("max_util_max")
                max_util_text = f"{max_util:.3g}" if max_util is not None else "n/a"
                print(
                    f"[run {run_idx}] No DAGs generated "
                    f"(util target={args.utilization}, max_util={max_util_text}, "
                    f"raw={stats['raw_total']}, infeasible={stats['infeasible']})"
                )
            else:
                print(f"[run {run_idx}] No DAGs generated.")
            continue

        u_min, u_avg, u_max = summarize_utilization(taskset, args.util_mode)
        print(
            f"[run {run_idx}] tasks={len(taskset)}/{stats['requested']} "
            f"batches={stats['batches']} infeasible={stats['infeasible']} "
            f"U[{args.util_mode}] min={u_min:.3g} avg={u_avg:.3g} max={u_max:.3g}"
        )

        federated = mcfq.min_cores(taskset, args.federated_max_cores, use_ilp=True)
        federated_text = format_core(federated)

        best_updates: List[dict] = []
        cluster_result = clustering_engine.optimize_taskset(
            taskset,
            time_limit=args.cluster_time_limit,
            max_cores=args.cluster_max_cores,
            seed=args.cluster_seed,
            temp_start=temp_start,
            temp_end=temp_end,
            penalty_weight_start=args.cluster_penalty_start,
            penalty_weight_end=args.cluster_penalty_end,
            apply_to_low_util=args.cluster_low_util,
            score_mode=args.cluster_score,
            score_epsilon=args.cluster_epsilon,
            best_log=best_updates,
            verbose=not args.cluster_quiet,
        )
        if best_updates:
            log_dir = args.cluster_log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"run_{run_idx:02d}_best_updates.csv"
            log_rows = [{"run_id": run_idx, **row} for row in best_updates]
            write_csv(log_rows, log_path)
        cluster_text = format_core(cluster_result.core_sum)

        print(
            f"[run {run_idx}] federated={federated_text} "
            f"cluster_multipath={cluster_text} "
            f"clusters={cluster_result.total_clusters} "
            f"score={cluster_result.best_score:.3g}"
        )

        run_dir = args.cluster_plot_dir / f"run_{run_idx:02d}"
        orig_dir = run_dir / "original"
        cluster_dir = run_dir / "clustered"
        for dag in taskset:
            graph, node_labels, node_colors = build_original_graph(dag, system.failure_rate)
            title = "\n".join(
                [
                    f"Run {run_idx} | {dag.id} (original)",
                    f"Nodes={len(dag.nodes)}",
                ]
            )
            out_path = orig_dir / f"{dag.id}.png"
            draw_dag(graph, dag, title, str(out_path), node_colors, node_labels)

        if cluster_result.reduced_dags:
            for reduced_dag in cluster_result.reduced_dags:
                graph, node_labels, node_colors = build_cluster_graph(reduced_dag)
                orig_count = sum(len(group) for group in reduced_dag.cluster_groups.values())
                title = "\n".join(
                    [
                        f"Run {run_idx} | {reduced_dag.id}",
                        f"Clusters={len(reduced_dag.nodes)} OrigNodes={orig_count}",
                    ]
                )
                out_path = cluster_dir / f"{reduced_dag.id}.png"
                draw_dag(graph, reduced_dag, title, str(out_path), node_colors, node_labels)


if __name__ == "__main__":
    main()
