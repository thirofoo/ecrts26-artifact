from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from src.analyzer.mcfq import MCFQAnalyzer, TaskParams
from src.analyzer.multipath import MultipathAnalyzer
from src.clustering.proposed_clustering import ProposedClusteringEngine
from src.common.dag_models import DAG, Node
from src.common.system_models import SystemModel
from src.generator.wrapper import RDGenWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RD-Gen DAG tasksets and compare federated (MCFQ) "
            "vs clustering+Multipath core counts."
        )
    )
    parser.add_argument("--sets", type=int, default=100, help="Number of task sets to generate.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for set evaluation (1 = sequential).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--set-seeds",
        type=Path,
        default=None,
        help="Optional file with per-set seeds (one integer per line).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run sets marked as gen_failed even if --skip-existing is enabled.",
    )
    parser.add_argument(
        "--processors",
        type=int,
        default=64,
        help="Processor count M used to normalize utilization bounds.",
    )
    parser.add_argument(
        "--ub",
        type=float,
        default=0.6,
        help="Normalized utilization bound (UB).",
    )
    parser.add_argument(
        "--ub-tolerance",
        type=float,
        default=0.1,
        help="Acceptance tolerance for UB.",
    )
    parser.add_argument("--phu", type=float, default=None, help="Probability of high-util tasks.")
    parser.add_argument("--phc", type=float, default=None, help="Probability of HI-critical tasks.")
    parser.add_argument(
        "--umax",
        type=float,
        default=2.0,
        help="Upper bound for u_O of high-util tasks.",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=2.0,
        help="Upper bound for HI scaling factor (C_HI / C_LO).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=1024,
        help="Hard cap on tasks per taskset to avoid runaway generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Max attempts to construct a taskset that satisfies UB.",
    )
    parser.add_argument(
        "--rdgen-config",
        type=Path,
        default=Path("src/generator/config_templates/test_config.yaml"),
        help="RD-Gen config path.",
    )
    parser.add_argument(
        "--rdgen-output-dir",
        type=Path,
        default=Path("data/raw_dags/ex05_runs"),
        help="RD-Gen output directory.",
    )
    parser.add_argument(
        "--rdgen-config-out-dir",
        type=Path,
        default=Path("data/results/ex05/rdgen_configs"),
        help="Directory to store generated RD-Gen configs.",
    )
    parser.add_argument("--nodes", type=int, default=None, help="Override number of nodes.")
    parser.add_argument(
        "--edges",
        type=int,
        default=None,
        help="Approximate edge count by fixing in/out degree (requires nodes).",
    )
    parser.add_argument(
        "--nodes-min",
        type=int,
        default=None,
        help="Minimum number of nodes (range).",
    )
    parser.add_argument(
        "--nodes-max",
        type=int,
        default=None,
        help="Maximum number of nodes (range).",
    )
    parser.add_argument(
        "--edges-min",
        type=int,
        default=None,
        help="Minimum edge count (range, requires nodes).",
    )
    parser.add_argument(
        "--edges-max",
        type=int,
        default=None,
        help="Maximum edge count (range, requires nodes).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of DAGs generated per RD-Gen batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Maximum number of RD-Gen batches per set attempt.",
    )
    parser.add_argument(
        "--rdgen-verbose",
        action="store_true",
        help="Show RD-Gen command/output.",
    )
    parser.add_argument(
        "--allow-infeasible",
        action="store_true",
        help="Keep DAGs even if utilization target is infeasible.",
    )
    parser.add_argument(
        "--federated-max-cores",
        type=int,
        default=1024,
        help="Upper bound for MCFQ core search.",
    )
    parser.add_argument(
        "--federated-debug",
        action="store_true",
        help="Log detailed MCFQ progress.",
    )
    parser.add_argument(
        "--federated-debug-every",
        type=int,
        default=10,
        help="Log MCFQ progress every N core counts.",
    )
    parser.add_argument(
        "--cluster-max-cores",
        type=int,
        default=1024,
        help="Upper bound for multipath bound inside clustering.",
    )
    parser.add_argument(
        "--cluster-time-limit",
        type=float,
        default=1.0,
        help="Time limit for clustering optimization (seconds).",
    )
    parser.add_argument(
        "--cluster-score",
        choices=["cores", "workload", "cores+critical", "cores+workload", "critical"],
        default="cores",
        help="Objective for clustering optimization.",
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.0,
        help="Epsilon term for tie-breaker objectives.",
    )
    parser.add_argument(
        "--cluster-temp-start",
        type=float,
        default=1e-2,
        help="Initial temperature for clustering (same as ex04).",
    )
    parser.add_argument(
        "--cluster-temp-end",
        type=float,
        default=1e-5,
        help="Final temperature for clustering (same as ex04).",
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
        "--cluster-log-dir",
        type=Path,
        default=Path("data/results/ex05/anneal_logs"),
        help="Directory to store annealing best-update logs.",
    )
    parser.add_argument(
        "--anneal-plot-set",
        type=int,
        default=1,
        help="Set index to visualize annealing progress (0 to disable).",
    )
    parser.add_argument(
        "--anneal-score-interval-ms",
        type=int,
        default=100,
        help="Interval in ms for annealing score time series (0 to disable).",
    )
    parser.add_argument(
        "--sched-curve-x",
        choices=["ub", "u_n_total", "u_o_total", "u_avg"],
        default="ub",
        help="X-axis value for the schedulability curve.",
    )
    parser.add_argument(
        "--sched-curve-bins",
        type=int,
        default=10,
        help="Bin count for the schedulability curve.",
    )
    parser.add_argument(
        "--cluster-quiet",
        action="store_true",
        help="Suppress clustering optimizer logs.",
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=None,
        help="Random seed for clustering optimizer.",
    )
    parser.add_argument(
        "--mcfq-log-omega",
        action="store_true",
        help="Log how often omega < 0 appears in SCHH checks.",
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
    parser.add_argument(
        "--fi-min",
        type=float,
        default=None,
        help="Minimum failure probability (per-job or per-hour) for sampling.",
    )
    parser.add_argument(
        "--fi-max",
        type=float,
        default=None,
        help="Maximum failure probability (per-job or per-hour) for sampling.",
    )
    parser.add_argument(
        "--fi-mode",
        choices=["per-job", "per-hour"],
        default="per-hour",
        help="Interpretation of --fi-min/--fi-max (per-job converts to per-hour).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/ex05"),
        help="Directory to store CSV and plots.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip sets already present in ex05_results.csv.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-evaluate all sets even if results exist.",
    )
    return parser


def choose_prob(rng: random.Random, override: Optional[float]) -> float:
    if override is not None:
        return override
    grid = [i / 10.0 for i in range(1, 11)]
    return rng.choice(grid)


def color_red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def format_core(value: int | None) -> str:
    if value is None:
        return color_red("None")
    return color_red(str(value))


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "nan"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "nan"}:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _normalize_range(
    min_value: int | None,
    max_value: int | None,
    name: str,
) -> tuple[int, int] | None:
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        raise ValueError(f"--{name}-min and --{name}-max must be set together.")
    if min_value < 0 or max_value < 0:
        raise ValueError(f"--{name}-min/max must be >= 0.")
    if min_value > max_value:
        raise ValueError(f"--{name}-min must be <= --{name}-max.")
    return min_value, max_value


def load_set_seeds(path: Path) -> List[int]:
    seeds: List[int] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            seeds.append(int(text))
        except ValueError as exc:
            raise ValueError(f"Invalid seed at {path}:{line_no}: {text}") from exc
    if not seeds:
        raise ValueError(f"No seeds found in {path}")
    return seeds


def write_set_seeds(seeds: Sequence[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(seed) for seed in seeds) + "\n")


def _format_run_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_format_run_value(item) for item in value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _sanitize_token(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=-")
    return "".join(ch if ch in allowed else "-" for ch in text)


def _build_run_id(
    args: argparse.Namespace,
    phu: float,
    phc: float,
    nodes_range: tuple[int, int] | None,
    edges_range: tuple[int, int] | None,
    fi_min: float | None,
    fi_max: float | None,
) -> str:
    parts = [
        ("seed", args.seed),
        ("set_seeds", args.set_seeds),
        ("sets", args.sets),
        ("proc", args.processors),
        ("ub", args.ub),
        ("ubtol", args.ub_tolerance),
        ("phu", phu),
        ("phc", phc),
        ("umax", args.umax),
        ("rmax", args.rmax),
        ("max_tasks", args.max_tasks),
        ("max_attempts", args.max_attempts),
        ("rdgen", args.rdgen_config),
        ("nodes", args.nodes),
        ("edges", args.edges),
        ("nodes_range", nodes_range),
        ("edges_range", edges_range),
        ("batch", args.batch_size),
        ("max_batches", args.max_batches),
        ("allow_infeasible", args.allow_infeasible),
        ("rdgen_verbose", args.rdgen_verbose),
        ("fmax", args.federated_max_cores),
        ("fdbg", args.federated_debug),
        ("fdbg_every", args.federated_debug_every),
        ("cmax", args.cluster_max_cores),
        ("ctime", args.cluster_time_limit),
        ("cscore", args.cluster_score),
        ("ceps", args.cluster_epsilon),
        ("ctemp", (args.cluster_temp_start, args.cluster_temp_end)),
        ("cpen", (args.cluster_penalty_start, args.cluster_penalty_end)),
        ("cseed", args.cluster_seed),
        ("cquiet", args.cluster_quiet),
        ("plotset", args.anneal_plot_set),
        ("scoreint", args.anneal_score_interval_ms),
        ("schedx", args.sched_curve_x),
        ("schedbins", args.sched_curve_bins),
        ("allow_fail", args.allowable_failure_prob),
        ("fail_rate", args.failure_rate),
        ("fi_min", fi_min),
        ("fi_max", fi_max),
        ("fi_mode", args.fi_mode),
    ]
    tokens = []
    for key, value in parts:
        token = f"{key}={_sanitize_token(_format_run_value(value))}"
        tokens.append(token)
    run_id = "__".join(tokens)
    max_len = 240
    if len(run_id) > max_len:
        import hashlib

        digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:10]
        run_id = f"{run_id[: max_len - 12]}__{digest}"
    return run_id


def _normalize_prob_range(
    min_value: float | None,
    max_value: float | None,
    name: str,
) -> tuple[float, float] | None:
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        raise ValueError(f"--{name}-min and --{name}-max must be set together.")
    if min_value < 0.0 or max_value < 0.0:
        raise ValueError(f"--{name}-min/max must be >= 0.")
    if min_value > 1.0 or max_value > 1.0:
        raise ValueError(f"--{name}-min/max must be <= 1.")
    if min_value > max_value:
        raise ValueError(f"--{name}-min must be <= --{name}-max.")
    return min_value, max_value


def _per_hour_prob(p_job: float, period_ms: int) -> float:
    if period_ms <= 0:
        return p_job
    releases = 3600000.0 / float(period_ms)
    if p_job <= 0.0:
        return 0.0
    if p_job >= 1.0:
        return 1.0
    return 1.0 - math.exp(releases * math.log1p(-p_job))


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
        raise ValueError("target utilization must be > 0.")
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


def sample_target_util(rng: random.Random, phu: float, umax: float) -> float:
    if rng.random() <= phu:
        upper = max(1.02, umax)
        return rng.uniform(1.02, upper)
    return rng.uniform(0.02, 1.0)


def apply_criticality_and_hi_params(
    dag: DAG,
    is_hi: bool,
    rng: random.Random,
    factor_max: float,
) -> None:
    dag.criticality = "HI" if is_hi else "LO"
    for node in dag.nodes.values():
        if is_hi:
            factor = rng.uniform(1.0, max(1.0, factor_max))
            calculated_hi = int(node.c_lo * factor)
            if calculated_hi <= node.c_lo:
                calculated_hi = node.c_lo + 1
            node.c_hi = calculated_hi
            node.sigma_c_hi = calculated_hi
            node.constituent_sigmas = [calculated_hi] if calculated_hi > 0 else []
        else:
            node.c_hi = node.c_lo
            node.sigma_c_hi = node.c_lo
            node.constituent_sigmas = [node.c_lo] if node.c_lo > 0 else []
        node.constituent_probs = []


def apply_failure_probs(
    dag: DAG,
    rng: random.Random,
    fi_min: float,
    fi_max: float,
    fi_mode: str,
) -> None:
    if dag.criticality != "HI":
        for node in dag.nodes.values():
            node.failure_prob = 0.0
            node.constituent_probs = [0.0]
        return
    period_ms = dag.period or dag.deadline
    for node in dag.nodes.values():
        p_sample = rng.uniform(fi_min, fi_max)
        if fi_mode == "per-job":
            p_hour = _per_hour_prob(p_sample, period_ms)
        else:
            p_hour = p_sample
        node.failure_prob = p_hour
        node.constituent_probs = [p_hour]


def task_utils(dag: DAG) -> tuple[float, float]:
    period = dag.period or dag.deadline
    if period <= 0:
        return 0.0, 0.0
    c_n = sum(node.c_lo for node in dag.nodes.values())
    if dag.criticality == "HI":
        c_o = sum(node.c_hi for node in dag.nodes.values())
    else:
        c_o = c_n
    return c_n / period, c_o / period


def count_edges(dag: DAG) -> int:
    return sum(len(node.successors) for node in dag.nodes.values())


def get_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_task_metrics(dags: Sequence[DAG], mcfq: MCFQAnalyzer) -> Dict[str, int | float | None]:
    tasks = mcfq.from_dags(dags)
    slack_values: List[float] = []
    count_lt = 0
    count_eq = 0
    omega_neg = 0
    for task in tasks:
        if task.deadline > 0:
            slack = task.deadline - task.l_n
            slack_values.append(slack)
            if slack < 0:
                count_lt += 1
            elif slack == 0:
                count_eq += 1
        omega = (task.c_o - task.c_n) - (task.l_o - task.l_n)
        if omega < 0:
            omega_neg += 1
    slack_min = min(slack_values) if slack_values else None
    return {
        "slack_min": slack_min,
        "count_D_lt_L": count_lt,
        "count_D_eq_L": count_eq,
        "omega_neg_count": omega_neg,
    }


def classify_federated_reason(
    details: Dict[str, int],
    count_d_lt_l: int,
    count_d_eq_l: int,
) -> str:
    if count_d_lt_l > 0 or count_d_eq_l > 0:
        return "D_le_L"
    if details.get("invalid_deadline"):
        return "invalid_deadline"
    if details.get("sfmc_invalid_deadline"):
        return "invalid_deadline"
    if details.get("sfmc_invalid_virtual_deadline"):
        return "virtual_deadline_invalid"
    if details.get("sfmc_denominator_nonpositive"):
        return "denominator_nonpositive"
    if details.get("sfmc_negative_load"):
        return "negative_load"
    if details.get("sfmc_capacity_fail_typical"):
        return "capacity_fail_typical"
    if details.get("sfmc_capacity_fail_critical"):
        return "capacity_fail_critical"
    if details.get("sfmc_capacity_fail_both"):
        return "capacity_fail_both"
    if details.get("mcfs_init_invalid"):
        return "mcfs_init_invalid"
    if details.get("low_util_unschedulable"):
        return "low_util_partition_fail"
    if details.get("lh_unschedulable"):
        return "lh_unschedulable"
    if details.get("hh_pairs_empty"):
        return "no_hh_pair"
    if details.get("hh_dp_empty"):
        return "capacity_fail"
    if details.get("capacity_fail_typical"):
        return "capacity_fail_typical"
    if details.get("capacity_fail_critical"):
        return "capacity_fail_critical"
    if details.get("capacity_fail_both"):
        return "capacity_fail_both"
    return "capacity_fail"


def generate_taskset_rdgen(
    wrapper: RDGenWrapper,
    config_path: Path,
    output_dir: Path,
    config_out_dir: Path,
    run_seed: int,
    rng: random.Random,
    phu: float,
    phc: float,
    umax: float,
    factor_max: float,
    processors: int,
    ub: float,
    ub_tolerance: float,
    max_tasks: int,
    batch_size: int,
    max_batches: int,
    nodes: int | None,
    edges: int | None,
    nodes_range: tuple[int, int] | None,
    edges_range: tuple[int, int] | None,
    allow_infeasible: bool,
    fi_min: float | None,
    fi_max: float | None,
    fi_mode: str,
    max_attempts: int,
    set_idx: int,
    log_progress: bool,
) -> tuple[List[DAG], dict]:
    for attempt in range(max_attempts):
        collected: List[DAG] = []
        raw_total = 0
        infeasible = 0
        batches = 0
        max_utils: List[float] = []
        total_u_n = 0.0
        total_u_o = 0.0
        done = False

        while len(collected) < max_tasks and batches < max_batches and not done:
            batch_seed = run_seed + attempt * max_batches + batches
            config_out_dir.mkdir(parents=True, exist_ok=True)
            config_out = config_out_dir / f"rdgen_config_{run_seed}_{attempt:02d}_{batches:02d}.yaml"
            batch_nodes = nodes
            if nodes_range is not None:
                batch_nodes = rng.randint(nodes_range[0], nodes_range[1])
            batch_edges = edges
            if edges_range is not None:
                batch_edges = rng.randint(edges_range[0], edges_range[1])
            write_seeded_config(
                config_path,
                batch_seed,
                config_out,
                max(1, batch_size),
                batch_nodes,
                batch_edges,
            )
            wrapper.run(str(config_out), str(output_dir))
            raw_dags = load_raw_dags(wrapper, output_dir)
            if not raw_dags:
                batches += 1
                continue

            rng.shuffle(raw_dags)
            for dag in raw_dags:
                raw_total += 1
                is_hi = rng.random() <= phc
                apply_criticality_and_hi_params(dag, is_hi, rng, factor_max)
                target_util = sample_target_util(rng, phu, umax)
                mode = "HI" if dag.criticality == "HI" else "LO"
                feasible, max_util = apply_utilization(dag, target_util, mode)
                if max_util is not None:
                    max_utils.append(max_util)
                if not feasible and not allow_infeasible:
                    infeasible += 1
                    continue
                if fi_min is not None and fi_max is not None:
                    apply_failure_probs(dag, rng, fi_min, fi_max, fi_mode)

                u_n, u_o = task_utils(dag)
                new_u_n = total_u_n + u_n
                new_u_o = total_u_o + (u_o if dag.criticality == "HI" else 0.0)
                if max(new_u_n / processors, new_u_o / processors) > ub:
                    done = True
                    break

                collected.append(dag)
                total_u_n = new_u_n
                total_u_o = new_u_o
                if len(collected) >= max_tasks:
                    done = True
                    break

            batches += 1
            if log_progress:
                current_ub = max(total_u_n / processors, total_u_o / processors) if processors > 0 else 0.0
                print(
                    f"[gen set {set_idx}] attempt={attempt + 1}/{max_attempts} "
                    f"batch={batches}/{max_batches} collected={len(collected)} "
                    f"raw={raw_total} infeasible={infeasible} "
                    f"U_n={total_u_n:.3f} U_o={total_u_o:.3f} ub={current_ub:.3f}"
                )

        if not collected:
            if log_progress:
                print(
                    f"[gen set {set_idx}] attempt={attempt + 1}/{max_attempts} "
                    "rejected: no DAGs collected"
                )
            continue

        ub_value = max(total_u_n / processors, total_u_o / processors)
        if ub_value >= ub - ub_tolerance:
            stats = {
                "raw_total": raw_total,
                "infeasible": infeasible,
                "batches": batches,
                "max_util_min": min(max_utils) if max_utils else None,
                "max_util_avg": (sum(max_utils) / len(max_utils)) if max_utils else None,
                "max_util_max": max(max_utils) if max_utils else None,
                "u_n": total_u_n,
                "u_o": total_u_o,
                "ub": ub_value,
                "attempts": attempt + 1,
            }
            return collected, stats
        if log_progress:
            print(
                f"[gen set {set_idx}] attempt={attempt + 1}/{max_attempts} "
                f"rejected: ub={ub_value:.3f} (< {ub - ub_tolerance:.3f}) "
                f"collected={len(collected)} raw={raw_total} infeasible={infeasible}"
            )

    return [], {
        "raw_total": 0,
        "infeasible": 0,
        "batches": 0,
        "max_util_min": None,
        "max_util_avg": None,
        "max_util_max": None,
        "u_n": 0.0,
        "u_o": 0.0,
        "ub": 0.0,
        "attempts": max_attempts,
    }


def split_taskset(
    dags: Sequence[DAG],
    mcfq: MCFQAnalyzer,
) -> tuple[List[tuple[DAG, TaskParams]], List[DAG], List[TaskParams], Dict[str, int]]:
    tasks = mcfq.from_dags(dags)
    hi_hi: List[tuple[DAG, TaskParams]] = []
    hi_lo: List[DAG] = []
    low_util: List[TaskParams] = []
    counts = {
        "hi_hi": 0,
        "hi_lo": 0,
        "lo_hi": 0,
        "lo_lo": 0,
    }
    for dag, task in zip(dags, tasks):
        if task.u_o > 1.0:
            if dag.criticality == "HI":
                hi_hi.append((dag, task))
                counts["hi_hi"] += 1
            else:
                hi_lo.append(dag)
                counts["hi_lo"] += 1
        else:
            low_util.append(task)
            if dag.criticality == "HI":
                counts["lo_hi"] += 1
            else:
                counts["lo_lo"] += 1
    return hi_hi, hi_lo, low_util, counts


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    fieldnames: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def load_existing_results(path: Path) -> tuple[List[Dict[str, object]], set[int]]:
    if not path.exists():
        return [], set()
    rows: List[Dict[str, object]] = []
    set_ids: set[int] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            set_id = _parse_optional_int(row.get("set_id"))
            if set_id is not None:
                set_ids.add(set_id)
    return rows, set_ids


def _rdgen_output_dir(args: argparse.Namespace) -> Path:
    if args.workers > 1:
        return args.rdgen_output_dir / f"worker_{os.getpid()}"
    return args.rdgen_output_dir


def evaluate_set_core(
    set_idx: int,
    set_seed: int,
    args: argparse.Namespace,
    phu: float,
    phc: float,
    nodes_range: tuple[int, int] | None,
    edges_range: tuple[int, int] | None,
    fi_min: float | None,
    fi_max: float | None,
    git_commit: str,
    wrapper: RDGenWrapper,
    mcfq: MCFQAnalyzer,
    clustering_engine: ProposedClusteringEngine,
) -> Dict[str, object]:
    set_start = time.monotonic()
    set_rng = random.Random(set_seed)
    rdgen_output_dir = _rdgen_output_dir(args)
    dags, stats = generate_taskset_rdgen(
        wrapper=wrapper,
        config_path=args.rdgen_config,
        output_dir=rdgen_output_dir,
        config_out_dir=args.rdgen_config_out_dir,
        run_seed=set_seed,
        rng=set_rng,
        phu=phu,
        phc=phc,
        umax=args.umax,
        factor_max=args.rmax,
        processors=args.processors,
        ub=args.ub,
        ub_tolerance=args.ub_tolerance,
        max_tasks=args.max_tasks,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        nodes=args.nodes,
        edges=args.edges,
        nodes_range=nodes_range,
        edges_range=edges_range,
        allow_infeasible=args.allow_infeasible,
        fi_min=fi_min,
        fi_max=fi_max,
        fi_mode=args.fi_mode,
        max_attempts=args.max_attempts,
        set_idx=set_idx,
        log_progress=args.rdgen_verbose,
    )

    log_lines: List[str] = []
    best_updates_for_plot: List[dict] = []
    score_timeseries: List[dict] = []
    if not dags:
        max_util = stats.get("max_util_max")
        max_util_text = f"{max_util:.3g}" if max_util is not None else "n/a"
        wall_time_sec = round(time.monotonic() - set_start, 3)
        wall_time_ms = int(wall_time_sec * 1000)
        failure_row = {
            "set_id": set_idx,
            "seed": set_seed,
            "git_commit": git_commit,
            "wall_time_sec": wall_time_sec,
            "wall_time_ms": wall_time_ms,
            "status_set": "gen_failed",
            "reason_set": "no_dags",
            "raw_total": stats["raw_total"],
            "infeasible": stats["infeasible"],
            "attempts": stats["attempts"],
            "batches": stats["batches"],
            "max_util_min": stats["max_util_min"],
            "max_util_avg": stats["max_util_avg"],
            "max_util_max": stats["max_util_max"],
            "tasks": 0,
            "nodes_min": None,
            "nodes_avg": None,
            "nodes_max": None,
            "edges_min": None,
            "edges_avg": None,
            "edges_max": None,
            "edges_total": None,
            "u_n_total": round(stats["u_n"], 6),
            "u_o_total": round(stats["u_o"], 6),
            "u_n_avg": None,
            "u_o_avg": None,
            "ub": round(stats["ub"], 6),
            "phu": phu,
            "phc": phc,
            "processors": args.processors,
        }
        log_lines.append(
            f"[set {set_idx}] No DAGs generated "
            f"(ub={args.ub}, max_util={max_util_text}, "
            f"raw={stats['raw_total']}, infeasible={stats['infeasible']})"
        )
        return {
            "set_id": set_idx,
            "row": failure_row,
            "log_lines": log_lines,
            "best_updates": best_updates_for_plot,
            "score_timeseries": score_timeseries,
        }

    task_count = len(dags)
    node_counts = [len(dag.nodes) for dag in dags]
    edge_counts = [count_edges(dag) for dag in dags]
    nodes_min = min(node_counts)
    nodes_max = max(node_counts)
    nodes_avg = sum(node_counts) / len(node_counts)
    edges_min = min(edge_counts)
    edges_max = max(edge_counts)
    edges_avg = sum(edge_counts) / len(edge_counts)
    edges_total = sum(edge_counts)
    task_metrics = compute_task_metrics(dags, mcfq)

    federated = None
    federated_status = "error"
    federated_reason = "error"
    federated_typical_used = None
    federated_critical_used = None
    omega_negative = None
    omega_total = None
    federated_ms = None
    federated_start = time.monotonic()
    try:
        federated = mcfq.min_cores(
            dags,
            args.federated_max_cores,
            use_ilp=True,
            debug=args.federated_debug,
            debug_every=args.federated_debug_every,
        )
        if federated is None:
            federated_status = "unsched"
            federated_result = mcfq.analyze(dags, args.federated_max_cores, use_ilp=True)
            federated_reason = classify_federated_reason(
                federated_result.details,
                int(task_metrics["count_D_lt_L"] or 0),
                int(task_metrics["count_D_eq_L"] or 0),
            )
        else:
            federated_result = mcfq.analyze(dags, federated, use_ilp=True)
            federated_status = "ok"
            federated_reason = "ok"
            federated_typical_used = federated_result.typical_used
            federated_critical_used = federated_result.critical_used
        omega_negative = federated_result.details.get("omega_negative", 0)
        omega_total = federated_result.details.get("omega_total", 0)
    except Exception:
        federated_status = "error"
        federated_reason = "error"
    finally:
        federated_ms = int((time.monotonic() - federated_start) * 1000)

    cluster_multipath_ms = None
    cluster_start = time.monotonic()
    hi_hi_pairs = []
    hi_lo_dags = []
    low_util_tasks = []
    counts = {"hi_hi": 0, "hi_lo": 0, "lo_hi": 0, "lo_lo": 0}
    low_util_cores = None
    hi_lo_cores = None
    hi_hi_cores = None
    cluster_status = "ok"
    cluster_reason = "ok"
    cluster_error = None
    cluster_total = None
    nocluster_total = None
    nocluster_status = "ok"
    nocluster_reason = "ok"
    nocluster_hi_hi_cores = None
    nocluster_ms = None
    try:
        hi_hi_pairs, hi_lo_dags, low_util_tasks, counts = split_taskset(dags, mcfq)
        hi_hi_dags = [dag for dag, _ in hi_hi_pairs]
        low_util_cores = mcfq._min_partition_cores_ut_075(
            low_util_tasks,
            args.federated_max_cores,
        )
        hi_lo_cores = 0
        if low_util_cores is None:
            hi_lo_cores = None
        else:
            for dag in hi_lo_dags:
                cores = MultipathAnalyzer.min_cores_for_dag(dag, args.cluster_max_cores)
                if cores is None:
                    hi_lo_cores = None
                    break
                hi_lo_cores += cores

        nocluster_start = time.monotonic()
        if low_util_cores is None:
            nocluster_status = "unsched"
            nocluster_reason = "low_util_partition_fail"
        elif hi_lo_cores is None:
            nocluster_status = "unsched"
            nocluster_reason = "multipath_none"
        else:
            hi_hi_sum = 0
            for dag in hi_hi_dags:
                cores = MultipathAnalyzer.min_cores_for_dag(dag, args.cluster_max_cores)
                if cores is None:
                    hi_hi_sum = None
                    nocluster_status = "unsched"
                    nocluster_reason = "multipath_none"
                    break
                hi_hi_sum += cores
            if hi_hi_sum is not None:
                nocluster_hi_hi_cores = hi_hi_sum
                nocluster_total = low_util_cores + hi_lo_cores + hi_hi_sum
        nocluster_ms = int((time.monotonic() - nocluster_start) * 1000)

        if hi_lo_cores is not None:
            if hi_hi_dags:
                best_updates: List[dict] = []
                collect_timeseries = (
                    args.anneal_plot_set > 0 and set_idx == args.anneal_plot_set
                )
                time_series_log = score_timeseries if collect_timeseries else None
                try:
                    cluster_result = clustering_engine.optimize_taskset(
                        hi_hi_dags,
                        time_limit=args.cluster_time_limit,
                        max_cores=args.cluster_max_cores,
                        seed=args.cluster_seed,
                        temp_start=args.cluster_temp_start,
                        temp_end=args.cluster_temp_end,
                        penalty_weight_start=args.cluster_penalty_start,
                        penalty_weight_end=args.cluster_penalty_end,
                        score_mode=args.cluster_score,
                        score_epsilon=args.cluster_epsilon,
                        best_log=best_updates,
                        time_series_log=time_series_log,
                        time_series_interval_ms=args.anneal_score_interval_ms,
                        verbose=not args.cluster_quiet,
                    )
                    hi_hi_cores = cluster_result.core_sum
                    if collect_timeseries:
                        best_updates_for_plot = list(best_updates)
                    if best_updates:
                        log_dir = args.cluster_log_dir
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_path = log_dir / f"set_{set_idx:04d}_best_updates.csv"
                        log_rows = [{"set_id": set_idx, **row} for row in best_updates]
                        write_csv(log_rows, log_path)
                    if score_timeseries:
                        log_dir = args.cluster_log_dir
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_path = log_dir / f"set_{set_idx:04d}_score_timeseries.csv"
                        log_rows = [{"set_id": set_idx, **row} for row in score_timeseries]
                        write_csv(log_rows, log_path)
                except Exception as exc:
                    cluster_status = "error"
                    cluster_reason = "cluster_error"
                    cluster_error = str(exc)
            else:
                hi_hi_cores = 0

        if low_util_cores is None or hi_lo_cores is None or hi_hi_cores is None:
            cluster_total = None
        else:
            cluster_total = low_util_cores + hi_lo_cores + hi_hi_cores
        if cluster_total is None and cluster_status != "error":
            cluster_status = "unsched"
            if low_util_cores is None:
                cluster_reason = "low_util_partition_fail"
            elif hi_lo_cores is None:
                cluster_reason = "multipath_none"
            else:
                cluster_reason = "cluster_failed"
    finally:
        cluster_multipath_ms = int((time.monotonic() - cluster_start) * 1000)

    federated_text = format_core(federated)
    nocluster_text = format_core(nocluster_total)
    cluster_text = format_core(cluster_total)
    delta_cores = None
    ratio_cores = None
    if federated is not None and cluster_total is not None:
        delta_cores = cluster_total - federated
        if federated > 0:
            ratio_cores = cluster_total / federated
    wall_time_sec = round(time.monotonic() - set_start, 3)
    wall_time_ms = int(wall_time_sec * 1000)
    low_util_tasks_count = counts["lo_hi"] + counts["lo_lo"]
    high_util_tasks_count = counts["hi_hi"] + counts["hi_lo"]
    hi_critical_tasks_count = counts["hi_hi"] + counts["lo_hi"]
    hi_critical_ratio = hi_critical_tasks_count / task_count if task_count > 0 else None
    u_n_avg = (stats["u_n"] / task_count) if task_count > 0 else None
    u_o_avg = (stats["u_o"] / task_count) if task_count > 0 else None
    row = {
        "set_id": set_idx,
        "seed": set_seed,
        "git_commit": git_commit,
        "wall_time_sec": wall_time_sec,
        "wall_time_ms": wall_time_ms,
        "status_set": "ok",
        "reason_set": "ok",
        "raw_total": stats["raw_total"],
        "infeasible": stats["infeasible"],
        "attempts": stats["attempts"],
        "batches": stats["batches"],
        "max_util_min": stats["max_util_min"],
        "max_util_avg": stats["max_util_avg"],
        "max_util_max": stats["max_util_max"],
        "tasks": task_count,
        "nodes_min": nodes_min,
        "nodes_avg": round(nodes_avg, 3),
        "nodes_max": nodes_max,
        "edges_min": edges_min,
        "edges_avg": round(edges_avg, 3),
        "edges_max": edges_max,
        "edges_total": edges_total,
        "u_n_total": round(stats["u_n"], 6),
        "u_o_total": round(stats["u_o"], 6),
        "u_n_avg": round(u_n_avg, 6) if u_n_avg is not None else None,
        "u_o_avg": round(u_o_avg, 6) if u_o_avg is not None else None,
        "ub": round(stats["ub"], 6),
        "phu": phu,
        "phc": phc,
        "processors": args.processors,
        "federated": federated,
        "time_federated_ms": federated_ms,
        "status_federated": federated_status,
        "reason_federated": federated_reason,
        "federated_typical_used": federated_typical_used,
        "federated_critical_used": federated_critical_used,
        "nocluster_multipath": nocluster_total,
        "time_nocluster_multipath_ms": nocluster_ms,
        "status_nocluster_multipath": nocluster_status,
        "reason_nocluster_multipath": nocluster_reason,
        "nocluster_hi_hi_cores": nocluster_hi_hi_cores,
        "cluster_multipath": cluster_total,
        "time_cluster_multipath_ms": cluster_multipath_ms,
        "status_cluster_multipath": cluster_status,
        "reason_cluster_multipath": cluster_reason,
        "low_util_cores": low_util_cores,
        "hi_lo_cores": hi_lo_cores,
        "hi_hi_cores": hi_hi_cores,
        "slack_min": task_metrics["slack_min"],
        "count_D_lt_L": task_metrics["count_D_lt_L"],
        "count_D_eq_L": task_metrics["count_D_eq_L"],
        "omega_neg_count": task_metrics["omega_neg_count"],
        "low_util_tasks": low_util_tasks_count,
        "high_util_tasks": high_util_tasks_count,
        "hi_critical_tasks": hi_critical_tasks_count,
        "hi_critical_ratio": hi_critical_ratio,
        "delta_cores": delta_cores,
        "ratio_cores": ratio_cores,
    }
    log_lines.append(
        f"[set {set_idx}] tasks={task_count} "
        f"nodes[min/avg/max]={nodes_min}/{nodes_avg:.1f}/{nodes_max} "
        f"U_n={stats['u_n']:.3f} U_o={stats['u_o']:.3f} "
        f"counts=[HI-HU:{counts['hi_hi']} LO-HU:{counts['hi_lo']} "
        f"HI-LU:{counts['lo_hi']} LO-LU:{counts['lo_lo']}] "
        f"federated_2018={federated_text} "
        f"federated_2016={federated_2016_text} "
        f"semi_federated={semi_federated_text} "
        f"nocluster_multipath={nocluster_text} "
        f"cluster_multipath={cluster_text}"
    )
    if args.mcfq_log_omega and omega_total is not None:
        log_lines.append(
            f"[set {set_idx}] omega_negative={omega_negative} omega_total={omega_total}"
        )
    for dag, task in hi_hi_pairs:
        edge_count = count_edges(dag)
        log_lines.append(
            f"[set {set_idx}] HI-HU dag={dag.id} "
            f"u_o={task.u_o:.4f} u_n={task.u_n:.4f} "
            f"nodes={len(dag.nodes)} edges={edge_count}"
        )
    return {
        "set_id": set_idx,
        "row": row,
        "log_lines": log_lines,
        "best_updates": best_updates_for_plot,
        "score_timeseries": score_timeseries,
    }


def evaluate_set_worker(
    set_idx: int,
    set_seed: int,
    args: argparse.Namespace,
    phu: float,
    phc: float,
    nodes_range: tuple[int, int] | None,
    edges_range: tuple[int, int] | None,
    fi_min: float | None,
    fi_max: float | None,
    git_commit: str,
) -> Dict[str, object]:
    system = SystemModel(
        num_cores=args.processors,
        allowable_failure_prob=args.allowable_failure_prob,
        failure_rate=args.failure_rate,
    )
    wrapper = RDGenWrapper(verbose=args.rdgen_verbose)
    mcfq = MCFQAnalyzer()
    clustering_engine = ProposedClusteringEngine(system)
    return evaluate_set_core(
        set_idx=set_idx,
        set_seed=set_seed,
        args=args,
        phu=phu,
        phc=phc,
        nodes_range=nodes_range,
        edges_range=edges_range,
        fi_min=fi_min,
        fi_max=fi_max,
        git_commit=git_commit,
        wrapper=wrapper,
        mcfq=mcfq,
        clustering_engine=clustering_engine,
    )


def plot_pairs(
    rows: Sequence[Dict[str, object]],
    methods: Sequence[str],
    out_dir: Path,
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            xs: List[int] = []
            ys: List[int] = []
            for row in rows:
                x = _parse_optional_int(row.get(m1))
                y = _parse_optional_int(row.get(m2))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if not xs or not ys:
                continue

            max_val = max(max(xs), max(ys))
            axis_max = min(max_val * 1.05, 200)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(xs, ys, s=18, alpha=0.7)
            ax.plot([0, axis_max], [0, axis_max], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel(m1)
            ax.set_ylabel(m2)
            ax.set_title(f"{title_prefix} {m1} vs {m2}")
            ax.set_xlim(0, axis_max)
            ax.set_ylim(0, axis_max)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
            out_path = out_dir / f"scatter_{m1}_vs_{m2}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def _sched_x_value(row: Dict[str, object], key: str) -> Optional[float]:
    if key == "u_avg":
        u_n = _parse_optional_float(row.get("u_n_total"))
        u_o = _parse_optional_float(row.get("u_o_total"))
        if u_n is None or u_o is None:
            return None
        return (u_n + u_o) / 2.0
    return _parse_optional_float(row.get(key))


def plot_schedulability_curve(
    rows: Sequence[Dict[str, object]],
    methods: Sequence[str],
    out_dir: Path,
    x_key: str,
    bins: int,
) -> None:
    values = [v for row in rows if (v := _sched_x_value(row, x_key)) is not None]
    if not values:
        return
    if bins <= 0:
        bins = 10
    x_min = min(values)
    x_max = max(values)
    if x_max <= x_min:
        x_max = x_min + 1e-9
    width = (x_max - x_min) / bins

    status_key = {
        "federated": "status_federated",
        "federated_2016": "status_federated_2016",
        "semi_federated": "status_semi_federated",
        "nocluster_multipath": "status_nocluster_multipath",
        "cluster_multipath": "status_cluster_multipath",
    }

    totals = [0 for _ in range(bins)]
    successes = {method: [0 for _ in range(bins)] for method in methods}

    for row in rows:
        x_val = _sched_x_value(row, x_key)
        if x_val is None:
            continue
        idx = int((x_val - x_min) / width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        totals[idx] += 1
        for method in methods:
            key = status_key.get(method)
            if key and row.get(key) == "ok":
                successes[method][idx] += 1

    centers = [x_min + (i + 0.5) * width for i in range(bins)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        ratios = [
            (successes[method][i] / totals[i]) if totals[i] > 0 else 0.0
            for i in range(bins)
        ]
        ax.plot(centers, ratios, marker="o", label=method)

    ax.set_xlabel(x_key)
    ax.set_ylabel("Acceptance ratio")
    ax.set_title(f"Schedulability curve ({x_key})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_path = out_dir / f"sched_curve_{x_key}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_anneal_best(
    best_updates: Sequence[Dict[str, float]],
    out_dir: Path,
    set_id: int,
    core_offset: int,
    method_cores: Dict[str, Optional[int]],
) -> None:
    if not best_updates:
        return
    xs_any: List[float] = []
    ys_any: List[float] = []
    xs_feas: List[float] = []
    ys_feas: List[float] = []
    for row in best_updates:
        elapsed = float(row.get("elapsed_ms", 0.0))
        core_sum = float(row.get("core_sum", 0.0)) + core_offset
        kind = row.get("kind")
        if kind == "best_any":
            xs_any.append(elapsed)
            ys_any.append(core_sum)
        elif kind == "best_feasible":
            xs_feas.append(elapsed)
            ys_feas.append(core_sum)
        elif kind == "initial" and not xs_any and not xs_feas:
            xs_any.append(elapsed)
            ys_any.append(core_sum)

    if not xs_any and not xs_feas:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if xs_any:
        ax.step(xs_any, ys_any, where="post", label="anneal (best any)", alpha=0.6)
    if xs_feas:
        ax.step(xs_feas, ys_feas, where="post", label="anneal (best feasible)")

    for name, value in method_cores.items():
        if value is None:
            continue
        ax.axhline(value, linestyle="--", linewidth=1, label=f"{name}={value}")

    ax.set_xlabel("Elapsed time [ms]")
    ax.set_ylabel("Required cores")
    ax.set_title(f"Annealing best updates (set {set_id})")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"anneal_best_set_{set_id:04d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_anneal_score(
    time_series: Sequence[Dict[str, float]],
    out_dir: Path,
    set_id: int,
) -> None:
    if not time_series:
        return
    xs: List[float] = []
    current: List[float] = []
    best_any: List[float] = []
    best_feasible: List[float] = []
    for row in time_series:
        xs.append(float(row.get("elapsed_ms", 0.0)))
        current.append(float(row.get("score_current", 0.0)))
        best_any.append(float(row.get("score_best_any", 0.0)))
        best_feasible.append(float(row.get("score_best_feasible", 0.0)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, current, label="current", alpha=0.7, marker="o", markersize=3, linewidth=1.5)
    ax.plot(xs, best_any, label="best any", marker="o", markersize=3, linewidth=1.5)
    ax.plot(
        xs,
        best_feasible,
        label="best feasible",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    if xs:
        max_x = max(xs)
        if max_x <= 0.0:
            max_x = 1.0
        ax.set_xlim(0.0, max_x)
    ax.set_xlabel("Elapsed time [ms]")
    ax.set_ylabel("Score")
    ax.set_title(f"Annealing score trajectory (set {set_id})")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"anneal_score_set_{set_id:04d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    phu = choose_prob(rng, args.phu)
    phc = choose_prob(rng, args.phc)
    try:
        nodes_range = _normalize_range(args.nodes_min, args.nodes_max, "nodes")
        edges_range = _normalize_range(args.edges_min, args.edges_max, "edges")
        fi_range = _normalize_prob_range(args.fi_min, args.fi_max, "fi")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    fi_min, fi_max = (fi_range if fi_range is not None else (None, None))

    if args.nodes is not None and nodes_range is not None:
        raise SystemExit("--nodes and --nodes-min/--nodes-max are mutually exclusive.")
    if args.edges is not None and edges_range is not None:
        raise SystemExit("--edges and --edges-min/--edges-max are mutually exclusive.")
    if args.nodes is not None and args.nodes < 1:
        raise SystemExit("--nodes must be >= 1.")
    if nodes_range is not None and nodes_range[0] < 1:
        raise SystemExit("--nodes-min must be >= 1.")
    if (args.edges is not None or edges_range is not None) and args.nodes is None and nodes_range is None:
        base_nodes = _extract_fixed_value(args.rdgen_config.read_text().splitlines(), "Number of nodes")
        if base_nodes is None:
            raise SystemExit(
                "--edges/--edges-min/--edges-max requires --nodes/--nodes-min/--nodes-max "
                "or fixed Number of nodes in the RD-Gen config."
            )
        if base_nodes < 2:
            raise SystemExit("Number of nodes in the RD-Gen config must be >= 2 when using edges.")
    if (args.edges is not None or edges_range is not None) and args.nodes is not None and args.nodes < 2:
        raise SystemExit("--nodes must be >= 2 when using edges.")
    if (args.edges is not None or edges_range is not None) and nodes_range is not None and nodes_range[0] < 2:
        raise SystemExit("--nodes-min must be >= 2 when using edges.")

    run_id = _build_run_id(args, phu, phc, nodes_range, edges_range, fi_min, fi_max)
    base_output_dir: Path = args.output_dir
    output_dir: Path = base_output_dir / run_id
    args.output_dir = output_dir
    default_cluster_log_dir = Path("data/results/ex05/anneal_logs")
    default_rdgen_config_out_dir = Path("data/results/ex05/rdgen_configs")
    if args.cluster_log_dir == default_cluster_log_dir:
        args.cluster_log_dir = output_dir / "anneal_logs"
    if args.rdgen_config_out_dir == default_rdgen_config_out_dir:
        args.rdgen_config_out_dir = output_dir / "rdgen_configs"

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    git_commit = get_git_commit(repo_root)
    print(f"[run] output_dir={output_dir}")
    set_seeds: List[int]
    if args.set_seeds is not None:
        try:
            set_seeds = load_set_seeds(args.set_seeds)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if len(set_seeds) < args.sets:
            raise SystemExit(
                f"--set-seeds has {len(set_seeds)} entries but --sets={args.sets}."
            )
        set_seeds = set_seeds[: args.sets]
    else:
        set_seeds = [args.seed + idx for idx in range(args.sets)]
    set_seeds_path = output_dir / "ex05_set_seeds.txt"
    write_set_seeds(set_seeds, set_seeds_path)
    meta = {
        "seed": args.seed,
        "sets": args.sets,
        "workers": args.workers,
        "run_id": run_id,
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "set_seeds_file": str(set_seeds_path),
        "set_seeds_source": str(args.set_seeds) if args.set_seeds else None,
        "processors": args.processors,
        "ub": args.ub,
        "ub_tolerance": args.ub_tolerance,
        "phu": phu,
        "phc": phc,
        "umax": args.umax,
        "rmax": args.rmax,
        "max_tasks": args.max_tasks,
        "max_attempts": args.max_attempts,
        "rdgen_config": str(args.rdgen_config),
        "rdgen_output_dir": str(args.rdgen_output_dir),
        "rdgen_config_out_dir": str(args.rdgen_config_out_dir),
        "nodes": args.nodes,
        "edges": args.edges,
        "nodes_range": list(nodes_range) if nodes_range else None,
        "edges_range": list(edges_range) if edges_range else None,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "cluster_temp_start": args.cluster_temp_start,
        "cluster_temp_end": args.cluster_temp_end,
        "cluster_penalty_start": args.cluster_penalty_start,
        "cluster_penalty_end": args.cluster_penalty_end,
        "cluster_log_dir": str(args.cluster_log_dir),
        "anneal_plot_set": args.anneal_plot_set,
        "anneal_score_interval_ms": args.anneal_score_interval_ms,
        "sched_curve_x": args.sched_curve_x,
        "sched_curve_bins": args.sched_curve_bins,
        "fi_min": fi_min,
        "fi_max": fi_max,
        "fi_mode": args.fi_mode,
        "skip_existing": args.skip_existing,
        "retry_failed": args.retry_failed,
        "git_commit": git_commit,
    }
    (output_dir / "ex05_meta.json").write_text(json.dumps(meta, indent=2))

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    methods = [
        "federated",
        "federated_2016",
        "semi_federated",
        "nocluster_multipath",
        "cluster_multipath",
    ]
    csv_path = output_dir / "ex05_results.csv"
    existing_rows: List[Dict[str, object]] = []
    existing_set_ids: set[int] = set()
    existing_failed_ids: set[int] = set()
    existing_by_set: Dict[int, Dict[str, object]] = {}
    extra_rows: List[Dict[str, object]] = []
    if csv_path.exists():
        existing_rows, existing_set_ids = load_existing_results(csv_path)
        if args.skip_existing and existing_set_ids:
            print(f"[resume] loaded {len(existing_set_ids)} existing sets from {csv_path}")
    if existing_rows:
        for row in existing_rows:
            set_id = _parse_optional_int(row.get("set_id"))
            if set_id is None:
                extra_rows.append(row)
            else:
                existing_by_set[set_id] = row
                if row.get("status_set") == "gen_failed":
                    existing_failed_ids.add(set_id)
    results_by_set: Dict[int, Dict[str, object]] = dict(existing_by_set)
    total_tasks = 0
    total_sets = 0
    min_tasks = None
    max_tasks = None
    pending_sets: List[tuple[int, int]] = []
    for set_idx in range(1, args.sets + 1):
        if args.skip_existing and set_idx in existing_set_ids:
            if args.retry_failed and set_idx in existing_failed_ids:
                print(f"[set {set_idx}] retry (previously gen_failed)")
            else:
                print(f"[set {set_idx}] skip (already in ex05_results.csv)")
                continue
        pending_sets.append((set_idx, set_seeds[set_idx - 1]))

    def handle_set_result(result: Dict[str, object]) -> None:
        nonlocal total_tasks, total_sets, min_tasks, max_tasks
        row = result.get("row")
        if not isinstance(row, dict):
            return
        set_id = _parse_optional_int(row.get("set_id"))
        if set_id is None:
            return
        results_by_set[set_id] = row
        existing_set_ids.add(set_id)
        if row.get("status_set") == "gen_failed":
            existing_failed_ids.add(set_id)
        if csv_path is not None:
            merged_rows = extra_rows + [results_by_set[k] for k in sorted(results_by_set)]
            write_csv(merged_rows, csv_path)
        for line in result.get("log_lines", []):
            print(line)
        if row.get("status_set") == "ok":
            task_count = _parse_optional_int(row.get("tasks")) or 0
            total_sets += 1
            total_tasks += task_count
            min_tasks = task_count if min_tasks is None else min(min_tasks, task_count)
            max_tasks = task_count if max_tasks is None else max(max_tasks, task_count)
        if args.anneal_plot_set > 0 and set_id == args.anneal_plot_set:
            best_updates = result.get("best_updates") or []
            score_timeseries = result.get("score_timeseries") or []
            if best_updates:
                low_util_cores = _parse_optional_int(row.get("low_util_cores"))
                hi_lo_cores = _parse_optional_int(row.get("hi_lo_cores"))
                if low_util_cores is not None and hi_lo_cores is not None:
                    core_offset = low_util_cores + hi_lo_cores
                    method_cores = {
                        "federated": _parse_optional_int(row.get("federated")),
                        "federated_2016": _parse_optional_int(row.get("federated_2016")),
                        "semi_federated": _parse_optional_int(row.get("semi_federated")),
                        "nocluster_multipath": _parse_optional_int(row.get("nocluster_multipath")),
                        "cluster_multipath": _parse_optional_int(row.get("cluster_multipath")),
                    }
                    plot_anneal_best(
                        best_updates,
                        args.cluster_log_dir,
                        set_id,
                        core_offset,
                        method_cores,
                    )
            if score_timeseries:
                plot_anneal_score(score_timeseries, args.cluster_log_dir, set_id)

    if pending_sets:
        if args.workers > 1 and len(pending_sets) > 1:
            max_workers = min(args.workers, len(pending_sets))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_set_worker,
                        set_idx,
                        set_seed,
                        args,
                        phu,
                        phc,
                        nodes_range,
                        edges_range,
                        fi_min,
                        fi_max,
                        git_commit,
                    )
                    for set_idx, set_seed in pending_sets
                ]
                for future in as_completed(futures):
                    handle_set_result(future.result())
        else:
            system = SystemModel(
                num_cores=args.processors,
                allowable_failure_prob=args.allowable_failure_prob,
                failure_rate=args.failure_rate,
            )
            wrapper = RDGenWrapper(verbose=args.rdgen_verbose)
            mcfq = MCFQAnalyzer()
            clustering_engine = ProposedClusteringEngine(system)
            for set_idx, set_seed in pending_sets:
                result = evaluate_set_core(
                    set_idx=set_idx,
                    set_seed=set_seed,
                    args=args,
                    phu=phu,
                    phc=phc,
                    nodes_range=nodes_range,
                    edges_range=edges_range,
                    fi_min=fi_min,
                    fi_max=fi_max,
                    git_commit=git_commit,
                    wrapper=wrapper,
                    mcfq=mcfq,
                    clustering_engine=clustering_engine,
                )
                handle_set_result(result)

    plot_rows = extra_rows + [results_by_set[k] for k in sorted(results_by_set)]
    if plot_rows and not csv_path.exists():
        write_csv(plot_rows, csv_path)

    title_prefix = f"phu={phu:.2f} phc={phc:.2f} UB={args.ub:.2f}"
    plot_pairs(plot_rows, methods, output_dir, title_prefix)
    plot_schedulability_curve(
        plot_rows,
        methods,
        output_dir,
        args.sched_curve_x,
        args.sched_curve_bins,
    )

    if total_sets > 0:
        avg_tasks = total_tasks / total_sets
        print(
            "[summary] "
            f"sets={total_sets} tasks_total={total_tasks} "
            f"tasks_min={min_tasks} tasks_max={max_tasks} "
            f"tasks_avg={avg_tasks:.2f}"
        )


if __name__ == "__main__":
    main()
