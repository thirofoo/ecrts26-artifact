from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.analyzer.mcfq import MCFQAnalyzer, TaskParams
from src.analyzer.multipath import MultipathAnalyzer
from src.clustering.proposed_clustering import ProposedClusteringEngine
from src.common.dag_models import DAG
from src.common.system_models import SystemModel
from src.generator.wrapper import RDGenWrapper

CP_BINS = [
    ("BIN_CP_1", 0.10, 0.20),
    ("BIN_CP_2", 0.20, 0.30),
    ("BIN_CP_3", 0.30, 0.40),
    ("BIN_CP_4", 0.40, 0.50),
    ("BIN_CP_5", 0.50, 0.60),
    ("BIN_CP_6", 0.60, 0.70),
    ("BIN_CP_7", 0.70, 0.80),
    ("BIN_CP_8", 0.80, 0.90),
]
CP_BIN_TO_ROSTER = {
    "BIN_CP_1": "0.1_0.2",
    "BIN_CP_2": "0.2_0.3",
    "BIN_CP_3": "0.3_0.4",
    "BIN_CP_4": "0.4_0.5",
    "BIN_CP_5": "0.5_0.6",
    "BIN_CP_6": "0.6_0.7",
    "BIN_CP_7": "0.7_0.8",
    "BIN_CP_8": "0.8_0.9",
}
TIGHTNESS_BINS = [
    ("BIN_T_1", 0.6, 0.7),
    ("BIN_T_2", 0.7, 0.8),
    ("BIN_T_3", 0.8, 0.9),
]
R_VALUES = [2, 4, 8]

DEFAULT_CP_BIN = "BIN_CP_4"
DEFAULT_TIGHTNESS_BIN = "BIN_T_2"
DEFAULT_R_VALUE = 4


def color_red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def format_core(value: int | None) -> str:
    if value is None:
        return color_red("None")
    return color_red(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RQ1 evaluation using ex05_pre DAG pool (ex05_1)."
    )
    parser.add_argument(
        "--pre-run-dir",
        type=Path,
        default=Path("data/results/ex05_pre/05_pre1"),
        help="ex05_pre run directory containing rosters/ and dag_tasks/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="ex05_1",
        help="Run identifier used for output subdirectory naming.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/ex05_1"),
        help="Base output directory for ex05_1 results.",
    )
    parser.add_argument(
        "--axis",
        choices=["cp", "ratio", "tightness", "cpxtightness", "all"],
        default="cp",
        help="Which axis to sweep for RQ1.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "fan-in", "chain", "mixed"],
        default="auto",
        help=(
            "Roster method to sample from. "
            "auto runs fan-in and chain separately; mixed uses combined rosters."
        ),
    )
    parser.add_argument(
        "--sets-per-bin",
        type=int,
        default=50,
        help="Number of tasksets per bin.",
    )
    parser.add_argument(
        "--fill-to",
        action="store_true",
        help="Extend plan to reach --sets-per-bin per bin based on existing results.",
    )
    parser.add_argument(
        "--fill-count",
        choices=["ok", "all", "plan"],
        default="ok",
        help="Count mode for --fill-to (ok=successful sets, all=all results, plan=plan rows).",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=20,
        help="Number of DAGs per taskset.",
    )
    parser.add_argument(
        "--taskset-nodes-max",
        type=int,
        default=None,
        help="Stop adding DAGs once total nodes >= this value (overrides --tasks).",
    )
    parser.add_argument(
        "--hi-critical-ratio",
        type=float,
        default=0.5,
        help="Ratio of HI-critical tasks per taskset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic generation.",
    )
    parser.add_argument(
        "--set-seeds",
        type=Path,
        default=None,
        help="Optional file with per-set seeds (one integer per line).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2000,
        help="Max attempts to build a taskset before giving up.",
    )
    parser.add_argument(
        "--deadline-tries",
        type=int,
        default=20,
        help="Max retries to sample a deadline/period for a DAG.",
    )
    parser.add_argument(
        "--deadline-mode",
        choices=["tightness", "utilization"],
        default="tightness",
        help="How to sample deadlines: tightness bin or utilization range.",
    )
    parser.add_argument(
        "--u-hi-min",
        type=float,
        default=1.0,
        help="Minimum target U (HI) when --deadline-mode=utilization.",
    )
    parser.add_argument(
        "--u-hi-max",
        type=float,
        default=4.0,
        help="Maximum target U (HI) when --deadline-mode=utilization.",
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=1.0,
        help="Minimum multiplier for period (T = ceil(kappa * D)).",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=2.0,
        help="Maximum multiplier for period (T = ceil(kappa * D)).",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=1.0,
        help="Minimum C^HI/C^LO ratio when sampling (used when axis != ratio).",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=4.0,
        help="Maximum C^HI/C^LO ratio when sampling (used when axis != ratio).",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        default=True,
        help="Allow duplicate DAGs within a taskset.",
    )
    parser.add_argument(
        "--no-allow-duplicates",
        action="store_false",
        dest="allow_duplicates",
        help="Disallow duplicate DAGs within a taskset.",
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
        default=10.0,
        help="Time limit for clustering optimization (seconds).",
    )
    parser.add_argument(
        "--cluster-score",
        choices=["cores", "workload", "cores+critical", "cores+workload", "critical", "both"],
        default="both",
        help="Objective for clustering optimization.",
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=1e-4,
        help="Epsilon term for tie-breaker objectives.",
    )
    parser.add_argument(
        "--cluster-temp-start",
        type=float,
        default=1e-2,
        help="Initial temperature for clustering.",
    )
    parser.add_argument(
        "--cluster-temp-end",
        type=float,
        default=1e-5,
        help="Final temperature for clustering.",
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
        "--cluster-quiet",
        action="store_true",
        default=False,
        help="Suppress clustering optimization logs.",
    )
    parser.add_argument(
        "--cluster-log-accepts",
        action="store_true",
        default=True,
        help="Log accepted annealing moves.",
    )
    parser.add_argument(
        "--no-cluster-log-accepts",
        action="store_false",
        dest="cluster_log_accepts",
        help="Disable accepted-move logs.",
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
        "--fi",
        type=float,
        default=1e-6,
        help="Per-task failure probability (per hour).",
    )
    parser.add_argument(
        "--fi-mode",
        choices=["per-job", "per-hour"],
        default="per-hour",
        help="Interpretation of --fi (per-job converts to per-hour).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip sets already present in ex05_1_results.csv.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-evaluate all sets even if results exist.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run sets marked as gen_failed even if --skip-existing is enabled.",
    )
    return parser


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


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_roster(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("status") != "ok":
                continue
            dag_id = _parse_int(row.get("dag_id"))
            xml_path = row.get("xml_path")
            if dag_id is None or not xml_path:
                continue
            rows.append(
                {
                    "dag_id": dag_id,
                    "seed": _parse_int(row.get("seed")),
                    "xml_path": xml_path,
                    "workload_hi": _parse_int(row.get("workload_hi")),
                    "critical_hi": _parse_int(row.get("critical_hi")),
                    "cpr": _parse_float(row.get("cpr")),
                    "cpr_class": row.get("cpr_class"),
                    "rdgen_method": row.get("rdgen_method"),
                }
            )
    return rows


def load_roster_pool(run_dir: Path, method: Optional[str]) -> Dict[str, List[Dict[str, object]]]:
    roster_dir = run_dir / "rosters"
    pools: Dict[str, List[Dict[str, object]]] = {}
    for label, _, _ in CP_BINS:
        roster_label = CP_BIN_TO_ROSTER.get(label, label)
        if method and method in {"fan-in", "chain"}:
            path = roster_dir / f"dag_roster_cpr_{roster_label}__{method}.csv"
        else:
            path = roster_dir / f"dag_roster_cpr_{roster_label}.csv"
        pools[label] = load_roster(path)
    return pools


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
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


def write_results(results_by_set: Dict[int, Dict[str, object]], out_path: Path) -> None:
    rows = [results_by_set[set_id] for set_id in sorted(results_by_set)]
    write_csv(rows, out_path)


def build_plan(args: argparse.Namespace) -> List[Dict[str, object]]:
    plan_rows: List[Dict[str, object]] = []
    set_id = 1

    def add_row(axis: str, cp_bin: str, r_value: int, t_bin: str) -> None:
        nonlocal set_id
        plan_rows.append(
            {
                "set_id": set_id,
                "axis": axis,
                "cp_bin": cp_bin,
                "r_value": r_value,
                "tightness_bin": t_bin,
            }
        )
        set_id += 1

    def add_axis(axis: str) -> None:
        if axis == "cp":
            for cp_label, _, _ in CP_BINS:
                for _ in range(args.sets_per_bin):
                    add_row(axis, cp_label, DEFAULT_R_VALUE, DEFAULT_TIGHTNESS_BIN)
        elif axis == "ratio":
            for r_value in R_VALUES:
                for _ in range(args.sets_per_bin):
                    add_row(axis, DEFAULT_CP_BIN, r_value, DEFAULT_TIGHTNESS_BIN)
        elif axis == "tightness":
            for t_label, _, _ in TIGHTNESS_BINS:
                for _ in range(args.sets_per_bin):
                    add_row(axis, DEFAULT_CP_BIN, DEFAULT_R_VALUE, t_label)
        elif axis == "cpxtightness":
            for cp_label, _, _ in CP_BINS:
                for t_label, _, _ in TIGHTNESS_BINS:
                    for _ in range(args.sets_per_bin):
                        add_row(axis, cp_label, DEFAULT_R_VALUE, t_label)

    if args.axis == "all":
        add_axis("cp")
        add_axis("ratio")
        add_axis("tightness")
    else:
        add_axis(args.axis)
    return plan_rows


def load_plan(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append(
                {
                    "set_id": _parse_int(row.get("set_id")),
                    "axis": row.get("axis"),
                    "cp_bin": row.get("cp_bin"),
                    "r_value": _parse_int(row.get("r_value")),
                    "tightness_bin": row.get("tightness_bin"),
                }
            )
    return rows


def _normalize_key(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _plan_key(row: Dict[str, object], axis: str) -> Optional[object]:
    if axis == "cp":
        return _normalize_key(row.get("cp_bin"))
    if axis == "ratio":
        return _normalize_key(row.get("r_value"))
    if axis == "tightness":
        return _normalize_key(row.get("tightness_bin"))
    if axis == "cpxtightness":
        cp_bin = _normalize_key(row.get("cp_bin"))
        t_bin = _normalize_key(row.get("tightness_bin"))
        if cp_bin and t_bin:
            return (cp_bin, t_bin)
        return None
    return None


def count_results_by_axis(
    existing: Dict[int, Dict[str, object]],
    axis: str,
    count_mode: str,
) -> Dict[object, int]:
    counts: Dict[object, int] = {}
    for row in existing.values():
        if str(row.get("axis")) != axis:
            continue
        if count_mode == "ok" and row.get("status_set") != "ok":
            continue
        key = _plan_key(row, axis)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_plan_by_axis(plan_rows: Sequence[Dict[str, object]], axis: str) -> Dict[object, int]:
    counts: Dict[object, int] = {}
    for row in plan_rows:
        if str(row.get("axis")) != axis:
            continue
        key = _plan_key(row, axis)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def extend_plan_to_target(
    plan_rows: List[Dict[str, object]],
    existing: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if args.sets_per_bin < 1:
        return plan_rows
    next_set_id = max((int(row.get("set_id") or 0) for row in plan_rows), default=0) + 1

    def add_row(axis: str, cp_bin: str, r_value: int, t_bin: str) -> None:
        nonlocal next_set_id
        plan_rows.append(
            {
                "set_id": next_set_id,
                "axis": axis,
                "cp_bin": cp_bin,
                "r_value": r_value,
                "tightness_bin": t_bin,
            }
        )
        next_set_id += 1

    def counts_for_axis(axis: str) -> Dict[object, int]:
        if args.fill_count == "plan":
            return count_plan_by_axis(plan_rows, axis)
        return count_results_by_axis(existing, axis, args.fill_count)

    def fill_cp() -> None:
        counts = counts_for_axis("cp")
        for cp_label, _, _ in CP_BINS:
            key = _normalize_key(cp_label)
            current = counts.get(key, 0)
            missing = args.sets_per_bin - current
            for _ in range(max(0, missing)):
                add_row("cp", cp_label, DEFAULT_R_VALUE, DEFAULT_TIGHTNESS_BIN)

    def fill_ratio() -> None:
        counts = counts_for_axis("ratio")
        for r_value in R_VALUES:
            key = _normalize_key(r_value)
            current = counts.get(key, 0)
            missing = args.sets_per_bin - current
            for _ in range(max(0, missing)):
                add_row("ratio", DEFAULT_CP_BIN, r_value, DEFAULT_TIGHTNESS_BIN)

    def fill_tightness() -> None:
        counts = counts_for_axis("tightness")
        for t_label, _, _ in TIGHTNESS_BINS:
            key = _normalize_key(t_label)
            current = counts.get(key, 0)
            missing = args.sets_per_bin - current
            for _ in range(max(0, missing)):
                add_row("tightness", DEFAULT_CP_BIN, DEFAULT_R_VALUE, t_label)

    def fill_cpxtightness() -> None:
        counts = counts_for_axis("cpxtightness")
        for cp_label, _, _ in CP_BINS:
            for t_label, _, _ in TIGHTNESS_BINS:
                key = (_normalize_key(cp_label), _normalize_key(t_label))
                current = counts.get(key, 0)
                missing = args.sets_per_bin - current
                for _ in range(max(0, missing)):
                    add_row("cpxtightness", cp_label, DEFAULT_R_VALUE, t_label)

    if args.axis == "all":
        fill_cp()
        fill_ratio()
        fill_tightness()
    elif args.axis == "cp":
        fill_cp()
    elif args.axis == "ratio":
        fill_ratio()
    elif args.axis == "tightness":
        fill_tightness()
    elif args.axis == "cpxtightness":
        fill_cpxtightness()

    return plan_rows


def _find_bin(label: str, bins: Sequence[Tuple[str, float, float]]) -> Tuple[float, float]:
    for name, low, high in bins:
        if name == label:
            return low, high
    raise ValueError(f"Unknown bin label: {label}")


def compute_workload_hi(dag: DAG) -> int:
    return sum(node.c_hi for node in dag.nodes.values())


def compute_critical_path_hi(dag: DAG) -> int:
    memo: Dict[int, int] = {}

    def _visit(node_id: int) -> int:
        if node_id in memo:
            return memo[node_id]
        node = dag.nodes[node_id]
        current = node.c_hi
        if not node.predecessors:
            memo[node_id] = current
            return current
        best = max(_visit(pred) for pred in node.predecessors)
        memo[node_id] = current + best
        return memo[node_id]

    if not dag.nodes:
        return 0
    return max(_visit(node_id) for node_id in dag.nodes)


def apply_failure_probs(dag: DAG, rng: random.Random, p_hour: float, mode: str) -> None:
    if dag.criticality != "HI":
        for node in dag.nodes.values():
            node.failure_prob = 0.0
            node.constituent_probs = [0.0]
        return
    period_ms = dag.period or dag.deadline
    for node in dag.nodes.values():
        p_value = p_hour
        if mode == "per-job":
            p_value = _per_hour_prob(p_hour, period_ms)
        node.failure_prob = p_value
        node.constituent_probs = [p_value]


def _per_hour_prob(p_job: float, period_ms: int) -> float:
    if p_job <= 0:
        return 0.0
    period_sec = period_ms / 1000.0
    if period_sec <= 0:
        return 0.0
    releases = 3600.0 / period_sec
    return 1.0 - math.exp(math.log(1.0 - p_job) * releases)


def split_taskset(
    dags: Sequence[DAG],
    mcfq: MCFQAnalyzer,
) -> Tuple[List[Tuple[DAG, TaskParams]], List[DAG], List[TaskParams], Dict[str, int]]:
    tasks = mcfq.from_dags(dags)
    hi_hi: List[Tuple[DAG, TaskParams]] = []
    hi_lo: List[DAG] = []
    low_util: List[TaskParams] = []
    counts = {"hi_hi": 0, "hi_lo": 0, "lo_hi": 0, "lo_lo": 0}
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


def classify_federated_reason(details: Dict[str, int], count_d_lt_l: int, count_d_eq_l: int) -> str:
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


def _pick_entry(
    pool: Sequence[Dict[str, object]],
    rng: random.Random,
    used_ids: set[int],
    allow_duplicates: bool,
) -> Optional[Dict[str, object]]:
    if not pool:
        return None
    if allow_duplicates:
        return rng.choice(list(pool))
    available = [entry for entry in pool if entry.get("dag_id") not in used_ids]
    if not available:
        return None
    return rng.choice(available)


def _sample_deadline(
    cp_hi: int,
    work_hi: int,
    tightness_label: str,
    rng: random.Random,
    kappa_min: float,
    kappa_max: float,
    deadline_tries: int,
) -> Optional[Tuple[int, int, float]]:
    low, high = _find_bin(tightness_label, TIGHTNESS_BINS)
    d_min = int(math.ceil(cp_hi / high))
    d_max = int(math.floor(cp_hi / low))
    if d_min < 1:
        d_min = 1
    if d_min > d_max:
        return None
    for _ in range(max(1, deadline_tries)):
        deadline = rng.randint(d_min, d_max)
        period = deadline
        tightness = cp_hi / deadline if deadline > 0 else 0.0
        u_hi = work_hi / period if period > 0 else 0.0
        if u_hi <= 1.0:
            continue
        return deadline, period, tightness
    return None


def _sample_deadline_by_utilization(
    cp_hi: int,
    work_hi: int,
    rng: random.Random,
    u_hi_min: float,
    u_hi_max: float,
    deadline_tries: int,
) -> Optional[Tuple[int, int, float]]:
    if cp_hi <= 0 or work_hi <= 0:
        return None
    u_cap = work_hi / cp_hi
    u_upper = min(u_hi_max, u_cap)
    u_lower = max(u_hi_min, 1.0 + 1e-9)
    if u_upper <= u_lower:
        return None
    for _ in range(max(1, deadline_tries)):
        u_target = rng.uniform(u_lower, u_upper)
        deadline = int(math.ceil(work_hi / u_target))
        if deadline <= cp_hi:
            deadline = int(cp_hi + 1)
        period = deadline
        tightness = cp_hi / deadline if deadline > 0 else 0.0
        u_hi = work_hi / period if period > 0 else 0.0
        if u_hi <= 1.0:
            continue
        return deadline, period, tightness
    return None


def _apply_ratio(dag: DAG, r_value: float) -> None:
    for node in dag.nodes.values():
        c_hi = node.c_hi
        c_lo = int(round(c_hi / r_value))
        if c_lo < 1:
            c_lo = 1
        node.c_lo = c_lo
        node.constituent_probs = []


def _set_criticality(dag: DAG, criticality: str) -> None:
    dag.criticality = criticality
    for node in dag.nodes.values():
        if criticality == "HI":
            node.sigma_c_hi = node.c_hi
            node.constituent_sigmas = [node.c_hi] if node.c_hi > 0 else []
        else:
            node.c_hi = node.c_lo
            node.sigma_c_hi = node.c_lo
            node.constituent_sigmas = [node.c_lo] if node.c_lo > 0 else []
        node.constituent_probs = []


def generate_taskset(
    set_id: int,
    set_seed: int,
    plan: Dict[str, object],
    args: argparse.Namespace,
    pools: Dict[str, List[Dict[str, object]]],
    wrapper: RDGenWrapper,
) -> Tuple[List[DAG], Dict[str, object]]:
    rng = random.Random(set_seed)
    cp_bin = str(plan.get("cp_bin") or DEFAULT_CP_BIN)
    axis = str(plan.get("axis") or args.axis)
    if axis == "ratio":
        r_value = float(plan.get("r_value") or DEFAULT_R_VALUE)
    else:
        r_value = rng.uniform(args.ratio_min, args.ratio_max)
    tightness_bin = str(plan.get("tightness_bin") or DEFAULT_TIGHTNESS_BIN)
    pool = pools.get(cp_bin, [])

    use_nodes_budget = args.taskset_nodes_max is not None
    target_nodes = args.taskset_nodes_max or 0
    used_ids: set[int] = set()
    tasks: List[DAG] = []
    task_rows: List[Dict[str, object]] = []
    rejects = {"no_dag": 0, "no_d_range": 0, "low_u_hi": 0}
    attempts = 0
    total_nodes = 0

    def should_continue() -> bool:
        if use_nodes_budget:
            return total_nodes < target_nodes
        return len(tasks) < args.tasks

    while should_continue() and attempts < args.max_attempts:
        attempts += 1
        entry = _pick_entry(pool, rng, used_ids, args.allow_duplicates)
        if entry is None:
            rejects["no_dag"] += 1
            break
        dag_id = entry.get("dag_id")
        if dag_id is None:
            rejects["no_dag"] += 1
            continue
        xml_path = entry.get("xml_path")
        if not xml_path:
            rejects["no_dag"] += 1
            continue
        xml_path = Path(str(xml_path))
        dags = wrapper.parse_xml(str(xml_path))
        if not dags:
            rejects["no_dag"] += 1
            continue
        dag = dags[0]

        cp_hi = entry.get("critical_hi")
        work_hi = entry.get("workload_hi")
        if cp_hi is None or work_hi is None:
            cp_hi = compute_critical_path_hi(dag)
            work_hi = compute_workload_hi(dag)

        deadline_result: Optional[Tuple[int, int, float]] = None
        if args.deadline_mode == "tightness":
            low, high = _find_bin(tightness_bin, TIGHTNESS_BINS)
            d_min = int(math.ceil(int(cp_hi) / high))
            d_max = int(math.floor(int(cp_hi) / low))
            if d_min < 1:
                d_min = 1
            if d_min > d_max:
                rejects["no_d_range"] += 1
                continue
            deadline_result = _sample_deadline(
                int(cp_hi),
                int(work_hi),
                tightness_bin,
                rng,
                args.kappa_min,
                args.kappa_max,
                args.deadline_tries,
            )
        else:
            deadline_result = _sample_deadline_by_utilization(
                int(cp_hi),
                int(work_hi),
                rng,
                args.u_hi_min,
                args.u_hi_max,
                args.deadline_tries,
            )
        if deadline_result is None:
            rejects["low_u_hi"] += 1
            continue
        deadline, period, tightness = deadline_result

        _apply_ratio(dag, r_value)
        dag.deadline = int(deadline)
        dag.period = int(period)

        tasks.append(dag)
        total_nodes += len(dag.nodes)
        used_ids.add(int(dag_id))
        task_rows.append(
            {
                "dag_id": dag_id,
                "seed": entry.get("seed"),
                "xml_path": str(xml_path),
                "workload_hi": work_hi,
                "critical_hi": cp_hi,
                "cpr": entry.get("cpr"),
                "rdgen_method": entry.get("rdgen_method"),
                "deadline": deadline,
                "period": period,
                "tightness": round(tightness, 6),
                "nodes": len(dag.nodes),
            }
        )

    if use_nodes_budget:
        if total_nodes < target_nodes:
            return [], {
                "status": "gen_failed",
                "reason": "insufficient_nodes",
                "attempts": attempts,
                "rejects": rejects,
            }
    elif len(tasks) < args.tasks:
        return [], {
            "status": "gen_failed",
            "reason": "insufficient_tasks",
            "attempts": attempts,
            "rejects": rejects,
        }

    task_count = len(tasks)
    hi_count = int(round(task_count * args.hi_critical_ratio))
    hi_count = max(0, min(task_count, hi_count))
    lo_count = task_count - hi_count
    indices = list(range(task_count))
    rng.shuffle(indices)
    hi_indices = set(indices[:hi_count])
    for idx, dag in enumerate(tasks):
        criticality = "HI" if idx in hi_indices else "LO"
        _set_criticality(dag, criticality)
        apply_failure_probs(dag, rng, args.fi, args.fi_mode)
        task_rows[idx]["criticality"] = criticality

    avg_cpr = None
    cprs = [row.get("cpr") for row in task_rows if row.get("cpr") is not None]
    if cprs:
        avg_cpr = float(sum(cprs) / len(cprs))
    avg_tightness = None
    tights = [row.get("tightness") for row in task_rows if row.get("tightness") is not None]
    if tights:
        avg_tightness = float(sum(tights) / len(tights))

    methods = [row.get("rdgen_method") for row in task_rows if row.get("rdgen_method")]
    method_set = sorted({str(m) for m in methods})
    if not method_set:
        set_method = "unknown"
    elif len(method_set) == 1:
        set_method = method_set[0]
    else:
        set_method = "mixed"

    return tasks, {
        "status": "ok",
        "reason": "ok",
        "attempts": attempts,
        "rejects": rejects,
        "task_rows": task_rows,
        "n_tasks": len(tasks),
        "total_nodes": total_nodes,
        "set_method": set_method,
        "hi_count": hi_count,
        "lo_count": lo_count,
        "avg_cpr": avg_cpr,
        "avg_tightness": avg_tightness,
        "avg_r": float(r_value),
        "r_value": float(r_value),
    }


def evaluate_taskset(
    dags: Sequence[DAG],
    args: argparse.Namespace,
    system: SystemModel,
    set_seed: int,
    score_mode: str,
    set_id: Optional[int] = None,
) -> Dict[str, object]:
    mcfq = MCFQAnalyzer()
    clustering_engine = ProposedClusteringEngine(system)

    task_metrics = compute_task_metrics(dags, mcfq)
    federated = None
    federated_status = "error"
    federated_reason = "error"
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
    except Exception:
        federated_status = "error"
        federated_reason = "error"
    finally:
        federated_ms = int((time.monotonic() - federated_start) * 1000)

    nocluster_total = None
    nocluster_status = "ok"
    nocluster_reason = "ok"
    nocluster_ms = None
    low_util_cores = None
    hi_lo_cores = None
    hi_hi_cores = None
    hi_hi_cores_critical = None
    cluster_error_stage = None
    cluster_error_detail = None
    cluster_total = None
    cluster_total_critical = None
    cluster_status = "ok"
    cluster_reason = "ok"
    cluster_ms = None
    cluster_ms_critical = None
    cluster_start = time.monotonic()
    try:
        cluster_error_stage = "split_taskset"
        hi_hi_pairs, hi_lo_dags, low_util_tasks, _ = split_taskset(dags, mcfq)
        hi_hi_dags = [dag for dag, _ in hi_hi_pairs]
        cluster_error_stage = "low_util_partition"
        low_util_cores = mcfq._min_partition_cores_ut_075(
            low_util_tasks,
            args.federated_max_cores,
        )
        hi_lo_cores = 0
        if low_util_cores is None:
            hi_lo_cores = None
        else:
            cluster_error_stage = "hi_lo_multipath"
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
                nocluster_total = low_util_cores + hi_lo_cores + hi_hi_sum
        nocluster_ms = int((time.monotonic() - nocluster_start) * 1000)

        hi_hi_cores = None
        hi_hi_cores_critical = None
        mode_primary = score_mode
        mode_secondary = None
        if score_mode == "both":
            mode_primary = "cores+critical"
            mode_secondary = "critical"
        elif score_mode == "critical":
            mode_secondary = "cores+critical"

        if hi_lo_cores is not None:
            if hi_hi_dags:
                cluster_error_stage = "hi_hi_optimize"
                cluster_start_primary = time.monotonic()
                cluster_result_primary = clustering_engine.optimize_taskset(
                    hi_hi_dags,
                    time_limit=args.cluster_time_limit,
                    max_cores=args.cluster_max_cores,
                    seed=set_seed,
                    temp_start=args.cluster_temp_start,
                    temp_end=args.cluster_temp_end,
                    penalty_weight_start=args.cluster_penalty_start,
                    penalty_weight_end=args.cluster_penalty_end,
                    score_mode=mode_primary,
                    score_epsilon=args.cluster_epsilon,
                    log_accepts=args.cluster_log_accepts,
                    verbose=not args.cluster_quiet,
                )
                primary_ms = int((time.monotonic() - cluster_start_primary) * 1000)
                if mode_primary == "critical":
                    hi_hi_cores_critical = cluster_result_primary.core_sum
                    cluster_ms_critical = primary_ms
                else:
                    hi_hi_cores = cluster_result_primary.core_sum
                    cluster_ms = primary_ms

                if mode_secondary is not None:
                    cluster_start_secondary = time.monotonic()
                    cluster_result_secondary = clustering_engine.optimize_taskset(
                        hi_hi_dags,
                        time_limit=args.cluster_time_limit,
                        max_cores=args.cluster_max_cores,
                        seed=set_seed,
                        temp_start=args.cluster_temp_start,
                        temp_end=args.cluster_temp_end,
                        penalty_weight_start=args.cluster_penalty_start,
                        penalty_weight_end=args.cluster_penalty_end,
                        score_mode=mode_secondary,
                        score_epsilon=args.cluster_epsilon,
                        log_accepts=args.cluster_log_accepts,
                        verbose=not args.cluster_quiet,
                    )
                    secondary_ms = int((time.monotonic() - cluster_start_secondary) * 1000)
                    if mode_secondary == "critical":
                        hi_hi_cores_critical = cluster_result_secondary.core_sum
                        cluster_ms_critical = secondary_ms
                    else:
                        hi_hi_cores = cluster_result_secondary.core_sum
                        cluster_ms = secondary_ms
            else:
                if mode_primary == "critical":
                    hi_hi_cores_critical = 0
                else:
                    hi_hi_cores = 0
                if mode_secondary == "critical":
                    hi_hi_cores_critical = 0
                if mode_secondary == "cores+critical":
                    hi_hi_cores = 0

        if low_util_cores is None or hi_lo_cores is None or hi_hi_cores is None:
            cluster_total = None
        else:
            cluster_error_stage = "combine_totals"
            cluster_total = low_util_cores + hi_lo_cores + hi_hi_cores
        if hi_hi_cores_critical is None:
            cluster_total_critical = None
        elif low_util_cores is None or hi_lo_cores is None:
            cluster_total_critical = None
        else:
            cluster_total_critical = low_util_cores + hi_lo_cores + hi_hi_cores_critical
        if cluster_total is None:
            cluster_status = "unsched"
            if low_util_cores is None:
                cluster_reason = "low_util_partition_fail"
            elif hi_lo_cores is None:
                cluster_reason = "multipath_none"
            else:
                cluster_reason = "cluster_failed"
    except Exception:
        cluster_status = "error"
        cluster_reason = "cluster_error"
        cluster_error_detail = traceback.format_exc().strip()
        label = f"set {set_id}" if set_id is not None else f"seed {set_seed}"
        stage = cluster_error_stage or "unknown"
        print(f"[cluster_error] {label} stage={stage}")
        if cluster_error_detail:
            print(cluster_error_detail)
    finally:
        if cluster_ms is None:
            cluster_ms = int((time.monotonic() - cluster_start) * 1000)

    delta_cores = None
    saving_rate = None
    if federated is not None and cluster_total is not None and federated > 0:
        delta_cores = federated - cluster_total
        saving_rate = delta_cores / federated

    return {
        "m_base": federated,
        "m_base_status": federated_status,
        "m_base_reason": federated_reason,
        "m_base_ms": federated_ms,
        "nocluster_multipath": nocluster_total,
        "nocluster_status": nocluster_status,
        "nocluster_reason": nocluster_reason,
        "nocluster_ms": nocluster_ms,
        "m_ours": cluster_total,
        "m_ours_status": cluster_status,
        "m_ours_reason": cluster_reason,
        "m_ours_ms": cluster_ms,
        "cluster_multipath": cluster_total,
        "cluster_status": cluster_status,
        "cluster_reason": cluster_reason,
        "cluster_ms": cluster_ms,
        "cluster_multipath_critical": cluster_total_critical,
        "cluster_ms_critical": cluster_ms_critical,
        "low_util_cores": low_util_cores,
        "hi_lo_cores": hi_lo_cores,
        "hi_hi_cores": hi_hi_cores,
        "hi_hi_cores_critical": hi_hi_cores_critical,
        "cluster_error_stage": cluster_error_stage,
        "cluster_error_detail": cluster_error_detail,
        "delta_cores": delta_cores,
        "saving_rate": saving_rate,
    }


def load_results(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, object]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get("set_id")
            if raw_id is None:
                continue
            try:
                set_id = int(float(raw_id))
            except ValueError:
                continue
            rows[set_id] = row
    return rows


def save_taskset(taskset: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(taskset, indent=2))


def load_taskset(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def reconstruct_taskset(taskset: Dict[str, object], args: argparse.Namespace) -> List[DAG]:
    wrapper = RDGenWrapper(verbose=False)
    r_value = float(taskset.get("r_value", DEFAULT_R_VALUE))
    dags: List[DAG] = []
    for task in taskset.get("tasks", []):
        xml_path = task.get("xml_path")
        if not xml_path:
            continue
        xml_path = Path(str(xml_path))
        parsed = wrapper.parse_xml(str(xml_path))
        if not parsed:
            continue
        dag = parsed[0]
        _apply_ratio(dag, r_value)
        dag.deadline = int(task.get("deadline") or 0)
        dag.period = int(task.get("period") or 0)
        criticality = task.get("criticality") or "LO"
        _set_criticality(dag, str(criticality))
        apply_failure_probs(dag, random.Random(taskset.get("seed", 0)), args.fi, args.fi_mode)
        dags.append(dag)
    return dags


def _run_eval_single(
    args: argparse.Namespace,
    method: Optional[str],
    score_mode: str,
) -> None:
    if args.tasks < 1:
        raise SystemExit("--tasks must be >= 1.")
    if args.sets_per_bin < 1:
        raise SystemExit("--sets-per-bin must be >= 1.")
    if not 0.0 <= args.hi_critical_ratio <= 1.0:
        raise SystemExit("--hi-critical-ratio must be between 0 and 1.")
    if args.kappa_min <= 0 or args.kappa_max <= 0 or args.kappa_min > args.kappa_max:
        raise SystemExit("--kappa-min/max must be > 0 and min <= max.")
    if args.u_hi_min < 1.0 or args.u_hi_max < 1.0:
        raise SystemExit("--u-hi-min/--u-hi-max must be >= 1.0 for high-util tasks.")
    if args.u_hi_min > args.u_hi_max:
        raise SystemExit("--u-hi-min must be <= --u-hi-max.")
    if args.ratio_min <= 0 or args.ratio_max <= 0 or args.ratio_min > args.ratio_max:
        raise SystemExit("--ratio-min/max must be > 0 and min <= max.")

    pre_run_dir = args.pre_run_dir
    pools = load_roster_pool(pre_run_dir, method)
    if not pools or all(len(v) == 0 for v in pools.values()):
        suffix = f" (method={method})" if method else ""
        raise SystemExit(f"No rosters found in {pre_run_dir}{suffix}")

    run_id = args.run_id if not method else f"{args.run_id}__{method}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    taskset_dir = run_dir / "tasksets"
    plan_path = run_dir / "ex05_1_plan.csv"
    results_path = run_dir / "ex05_1_results.csv"
    set_seeds_path = run_dir / "ex05_1_set_seeds.txt"

    plan_rows = load_plan(plan_path)
    if not plan_rows:
        plan_rows = build_plan(args)
    existing = load_results(results_path)
    if args.fill_to:
        plan_rows = extend_plan_to_target(plan_rows, existing, args)
    write_csv(plan_rows, plan_path)

    total_sets = len(plan_rows)
    if args.set_seeds is not None:
        set_seeds = load_set_seeds(args.set_seeds)
        if len(set_seeds) < total_sets:
            raise SystemExit(
                f"--set-seeds has {len(set_seeds)} entries but needs {total_sets}."
            )
        set_seeds = set_seeds[:total_sets]
        write_set_seeds(set_seeds, set_seeds_path)
    elif set_seeds_path.exists():
        set_seeds = load_set_seeds(set_seeds_path)
        if len(set_seeds) < total_sets:
            extra = [args.seed + idx for idx in range(len(set_seeds), total_sets)]
            set_seeds = list(set_seeds) + extra
        set_seeds = set_seeds[:total_sets]
        write_set_seeds(set_seeds, set_seeds_path)
    else:
        set_seeds = [args.seed + idx for idx in range(total_sets)]
        write_set_seeds(set_seeds, set_seeds_path)

    meta = {
        "run_id": run_id,
        "method": method,
        "axis": args.axis,
        "sets_per_bin": args.sets_per_bin,
        "fill_to": args.fill_to,
        "fill_count": args.fill_count,
        "tasks": args.tasks,
        "taskset_nodes_max": args.taskset_nodes_max,
        "hi_critical_ratio": args.hi_critical_ratio,
        "seed": args.seed,
        "set_seeds_file": str(set_seeds_path),
        "pre_run_dir": str(pre_run_dir),
        "kappa_min": args.kappa_min,
        "kappa_max": args.kappa_max,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
        "deadline_tries": args.deadline_tries,
        "deadline_mode": args.deadline_mode,
        "u_hi_min": args.u_hi_min,
        "u_hi_max": args.u_hi_max,
        "allow_duplicates": args.allow_duplicates,
        "allowable_failure_prob": args.allowable_failure_prob,
        "failure_rate": args.failure_rate,
        "fi": args.fi,
        "fi_mode": args.fi_mode,
        "cluster_time_limit": args.cluster_time_limit,
        "cluster_score": score_mode,
        "cluster_epsilon": args.cluster_epsilon,
        "cluster_temp_start": args.cluster_temp_start,
        "cluster_temp_end": args.cluster_temp_end,
        "cluster_penalty_start": args.cluster_penalty_start,
        "cluster_penalty_end": args.cluster_penalty_end,
        "federated_max_cores": args.federated_max_cores,
    }
    (run_dir / "ex05_1_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        "[config] "
        f"run_id={run_id} method={method} axis={args.axis} sets_per_bin={args.sets_per_bin} "
        f"tasks={args.tasks} taskset_nodes_max={args.taskset_nodes_max} "
        f"hi_critical_ratio={args.hi_critical_ratio:.2f} "
        f"fi={args.fi:.3g} fi_mode={args.fi_mode} "
        f"failure_rate={args.failure_rate:.3g} "
        f"allowable_failure_prob={args.allowable_failure_prob:.3g} "
        f"cluster_score={score_mode} cluster_epsilon={args.cluster_epsilon:g} "
        f"fill_to={args.fill_to} fill_count={args.fill_count} "
        f"deadline_mode={args.deadline_mode} "
        f"u_hi=[{args.u_hi_min:g},{args.u_hi_max:g}] "
        f"kappa=[{args.kappa_min:g},{args.kappa_max:g}] "
        f"ratio=[{args.ratio_min:g},{args.ratio_max:g}] "
        f"results={results_path}"
    )
    existing_failed = {
        set_id for set_id, row in existing.items() if row.get("status_set") == "gen_failed"
    }
    results_by_set = dict(existing)

    system = SystemModel(
        num_cores=args.federated_max_cores,
        allowable_failure_prob=args.allowable_failure_prob,
        failure_rate=args.failure_rate,
    )

    for idx, plan in enumerate(plan_rows, start=1):
        set_id = int(plan.get("set_id") or idx)
        if args.skip_existing and set_id in existing:
            if args.retry_failed and set_id in existing_failed:
                print(f"[set {set_id}] retry (previously gen_failed)")
            else:
                print(f"[set {set_id}] skip (already in ex05_1_results.csv)")
                continue

        set_seed = set_seeds[set_id - 1]
        taskset_path = taskset_dir / f"taskset_{set_id:04d}.json"
        if taskset_path.exists():
            taskset = load_taskset(taskset_path)
            dags = reconstruct_taskset(taskset, args)
            gen_status = taskset.get("status", "ok")
            if args.taskset_nodes_max is not None:
                total_nodes = taskset.get("total_nodes")
                if total_nodes is None:
                    total_nodes = sum(len(dag.nodes) for dag in dags)
                    taskset["total_nodes"] = total_nodes
                if total_nodes < args.taskset_nodes_max:
                    gen_status = "gen_failed"
                    taskset["reason"] = taskset.get("reason") or "taskset_incomplete"
            else:
                if len(dags) < args.tasks:
                    gen_status = "gen_failed"
                    taskset["reason"] = taskset.get("reason") or "taskset_incomplete"
        else:
            wrapper = RDGenWrapper(verbose=False)
            dags, gen_info = generate_taskset(set_id, set_seed, plan, args, pools, wrapper)
            taskset = {
                "set_id": set_id,
                "seed": set_seed,
                "axis": plan.get("axis"),
                "cp_bin": plan.get("cp_bin"),
                "r_value": gen_info.get("r_value"),
                "tightness_bin": plan.get("tightness_bin"),
                "n_tasks": gen_info.get("n_tasks") or len(dags),
                "total_nodes": gen_info.get("total_nodes"),
                "set_method": gen_info.get("set_method"),
                "hi_count": gen_info.get("hi_count"),
                "lo_count": gen_info.get("lo_count"),
                "avg_cp_ratio_hi": gen_info.get("avg_cpr"),
                "avg_tightness": gen_info.get("avg_tightness"),
                "avg_r": gen_info.get("avg_r"),
                "rejects": gen_info.get("rejects"),
                "attempts": gen_info.get("attempts"),
                "status": gen_info.get("status"),
                "reason": gen_info.get("reason"),
                "tasks": gen_info.get("task_rows"),
            }
            save_taskset(taskset, taskset_path)
            gen_status = gen_info.get("status", "gen_failed")

        if gen_status == "ok":
            print(
                f"[taskset {set_id}] "
                f"axis={plan.get('axis')} cp={plan.get('cp_bin')} "
                f"r={taskset.get('r_value')} t={plan.get('tightness_bin')} "
                f"tasks={taskset.get('n_tasks') or len(dags)} "
                f"nodes={taskset.get('total_nodes') or sum(len(d.nodes) for d in dags)} "
                f"avg_cpr={taskset.get('avg_cp_ratio_hi')} "
                f"avg_tight={taskset.get('avg_tightness')} "
                f"method={taskset.get('set_method')}"
            )

        if not dags or gen_status != "ok":
            row = {
                "set_id": set_id,
                "seed": set_seed,
                "axis": plan.get("axis"),
                "cp_bin": plan.get("cp_bin"),
                "r_value": taskset.get("r_value"),
                "tightness_bin": plan.get("tightness_bin"),
                "n_tasks": taskset.get("n_tasks") or args.tasks,
                "total_nodes": taskset.get("total_nodes"),
                "set_method": taskset.get("set_method"),
                "status_set": "gen_failed",
                "reason_set": taskset.get("reason"),
                "attempts": taskset.get("attempts"),
                "reject_no_dag": (taskset.get("rejects") or {}).get("no_dag"),
                "reject_no_d_range": (taskset.get("rejects") or {}).get("no_d_range"),
                "reject_low_u_hi": (taskset.get("rejects") or {}).get("low_u_hi"),
                "avg_cp_ratio_hi": taskset.get("avg_cp_ratio_hi"),
                "avg_tightness": taskset.get("avg_tightness"),
                "avg_r": taskset.get("avg_r"),
            }
            results_by_set[set_id] = row
            write_results(results_by_set, results_path)
            print(f"[set {set_id}] gen_failed")
            continue

        set_start = time.monotonic()
        eval_result = evaluate_taskset(dags, args, system, set_seed, score_mode, set_id)
        wall_time_ms = int((time.monotonic() - set_start) * 1000)
        row = {
            "set_id": set_id,
            "seed": set_seed,
            "axis": plan.get("axis"),
            "cp_bin": plan.get("cp_bin"),
            "r_value": taskset.get("r_value"),
            "tightness_bin": plan.get("tightness_bin"),
            "n_tasks": taskset.get("n_tasks") or args.tasks,
            "total_nodes": taskset.get("total_nodes"),
            "set_method": taskset.get("set_method"),
            "hi_count": taskset.get("hi_count"),
            "lo_count": taskset.get("lo_count"),
            "avg_cp_ratio_hi": taskset.get("avg_cp_ratio_hi"),
            "avg_tightness": taskset.get("avg_tightness"),
            "avg_r": taskset.get("avg_r"),
            "status_set": "ok",
            "reason_set": "ok",
            "attempts": taskset.get("attempts"),
            "reject_no_dag": (taskset.get("rejects") or {}).get("no_dag"),
            "reject_no_d_range": (taskset.get("rejects") or {}).get("no_d_range"),
            "reject_low_u_hi": (taskset.get("rejects") or {}).get("low_u_hi"),
            "wall_time_ms": wall_time_ms,
            **eval_result,
        }
        results_by_set[set_id] = row
        write_results(results_by_set, results_path)
        fed_text = format_core(eval_result.get("m_base"))
        nocluster_text = format_core(eval_result.get("nocluster_multipath"))
        cluster_text = format_core(eval_result.get("cluster_multipath"))
        cluster_text_critical = format_core(eval_result.get("cluster_multipath_critical"))
        print(
            f"[set {set_id}] "
            f"federated_2018={fed_text} "
            f"nocluster_multipath={nocluster_text} "
            f"cluster_multipath (critical path + core)={cluster_text} "
            f"cluster_multipath (critical path)={cluster_text_critical} "
            f"saving={row.get('saving_rate')}"
        )


def _run_eval(args: argparse.Namespace, method: Optional[str]) -> None:
    _run_eval_single(args, method, args.cluster_score)


def main() -> None:
    args = build_parser().parse_args()
    if args.tasks < 1:
        raise SystemExit("--tasks must be >= 1.")
    if args.sets_per_bin < 1:
        raise SystemExit("--sets-per-bin must be >= 1.")
    if not 0.0 <= args.hi_critical_ratio <= 1.0:
        raise SystemExit("--hi-critical-ratio must be between 0 and 1.")
    if args.kappa_min <= 0 or args.kappa_max <= 0 or args.kappa_min > args.kappa_max:
        raise SystemExit("--kappa-min/max must be > 0 and min <= max.")
    if args.u_hi_min < 1.0 or args.u_hi_max < 1.0:
        raise SystemExit("--u-hi-min/--u-hi-max must be >= 1.0 for high-util tasks.")
    if args.u_hi_min > args.u_hi_max:
        raise SystemExit("--u-hi-min must be <= --u-hi-max.")
    if args.ratio_min <= 0 or args.ratio_max <= 0 or args.ratio_min > args.ratio_max:
        raise SystemExit("--ratio-min/max must be > 0 and min <= max.")

    if args.method == "auto":
        for method in ("fan-in", "chain"):
            _run_eval(args, method)
    elif args.method == "mixed":
        _run_eval(args, None)
    else:
        _run_eval(args, args.method)


if __name__ == "__main__":
    main()
