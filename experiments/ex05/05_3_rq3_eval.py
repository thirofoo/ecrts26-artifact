from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

from src.common.system_models import SystemModel

SCORE_CHOICES = [
    "cores",
    "workload",
    "cores+critical",
    "cores+workload",
    "critical",
    "both",
]


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: object) -> Optional[int]:
    fval = _to_float(value)
    if fval is None:
        return None
    return int(fval)


def load_rq1_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("rq1_eval_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_set_ids(text: str) -> List[int]:
    ids: List[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if start <= end else -1
            for value in range(start, end + step, step):
                ids.append(value)
        else:
            ids.append(int(item))
    unique = sorted(set(ids))
    return unique


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_results_by_set(path: Path) -> Dict[int, Dict[str, str]]:
    rows = load_csv_rows(path)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        set_id = _to_int(row.get("set_id"))
        if set_id is None:
            continue
        out[set_id] = row
    return out


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RQ3: Re-evaluate existing RQ1 tasksets with longer clustering time limit."
    )
    parser.add_argument(
        "--rq1-run-dir",
        type=Path,
        required=True,
        help="Path to an existing RQ1 run directory (contains tasksets/ and ex05_1_results.csv).",
    )
    parser.add_argument(
        "--rq1-script",
        type=Path,
        default=Path("experiments/ex05/05_1_rq1_eval.py"),
        help="Path to RQ1 evaluator script used for reconstruction/evaluation functions.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="ex05_3",
        help="Output run ID under --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/ex05_3"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--max-sets",
        type=int,
        default=100,
        help="Maximum number of tasksets to evaluate from source.",
    )
    parser.add_argument(
        "--set-id-start",
        type=int,
        default=1,
        help="Start set_id (inclusive) when selecting tasksets.",
    )
    parser.add_argument(
        "--set-ids",
        type=str,
        default="",
        help="Explicit set IDs (e.g., '1,2,10-20'). If set, --max-sets is ignored.",
    )
    parser.add_argument(
        "--cluster-time-limit",
        type=float,
        default=10.0,
        help="Clustering optimization time limit in seconds for RQ3 run.",
    )
    parser.add_argument(
        "--cluster-score",
        choices=["from-rq1", *SCORE_CHOICES],
        default="from-rq1",
        help="Clustering score mode. 'from-rq1' reads ex05_1_meta.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip set_id already present in ex05_3_results.csv.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-run all selected set_ids.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run rows where status_set is not ok even if skip-existing is enabled.",
    )
    parser.add_argument(
        "--allowable-failure-prob",
        type=float,
        default=None,
        help="Override allowable failure probability (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=None,
        help="Override failure rate lambda (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--fi",
        type=float,
        default=None,
        help="Override per-task/node failure probability (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--fi-mode",
        choices=["per-hour", "per-job"],
        default=None,
        help="Override fi mode (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--federated-max-cores",
        type=int,
        default=None,
        help="Override max cores for federated analyzers.",
    )
    parser.add_argument(
        "--cluster-max-cores",
        type=int,
        default=1024,
        help="Max cores for multipath/cluster bound.",
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=None,
        help="Override cluster epsilon (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--cluster-temp-start",
        type=float,
        default=None,
        help="Override SA start temperature (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--cluster-temp-end",
        type=float,
        default=None,
        help="Override SA end temperature (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--cluster-penalty-start",
        type=float,
        default=None,
        help="Override SA penalty start (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--cluster-penalty-end",
        type=float,
        default=None,
        help="Override SA penalty end (default: source RQ1 meta).",
    )
    parser.add_argument(
        "--cluster-quiet",
        action="store_true",
        default=True,
        help="Suppress clustering optimization logs.",
    )
    parser.add_argument(
        "--no-cluster-quiet",
        action="store_false",
        dest="cluster_quiet",
        help="Enable clustering optimization logs.",
    )
    parser.add_argument(
        "--cluster-log-accepts",
        action="store_true",
        default=False,
        help="Log accepted annealing moves.",
    )
    parser.add_argument(
        "--federated-debug",
        action="store_true",
        default=False,
        help="Enable federated debug logs.",
    )
    parser.add_argument(
        "--federated-debug-every",
        type=int,
        default=10,
        help="Federated debug interval.",
    )
    return parser


def resolve_value(override: Optional[float], meta: Dict[str, object], key: str, default: float) -> float:
    if override is not None:
        return float(override)
    value = _to_float(meta.get(key))
    if value is not None:
        return value
    return float(default)


def resolve_int_value(override: Optional[int], meta: Dict[str, object], key: str, default: int) -> int:
    if override is not None:
        return int(override)
    value = _to_int(meta.get(key))
    if value is not None:
        return value
    return int(default)


def select_set_ids(
    source_rows: Dict[int, Dict[str, str]],
    taskset_dir: Path,
    set_id_start: int,
    max_sets: int,
    explicit_set_ids: Sequence[int],
) -> List[int]:
    if explicit_set_ids:
        return [set_id for set_id in explicit_set_ids if (taskset_dir / f"taskset_{set_id:04d}.json").exists()]

    selected: List[int] = []
    for set_id in sorted(source_rows):
        if set_id < set_id_start:
            continue
        if source_rows[set_id].get("status_set") != "ok":
            continue
        taskset_path = taskset_dir / f"taskset_{set_id:04d}.json"
        if not taskset_path.exists():
            continue
        selected.append(set_id)
        if len(selected) >= max_sets:
            break
    return selected


def parse_score_mode(arg_value: str, meta: Dict[str, object]) -> str:
    if arg_value != "from-rq1":
        return arg_value
    score = str(meta.get("cluster_score") or "both").strip()
    if score not in SCORE_CHOICES:
        return "both"
    return score


def main() -> None:
    args = build_parser().parse_args()
    if args.max_sets < 1:
        raise SystemExit("--max-sets must be >= 1.")
    if args.cluster_time_limit <= 0:
        raise SystemExit("--cluster-time-limit must be > 0.")

    rq1_run_dir = args.rq1_run_dir
    rq1_results_path = rq1_run_dir / "ex05_1_results.csv"
    rq1_meta_path = rq1_run_dir / "ex05_1_meta.json"
    taskset_dir = rq1_run_dir / "tasksets"

    if not taskset_dir.exists():
        raise SystemExit(f"tasksets directory not found: {taskset_dir}")

    source_rows = load_results_by_set(rq1_results_path)
    if not source_rows:
        raise SystemExit(f"No source rows found: {rq1_results_path}")

    source_meta: Dict[str, object] = {}
    if rq1_meta_path.exists():
        source_meta = json.loads(rq1_meta_path.read_text())

    rq1_module = load_rq1_module(args.rq1_script)

    score_mode = parse_score_mode(args.cluster_score, source_meta)
    allowable_failure_prob = resolve_value(
        args.allowable_failure_prob,
        source_meta,
        "allowable_failure_prob",
        1e-9,
    )
    failure_rate = resolve_value(args.failure_rate, source_meta, "failure_rate", 1e-9)
    fi_value = resolve_value(args.fi, source_meta, "fi", 1e-6)
    fi_mode = args.fi_mode or str(source_meta.get("fi_mode") or "per-hour")
    if fi_mode not in {"per-hour", "per-job"}:
        raise SystemExit(f"Invalid fi_mode: {fi_mode}")

    eval_args = SimpleNamespace(
        fi=fi_value,
        fi_mode=fi_mode,
        federated_max_cores=resolve_int_value(
            args.federated_max_cores,
            source_meta,
            "federated_max_cores",
            1024,
        ),
        federated_debug=args.federated_debug,
        federated_debug_every=args.federated_debug_every,
        cluster_max_cores=int(args.cluster_max_cores),
        cluster_time_limit=float(args.cluster_time_limit),
        cluster_epsilon=resolve_value(args.cluster_epsilon, source_meta, "cluster_epsilon", 1e-4),
        cluster_temp_start=resolve_value(args.cluster_temp_start, source_meta, "cluster_temp_start", 1e-2),
        cluster_temp_end=resolve_value(args.cluster_temp_end, source_meta, "cluster_temp_end", 1e-5),
        cluster_penalty_start=resolve_value(
            args.cluster_penalty_start,
            source_meta,
            "cluster_penalty_start",
            0.5,
        ),
        cluster_penalty_end=resolve_value(
            args.cluster_penalty_end,
            source_meta,
            "cluster_penalty_end",
            5.0,
        ),
        cluster_log_accepts=args.cluster_log_accepts,
        cluster_quiet=args.cluster_quiet,
    )

    explicit_set_ids = parse_set_ids(args.set_ids) if args.set_ids.strip() else []
    selected_set_ids = select_set_ids(
        source_rows,
        taskset_dir,
        args.set_id_start,
        args.max_sets,
        explicit_set_ids,
    )
    if not selected_set_ids:
        raise SystemExit("No tasksets selected. Check --rq1-run-dir / --set-ids.")

    out_run_dir = args.output_dir / args.run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_results_path = out_run_dir / "ex05_3_results.csv"
    existing_out = load_results_by_set(out_results_path)

    meta = {
        "run_id": args.run_id,
        "source_rq1_run_dir": str(rq1_run_dir),
        "source_rq1_results": str(rq1_results_path),
        "source_rq1_meta": str(rq1_meta_path),
        "source_rq1_script": str(args.rq1_script),
        "max_sets": args.max_sets,
        "set_id_start": args.set_id_start,
        "explicit_set_ids": explicit_set_ids,
        "selected_set_count": len(selected_set_ids),
        "selected_set_ids": selected_set_ids,
        "cluster_time_limit": eval_args.cluster_time_limit,
        "cluster_score": score_mode,
        "allowable_failure_prob": allowable_failure_prob,
        "failure_rate": failure_rate,
        "fi": fi_value,
        "fi_mode": fi_mode,
        "federated_max_cores": eval_args.federated_max_cores,
        "cluster_max_cores": eval_args.cluster_max_cores,
        "cluster_epsilon": eval_args.cluster_epsilon,
        "cluster_temp_start": eval_args.cluster_temp_start,
        "cluster_temp_end": eval_args.cluster_temp_end,
        "cluster_penalty_start": eval_args.cluster_penalty_start,
        "cluster_penalty_end": eval_args.cluster_penalty_end,
    }
    (out_run_dir / "ex05_3_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        "[config] "
        f"source={rq1_run_dir} selected_sets={len(selected_set_ids)} "
        f"time_limit={eval_args.cluster_time_limit}s score={score_mode} "
        f"fi={fi_value:.3g} fi_mode={fi_mode} "
        f"F_S={allowable_failure_prob:.3g} failure_rate={failure_rate:.3g} "
        f"results={out_results_path}"
    )

    system = SystemModel(
        num_cores=eval_args.federated_max_cores,
        allowable_failure_prob=allowable_failure_prob,
        failure_rate=failure_rate,
    )

    results_by_set: Dict[int, Dict[str, object]] = {k: dict(v) for k, v in existing_out.items()}

    for set_id in selected_set_ids:
        if args.skip_existing and set_id in existing_out:
            status = str(existing_out[set_id].get("status_set") or "")
            if args.retry_failed and status != "ok":
                print(f"[set {set_id}] retry (previous status={status})")
            else:
                print(f"[set {set_id}] skip (already exists)")
                continue

        source_row = source_rows.get(set_id, {})
        taskset_path = taskset_dir / f"taskset_{set_id:04d}.json"
        if not taskset_path.exists():
            row = {
                "set_id": set_id,
                "status_set": "gen_failed",
                "reason_set": "missing_taskset_file",
                "source_status_set": source_row.get("status_set"),
            }
            results_by_set[set_id] = row
            write_csv([results_by_set[k] for k in sorted(results_by_set)], out_results_path)
            print(f"[set {set_id}] gen_failed (missing taskset)")
            continue

        taskset = rq1_module.load_taskset(taskset_path)
        dags = rq1_module.reconstruct_taskset(taskset, eval_args)
        if not dags:
            row = {
                "set_id": set_id,
                "seed": taskset.get("seed"),
                "status_set": "gen_failed",
                "reason_set": "reconstruct_failed",
                "source_status_set": source_row.get("status_set"),
            }
            results_by_set[set_id] = row
            write_csv([results_by_set[k] for k in sorted(results_by_set)], out_results_path)
            print(f"[set {set_id}] gen_failed (reconstruct)")
            continue

        set_seed = _to_int(taskset.get("seed"))
        if set_seed is None:
            set_seed = _to_int(source_row.get("seed")) or set_id

        start = time.monotonic()
        eval_result = rq1_module.evaluate_taskset(
            dags,
            eval_args,
            system,
            set_seed,
            score_mode,
            set_id,
        )
        wall_time_ms = int((time.monotonic() - start) * 1000)

        src_cluster = _to_int(source_row.get("cluster_multipath"))
        src_cluster_critical = _to_int(source_row.get("cluster_multipath_critical"))
        new_cluster = _to_int(eval_result.get("cluster_multipath"))
        new_cluster_critical = _to_int(eval_result.get("cluster_multipath_critical"))

        improve = None
        if src_cluster is not None and new_cluster is not None:
            improve = src_cluster - new_cluster
        improve_critical = None
        if src_cluster_critical is not None and new_cluster_critical is not None:
            improve_critical = src_cluster_critical - new_cluster_critical

        row = {
            "set_id": set_id,
            "seed": set_seed,
            "axis": source_row.get("axis") or taskset.get("axis"),
            "cp_bin": source_row.get("cp_bin") or taskset.get("cp_bin"),
            "r_value": source_row.get("r_value") or taskset.get("r_value"),
            "tightness_bin": source_row.get("tightness_bin") or taskset.get("tightness_bin"),
            "n_tasks": source_row.get("n_tasks") or taskset.get("n_tasks") or len(dags),
            "total_nodes": source_row.get("total_nodes") or taskset.get("total_nodes"),
            "set_method": source_row.get("set_method") or taskset.get("set_method"),
            "status_set": "ok",
            "reason_set": "ok",
            "source_status_set": source_row.get("status_set"),
            "source_cluster_multipath": source_row.get("cluster_multipath"),
            "source_cluster_multipath_critical": source_row.get("cluster_multipath_critical"),
            "source_cluster_time_limit": source_meta.get("cluster_time_limit"),
            "rq3_cluster_time_limit": eval_args.cluster_time_limit,
            "cluster_score": score_mode,
            "delta_source_minus_rq3_cluster_multipath": improve,
            "delta_source_minus_rq3_cluster_multipath_critical": improve_critical,
            "wall_time_ms": wall_time_ms,
            **eval_result,
        }
        results_by_set[set_id] = row
        write_csv([results_by_set[k] for k in sorted(results_by_set)], out_results_path)

        fed_text = rq1_module.format_core(eval_result.get("m_base"))
        nocluster_text = rq1_module.format_core(eval_result.get("nocluster_multipath"))
        cluster_text = rq1_module.format_core(eval_result.get("cluster_multipath"))
        critical_text = rq1_module.format_core(eval_result.get("cluster_multipath_critical"))
        print(
            f"[set {set_id}] federated_2018={fed_text} "
            f"nocluster_multipath={nocluster_text} "
            f"cluster_multipath (critical path + core)={cluster_text} "
            f"cluster_multipath (critical path)={critical_text} "
            f"delta_vs_source={improve}"
        )


if __name__ == "__main__":
    main()
