from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

FONT_SCALE = 1.3
plt.rcParams.update(
    {
        "font.size": 10 * FONT_SCALE,
        "axes.labelsize": 10 * FONT_SCALE,
        "xtick.labelsize": 9 * FONT_SCALE,
        "ytick.labelsize": 9 * FONT_SCALE,
        "legend.fontsize": 9 * FONT_SCALE,
    }
)

AXIS_CONFIG = {
    "cp": {
        "label": "CPR bin",
        "field": "cp_bin",
        "order": [
            "BIN_CP_1",
            "BIN_CP_2",
            "BIN_CP_3",
            "BIN_CP_4",
            "BIN_CP_5",
            "BIN_CP_6",
        ],
        "display_map": {
            "BIN_CP_1": "[0.1, 0.2)",
            "BIN_CP_2": "[0.2, 0.3)",
            "BIN_CP_3": "[0.3, 0.4)",
            "BIN_CP_4": "[0.4, 0.5)",
            "BIN_CP_5": "[0.5, 0.6)",
            "BIN_CP_6": "[0.6, 0.7)",
        },
        "prefix": "",
    },
    "ratio": {
        "label": "C^HI/C^LO ratio",
        "field": "r_value",
        "order": ["2", "4", "8"],
        "prefix": "r=",
    },
    "tightness": {
        "label": "Tightness bin",
        "field": "tightness_bin",
        "order": ["BIN_T_1", "BIN_T_2", "BIN_T_3"],
        "prefix": "",
    },
}
METRIC_CONFIG = {
    "saving_rate": {
        "label": "saving_rate (Fed-2018 vs Proposed)",
        "kind": "single",
    },
    "saving_rate_nocluster": {
        "label": "saving_rate (NoCluster vs Proposed)",
        "kind": "single",
    },
    "delta_cores": {
        "label": "delta_cores (Fed-2018 - Proposed)",
        "kind": "single",
    },
    "delta_cores_nocluster": {
        "label": "delta_cores (NoCluster - Proposed)",
        "kind": "single",
    },
    "cores": {
        "label": "required cores",
        "kind": "grouped",
    },
    "fail_counts": {
        "label": "failure counts (missing/unsched)",
        "kind": "counts",
    },
    "avg_utilization": {
        "label": "avg utilization (LO/HI)",
        "kind": "lohi",
    },
}

METHOD_LABELS = {
    "fed2018": "Fed-2018",
    "nocluster": "NoCluster",
    "ours": "Proposed",
    "critical": "Proposed (critical)",
}


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RQ1 boxplots from ex05_1 results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/ex05_1"),
        help="ex05_1_results.csv or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store plots (default: <run_dir>/plots).",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="cp,ratio,tightness",
        help="Comma-separated axes to plot: cp,ratio,tightness.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="saving_rate,cores",
        help=(
            "Comma-separated metrics: saving_rate, saving_rate_nocluster, "
            "delta_cores, delta_cores_nocluster, cores, fail_counts."
        ),
    )
    parser.add_argument(
        "--whis",
        type=str,
        default="minmax",
        help="Whisker length (e.g., 1.5) or 'minmax' to extend to min/max.",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        default=False,
        help="Show outliers (fliers) beyond whiskers.",
    )
    parser.add_argument(
        "--no-show-fliers",
        action="store_false",
        dest="show_fliers",
        help="Hide outliers (fliers).",
    )
    parser.add_argument(
        "--show-means",
        action="store_true",
        default=True,
        help="Show mean markers on boxplots.",
    )
    parser.add_argument(
        "--no-show-means",
        action="store_false",
        dest="show_means",
        help="Hide mean markers on boxplots.",
    )
    parser.add_argument(
        "--y-scale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale (log is useful when outliers dominate).",
    )
    parser.add_argument(
        "--require-ok",
        action="store_true",
        default=True,
        help="Use only rows where set/base/ours status are ok.",
    )
    parser.add_argument(
        "--no-require-ok",
        action="store_false",
        dest="require_ok",
        help="Include rows even if status is not ok.",
    )
    parser.add_argument(
        "--split-by-method",
        action="store_true",
        default=True,
        help="Also output separate plots per set_method (default: on).",
    )
    parser.add_argument(
        "--no-split-by-method",
        action="store_false",
        dest="split_by_method",
        help="Disable per-method split outputs.",
    )
    parser.add_argument(
        "--exclude-bins",
        type=str,
        default="",
        help="Comma-separated bin labels to exclude (e.g., BIN_CP_6).",
    )
    parser.add_argument(
        "--include-critical",
        action="store_true",
        default=False,
        help="Include critical-path-only results in core boxplots.",
    )
    parser.add_argument(
        "--no-include-critical",
        action="store_false",
        dest="include_critical",
        help="Exclude critical-path-only results from core boxplots.",
    )
    return parser


def resolve_results_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    candidate = input_path / "ex05_1_results.csv"
    if candidate.exists():
        return candidate
    matches = list(input_path.rglob("ex05_1_results.csv"))
    if not matches:
        raise FileNotFoundError(f"ex05_1_results.csv not found under {input_path}")
    if len(matches) > 1:
        names = "\n".join(str(path) for path in matches)
        raise FileNotFoundError(
            "Multiple ex05_1_results.csv found. Pass --input with the file path:\n"
            f"{names}"
        )
    return matches[0]


def parse_axes(value: str) -> List[str]:
    axes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [axis for axis in axes if axis not in AXIS_CONFIG]
    if unknown:
        raise ValueError(f"Unknown axes: {', '.join(unknown)}")
    return axes


def parse_metrics(value: str) -> List[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [metric for metric in metrics if metric not in METRIC_CONFIG]
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}")
    return metrics


def parse_whis(value: str) -> float | Tuple[float, float]:
    text = value.strip().lower()
    if text in {"minmax", "min-max", "full", "0-100", "0,100"}:
        return (0.0, 100.0)
    return float(text)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    return float(text)




def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _compute_taskset_util(taskset: Dict[str, object], fallback_r: Optional[float]) -> Optional[Tuple[float, float]]:
    tasks = taskset.get("tasks") or []
    if not tasks:
        return None
    r_value = _to_float(taskset.get("r_value"))
    if r_value is None or r_value <= 0:
        r_value = fallback_r
    if r_value is None or r_value <= 0:
        r_value = 1.0

    lo_vals: List[float] = []
    hi_vals: List[float] = []
    for task in tasks:
        work_hi = _to_float(task.get("workload_hi"))
        period = _to_float(task.get("period")) or _to_float(task.get("deadline"))
        if work_hi is None or period in (None, 0.0):
            continue
        hi_vals.append(work_hi / period)
        lo_vals.append((work_hi / r_value) / period)

    if not hi_vals:
        return None
    return (float(sum(lo_vals) / len(lo_vals)), float(sum(hi_vals) / len(hi_vals)))


def load_taskset_utilization(
    rows: Sequence[Dict[str, str]],
    run_dir: Path,
) -> Dict[int, Tuple[float, float]]:
    taskset_dir = run_dir / "tasksets"
    util_map: Dict[int, Tuple[float, float]] = {}
    missing = 0
    for row in rows:
        set_id = _to_int(row.get("set_id"))
        if set_id is None or set_id in util_map:
            continue
        taskset_path = taskset_dir / f"taskset_{set_id:04d}.json"
        if not taskset_path.exists():
            missing += 1
            continue
        try:
            taskset = json.loads(taskset_path.read_text())
        except json.JSONDecodeError:
            continue
        util = _compute_taskset_util(taskset, _to_float(row.get("r_value")))
        if util is None:
            continue
        util_map[set_id] = util
    if missing:
        print(f"[warn] tasksets missing: {missing}")
    return util_map


def extract_utilization_series(
    rows: Sequence[Dict[str, str]],
    axis: str,
    util_map: Dict[int, Tuple[float, float]],
) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    grouped_lo: Dict[str, List[float]] = {}
    grouped_hi: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("axis") != axis:
            continue
        set_id = _to_int(row.get("set_id"))
        if set_id is None:
            continue
        util = util_map.get(set_id)
        if util is None:
            continue
        key = str(row.get(AXIS_CONFIG[axis]["field"], "")).strip()
        if not key:
            continue
        grouped_lo.setdefault(key, []).append(util[0])
        grouped_hi.setdefault(key, []).append(util[1])

    bins = order_bins(set(grouped_lo.keys()) | set(grouped_hi.keys()), axis)
    lo_data = [grouped_lo.get(label, []) for label in bins]
    hi_data = [grouped_hi.get(label, []) for label in bins]
    return bins, lo_data, hi_data


def make_labels_with_counts(bins: Sequence[str], counts: Sequence[int], axis: str) -> List[str]:
    prefix = AXIS_CONFIG[axis]["prefix"]
    display_map = AXIS_CONFIG[axis].get("display_map", {})
    labels: List[str] = []
    for label, count in zip(bins, counts):
        display_label = display_map.get(label, label)
        name = f"{prefix}{display_label}" if prefix else display_label
        labels.append(f"{name}\n(n={count})")
    return labels


def plot_utilization_boxplot(
    bins: Sequence[str],
    lo_data: Sequence[Sequence[float]],
    hi_data: Sequence[Sequence[float]],
    axis: str,
    y_scale: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    out_dir: Path,
) -> Optional[Path]:
    entries = [
        (label, list(lo_vals), list(hi_vals))
        for label, lo_vals, hi_vals in zip(bins, lo_data, hi_data)
        if lo_vals or hi_vals
    ]
    if not entries:
        return None

    bins_f = [label for label, _, _ in entries]
    lo_vals_f = [vals for _, vals, _ in entries]
    hi_vals_f = [vals for _, _, vals in entries]

    positions = [float(i + 1) for i in range(len(bins_f))]
    width = 0.32
    offsets = 0.18
    lo_pos = [p - offsets for p in positions]
    hi_pos = [p + offsets for p in positions]

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    if any(lo_vals_f):
        lo_bp = ax.boxplot(
            lo_vals_f,
            positions=lo_pos,
            widths=width,
            patch_artist=True,
            showmeans=show_means,
            showfliers=show_fliers,
            whis=whis,
        )
        for patch in lo_bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
    if any(hi_vals_f):
        hi_bp = ax.boxplot(
            hi_vals_f,
            positions=hi_pos,
            widths=width,
            patch_artist=True,
            showmeans=show_means,
            showfliers=show_fliers,
            whis=whis,
        )
        for patch in hi_bp["boxes"]:
            patch.set_facecolor("#DD8452")
            patch.set_alpha(0.6)

    counts = [max(len(lo), len(hi)) for lo, hi in zip(lo_vals_f, hi_vals_f)]
    labels = make_labels_with_counts(bins_f, counts, axis)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average utilization")
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_title(f"RQ1 avg utilization by {axis} (LO vs HI)")
    if not _apply_y_scale(ax, _flatten([lo_vals_f, hi_vals_f]), y_scale):
        print(f"[warn] y-scale=log skipped for avg_utilization ({axis})")
    ax.legend(
        handles=[
            Patch(facecolor="#4C72B0", label="LO", alpha=0.6),
            Patch(facecolor="#DD8452", label="HI", alpha=0.6),
            *(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="none",
                        markerfacecolor="#2CA02C",
                        markersize=7,
                        label="mean",
                    )
                ]
                if show_means
                else []
            ),
        ],
        loc="best",
        frameon=False,
    )
    fig.tight_layout()
    out_path = out_dir / f"rq1_avg_utilization_{axis}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path



def load_rows(path: Path, require_ok: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if require_ok:
                if row.get("status_set") != "ok":
                    continue
            rows.append(row)
    return rows


def _row_ok(
    row: Dict[str, str],
    fields: Sequence[Tuple[str, str]],
    require_ok: bool,
) -> bool:
    for value_field, status_field in fields:
        if require_ok and status_field and row.get(status_field) != "ok":
            return False
        if _to_float(row.get(value_field)) is None:
            return False
    return True


def order_bins(values: Iterable[str], axis: str) -> List[str]:
    config = AXIS_CONFIG[axis]
    order = list(config["order"])
    existing = set(values)
    ordered = [label for label in order if label in existing]
    remaining = sorted(existing - set(ordered))
    return ordered + remaining


def extract_series(
    rows: Sequence[Dict[str, str]], axis: str
) -> Tuple[List[str], List[List[float]]]:
    config = AXIS_CONFIG[axis]
    field = config["field"]
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("axis") != axis:
            continue
        key = str(row.get(field, "")).strip()
        if not key:
            continue
        value = _to_float(row.get("saving_rate"))
        if value is None:
            continue
        grouped.setdefault(key, []).append(value)
    bins = order_bins(grouped.keys(), axis)
    data = [grouped[label] for label in bins]
    return bins, data


def _flatten(values: Sequence[object]) -> List[float]:
    flat: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
            continue
        if item is None:
            continue
        try:
            flat.append(float(item))
        except (TypeError, ValueError):
            continue
    return flat


def _filter_bins(bins: Sequence[str], exclude: set[str]) -> Tuple[List[str], List[int]]:
    keep = [i for i, label in enumerate(bins) if label not in exclude]
    filtered = [bins[i] for i in keep]
    return filtered, keep


def _filter_data_by_idx(
    data: Sequence[Sequence[float]],
    keep: Sequence[int],
) -> List[List[float]]:
    return [list(data[i]) for i in keep]


def _filter_series_list(
    series: Sequence[Tuple[str, Sequence[Sequence[float]]]],
    keep: Sequence[int],
) -> List[Tuple[str, List[List[float]]]]:
    filtered: List[Tuple[str, List[List[float]]]] = []
    for name, data in series:
        filtered.append((name, _filter_data_by_idx(data, keep)))
    return filtered


def _filter_series_dict(
    series: Dict[str, Sequence[float]],
    keep: Sequence[int],
) -> Dict[str, List[float]]:
    return {key: [values[i] for i in keep] for key, values in series.items()}


def _parse_exclude_bins(raw: str) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _apply_y_scale(ax: plt.Axes, values: Sequence[float], y_scale: str) -> bool:
    if y_scale != "log":
        return True
    positives = [value for value in values if value > 0.0]
    if not positives:
        return False
    ax.set_yscale("log")
    ticks = _log_ticks(positives)
    if ticks:
        ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
    return True


def _log_ticks(values: Sequence[float]) -> List[float]:
    positives = [value for value in values if value > 0.0]
    if not positives:
        return []
    vmin = min(positives)
    vmax = max(positives)
    if vmin == vmax:
        return [vmin]
    exp_min = int(math.floor(math.log10(vmin)))
    exp_max = int(math.ceil(math.log10(vmax)))
    candidates: List[float] = []
    for exp in range(exp_min - 1, exp_max + 2):
        for base in (1.0, 2.0, 5.0):
            candidates.append(base * (10 ** exp))
    candidates = sorted(set(candidates))
    ticks = [tick for tick in candidates if vmin <= tick <= vmax]
    if len(ticks) < 2:
        below = [tick for tick in candidates if tick < vmin]
        above = [tick for tick in candidates if tick > vmax]
        if below:
            ticks.append(max(below))
        if above:
            ticks.append(min(above))
        ticks = sorted(set(ticks))
    return ticks


def extract_metric_series(
    rows: Sequence[Dict[str, str]],
    axis: str,
    metric: str,
    require_ok: bool,
) -> Tuple[List[str], List[List[float]]]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("axis") != axis:
            continue
        key = str(row.get(AXIS_CONFIG[axis]["field"], "")).strip()
        if not key:
            continue
        if metric == "saving_rate":
            if not _row_ok(
                row, [("m_base", "m_base_status"), ("m_ours", "m_ours_status")], require_ok
            ):
                continue
            value = _to_float(row.get("saving_rate"))
        elif metric == "saving_rate_nocluster":
            if not _row_ok(
                row,
                [("nocluster_multipath", "nocluster_status"), ("m_ours", "m_ours_status")],
                require_ok,
            ):
                continue
            nocluster = _to_float(row.get("nocluster_multipath"))
            ours = _to_float(row.get("m_ours"))
            if nocluster in (None, 0.0) or ours is None:
                continue
            value = (nocluster - ours) / nocluster
        elif metric == "delta_cores":
            if not _row_ok(
                row, [("m_base", "m_base_status"), ("m_ours", "m_ours_status")], require_ok
            ):
                continue
            base = _to_float(row.get("m_base"))
            ours = _to_float(row.get("m_ours"))
            if base is None or ours is None:
                continue
            value = base - ours
        elif metric == "delta_cores_nocluster":
            if not _row_ok(
                row,
                [("nocluster_multipath", "nocluster_status"), ("m_ours", "m_ours_status")],
                require_ok,
            ):
                continue
            nocluster = _to_float(row.get("nocluster_multipath"))
            ours = _to_float(row.get("m_ours"))
            if nocluster is None or ours is None:
                continue
            value = nocluster - ours
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if value is None:
            continue
        grouped.setdefault(key, []).append(float(value))
    bins = order_bins(grouped.keys(), axis)
    data = [grouped[label] for label in bins]
    return bins, data


def extract_core_series(
    rows: Sequence[Dict[str, str]],
    axis: str,
    require_ok: bool,
    include_critical: bool,
) -> Tuple[List[str], List[Tuple[str, List[List[float]]]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if row.get("axis") != axis:
            continue
        key = str(row.get(AXIS_CONFIG[axis]["field"], "")).strip()
        if not key:
            continue
        if not _row_ok(
            row,
            [
                ("m_base", "m_base_status"),
                ("nocluster_multipath", "nocluster_status"),
                ("m_ours", "m_ours_status"),
            ],
            require_ok,
        ):
            continue
        base = _to_float(row.get("m_base"))
        nocluster = _to_float(row.get("nocluster_multipath"))
        ours = _to_float(row.get("m_ours"))
        if base is None or nocluster is None or ours is None:
            continue
        grouped.setdefault(key, {"fed2018": [], "nocluster": [], "ours": [], "critical": []})
        grouped[key]["fed2018"].append(base)
        grouped[key]["nocluster"].append(nocluster)
        grouped[key]["ours"].append(ours)
        if include_critical:
            critical = _to_float(row.get("cluster_multipath_critical"))
            if critical is not None:
                grouped[key]["critical"].append(critical)
    bins = order_bins(grouped.keys(), axis)
    series = [
        ("fed2018", [grouped[label]["fed2018"] for label in bins]),
        ("nocluster", [grouped[label]["nocluster"] for label in bins]),
        ("ours", [grouped[label]["ours"] for label in bins]),
    ]
    if include_critical:
        series.append(("critical", [grouped[label]["critical"] for label in bins]))
    return bins, series


def extract_fail_counts(
    rows: Sequence[Dict[str, str]],
    axis: str,
) -> Tuple[List[str], Dict[str, List[int]]]:
    grouped: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if row.get("axis") != axis:
            continue
        key = str(row.get(AXIS_CONFIG[axis]["field"], "")).strip()
        if not key:
            continue
        grouped.setdefault(key, {"fed2018": 0, "nocluster": 0, "ours": 0})
        for name, value_field, status_field in (
            ("fed2018", "m_base", "m_base_status"),
            ("nocluster", "nocluster_multipath", "nocluster_status"),
            ("ours", "m_ours", "m_ours_status"),
        ):
            status_ok = row.get(status_field) == "ok"
            value = _to_float(row.get(value_field))
            if (not status_ok) or value is None:
                grouped[key][name] += 1
    bins = order_bins(grouped.keys(), axis)
    series = {
        "fed2018": [grouped[label]["fed2018"] for label in bins],
        "nocluster": [grouped[label]["nocluster"] for label in bins],
        "ours": [grouped[label]["ours"] for label in bins],
    }
    return bins, series


def make_labels(bins: Sequence[str], data: Sequence[Sequence[float]], axis: str) -> List[str]:
    prefix = AXIS_CONFIG[axis]["prefix"]
    display_map = AXIS_CONFIG[axis].get("display_map", {})
    labels: List[str] = []
    for label, values in zip(bins, data):
        display_label = display_map.get(label, label)
        name = f"{prefix}{display_label}" if prefix else display_label
        labels.append(f"{name}\n(n={len(values)})")
    return labels


def plot_boxplot(
    bins: Sequence[str],
    data: Sequence[Sequence[float]],
    axis: str,
    metric: str,
    y_scale: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    out_dir: Path,
) -> Optional[Path]:
    if not data:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = make_labels(bins, data, axis)
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.boxplot(
        data,
        labels=labels,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
    )
    if y_scale != "log":
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_ylabel(METRIC_CONFIG[metric]["label"])
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_title(None)
    if not _apply_y_scale(ax, _flatten(data), y_scale):
        print(f"[warn] y-scale=log skipped for {metric} (non-positive values).")
    fig.tight_layout()
    out_path = out_dir / f"rq1_{metric}_{axis}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_grouped_cores(
    bins: Sequence[str],
    series: Sequence[Tuple[str, List[List[float]]]],
    axis: str,
    y_scale: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    out_dir: Path,
) -> Optional[Path]:
    if not bins or not series:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = [float(i + 1) for i in range(len(bins))]
    width = 0.22
    colors = {
        "fed2018": "#4C72B0",
        "nocluster": "#55A868",
        "ours": "#C44E52",
        "critical": "#8172B2",
    }
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    for idx, (name, data) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        pos = [p + offset for p in positions]
        ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showmeans=show_means,
            showfliers=show_fliers,
            whis=whis,
            boxprops={"facecolor": colors.get(name, "#999999"), "alpha": 0.5},
            medianprops={"color": "#333333"},
        )
    labels = make_labels(bins, series[0][1], axis)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(METRIC_CONFIG["cores"]["label"])
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_title(None)
    if not _apply_y_scale(ax, _flatten([values for _, values in series]), y_scale):
        print("[warn] y-scale=log skipped for cores (non-positive values).")
    legend_handles = [
        Patch(
            facecolor=colors.get(name, "#999999"),
            label=METHOD_LABELS.get(name, name),
            alpha=0.5,
        )
        for name, _ in series
    ]
    if show_means:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#2CA02C",
                markersize=7,
                label="mean",
            )
        )
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    out_path = out_dir / f"rq1_cores_{axis}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fail_counts(
    bins: Sequence[str],
    series: Dict[str, List[int]],
    axis: str,
    out_dir: Path,
) -> Optional[Path]:
    if not bins:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = [float(i + 1) for i in range(len(bins))]
    width = 0.22
    colors = {
        "fed2018": "#4C72B0",
        "nocluster": "#55A868",
        "ours": "#C44E52",
    }
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    for idx, name in enumerate(("fed2018", "nocluster", "ours")):
        offset = (idx - 1) * width
        pos = [p + offset for p in positions]
        ax.bar(
            pos,
            series.get(name, []),
            width=width * 0.9,
            color=colors.get(name, "#999999"),
            alpha=0.7,
            label=METHOD_LABELS.get(name, name),
        )
    labels = make_labels(bins, [series.get("fed2018", [])], axis)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("failed cases")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_title(None)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / f"rq1_fail_counts_{axis}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_path = resolve_results_path(args.input)
    run_dir = results_path.parent
    output_dir = args.output_dir or (run_dir / "plots")

    rows = load_rows(results_path, args.require_ok)
    if not rows:
        raise SystemExit("No rows found to plot (check --require-ok or input path).")

    axes = parse_axes(args.axes)
    metrics = parse_metrics(args.metrics)
    whis = parse_whis(args.whis)
    util_map: Dict[int, Tuple[float, float]] = {}
    if "avg_utilization" in metrics:
        util_map = load_taskset_utilization(rows, run_dir)

    def render(rows_subset: Sequence[Dict[str, str]], out_dir: Path) -> None:
        exclude_bins = _parse_exclude_bins(args.exclude_bins)
        for axis in axes:
            for metric in metrics:
                if metric == "avg_utilization":
                    bins, lo_data, hi_data = extract_utilization_series(
                        rows_subset, axis, util_map
                    )
                    if exclude_bins:
                        bins, keep = _filter_bins(bins, exclude_bins)
                        lo_data = _filter_data_by_idx(lo_data, keep)
                        hi_data = _filter_data_by_idx(hi_data, keep)
                    plot_utilization_boxplot(
                        bins,
                        lo_data,
                        hi_data,
                        axis,
                        args.y_scale,
                        whis,
                        args.show_fliers,
                        args.show_means,
                        out_dir,
                    )
                    continue
                if METRIC_CONFIG[metric]["kind"] == "grouped":
                    bins, series = extract_core_series(
                        rows_subset,
                        axis,
                        args.require_ok,
                        args.include_critical,
                    )
                    if exclude_bins:
                        bins, keep = _filter_bins(bins, exclude_bins)
                        series = _filter_series_list(series, keep)
                    plot_grouped_cores(
                        bins,
                        series,
                        axis,
                        args.y_scale,
                        whis,
                        args.show_fliers,
                        args.show_means,
                        out_dir,
                    )
                    continue
                if METRIC_CONFIG[metric]["kind"] == "counts":
                    bins, series = extract_fail_counts(rows_subset, axis)
                    if exclude_bins:
                        bins, keep = _filter_bins(bins, exclude_bins)
                        series = _filter_series_dict(series, keep)
                    plot_fail_counts(bins, series, axis, out_dir)
                    continue
                bins, data = extract_metric_series(rows_subset, axis, metric, args.require_ok)
                if exclude_bins:
                    bins, keep = _filter_bins(bins, exclude_bins)
                    data = _filter_data_by_idx(data, keep)
                plot_boxplot(
                    bins,
                    data,
                    axis,
                    metric,
                    args.y_scale,
                    whis,
                    args.show_fliers,
                    args.show_means,
                    out_dir,
                )

    render(rows, output_dir)

    if args.split_by_method:
        methods = sorted(
            {
                str(row.get("set_method"))
                for row in rows
                if row.get("set_method") not in (None, "", "nan")
            }
        )
        for method in methods:
            subset = [row for row in rows if str(row.get("set_method")) == method]
            if not subset:
                continue
            render(subset, output_dir / f"by_method_{method}")

    print(f"[rq1] rows={len(rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
