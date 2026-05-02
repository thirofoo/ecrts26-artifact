from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter


BASELINE_METHODS = [
    ("Fed-2018", "m_base", "m_base_status", "#4C72B0"),
]
PER_BIN_METHODS = [
    ("NoCluster", "nocluster_multipath", "nocluster_status", "#64B5CD"),
    ("Proposed", "cluster_multipath", "cluster_status", "#C44E52"),
]
CRITICAL_METHOD = (
    "Proposed (critical)",
    "cluster_multipath_critical",
    "cluster_status",
    "#CC79A7",
)


def save_plot(fig: plt.Figure, out_path: Path, pad_inches: float = 0.02) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad_inches)


def _clean_heatmap_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.minorticks_off()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(which="minor", length=0)
    ax.set_facecolor(plt.cm.viridis(0.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RQ2 boxplots from ex05_2 results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/ex05_2"),
        help="ex05_2_results.csv or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store plots (default: <run_dir>/plots).",
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
        "--include-baselines",
        action="store_true",
        default=False,
        help="Include baseline methods as a single bin (default: off).",
    )
    parser.add_argument(
        "--no-include-baselines",
        action="store_false",
        dest="include_baselines",
        help="Exclude baseline methods from boxplots.",
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
        "--include-critical",
        action="store_true",
        default=False,
        help="Include critical-path-only results in per-bin boxplots.",
    )
    parser.add_argument(
        "--no-include-critical",
        action="store_false",
        dest="include_critical",
        help="Exclude critical-path-only results from per-bin boxplots.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="fixed",
        help="Label for the single baseline bin.",
    )
    parser.add_argument(
        "--fi-profile",
        type=str,
        default="",
        help="Comma-separated fi profiles to include (e.g., fixed,loguniform).",
    )
    parser.add_argument(
        "--fi-compare",
        action="store_true",
        default=False,
        help="Generate plots for each fi profile separately.",
    )
    parser.add_argument(
        "--fi-compare-scatter",
        action="store_true",
        default=False,
        help="Generate scatter comparing fixed vs loguniform for the same tasksets.",
    )
    parser.add_argument(
        "--fi-compare-heatmap",
        action="store_true",
        default=True,
        help="Also generate heatmap for fi-compare scatter.",
    )
    parser.add_argument(
        "--no-fi-compare-heatmap",
        action="store_false",
        dest="fi_compare_heatmap",
        help="Disable fi-compare heatmap output.",
    )
    parser.add_argument(
        "--fi-compare-delta",
        action="store_true",
        default=True,
        help="Generate paired delta boxplots (log uniform - fixed value) across F_S.",
    )
    parser.add_argument(
        "--no-fi-compare-delta",
        action="store_false",
        dest="fi_compare_delta",
        help="Disable paired delta boxplots for fi-profile comparison.",
    )
    parser.add_argument(
        "--fi-compare-winloss",
        action="store_true",
        default=True,
        help="Generate win/tie/loss stacked bars for fi-profile comparison.",
    )
    parser.add_argument(
        "--no-fi-compare-winloss",
        action="store_false",
        dest="fi_compare_winloss",
        help="Disable win/tie/loss stacked bars for fi-profile comparison.",
    )
    parser.add_argument(
        "--fi-compare-heatmap-fs",
        type=str,
        default="",
        help="Comma-separated F_S values to render side-by-side fi-compare heatmaps (e.g., 1e-9,1e-8).",
    )
    parser.add_argument(
        "--fi-compare-heatmap-merge",
        action="store_true",
        default=True,
        help="Generate merged heatmap grid with fixed/log uniform rows.",
    )
    parser.add_argument(
        "--no-fi-compare-heatmap-merge",
        action="store_false",
        dest="fi_compare_heatmap_merge",
        help="Disable merged fi-compare heatmap grid.",
    )
    parser.add_argument(
        "--fi-compare-x",
        type=str,
        default="fixed",
        help="fi profile for x-axis in fi-compare scatter.",
    )
    parser.add_argument(
        "--fi-compare-y",
        type=str,
        default="loguniform",
        help="fi profile for y-axis in fi-compare scatter.",
    )
    parser.add_argument(
        "--fi-compare-fs",
        type=str,
        default="",
        help="Comma-separated F_S values for fi-compare scatter (default: all available).",
    )
    parser.add_argument(
        "--scatter-fs-x",
        type=str,
        default="1e-6",
        help="F_S value for x-axis scatter comparison.",
    )
    parser.add_argument(
        "--scatter-fs-y",
        type=str,
        default="1e-9",
        help="F_S value for y-axis scatter comparison.",
    )
    parser.add_argument(
        "--scatter-fs-pairs",
        type=str,
        default="",
        help="Comma-separated F_S pairs (e.g., 1e-7:1e-8,1e-7:1e-9) or 'all'.",
    )
    parser.add_argument(
        "--scatter-fs-skip",
        type=str,
        default="",
        help="Comma-separated F_S pairs to skip (e.g., 1e-7:1e-9).",
    )
    parser.add_argument(
        "--scatter-methods",
        type=str,
        default="ours",
        help="Comma-separated methods for scatter plot.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="Generate scatter-density heatmaps for F_S pairs.",
    )
    parser.add_argument(
        "--heatmap-bin-size",
        type=float,
        default=2.0,
        help="Square bin size for scatter heatmap (default: 2.0).",
    )
    parser.add_argument(
        "--heatmap-combine",
        action="store_true",
        default=True,
        help="Combine heatmaps for all F_S pairs into a single figure.",
    )
    parser.add_argument(
        "--no-heatmap-combine",
        action="store_false",
        dest="heatmap_combine",
        help="Render separate heatmaps per F_S pair.",
    )
    parser.add_argument(
        "--fs-compare-delta",
        action="store_true",
        default=True,
        help="Generate paired delta boxplots for F_S pairs (y - x).",
    )
    parser.add_argument(
        "--no-fs-compare-delta",
        action="store_false",
        dest="fs_compare_delta",
        help="Disable paired delta boxplots for F_S pairs.",
    )
    parser.add_argument(
        "--fs-compare-winloss",
        action="store_true",
        default=True,
        help="Generate win/tie/loss stacked bars for F_S pairs.",
    )
    parser.add_argument(
        "--no-fs-compare-winloss",
        action="store_false",
        dest="fs_compare_winloss",
        help="Disable win/tie/loss stacked bars for F_S pairs.",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        default=False,
        help="Generate boxplot + all scatter pairs + heatmaps in one run.",
    )
    return parser


def resolve_results_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    candidate = input_path / "ex05_2_results.csv"
    if candidate.exists():
        return candidate
    matches = list(input_path.rglob("ex05_2_results.csv"))
    if not matches:
        raise FileNotFoundError(f"ex05_2_results.csv not found under {input_path}")
    if len(matches) > 1:
        names = "\n".join(str(path) for path in matches)
        raise FileNotFoundError(
            "Multiple ex05_2_results.csv found. Pass --input with the file path:\n"
            f"{names}"
        )
    return matches[0]


def parse_whis(value: str) -> float | Tuple[float, float]:
    text = value.strip().lower()
    if text in {"minmax", "min-max", "full", "0-100", "0,100"}:
        return (0.0, 100.0)
    return float(text)


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
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_fs_value(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("F_S value must not be empty.")
    return float(text)


def _parse_fs_pairs(
    text: str,
    available: Optional[Sequence[float]] = None,
    skip: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    raw = text.strip()
    if not raw:
        return []
    if raw.lower() == "all":
        if not available:
            raise ValueError("scatter fs pairs 'all' requires available F_S values.")
        values = sorted(set(available))
        pairs: List[Tuple[float, float]] = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs.append((values[i], values[j]))
        if skip:
            skip_set = {(min(a, b), max(a, b)) for a, b in skip}
            pairs = [
                (a, b)
                for a, b in pairs
                if (min(a, b), max(a, b)) not in skip_set
            ]
        return pairs
    pairs: List[Tuple[float, float]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "vs" in token:
            left, right = token.split("vs", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            raise ValueError(f"Invalid F_S pair: {token}")
        pairs.append((_parse_fs_value(left), _parse_fs_value(right)))
    if skip:
        skip_set = {(min(a, b), max(a, b)) for a, b in skip}
        pairs = [
            (a, b)
            for a, b in pairs
            if (min(a, b), max(a, b)) not in skip_set
        ]
    return pairs


def _parse_fs_list(text: str, available: Sequence[float]) -> List[float]:
    raw = text.strip()
    if not raw:
        return list(available)
    values: List[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(_parse_fs_value(token))
    return values


def _format_fi_profile(profile: str) -> str:
    text = profile.strip().lower()
    if text in {"loguniform", "log_uniform", "log-uniform", "log uniform"}:
        return "log uniform"
    if text == "fixed":
        return "fixed value"
    return text


def load_rows(path: Path, require_ok: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if require_ok and row.get("status_set") != "ok":
                continue
            rows.append(row)
    return rows


def filter_rows_by_fi_profile(
    rows: Sequence[Dict[str, str]],
    profiles: Sequence[str],
) -> List[Dict[str, str]]:
    if not profiles:
        return list(rows)
    allowed = {profile.strip().lower() for profile in profiles if profile.strip()}
    if not allowed:
        return list(rows)
    filtered: List[Dict[str, str]] = []
    for row in rows:
        value = str(row.get("fi_profile") or "fixed").strip().lower()
        if value in allowed:
            filtered.append(row)
    return filtered


def _row_ok(
    row: Dict[str, str],
    value_field: str,
    status_field: Optional[str],
    require_ok: bool,
) -> bool:
    if require_ok and status_field:
        if row.get(status_field) != "ok":
            return False
    if _to_float(row.get(value_field)) is None:
        return False
    return True


def _format_fs(value: float) -> str:
    label = f"{value:.0e}"
    label = label.replace("e-0", "e-").replace("e+0", "e+")
    return label


def _compute_offsets(count: int, width: float = 0.6) -> List[float]:
    if count <= 1:
        return [0.0]
    step = width / count
    start = -width / 2 + step / 2
    return [start + idx * step for idx in range(count)]


def collect_baseline(
    rows: Sequence[Dict[str, str]],
    value_field: str,
    status_field: Optional[str],
    require_ok: bool,
) -> List[float]:
    values_by_set: Dict[int, float] = {}
    for row in rows:
        if not _row_ok(row, value_field, status_field, require_ok):
            continue
        set_id = _to_int(row.get("set_id"))
        if set_id is None:
            continue
        if set_id in values_by_set:
            continue
        value = _to_float(row.get(value_field))
        if value is None:
            continue
        values_by_set[set_id] = value
    return list(values_by_set.values())


def collect_per_bin(
    rows: Sequence[Dict[str, str]],
    fs_values: Sequence[float],
    value_field: str,
    status_field: Optional[str],
    require_ok: bool,
) -> List[List[float]]:
    grouped: Dict[float, List[float]] = {fs: [] for fs in fs_values}
    for row in rows:
        fs_value = _to_float(row.get("allowable_failure_prob"))
        if fs_value is None or fs_value not in grouped:
            continue
        if not _row_ok(row, value_field, status_field, require_ok):
            continue
        value = _to_float(row.get(value_field))
        if value is None:
            continue
        grouped[fs_value].append(value)
    return [grouped[fs] for fs in fs_values]


def plot_core_boxplot(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    y_scale: str,
    include_critical: bool,
    include_baselines: bool,
    baseline_label: str,
) -> Path:
    fs_values = sorted(
        {fs for fs in (_to_float(row.get("allowable_failure_prob")) for row in rows) if fs}
    )
    if not fs_values:
        raise ValueError("No allowable_failure_prob values found in data.")

    baseline_methods = list(BASELINE_METHODS) if include_baselines else []
    per_bin_methods = list(PER_BIN_METHODS)
    if include_critical:
        per_bin_methods.append(CRITICAL_METHOD)

    baseline_offsets = _compute_offsets(len(baseline_methods))
    per_bin_offsets = _compute_offsets(len(per_bin_methods))

    fig, ax = plt.subplots(figsize=(11, 5))

    start_index = 0
    if baseline_methods:
        baseline_x = 0
        start_index = 1
        for (label, field, status, color), offset in zip(baseline_methods, baseline_offsets):
            data = collect_baseline(rows, field, status, require_ok)
            if not data:
                continue
            ax.boxplot(
                data,
                positions=[baseline_x + offset],
                widths=0.5 / max(1, len(baseline_methods)),
                patch_artist=True,
                showfliers=show_fliers,
                showmeans=show_means,
                whis=whis,
                boxprops=dict(facecolor=color, alpha=0.55, edgecolor="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                meanprops=dict(
                    marker="^",
                    markerfacecolor="#2CA02C",
                    markeredgecolor="black",
                    markersize=6,
                ),
            )

    # Per-bin (fs)
    for idx, fs_value in enumerate(fs_values, start=start_index):
        for (label, field, status, color), offset in zip(per_bin_methods, per_bin_offsets):
            data = collect_per_bin(rows, fs_values, field, status, require_ok)[
                idx - start_index
            ]
            if not data:
                continue
            ax.boxplot(
                data,
                positions=[idx + offset],
                widths=0.6 / max(1, len(per_bin_methods)),
                patch_artist=True,
                showfliers=show_fliers,
                showmeans=show_means,
                whis=whis,
                boxprops=dict(facecolor=color, alpha=0.55, edgecolor="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                meanprops=dict(
                    marker="^",
                    markerfacecolor="#2CA02C",
                    markeredgecolor="black",
                    markersize=6,
                ),
            )

    if baseline_methods:
        xticks = [0] + list(range(1, len(fs_values) + 1))
        xticklabels = [baseline_label] + [_format_fs(fs) for fs in fs_values]
    else:
        xticks = list(range(0, len(fs_values)))
        xticklabels = [_format_fs(fs) for fs in fs_values]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(r"allowable failure probability $F_S$")
    ax.set_ylabel("required cores")
    if y_scale == "log":
        ax.set_yscale("log")

    legend_patches = [
        Patch(facecolor=color, edgecolor="black", alpha=0.55, label=label)
        for label, _, _, color in baseline_methods + per_bin_methods
    ]
    mean_handle = (
        [Line2D([0], [0], marker="^", color="none", markerfacecolor="#2CA02C", markersize=6, label="mean")]
        if show_means
        else []
    )
    ax.legend(handles=legend_patches + mean_handle, loc="best", frameon=False)
    fig.tight_layout()
    out_path = out_dir / "rq2_cores_fs.png"
    save_plot(fig, out_path, pad_inches=0.05)
    plt.close(fig)
    return out_path


def plot_fs_scatter_heatmap(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_x: float,
    fs_y: float,
    method: str,
    bin_size: float = 1.0,
) -> Optional[Path]:
    label, field, status, color = _method_config(method)
    by_fs: Dict[float, Dict[int, Dict[str, str]]] = {}
    for row in rows:
        fs_value = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs_value is None or set_id is None:
            continue
        by_fs.setdefault(fs_value, {})[set_id] = row

    if fs_x not in by_fs or fs_y not in by_fs:
        return None

    points_x: List[float] = []
    points_y: List[float] = []
    for set_id, row_x in by_fs[fs_x].items():
        row_y = by_fs[fs_y].get(set_id)
        if row_y is None:
            continue
        if not _row_ok(row_x, field, status, require_ok):
            continue
        if not _row_ok(row_y, field, status, require_ok):
            continue
        x_val = _to_float(row_x.get(field))
        y_val = _to_float(row_y.get(field))
        if x_val is None or y_val is None:
            continue
        points_x.append(x_val)
        points_y.append(y_val)

    if not points_x:
        return None

    if bin_size <= 0:
        raise ValueError("heatmap bin size must be > 0.")
    x_min = math.floor(min(points_x) / bin_size) * bin_size
    x_max = math.ceil(max(points_x) / bin_size) * bin_size
    y_min = math.floor(min(points_y) / bin_size) * bin_size
    y_max = math.ceil(max(points_y) / bin_size) * bin_size
    xedges = np.arange(x_min, x_max + bin_size, bin_size)
    yedges = np.arange(y_min, y_max + bin_size, bin_size)
    if len(xedges) < 2:
        xedges = np.array([x_min, x_min + bin_size])
    if len(yedges) < 2:
        yedges = np.array([y_min, y_min + bin_size])

    fig, ax = plt.subplots(figsize=(6, 6))
    counts, _, _ = np.histogram2d(points_x, points_y, bins=[xedges, yedges])
    counts, xedges, yedges = _pad_square_hist(counts, xedges, yedges, bin_size)
    vmax = float(np.nanmax(counts)) if counts.size else None
    im = ax.pcolormesh(
        xedges,
        yedges,
        counts.T,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=vmax if vmax and vmax > 0 else None,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    min_val = xedges[0]
    max_val = xedges[-1]
    pad = bin_size * 0.5
    max_val_padded = max_val + pad
    ax.plot([min_val, max_val_padded], [min_val, max_val_padded], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(min_val, max_val_padded)
    ax.set_ylim(min_val, max_val_padded)
    ax.margins(0)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.set_aspect("equal", adjustable="box")
    _clean_heatmap_axes(ax)
    ax.set_xlabel(rf"cores @ $F_S$={_format_fs(fs_x)}")
    ax.set_ylabel(rf"cores @ $F_S$={_format_fs(fs_y)}")
    fig.tight_layout(pad=0.6)
    out_path = out_dir / (
        f"rq2_cores_scatter_heatmap_{_format_fs(fs_x)}_vs_{_format_fs(fs_y)}_"
        f"{label.replace(' ', '_').lower()}.png"
    )
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def _scatter_hist2d(
    points_x: Sequence[float],
    points_y: Sequence[float],
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bin_size <= 0:
        raise ValueError("heatmap bin size must be > 0.")
    x_min = math.floor(min(points_x) / bin_size) * bin_size
    x_max = math.ceil(max(points_x) / bin_size) * bin_size
    y_min = math.floor(min(points_y) / bin_size) * bin_size
    y_max = math.ceil(max(points_y) / bin_size) * bin_size
    xedges = np.arange(x_min, x_max + bin_size, bin_size)
    yedges = np.arange(y_min, y_max + bin_size, bin_size)
    if len(xedges) < 2:
        xedges = np.array([x_min, x_min + bin_size])
    if len(yedges) < 2:
        yedges = np.array([y_min, y_min + bin_size])
    counts, _, _ = np.histogram2d(points_x, points_y, bins=[xedges, yedges])
    return counts, xedges, yedges


def _pad_square_hist(
    counts: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bin_size <= 0:
        raise ValueError("heatmap bin size must be > 0.")
    x_min = xedges[0]
    x_max = xedges[-1]
    y_min = yedges[0]
    y_max = yedges[-1]
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    new_xedges = np.arange(min_val, max_val + bin_size, bin_size)
    new_yedges = np.arange(min_val, max_val + bin_size, bin_size)
    new_counts = np.zeros((len(new_xedges) - 1, len(new_yedges) - 1))
    x_off = int(round((x_min - min_val) / bin_size))
    y_off = int(round((y_min - min_val) / bin_size))
    new_counts[x_off : x_off + counts.shape[0], y_off : y_off + counts.shape[1]] = counts
    return new_counts, new_xedges, new_yedges


def plot_fs_scatter_heatmap_grid(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    pairs: Sequence[Tuple[float, float]],
    method: str,
    bin_size: float = 1.0,
) -> Optional[Path]:
    label, field, status, _ = _method_config(method)
    by_fs: Dict[float, Dict[int, Dict[str, str]]] = {}
    for row in rows:
        fs_value = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs_value is None or set_id is None:
            continue
        by_fs.setdefault(fs_value, {})[set_id] = row

    hist_items: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    max_count = 0.0
    for fs_x, fs_y in pairs:
        if fs_x not in by_fs or fs_y not in by_fs:
            continue
        points_x: List[float] = []
        points_y: List[float] = []
        for set_id, row_x in by_fs[fs_x].items():
            row_y = by_fs[fs_y].get(set_id)
            if row_y is None:
                continue
            if not _row_ok(row_x, field, status, require_ok):
                continue
            if not _row_ok(row_y, field, status, require_ok):
                continue
            x_val = _to_float(row_x.get(field))
            y_val = _to_float(row_y.get(field))
            if x_val is None or y_val is None:
                continue
            points_x.append(x_val)
            points_y.append(y_val)
        if not points_x:
            continue
        counts, xedges, yedges = _scatter_hist2d(points_x, points_y, bin_size)
        counts, xedges, yedges = _pad_square_hist(counts, xedges, yedges, bin_size)
        max_count = max(max_count, float(np.nanmax(counts)))
        hist_items.append((fs_x, fs_y, counts, xedges, yedges))

    if not hist_items:
        return None

    n_panels = len(hist_items)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig_w = max(6.8, 4.4 * ncols)
    fig_h = max(4.2, 3.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes_list = np.array(axes).reshape(-1)

    for ax in axes_list[n_panels:]:
        ax.axis("off")

    im = None
    for ax, (fs_x, fs_y, counts, xedges, yedges) in zip(axes_list, hist_items):
        im = ax.pcolormesh(
            xedges,
            yedges,
            counts.T,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max_count if max_count > 0 else None,
        )
        min_val = xedges[0]
        max_val = xedges[-1]
        pad = bin_size * 0.5
        max_val_padded = max_val + pad
        ax.plot([min_val, max_val_padded], [min_val, max_val_padded], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(min_val, max_val_padded)
        ax.set_ylim(min_val, max_val_padded)
        ax.margins(0)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_aspect("equal", adjustable="box")
        _clean_heatmap_axes(ax)
        ax.set_xlabel(rf"cores @ $F_S$={_format_fs(fs_x)}")
        ax.set_ylabel(rf"cores @ $F_S$={_format_fs(fs_y)}")

    if im is not None:
        fig.subplots_adjust(left=0.05, right=0.96, top=0.98, bottom=0.12, wspace=0.22, hspace=0.24)
        cax = fig.add_axes([0.98, 0.14, 0.02, 0.72])
        cbar = fig.colorbar(im, cax=cax, label="count")
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
    else:
        fig.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.12, wspace=0.22, hspace=0.24)

    out_path = out_dir / f"rq2_cores_scatter_heatmap_grid_{label.replace(' ', '_').lower()}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def _method_config(name: str) -> Tuple[str, str, Optional[str], str]:
    lookup = {
        "fed2018": ("Fed-2018", "m_base", "m_base_status", "#4C72B0"),
        "nocluster": ("NoCluster", "nocluster_multipath", "nocluster_status", "#64B5CD"),
        "ours": ("Proposed", "cluster_multipath", "cluster_status", "#C44E52"),
        "critical": ("Proposed (critical)", "cluster_multipath_critical", "cluster_status", "#CC79A7"),
    }
    if name not in lookup:
        raise ValueError(f"Unknown method: {name}")
    return lookup[name]


def _index_rows_by_fi_profile_fs(
    rows: Sequence[Dict[str, str]],
) -> Dict[str, Dict[float, Dict[int, Dict[str, str]]]]:
    by_profile_fs: Dict[str, Dict[float, Dict[int, Dict[str, str]]]] = {}
    for row in rows:
        profile = str(row.get("fi_profile") or "fixed").strip().lower()
        fs = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs is None or set_id is None:
            continue
        by_profile_fs.setdefault(profile, {}).setdefault(fs, {})[set_id] = row
    return by_profile_fs


def _index_rows_by_fs(
    rows: Sequence[Dict[str, str]],
) -> Dict[float, Dict[int, Dict[str, str]]]:
    by_fs: Dict[float, Dict[int, Dict[str, str]]] = {}
    for row in rows:
        fs_value = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs_value is None or set_id is None:
            continue
        by_fs.setdefault(fs_value, {})[set_id] = row
    return by_fs


def _collect_fs_pairs_for_values(
    by_fs: Dict[float, Dict[int, Dict[str, str]]],
    require_ok: bool,
    fs_x: float,
    fs_y: float,
    value_field: str,
    status_field: Optional[str],
) -> List[Tuple[int, float, float]]:
    if fs_x not in by_fs or fs_y not in by_fs:
        return []
    pairs: List[Tuple[int, float, float]] = []
    for set_id, row_x in by_fs[fs_x].items():
        row_y = by_fs[fs_y].get(set_id)
        if row_y is None:
            continue
        if not _row_ok(row_x, value_field, status_field, require_ok):
            continue
        if not _row_ok(row_y, value_field, status_field, require_ok):
            continue
        x_val = _to_float(row_x.get(value_field))
        y_val = _to_float(row_y.get(value_field))
        if x_val is None or y_val is None:
            continue
        pairs.append((set_id, x_val, y_val))
    return pairs


def _collect_fi_profile_pairs_for_fs(
    by_profile_fs: Dict[str, Dict[float, Dict[int, Dict[str, str]]]],
    require_ok: bool,
    fs_value: float,
    value_field: str,
    status_field: Optional[str],
    x_profile: str,
    y_profile: str,
) -> List[Tuple[int, float, float]]:
    x_key = x_profile.strip().lower()
    y_key = y_profile.strip().lower()
    if x_key not in by_profile_fs or y_key not in by_profile_fs:
        return []
    if fs_value not in by_profile_fs[x_key] or fs_value not in by_profile_fs[y_key]:
        return []

    pairs: List[Tuple[int, float, float]] = []
    for set_id, row_x in by_profile_fs[x_key][fs_value].items():
        row_y = by_profile_fs[y_key][fs_value].get(set_id)
        if row_y is None:
            continue
        if not _row_ok(row_x, value_field, status_field, require_ok):
            continue
        if not _row_ok(row_y, value_field, status_field, require_ok):
            continue
        x_val = _to_float(row_x.get(value_field))
        y_val = _to_float(row_y.get(value_field))
        if x_val is None or y_val is None:
            continue
        pairs.append((set_id, x_val, y_val))
    return pairs


def plot_fs_scatter(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_x: float,
    fs_y: float,
    methods: Sequence[str],
) -> Path:
    by_fs = _index_rows_by_fs(rows)

    available = sorted(by_fs.keys())
    if fs_x not in by_fs or fs_y not in by_fs:
        raise ValueError(
            "scatter fs values not found in data. "
            f"available={', '.join(_format_fs(v) for v in available)}"
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    for method in methods:
        label, field, status, color = _method_config(method)
        points_x: List[float] = []
        points_y: List[float] = []
        for set_id, row_x in by_fs[fs_x].items():
            row_y = by_fs[fs_y].get(set_id)
            if row_y is None:
                continue
            if not _row_ok(row_x, field, status, require_ok):
                continue
            if not _row_ok(row_y, field, status, require_ok):
                continue
            x_val = _to_float(row_x.get(field))
            y_val = _to_float(row_y.get(field))
            if x_val is None or y_val is None:
                continue
            points_x.append(x_val)
            points_y.append(y_val)
        if points_x:
            ax.scatter(points_x, points_y, s=30, alpha=0.75, label=label, color=color)

    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(rf"cores @ $F_S$={_format_fs(fs_x)}")
    ax.set_ylabel(rf"cores @ $F_S$={_format_fs(fs_y)}")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / f"rq2_cores_scatter_fs_{_format_fs(fs_x)}_vs_{_format_fs(fs_y)}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fs_delta_boxplot(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    pairs: Sequence[Tuple[float, float]],
    method: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
) -> Optional[Path]:
    if not pairs:
        return None
    label, field, status, color = _method_config(method)
    by_fs = _index_rows_by_fs(rows)

    pair_items: List[Tuple[float, float, List[float]]] = []
    for fs_x, fs_y in pairs:
        pair_rows = _collect_fs_pairs_for_values(
            by_fs,
            require_ok,
            fs_x,
            fs_y,
            field,
            status,
        )
        deltas = [y - x for _, x, y in pair_rows]
        if deltas:
            pair_items.append((fs_x, fs_y, deltas))
    if not pair_items:
        return None

    fig_w = max(8.4, 2.9 * len(pair_items))
    fig, ax = plt.subplots(figsize=(fig_w, 3.1))
    positions = list(range(1, len(pair_items) + 1))
    box = ax.boxplot(
        [d for _, _, d in pair_items],
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2ca02c",
            "markeredgecolor": "#2ca02c",
            "markersize": 7,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_xticks(positions)
    totals = [len(deltas) for _, _, deltas in pair_items]
    ax.set_xticklabels(
        [
            f"{_format_fs(fs_x)} -> {_format_fs(fs_y)}"
            for fs_x, fs_y, _ in pair_items
        ],
        fontsize=10,
    )
    if len(set(totals)) == 1:
        ax.set_xlabel(rf"$F_S$ pair (x -> y), n={totals[0]} each")
    else:
        ax.set_xlabel(r"$F_S$ pair (x -> y)")
    ax.set_ylabel(r"$\Delta$ cores (y - x)")
    ax.tick_params(axis="y", labelsize=10)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.96)
    out_path = out_dir / f"rq2_cores_delta_fs_pairs_{label.replace(' ', '_').lower()}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fs_win_tie_loss(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    pairs: Sequence[Tuple[float, float]],
    method: str,
) -> Optional[Path]:
    if not pairs:
        return None
    label, field, status, _ = _method_config(method)
    by_fs = _index_rows_by_fs(rows)

    pair_items: List[Tuple[float, float, int, int, int]] = []
    for fs_x, fs_y in pairs:
        pair_rows = _collect_fs_pairs_for_values(
            by_fs,
            require_ok,
            fs_x,
            fs_y,
            field,
            status,
        )
        if not pair_rows:
            continue
        y_better = 0
        tie = 0
        x_better = 0
        for _, x_val, y_val in pair_rows:
            if y_val < x_val:
                y_better += 1
            elif y_val > x_val:
                x_better += 1
            else:
                tie += 1
        pair_items.append((fs_x, fs_y, y_better, tie, x_better))
    if not pair_items:
        return None
    preferred_order = {
        ("1e-9", "1e-7"): 0,
        ("1e-9", "1e-8"): 1,
        ("1e-8", "1e-7"): 2,
    }
    pair_items.sort(
        key=lambda item: (
            preferred_order.get((_format_fs(item[0]), _format_fs(item[1])), 100),
            -(math.log10(item[1]) - math.log10(item[0])),
            item[0],
            item[1],
        )
    )

    fig_w = max(8.4, 2.9 * len(pair_items))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    xs = np.arange(len(pair_items), dtype=float)
    totals = np.array([max(1, a + b + c) for _, _, a, b, c in pair_items], dtype=float)
    y_better = np.array([a for _, _, a, _, _ in pair_items], dtype=float) / totals
    ties = np.array([b for _, _, _, b, _ in pair_items], dtype=float) / totals
    x_better = np.array([c for _, _, _, _, c in pair_items], dtype=float) / totals

    bars1 = ax.bar(xs, y_better, width=0.72, color="#4CAF50", label="y-side better")
    bars2 = ax.bar(xs, ties, width=0.72, bottom=y_better, color="#9E9E9E", label="tie")
    bars3 = ax.bar(
        xs,
        x_better,
        width=0.72,
        bottom=y_better + ties,
        color="#E57373",
        label="x-side better",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{_format_fs(fs_x)} -> {_format_fs(fs_y)}" for fs_x, fs_y, *_ in pair_items],
        fontsize=10,
    )
    if len(set(int(t) for t in totals)) == 1:
        ax.set_xlabel(rf"$F_S$ pair (x -> y), n={int(totals[0])} each")
    else:
        ax.set_xlabel(r"$F_S$ pair (x -> y)")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of paired tasksets")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fontsize=8.8,
        borderaxespad=0.0,
    )

    for bars, counts, bottoms in [
        (bars1, [a for _, _, a, _, _ in pair_items], np.zeros_like(y_better)),
        (bars2, [b for _, _, _, b, _ in pair_items], y_better),
        (bars3, [c for _, _, _, _, c in pair_items], y_better + ties),
    ]:
        heights = [b.get_height() for b in bars]
        for bar, count, bottom, frac in zip(bars, counts, bottoms, heights):
            if count <= 0 or frac < 0.08:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(bottom) + float(frac) / 2.0,
                str(int(count)),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if frac > 0.16 else "black",
            )

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.21, top=0.80)
    out_path = out_dir / f"rq2_cores_win_tie_loss_fs_pairs_{label.replace(' ', '_').lower()}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_profile_scatter(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_value: float,
    method: str,
    x_profile: str,
    y_profile: str,
) -> Optional[Path]:
    label, field, status, color = _method_config(method)
    by_profile_fs = _index_rows_by_fi_profile_fs(rows)

    x_profile = x_profile.strip().lower()
    y_profile = y_profile.strip().lower()
    x_label = _format_fi_profile(x_profile)
    y_label = _format_fi_profile(y_profile)
    if x_profile not in by_profile_fs or y_profile not in by_profile_fs:
        return None
    if fs_value not in by_profile_fs[x_profile] or fs_value not in by_profile_fs[y_profile]:
        return None

    pairs = _collect_fi_profile_pairs_for_fs(
        by_profile_fs,
        require_ok,
        fs_value,
        field,
        status,
        x_profile,
        y_profile,
    )
    points_x = [x for _, x, _ in pairs]
    points_y = [y for _, _, y in pairs]

    if not points_x:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_x, points_y, s=30, alpha=0.75, color=color, label=label)
    min_val = min(min(points_x), min(points_y))
    max_val = max(max(points_x), max(points_y))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(rf"cores ($f_i$: {x_label})")
    ax.set_ylabel(rf"cores ($f_i$: {y_label})")
    ax.set_title(rf"$F_S$={_format_fs(fs_value)}", pad=6)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / (
        f"rq2_cores_scatter_fi_{x_profile}_vs_{y_profile}_fs_{_format_fs(fs_value)}_"
        f"{label.replace(' ', '_').lower()}.png"
    )
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_profile_heatmap(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_value: float,
    method: str,
    x_profile: str,
    y_profile: str,
    bin_size: float = 1.0,
) -> Optional[Path]:
    label, field, status, _ = _method_config(method)
    by_profile_fs = _index_rows_by_fi_profile_fs(rows)

    x_profile = x_profile.strip().lower()
    y_profile = y_profile.strip().lower()
    x_label = _format_fi_profile(x_profile)
    y_label = _format_fi_profile(y_profile)
    if x_profile not in by_profile_fs or y_profile not in by_profile_fs:
        return None
    if fs_value not in by_profile_fs[x_profile] or fs_value not in by_profile_fs[y_profile]:
        return None

    pairs = _collect_fi_profile_pairs_for_fs(
        by_profile_fs,
        require_ok,
        fs_value,
        field,
        status,
        x_profile,
        y_profile,
    )
    points_x = [x for _, x, _ in pairs]
    points_y = [y for _, _, y in pairs]

    if not points_x:
        return None

    if bin_size <= 0:
        raise ValueError("heatmap bin size must be > 0.")
    x_min = math.floor(min(points_x) / bin_size) * bin_size
    x_max = math.ceil(max(points_x) / bin_size) * bin_size
    y_min = math.floor(min(points_y) / bin_size) * bin_size
    y_max = math.ceil(max(points_y) / bin_size) * bin_size
    xedges = np.arange(x_min, x_max + bin_size, bin_size)
    yedges = np.arange(y_min, y_max + bin_size, bin_size)
    if len(xedges) < 2:
        xedges = np.array([x_min, x_min + bin_size])
    if len(yedges) < 2:
        yedges = np.array([y_min, y_min + bin_size])

    fig, ax = plt.subplots(figsize=(6, 6))
    counts, _, _ = np.histogram2d(points_x, points_y, bins=[xedges, yedges])
    counts, xedges, yedges = _pad_square_hist(counts, xedges, yedges, bin_size)
    vmax = float(np.nanmax(counts)) if counts.size else None
    im = ax.pcolormesh(
        xedges,
        yedges,
        counts.T,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=vmax if vmax and vmax > 0 else None,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    min_val = xedges[0]
    max_val = xedges[-1]
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.margins(0)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.set_aspect("equal", adjustable="box")
    _clean_heatmap_axes(ax)
    ax.set_xlabel(rf"cores ($f_i$: {x_label})")
    ax.set_ylabel(rf"cores ($f_i$: {y_label})")
    ax.set_title(rf"$F_S$={_format_fs(fs_value)}", pad=6)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / (
        f"rq2_cores_scatter_heatmap_fi_{x_profile}_vs_{y_profile}_fs_{_format_fs(fs_value)}_"
        f"{label.replace(' ', '_').lower()}.png"
    )
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_profile_heatmap_grid(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    pairs: Sequence[Tuple[float, float]],
    method: str,
    profiles: Sequence[str],
    bin_size: float = 1.0,
) -> Optional[Path]:
    if not pairs:
        return None
    label, field, status, _ = _method_config(method)
    profiles_norm = [p.strip().lower() for p in profiles if p.strip()]
    if not profiles_norm:
        return None

    by_profile_fs: Dict[str, Dict[float, Dict[int, Dict[str, str]]]] = {}
    for row in rows:
        profile = str(row.get("fi_profile") or "fixed").strip().lower()
        fs = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs is None or set_id is None:
            continue
        by_profile_fs.setdefault(profile, {}).setdefault(fs, {})[set_id] = row

    hist_items: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_count = 0.0
    for r_idx, profile in enumerate(profiles_norm):
        if profile not in by_profile_fs:
            continue
        for c_idx, (fs_x, fs_y) in enumerate(pairs):
            if fs_x not in by_profile_fs[profile] or fs_y not in by_profile_fs[profile]:
                continue
            points_x: List[float] = []
            points_y: List[float] = []
            for set_id, row_x in by_profile_fs[profile][fs_x].items():
                row_y = by_profile_fs[profile][fs_y].get(set_id)
                if row_y is None:
                    continue
                if not _row_ok(row_x, field, status, require_ok):
                    continue
                if not _row_ok(row_y, field, status, require_ok):
                    continue
                x_val = _to_float(row_x.get(field))
                y_val = _to_float(row_y.get(field))
                if x_val is None or y_val is None:
                    continue
                points_x.append(x_val)
                points_y.append(y_val)
            if not points_x:
                continue
            counts, xedges, yedges = _scatter_hist2d(points_x, points_y, bin_size)
            counts, xedges, yedges = _pad_square_hist(counts, xedges, yedges, bin_size)
            max_count = max(max_count, float(np.nanmax(counts)))
            hist_items[(r_idx, c_idx)] = (counts, xedges, yedges)

    if not hist_items:
        return None

    nrows = len(profiles_norm)
    ncols = len(pairs)
    fig_w = max(6.8, 4.4 * ncols)
    fig_h = max(4.2, 3.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes_arr = np.array(axes).reshape(nrows, ncols)

    im = None
    for r_idx, profile in enumerate(profiles_norm):
        for c_idx, (fs_x, fs_y) in enumerate(pairs):
            ax = axes_arr[r_idx, c_idx]
            item = hist_items.get((r_idx, c_idx))
            if item is None:
                ax.axis("off")
                continue
            counts, xedges, yedges = item
            im = ax.pcolormesh(
                xedges,
                yedges,
                counts.T,
                cmap="viridis",
                shading="auto",
                vmin=0.0,
                vmax=max_count if max_count > 0 else None,
            )
            min_val = xedges[0]
            max_val = xedges[-1]
            pad = bin_size * 0.5
            max_val_padded = max_val + pad
            ax.plot([min_val, max_val_padded], [min_val, max_val_padded], linestyle="--", color="gray", linewidth=1)
            ax.set_xlim(min_val, max_val_padded)
            ax.set_ylim(min_val, max_val_padded)
            ax.margins(0)
            ax.set_xmargin(0)
            ax.set_ymargin(0)
            ax.set_aspect("equal", adjustable="box")
            _clean_heatmap_axes(ax)

            if r_idx == nrows - 1:
                ax.set_xlabel(rf"cores @ $F_S$={_format_fs(fs_x)}")
            else:
                ax.set_xlabel("")

            if c_idx == 0:
                profile_label = _format_fi_profile(profile)
                ax.set_ylabel(
                    rf"cores @ $F_S$={_format_fs(fs_y)}" + "\n" + rf"$f_i$={profile_label}"
                )
            else:
                ax.set_ylabel("")

    if im is not None:
        fig.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.12, wspace=0.22, hspace=0.30)
        cax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
        cbar = fig.colorbar(im, cax=cax, label="count")
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
    else:
        fig.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.12, wspace=0.22, hspace=0.30)

    out_path = out_dir / f"rq2_cores_scatter_heatmap_grid_fi_compare_{label.replace(' ', '_').lower()}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_compare_heatmap_row(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_values: Sequence[float],
    method: str,
    x_profile: str,
    y_profile: str,
    bin_size: float = 1.0,
) -> Optional[Path]:
    if not fs_values:
        return None
    label, field, status, _ = _method_config(method)
    x_profile = x_profile.strip().lower()
    y_profile = y_profile.strip().lower()
    x_label = _format_fi_profile(x_profile)
    y_label = _format_fi_profile(y_profile)

    by_profile_fs: Dict[str, Dict[float, Dict[int, Dict[str, str]]]] = {}
    for row in rows:
        profile = str(row.get("fi_profile") or "fixed").strip().lower()
        fs = _to_float(row.get("allowable_failure_prob"))
        set_id = _to_int(row.get("set_id"))
        if fs is None or set_id is None:
            continue
        by_profile_fs.setdefault(profile, {}).setdefault(fs, {})[set_id] = row

    if x_profile not in by_profile_fs or y_profile not in by_profile_fs:
        return None

    hist_items: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    max_count = 0.0
    for fs_value in fs_values:
        if fs_value not in by_profile_fs[x_profile] or fs_value not in by_profile_fs[y_profile]:
            continue
        points_x: List[float] = []
        points_y: List[float] = []
        for set_id, row_x in by_profile_fs[x_profile][fs_value].items():
            row_y = by_profile_fs[y_profile][fs_value].get(set_id)
            if row_y is None:
                continue
            if not _row_ok(row_x, field, status, require_ok):
                continue
            if not _row_ok(row_y, field, status, require_ok):
                continue
            x_val = _to_float(row_x.get(field))
            y_val = _to_float(row_y.get(field))
            if x_val is None or y_val is None:
                continue
            points_x.append(x_val)
            points_y.append(y_val)
        if not points_x:
            continue
        counts, xedges, yedges = _scatter_hist2d(points_x, points_y, bin_size)
        counts, xedges, yedges = _pad_square_hist(counts, xedges, yedges, bin_size)
        max_count = max(max_count, float(np.nanmax(counts)))
        hist_items.append((fs_value, counts, xedges, yedges))

    if not hist_items:
        return None

    ncols = len(hist_items)
    fig_w = max(6.8, 4.4 * ncols)
    fig_h = 4.6
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_w, fig_h))
    axes_list = np.array(axes).reshape(-1)

    im = None
    for ax, (fs_value, counts, xedges, yedges) in zip(axes_list, hist_items):
        im = ax.pcolormesh(
            xedges,
            yedges,
            counts.T,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max_count if max_count > 0 else None,
        )
        min_val = xedges[0]
        max_val = xedges[-1]
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.margins(0)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_aspect("equal", adjustable="box")
        _clean_heatmap_axes(ax)
        ax.set_title(rf"$F_S$={_format_fs(fs_value)}", pad=6)
        ax.set_xlabel(rf"cores ($f_i$: {x_label})")
        ax.set_ylabel(rf"cores ($f_i$: {y_label})")

    if im is not None:
        fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.16, wspace=0.28)
        cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
        cbar = fig.colorbar(im, cax=cax, label="count")
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
    else:
        fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.16, wspace=0.28)

    out_path = out_dir / f"rq2_cores_scatter_heatmap_fi_compare_row_{label.replace(' ', '_').lower()}.png"
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_profile_delta_boxplot(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_values: Sequence[float],
    method: str,
    x_profile: str,
    y_profile: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
) -> Optional[Path]:
    label, field, status, color = _method_config(method)
    by_profile_fs = _index_rows_by_fi_profile_fs(rows)

    x_key = x_profile.strip().lower()
    y_key = y_profile.strip().lower()
    x_label = _format_fi_profile(x_key)
    y_label = _format_fi_profile(y_key)

    fs_items: List[Tuple[float, List[float]]] = []
    for fs_value in fs_values:
        pairs = _collect_fi_profile_pairs_for_fs(
            by_profile_fs,
            require_ok,
            fs_value,
            field,
            status,
            x_key,
            y_key,
        )
        deltas = [y - x for _, x, y in pairs]
        if deltas:
            fs_items.append((fs_value, deltas))

    if not fs_items:
        return None

    fig_w = max(6.4, 2.2 * len(fs_items))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    data = [deltas for _, deltas in fs_items]
    positions = list(range(1, len(data) + 1))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2ca02c",
            "markeredgecolor": "#2ca02c",
            "markersize": 7,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [rf"$F_S$={_format_fs(fs)}" + f"\n(n={len(deltas)})" for fs, deltas in fs_items]
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        rf"$\Delta$ cores ($f_i$: {y_label} $-$ {x_label})"
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / (
        f"rq2_cores_delta_fi_{x_key}_vs_{y_key}_{label.replace(' ', '_').lower()}.png"
    )
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_fi_profile_win_tie_loss(
    rows: Sequence[Dict[str, str]],
    out_dir: Path,
    require_ok: bool,
    fs_values: Sequence[float],
    method: str,
    x_profile: str,
    y_profile: str,
) -> Optional[Path]:
    label, field, status, _ = _method_config(method)
    by_profile_fs = _index_rows_by_fi_profile_fs(rows)

    x_key = x_profile.strip().lower()
    y_key = y_profile.strip().lower()
    x_label = _format_fi_profile(x_key)
    y_label = _format_fi_profile(y_key)

    fs_items: List[Tuple[float, int, int, int]] = []
    for fs_value in fs_values:
        pairs = _collect_fi_profile_pairs_for_fs(
            by_profile_fs,
            require_ok,
            fs_value,
            field,
            status,
            x_key,
            y_key,
        )
        if not pairs:
            continue
        y_better = 0
        tie = 0
        x_better = 0
        for _, x_val, y_val in pairs:
            if y_val < x_val:
                y_better += 1
            elif y_val > x_val:
                x_better += 1
            else:
                tie += 1
        fs_items.append((fs_value, y_better, tie, x_better))

    if not fs_items:
        return None

    fig_w = max(6.4, 2.2 * len(fs_items))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    xs = np.arange(len(fs_items), dtype=float)
    totals = np.array([max(1, yb + t + xb) for _, yb, t, xb in fs_items], dtype=float)
    y_better = np.array([yb for _, yb, _, _ in fs_items], dtype=float) / totals
    ties = np.array([t for _, _, t, _ in fs_items], dtype=float) / totals
    x_better = np.array([xb for _, _, _, xb in fs_items], dtype=float) / totals

    bars1 = ax.bar(xs, y_better, width=0.7, color="#4CAF50", label=f"{y_label} better")
    bars2 = ax.bar(xs, ties, width=0.7, bottom=y_better, color="#9E9E9E", label="tie")
    bars3 = ax.bar(
        xs,
        x_better,
        width=0.7,
        bottom=y_better + ties,
        color="#E57373",
        label=f"{x_label} better",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [rf"$F_S$={_format_fs(fs)}" + f"\n(n={int(total)})" for (fs, *_), total in zip(fs_items, totals)]
    )
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of paired tasksets")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    # Show counts inside segments when there is enough height.
    for bars, counts, bottoms in [
        (bars1, [yb for _, yb, _, _ in fs_items], np.zeros_like(y_better)),
        (bars2, [t for _, _, t, _ in fs_items], y_better),
        (bars3, [xb for _, _, _, xb in fs_items], y_better + ties),
    ]:
        for bar, count, bottom, frac in zip(bars, counts, bottoms, [b.get_height() for b in bars]):
            if count <= 0 or frac < 0.08:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(bottom) + float(frac) / 2.0,
                str(int(count)),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if frac > 0.16 else "black",
            )

    fig.tight_layout(pad=0.6)
    out_path = out_dir / (
        f"rq2_cores_win_tie_loss_fi_{x_key}_vs_{y_key}_{label.replace(' ', '_').lower()}.png"
    )
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    results_path = resolve_results_path(args.input)
    out_dir = args.output_dir or (results_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results_path, args.require_ok)
    rows_all = list(rows)
    profile_list = [item.strip() for item in args.fi_profile.split(",") if item.strip()]
    if not profile_list:
        profile_list = ["fixed", "loguniform"]

    def run_plots(selected_rows: Sequence[Dict[str, str]], target_dir: Path) -> None:
        if not selected_rows:
            return
        fs_available = sorted(
            {fs for fs in (_to_float(row.get("allowable_failure_prob")) for row in selected_rows) if fs}
        )
        plot_core_boxplot(
            selected_rows,
            target_dir,
            args.require_ok,
            parse_whis(args.whis),
            args.show_fliers,
            args.show_means,
            args.y_scale,
            args.include_critical,
            args.include_baselines,
            args.baseline_label,
        )
        methods = [item.strip() for item in args.scatter_methods.split(",") if item.strip()]
        if methods:
            scatter_pairs = args.scatter_fs_pairs
            if args.all_plots and not scatter_pairs:
                scatter_pairs = "all"
            skip_pairs = _parse_fs_pairs(args.scatter_fs_skip, fs_available)
            pairs = _parse_fs_pairs(scatter_pairs, fs_available, skip_pairs)
            if not pairs:
                fs_x = _parse_fs_value(args.scatter_fs_x)
                fs_y = _parse_fs_value(args.scatter_fs_y)
                if fs_x in fs_available and fs_y in fs_available:
                    pairs = [(fs_x, fs_y)]
                else:
                    return
            for fs_x, fs_y in pairs:
                plot_fs_scatter(
                    selected_rows,
                    target_dir,
                    args.require_ok,
                    fs_x,
                    fs_y,
                    methods,
                )
            if args.heatmap or args.all_plots:
                for method in methods:
                    if args.heatmap_combine:
                        plot_fs_scatter_heatmap_grid(
                            selected_rows,
                            target_dir,
                            args.require_ok,
                            pairs,
                            method,
                            args.heatmap_bin_size,
                        )
                    else:
                        for fs_x, fs_y in pairs:
                            plot_fs_scatter_heatmap(
                                selected_rows,
                                target_dir,
                                args.require_ok,
                                fs_x,
                                fs_y,
                                method,
                                args.heatmap_bin_size,
                            )
            for method in methods:
                if args.fs_compare_delta:
                    plot_fs_delta_boxplot(
                        selected_rows,
                        target_dir,
                        args.require_ok,
                        pairs,
                        method,
                        parse_whis(args.whis),
                        args.show_fliers,
                        args.show_means,
                    )
                if args.fs_compare_winloss:
                    plot_fs_win_tie_loss(
                        selected_rows,
                        target_dir,
                        args.require_ok,
                        pairs,
                        method,
                    )

    if args.fi_compare:
        for profile in profile_list:
            filtered = filter_rows_by_fi_profile(rows, [profile])
            if not filtered:
                continue
            subdir = out_dir / f"fi_{profile}"
            subdir.mkdir(parents=True, exist_ok=True)
            run_plots(filtered, subdir)
    else:
        rows = filter_rows_by_fi_profile(rows, profile_list)
        if not rows:
            raise SystemExit("No rows to plot after filtering.")
        run_plots(rows, out_dir)

    if args.fi_compare_scatter:
        methods = [item.strip() for item in args.scatter_methods.split(",") if item.strip()]
        if not methods:
            methods = ["ours"]
        fs_available_all = sorted(
            {
                fs
                for fs in (_to_float(row.get("allowable_failure_prob")) for row in rows_all)
                if fs
            }
        )
        fs_values = _parse_fs_list(args.fi_compare_fs, fs_available_all)
        for fs_value in fs_values:
            for method in methods:
                plot_fi_profile_scatter(
                    rows_all,
                    out_dir,
                    args.require_ok,
                    fs_value,
                    method,
                    args.fi_compare_x,
                    args.fi_compare_y,
                )
                if args.fi_compare_heatmap:
                    plot_fi_profile_heatmap(
                        rows_all,
                        out_dir,
                        args.require_ok,
                        fs_value,
                        method,
                        args.fi_compare_x,
                        args.fi_compare_y,
                        args.heatmap_bin_size,
                    )
        for method in methods:
            if args.fi_compare_delta:
                plot_fi_profile_delta_boxplot(
                    rows_all,
                    out_dir,
                    args.require_ok,
                    fs_values,
                    method,
                    args.fi_compare_x,
                    args.fi_compare_y,
                    parse_whis(args.whis),
                    args.show_fliers,
                    args.show_means,
                )
            if args.fi_compare_winloss:
                plot_fi_profile_win_tie_loss(
                    rows_all,
                    out_dir,
                    args.require_ok,
                    fs_values,
                    method,
                    args.fi_compare_x,
                    args.fi_compare_y,
                )
        if args.fi_compare_heatmap_merge:
            scatter_pairs = args.scatter_fs_pairs
            if args.all_plots and not scatter_pairs:
                scatter_pairs = "all"
            skip_pairs = _parse_fs_pairs(args.scatter_fs_skip, fs_available_all)
            pairs = _parse_fs_pairs(scatter_pairs, fs_available_all, skip_pairs)
            if not pairs:
                fs_x = _parse_fs_value(args.scatter_fs_x)
                fs_y = _parse_fs_value(args.scatter_fs_y)
                if fs_x in fs_available_all and fs_y in fs_available_all:
                    pairs = [(fs_x, fs_y)]
            for method in methods:
                plot_fi_profile_heatmap_grid(
                    rows_all,
                    out_dir,
                    args.require_ok,
                    pairs,
                    method,
                    profile_list,
                    args.heatmap_bin_size,
                )
        if args.fi_compare_heatmap_fs:
            fs_row = _parse_fs_list(args.fi_compare_heatmap_fs, fs_available_all)
            for method in methods:
                plot_fi_compare_heatmap_row(
                    rows_all,
                    out_dir,
                    args.require_ok,
                    fs_row,
                    method,
                    args.fi_compare_x,
                    args.fi_compare_y,
                    args.heatmap_bin_size,
                )


if __name__ == "__main__":
    main()
