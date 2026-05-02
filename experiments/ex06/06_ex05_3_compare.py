from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

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

STATUS_FIELD_MAP = {
    "m_base": "m_base_status",
    "federated_2016": "federated_2016_status",
    "semi_federated": "semi_federated_status",
    "nocluster_multipath": "nocluster_status",
    "m_ours": "m_ours_status",
    "cluster_multipath": "cluster_status",
    "cluster_multipath_critical": "cluster_status",
}


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: object) -> Optional[int]:
    val = _to_float(value)
    if val is None:
        return None
    return int(val)


def parse_whis(value: str) -> float | Tuple[float, float]:
    text = value.strip().lower()
    if text in {"minmax", "min-max", "full", "0-100", "0,100"}:
        return (0.0, 100.0)
    return float(text)


def parse_axes(value: str) -> List[str]:
    axes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [axis for axis in axes if axis not in AXIS_CONFIG]
    if unknown:
        raise ValueError(f"Unknown axes: {', '.join(unknown)}")
    return axes


def parse_fields(value: str) -> List[str]:
    fields = [item.strip() for item in value.split(",") if item.strip()]
    if not fields:
        raise ValueError("--fields must not be empty.")
    return fields


def parse_exclude_bins(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_results_path(input_path: Path, filename: str) -> Path:
    if input_path.is_file():
        return input_path
    candidate = input_path / filename
    if candidate.exists():
        return candidate
    matches = list(input_path.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {input_path}")
    if len(matches) > 1:
        names = "\n".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Multiple {filename} found. Pass the file path directly:\n{names}"
        )
    return matches[0]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_rows_by_set(path: Path) -> Dict[int, Dict[str, str]]:
    rows = load_rows(path)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        set_id = _to_int(row.get("set_id"))
        if set_id is None:
            continue
        out[set_id] = row
    return out


def load_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def time_label_from_meta(meta: Dict[str, object], fallback: str) -> str:
    seconds = _to_float(meta.get("cluster_time_limit"))
    if seconds is None:
        return fallback
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}s"
    return f"{seconds:g}s"


def row_ok(row: Dict[str, str], field: str, require_ok: bool) -> bool:
    if _to_float(row.get(field)) is None:
        return False
    if not require_ok:
        return True
    if row.get("status_set") != "ok":
        return False
    status_field = STATUS_FIELD_MAP.get(field)
    if status_field and row.get(status_field) != "ok":
        return False
    return True


def order_bins(values: Iterable[str], axis: str) -> List[str]:
    config = AXIS_CONFIG[axis]
    order = list(config["order"])
    existing = set(values)
    ordered = [label for label in order if label in existing]
    remaining = sorted(existing - set(ordered))
    return ordered + remaining


def axis_key(row: Dict[str, object], axis: str) -> str:
    return str(row.get(AXIS_CONFIG[axis]["field"], "")).strip()


def make_labels(bins: Sequence[str], counts: Sequence[int], axis: str) -> List[str]:
    config = AXIS_CONFIG[axis]
    prefix = config.get("prefix", "")
    display_map = config.get("display_map", {})
    labels: List[str] = []
    for label, count in zip(bins, counts):
        display = display_map.get(label, label)
        name = f"{prefix}{display}" if prefix else display
        labels.append(f"{name}\n(n={count})")
    return labels


def _apply_y_scale(ax: plt.Axes, values: Sequence[float], y_scale: str) -> None:
    if y_scale == "log":
        positives = [value for value in values if value > 0.0]
        if positives:
            ax.set_yscale("log")


def _style_heatmap_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.minorticks_off()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(which="minor", length=0)
    ax.set_facecolor(plt.cm.viridis(0.0))


def _apply_heatmap_tick_style(
    ax: plt.Axes,
    display_min: float,
    display_max: float,
    *,
    panel: bool = False,
) -> None:
    span = float(display_max) - float(display_min)
    if panel:
        step = 20 if span > 120 else 10
        tick_size = 9
        label_size = 11
    else:
        step = 20 if span > 140 else 10
        tick_size = 10
        label_size = 12
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)


def _heatmap_mesh_and_norm(
    counts_t: np.ndarray,
    color_scale: str,
    vmax_override: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[mcolors.Normalize]]:
    scale = color_scale.strip().lower()
    vmax = float(np.nanmax(counts_t)) if counts_t.size else 0.0
    if vmax_override is not None and vmax_override > 0.0:
        vmax = min(vmax, float(vmax_override))
    if vmax <= 0.0 or scale == "linear":
        return counts_t, None
    if scale == "sqrt":
        return counts_t, mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    if scale == "log":
        masked = np.ma.masked_less_equal(counts_t, 0.0)
        return masked, mcolors.LogNorm(vmin=1.0, vmax=vmax)
    raise ValueError(f"Unknown heatmap color scale: {color_scale}")


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        out_path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _scatter_hist2d(
    x_values: Sequence[float],
    y_values: Sequence[float],
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = math.floor(min(x_values) / bin_size) * bin_size
    x_max = math.ceil(max(x_values) / bin_size) * bin_size
    y_min = math.floor(min(y_values) / bin_size) * bin_size
    y_max = math.ceil(max(y_values) / bin_size) * bin_size
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    if len(x_edges) < 2:
        x_edges = np.array([x_min, x_min + bin_size])
    if len(y_edges) < 2:
        y_edges = np.array([y_min, y_min + bin_size])
    counts, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
    return counts, x_edges, y_edges


def _pad_square_hist(
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_edge = min(float(x_edges[0]), float(y_edges[0]))
    max_edge = max(float(x_edges[-1]), float(y_edges[-1]))
    square_edges = np.arange(min_edge, max_edge + bin_size, bin_size)
    if len(square_edges) < 2:
        square_edges = np.array([min_edge, min_edge + bin_size], dtype=float)

    square_counts = np.zeros((len(square_edges) - 1, len(square_edges) - 1), dtype=float)
    x_offset = int(round((float(x_edges[0]) - min_edge) / bin_size))
    y_offset = int(round((float(y_edges[0]) - min_edge) / bin_size))
    square_counts[
        x_offset : x_offset + counts.shape[0],
        y_offset : y_offset + counts.shape[1],
    ] = counts
    return square_counts, square_edges, square_edges


def _pad_hist_to_edges(
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    target_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    out = np.zeros((len(target_edges) - 1, len(target_edges) - 1), dtype=float)
    x_offset = int(round((float(x_edges[0]) - float(target_edges[0])) / bin_size))
    y_offset = int(round((float(y_edges[0]) - float(target_edges[0])) / bin_size))
    out[
        x_offset : x_offset + counts.shape[0],
        y_offset : y_offset + counts.shape[1],
    ] = counts
    return out


def infer_panel_label(default_label: str, pairs: Sequence[Dict[str, object]]) -> str:
    if default_label:
        return default_label
    methods = sorted(
        {
            str(item.get("set_method")).strip()
            for item in pairs
            if str(item.get("set_method", "")).strip()
        }
    )
    if len(methods) == 1:
        return methods[0]
    return ""


def ensure_single_set_method(pairs: Sequence[Dict[str, object]], label: str) -> None:
    methods = sorted(
        {
            str(item.get("set_method", "")).strip()
            for item in pairs
            if str(item.get("set_method", "")).strip()
        }
    )
    if len(methods) > 1:
        raise SystemExit(
            f"Mixed set_method values in {label}: {', '.join(methods)}. "
            "Use a single-method input run (chain or fan-in) for RQ3 comparison plots."
        )


def build_pairs(
    rq1_by_set: Dict[int, Dict[str, str]],
    rq3_by_set: Dict[int, Dict[str, str]],
    source_field: str,
    target_field: str,
    require_ok: bool,
    axes: Sequence[str],
) -> List[Dict[str, object]]:
    axis_set = set(axes)
    pairs: List[Dict[str, object]] = []
    for set_id in sorted(set(rq1_by_set) & set(rq3_by_set)):
        source = rq1_by_set[set_id]
        target = rq3_by_set[set_id]
        if not row_ok(source, source_field, require_ok):
            continue
        if not row_ok(target, target_field, require_ok):
            continue

        source_val = _to_float(source.get(source_field))
        target_val = _to_float(target.get(target_field))
        if source_val is None or target_val is None:
            continue

        axis_value = str(target.get("axis") or source.get("axis") or "").strip()
        if axis_value not in axis_set:
            continue

        pairs.append(
            {
                "set_id": set_id,
                "axis": axis_value,
                "cp_bin": target.get("cp_bin") or source.get("cp_bin"),
                "r_value": target.get("r_value") or source.get("r_value"),
                "tightness_bin": target.get("tightness_bin") or source.get("tightness_bin"),
                "set_method": target.get("set_method") or source.get("set_method"),
                "source": float(source_val),
                "target": float(target_val),
                "delta": float(source_val - target_val),
            }
        )
    return pairs


def extract_bin_data(
    pairs: Sequence[Dict[str, object]],
    axis: str,
    exclude_bins: set[str],
) -> Tuple[List[str], List[List[float]], List[List[float]], List[List[float]]]:
    grouped_source: Dict[str, List[float]] = {}
    grouped_target: Dict[str, List[float]] = {}
    grouped_delta: Dict[str, List[float]] = {}

    for item in pairs:
        if item.get("axis") != axis:
            continue
        key = axis_key(item, axis)
        if not key or key in exclude_bins:
            continue
        grouped_source.setdefault(key, []).append(float(item["source"]))
        grouped_target.setdefault(key, []).append(float(item["target"]))
        grouped_delta.setdefault(key, []).append(float(item["delta"]))

    bins = order_bins(grouped_source.keys(), axis)
    source_data = [grouped_source.get(label, []) for label in bins]
    target_data = [grouped_target.get(label, []) for label in bins]
    delta_data = [grouped_delta.get(label, []) for label in bins]
    return bins, source_data, target_data, delta_data


def plot_cores_boxplot(
    bins: Sequence[str],
    source_data: Sequence[Sequence[float]],
    target_data: Sequence[Sequence[float]],
    axis: str,
    source_label: str,
    target_label: str,
    y_scale: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    out_path: Path,
) -> Optional[Path]:
    if not bins:
        return None

    positions = [float(i + 1) for i in range(len(bins))]
    width = 0.33
    source_pos = [p - 0.19 for p in positions]
    target_pos = [p + 0.19 for p in positions]

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    source_bp = ax.boxplot(
        source_data,
        positions=source_pos,
        widths=width,
        patch_artist=True,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
    )
    target_bp = ax.boxplot(
        target_data,
        positions=target_pos,
        widths=width,
        patch_artist=True,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
    )

    for patch in source_bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.55)
    for patch in target_bp["boxes"]:
        patch.set_facecolor("#C44E52")
        patch.set_alpha(0.55)

    counts = [max(len(s), len(t)) for s, t in zip(source_data, target_data)]
    ax.set_xticks(positions)
    ax.set_xticklabels(make_labels(bins, counts, axis))
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_ylabel("required cores")

    all_values: List[float] = [v for seq in source_data for v in seq] + [v for seq in target_data for v in seq]
    _apply_y_scale(ax, all_values, y_scale)

    ax.legend(
        handles=[
            Patch(facecolor="#4C72B0", alpha=0.55, label=source_label),
            Patch(facecolor="#C44E52", alpha=0.55, label=target_label),
        ],
        loc="best",
        frameon=False,
    )

    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_delta_boxplot(
    bins: Sequence[str],
    delta_data: Sequence[Sequence[float]],
    axis: str,
    source_label: str,
    target_label: str,
    whis: float | Tuple[float, float],
    show_fliers: bool,
    show_means: bool,
    out_path: Path,
) -> Optional[Path]:
    if not bins:
        return None

    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    positions = [float(i + 1) for i in range(len(bins))]
    ax.boxplot(
        delta_data,
        positions=positions,
        showmeans=show_means,
        showfliers=show_fliers,
        whis=whis,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(make_labels(bins, [len(v) for v in delta_data], axis))
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel(AXIS_CONFIG[axis]["label"])
    ax.set_ylabel(f"delta cores ({source_label} - {target_label})")

    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_scatter(
    pairs: Sequence[Dict[str, object]],
    axis: str,
    source_label: str,
    target_label: str,
    out_path: Path,
) -> Optional[Path]:
    points = [item for item in pairs if item.get("axis") == axis]
    if not points:
        return None

    x_vals = [float(item["source"]) for item in points]
    y_vals = [float(item["target"]) for item in points]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(x_vals, y_vals, s=26, alpha=0.7, color="#4C72B0")

    min_val = min(min(x_vals), min(y_vals))
    max_val = max(max(x_vals), max(y_vals))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(f"cores @ {source_label}")
    ax.set_ylabel(f"cores @ {target_label}")

    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_scatter_heatmap(
    pairs: Sequence[Dict[str, object]],
    axis: str,
    source_label: str,
    target_label: str,
    bin_size: float,
    color_scale: str,
    vmax_quantile: float,
    out_path: Path,
) -> Optional[Path]:
    points = [item for item in pairs if item.get("axis") == axis]
    if not points:
        return None
    if bin_size <= 0:
        raise ValueError("--heatmap-bin-size must be > 0.")

    x_vals = [float(item["source"]) for item in points]
    y_vals = [float(item["target"]) for item in points]
    counts, x_edges, y_edges = _scatter_hist2d(x_vals, y_vals, bin_size)
    counts, x_edges, y_edges = _pad_square_hist(counts, x_edges, y_edges, bin_size)
    display_min = min(min(x_vals), min(y_vals))
    display_max = max(max(x_vals), max(y_vals))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    vmax_override: Optional[float] = None
    if 0.0 < vmax_quantile < 1.0:
        positives = counts[counts > 0]
        if positives.size:
            vmax_override = float(np.quantile(positives, vmax_quantile))
    mesh_data, norm = _heatmap_mesh_and_norm(counts.T, color_scale, vmax_override)
    actual_max = float(np.nanmax(counts.T)) if counts.size else 0.0
    clipped = vmax_override is not None and actual_max > float(vmax_override)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        mesh_data,
        cmap="viridis",
        shading="auto",
        vmin=None if norm is not None else 0.0,
        norm=norm,
    )

    ax.plot([display_min, display_max], [display_min, display_max], "--", color="gray", linewidth=1)
    ax.set_xlim(display_min, display_max)
    ax.set_ylim(display_min, display_max)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    _style_heatmap_axes(ax)
    _apply_heatmap_tick_style(ax, display_min, display_max, panel=False)
    ax.set_xlabel(f"cores @ {source_label}")
    ax.set_ylabel(f"cores @ {target_label}")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="count",
        extend="max" if clipped else "neither",
    )
    if color_scale.strip().lower() != "log":
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()

    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_scatter_heatmap_panel(
    left_pairs: Sequence[Dict[str, object]],
    right_pairs: Sequence[Dict[str, object]],
    axis: str,
    source_label: str,
    target_label: str,
    left_title: str,
    right_title: str,
    bin_size: float,
    color_scale: str,
    vmax_quantile: float,
    out_path: Path,
) -> Optional[Path]:
    left_points = [item for item in left_pairs if item.get("axis") == axis]
    right_points = [item for item in right_pairs if item.get("axis") == axis]
    if not left_points or not right_points:
        return None
    if bin_size <= 0:
        raise ValueError("--heatmap-bin-size must be > 0.")

    left_x = [float(item["source"]) for item in left_points]
    left_y = [float(item["target"]) for item in left_points]
    right_x = [float(item["source"]) for item in right_points]
    right_y = [float(item["target"]) for item in right_points]
    display_min = min(min(left_x), min(left_y), min(right_x), min(right_y))
    display_max = max(max(left_x), max(left_y), max(right_x), max(right_y))

    left_counts, left_x_edges, left_y_edges = _scatter_hist2d(left_x, left_y, bin_size)
    left_counts, left_x_edges, left_y_edges = _pad_square_hist(
        left_counts, left_x_edges, left_y_edges, bin_size
    )
    right_counts, right_x_edges, right_y_edges = _scatter_hist2d(right_x, right_y, bin_size)
    right_counts, right_x_edges, right_y_edges = _pad_square_hist(
        right_counts, right_x_edges, right_y_edges, bin_size
    )

    global_min = min(
        float(left_x_edges[0]),
        float(left_y_edges[0]),
        float(right_x_edges[0]),
        float(right_y_edges[0]),
    )
    global_max = max(
        float(left_x_edges[-1]),
        float(left_y_edges[-1]),
        float(right_x_edges[-1]),
        float(right_y_edges[-1]),
    )
    global_edges = np.arange(global_min, global_max + bin_size, bin_size)
    if len(global_edges) < 2:
        global_edges = np.array([global_min, global_min + bin_size], dtype=float)

    left_counts = _pad_hist_to_edges(
        left_counts, left_x_edges, left_y_edges, global_edges, bin_size
    )
    right_counts = _pad_hist_to_edges(
        right_counts, right_x_edges, right_y_edges, global_edges, bin_size
    )
    vmax = max(float(np.nanmax(left_counts)), float(np.nanmax(right_counts)))
    vmax_override: Optional[float] = None
    if 0.0 < vmax_quantile < 1.0:
        positives = np.concatenate(
            [left_counts[left_counts > 0].ravel(), right_counts[right_counts > 0].ravel()]
        )
        if positives.size:
            vmax_override = float(np.quantile(positives, vmax_quantile))
    scale = color_scale.strip().lower()
    shared_norm: Optional[mcolors.Normalize] = None
    norm_vmax = min(vmax, vmax_override) if (vmax_override is not None and vmax > 0) else vmax
    if norm_vmax > 0 and scale == "sqrt":
        shared_norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=norm_vmax)
    elif norm_vmax > 0 and scale == "log":
        shared_norm = mcolors.LogNorm(vmin=1.0, vmax=norm_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    im = None
    for idx, (ax, counts, title) in enumerate(
        (
            (axes[0], left_counts, left_title),
            (axes[1], right_counts, right_title),
        )
    ):
        mesh_data = counts.T
        if scale == "log":
            mesh_data = np.ma.masked_less_equal(mesh_data, 0.0)
        im = ax.pcolormesh(
            global_edges,
            global_edges,
            mesh_data,
            cmap="viridis",
            shading="auto",
            vmin=None if shared_norm is not None else 0.0,
            vmax=(vmax if vmax > 0 else None) if shared_norm is None else None,
            norm=shared_norm,
        )
        ax.plot([display_min, display_max], [display_min, display_max], "--", color="gray", linewidth=1)
        ax.set_xlim(display_min, display_max)
        ax.set_ylim(display_min, display_max)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        _style_heatmap_axes(ax)
        _apply_heatmap_tick_style(ax, display_min, display_max, panel=True)
        ax.set_xlabel(f"cores @ {source_label}")
        if idx == 0:
            ax.set_ylabel(f"cores @ {target_label}")
        else:
            ax.set_ylabel("")
        if title:
            ax.set_title(title, pad=4, fontsize=12)

    fig.subplots_adjust(left=0.065, right=0.898, top=0.90, bottom=0.16, wspace=0.06)
    # Make the shared colorbar taller/wider so it visually balances two panels.
    cax = fig.add_axes([0.905, 0.145, 0.026, 0.77])
    clipped = vmax_override is not None and vmax > norm_vmax
    cbar = fig.colorbar(im, cax=cax, label="count", extend="max" if clipped else "neither")
    if color_scale.strip().lower() != "log":
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
    cbar.ax.tick_params(labelsize=10)

    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def build_summary_rows(
    pairs: Sequence[Dict[str, object]],
    field: str,
    axis: str,
    bins: Sequence[str],
    source_data: Sequence[Sequence[float]],
    target_data: Sequence[Sequence[float]],
    delta_data: Sequence[Sequence[float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    def append_row(bin_label: str, src: Sequence[float], tgt: Sequence[float], dlt: Sequence[float]) -> None:
        n = len(dlt)
        win = sum(1 for value in dlt if value > 0)
        tie = sum(1 for value in dlt if value == 0)
        lose = sum(1 for value in dlt if value < 0)
        rows.append(
            {
                "field": field,
                "axis": axis,
                "bin": bin_label,
                "n": n,
                "mean_source": _safe_mean(src),
                "mean_target": _safe_mean(tgt),
                "mean_delta": _safe_mean(dlt),
                "median_source": _safe_median(src),
                "median_target": _safe_median(tgt),
                "median_delta": _safe_median(dlt),
                "win_count": win,
                "tie_count": tie,
                "lose_count": lose,
                "win_rate": (win / n) if n else None,
                "tie_rate": (tie / n) if n else None,
                "lose_rate": (lose / n) if n else None,
            }
        )

    for bin_label, src, tgt, dlt in zip(bins, source_data, target_data, delta_data):
        append_row(bin_label, src, tgt, dlt)

    all_src = [value for seq in source_data for value in seq]
    all_tgt = [value for seq in target_data for value in seq]
    all_dlt = [value for seq in delta_data for value in seq]
    append_row("ALL", all_src, all_tgt, all_dlt)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare RQ1 (short SA) vs RQ3 (long SA) on the same tasksets."
    )
    parser.add_argument(
        "--input-rq1",
        type=Path,
        required=True,
        help="ex05_1_results.csv or directory containing it.",
    )
    parser.add_argument(
        "--input-rq3",
        type=Path,
        required=True,
        help="ex05_3_results.csv or directory containing it.",
    )
    parser.add_argument(
        "--input-rq1-right",
        type=Path,
        default=None,
        help="Optional right-panel ex05_1 input for side-by-side heatmap panel.",
    )
    parser.add_argument(
        "--input-rq3-right",
        type=Path,
        default=None,
        help="Optional right-panel ex05_3 input for side-by-side heatmap panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <rq3_run_dir>/plots).",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="cp",
        help="Comma-separated axes to render: cp,ratio,tightness.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="cluster_multipath",
        help="Comma-separated fields to compare (e.g., cluster_multipath,cluster_multipath_critical).",
    )
    parser.add_argument(
        "--source-field",
        type=str,
        default="",
        help="Override source field when a single field is compared.",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="",
        help="Override target field when a single field is compared.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default="",
        help="Legend label for source run (default: inferred from RQ1 meta).",
    )
    parser.add_argument(
        "--label-target",
        type=str,
        default="",
        help="Legend label for target run (default: inferred from RQ3 meta).",
    )
    parser.add_argument(
        "--panel-left-label",
        type=str,
        default="",
        help="Panel title for left heatmap (default: inferred from set_method).",
    )
    parser.add_argument(
        "--panel-right-label",
        type=str,
        default="",
        help="Panel title for right heatmap (default: inferred from set_method).",
    )
    parser.add_argument(
        "--require-ok",
        action="store_true",
        default=True,
        help="Use only rows where status_set and method status are ok.",
    )
    parser.add_argument(
        "--no-require-ok",
        action="store_false",
        dest="require_ok",
        help="Allow rows regardless of status fields.",
    )
    parser.add_argument(
        "--exclude-bins",
        type=str,
        default="",
        help="Comma-separated bins to exclude (e.g., BIN_CP_6).",
    )
    parser.add_argument(
        "--y-scale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for cores boxplot.",
    )
    parser.add_argument(
        "--whis",
        type=str,
        default="minmax",
        help="Whisker range (e.g., 1.5 or minmax).",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        default=False,
        help="Show outliers in boxplots.",
    )
    parser.add_argument(
        "--show-means",
        action="store_true",
        default=True,
        help="Show means in boxplots.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="Also output scatter heatmaps.",
    )
    parser.add_argument(
        "--heatmap-bin-size",
        type=float,
        default=2.0,
        help="Bin size for scatter heatmap (default: 2.0).",
    )
    parser.add_argument(
        "--heatmap-color-scale",
        choices=["linear", "sqrt", "log"],
        default="sqrt",
        help="Color scaling for heatmap counts (default: sqrt).",
    )
    parser.add_argument(
        "--heatmap-vmax-quantile",
        type=float,
        default=1.0,
        help="Clip heatmap color vmax to this quantile of nonzero cell counts (0<q<=1, default: 1.0=no clip).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rq1_path = resolve_results_path(args.input_rq1, "ex05_1_results.csv")
    rq3_path = resolve_results_path(args.input_rq3, "ex05_3_results.csv")

    rq1_by_set = load_rows_by_set(rq1_path)
    rq3_by_set = load_rows_by_set(rq3_path)

    rq1_meta = load_meta(rq1_path.parent / "ex05_1_meta.json")
    rq3_meta = load_meta(rq3_path.parent / "ex05_3_meta.json")

    rq1_right_by_set: Optional[Dict[int, Dict[str, str]]] = None
    rq3_right_by_set: Optional[Dict[int, Dict[str, str]]] = None
    if (args.input_rq1_right is None) ^ (args.input_rq3_right is None):
        raise SystemExit("Specify both --input-rq1-right and --input-rq3-right together.")
    if args.input_rq1_right is not None and args.input_rq3_right is not None:
        rq1_right_path = resolve_results_path(args.input_rq1_right, "ex05_1_results.csv")
        rq3_right_path = resolve_results_path(args.input_rq3_right, "ex05_3_results.csv")
        rq1_right_by_set = load_rows_by_set(rq1_right_path)
        rq3_right_by_set = load_rows_by_set(rq3_right_path)

    source_label = args.label_source or time_label_from_meta(rq1_meta, "source")
    target_label = args.label_target or time_label_from_meta(rq3_meta, "target")

    output_dir = args.output_dir or (rq3_path.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = parse_axes(args.axes)
    fields = parse_fields(args.fields)
    exclude_bins = parse_exclude_bins(args.exclude_bins)
    whis = parse_whis(args.whis)

    all_summary_rows: List[Dict[str, object]] = []
    all_pair_rows: List[Dict[str, object]] = []

    for field in fields:
        source_field = args.source_field or field
        target_field = args.target_field or field
        if len(fields) > 1 and (args.source_field or args.target_field):
            raise SystemExit("--source-field/--target-field can be used only with a single --fields value.")

        pairs = build_pairs(
            rq1_by_set,
            rq3_by_set,
            source_field,
            target_field,
            args.require_ok,
            axes,
        )
        if not pairs:
            print(f"[warn] no comparable rows for field={field}")
            continue
        ensure_single_set_method(pairs, f"left pairs for field={field}")

        for item in pairs:
            row = dict(item)
            row["field"] = field
            row["source_field"] = source_field
            row["target_field"] = target_field
            all_pair_rows.append(row)

        field_tag = field.replace("+", "_plus_")
        right_pairs: Optional[List[Dict[str, object]]] = None
        if rq1_right_by_set is not None and rq3_right_by_set is not None:
            right_pairs = build_pairs(
                rq1_right_by_set,
                rq3_right_by_set,
                source_field,
                target_field,
                args.require_ok,
                axes,
            )
            ensure_single_set_method(right_pairs, f"right pairs for field={field}")
        for axis in axes:
            bins, source_data, target_data, delta_data = extract_bin_data(pairs, axis, exclude_bins)
            if not bins:
                print(f"[warn] no data for field={field}, axis={axis}")
                continue

            plot_cores_boxplot(
                bins,
                source_data,
                target_data,
                axis,
                source_label,
                target_label,
                args.y_scale,
                whis,
                args.show_fliers,
                args.show_means,
                output_dir / f"rq3_compare_cores_{field_tag}_{axis}.png",
            )
            plot_delta_boxplot(
                bins,
                delta_data,
                axis,
                source_label,
                target_label,
                whis,
                args.show_fliers,
                args.show_means,
                output_dir / f"rq3_compare_delta_{field_tag}_{axis}.png",
            )
            plot_scatter(
                pairs,
                axis,
                source_label,
                target_label,
                output_dir / f"rq3_compare_scatter_{field_tag}_{axis}.png",
            )
            if args.heatmap:
                plot_scatter_heatmap(
                    pairs,
                    axis,
                    source_label,
                    target_label,
                    args.heatmap_bin_size,
                    args.heatmap_color_scale,
                    args.heatmap_vmax_quantile,
                    output_dir / f"rq3_compare_scatter_heatmap_{field_tag}_{axis}.png",
                )
                if right_pairs:
                    left_title = infer_panel_label(args.panel_left_label, pairs)
                    right_title = infer_panel_label(args.panel_right_label, right_pairs)
                    plot_scatter_heatmap_panel(
                        pairs,
                        right_pairs,
                        axis,
                        source_label,
                        target_label,
                        left_title,
                        right_title,
                        args.heatmap_bin_size,
                        args.heatmap_color_scale,
                        args.heatmap_vmax_quantile,
                        output_dir
                        / f"rq3_compare_scatter_heatmap_panel_{field_tag}_{axis}.png",
                    )

            all_summary_rows.extend(
                build_summary_rows(
                    pairs,
                    field,
                    axis,
                    bins,
                    source_data,
                    target_data,
                    delta_data,
                )
            )

    write_csv(all_pair_rows, output_dir / "rq3_compare_pairs.csv")
    write_csv(all_summary_rows, output_dir / "rq3_compare_summary.csv")

    print(
        f"[rq3-compare] rq1={rq1_path} rq3={rq3_path} "
        f"pairs={len(all_pair_rows)} summary_rows={len(all_summary_rows)} out={output_dir}"
    )


if __name__ == "__main__":
    main()
