from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METHOD_STATUS = {
    "federated": "status_federated",
    "nocluster_multipath": "status_nocluster_multipath",
    "cluster_multipath": "status_cluster_multipath",
}


def save_plot(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots from ex05 results, highlighting where methods win/lose "
            "and how required cores change across task-set features."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/ex05"),
        help="Input CSV from ex05 or a directory containing run subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/ex06"),
        help="Directory to store the plots.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=10,
        help="Bin size for fed_2018 required cores.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=20,
        help="Bin count for the delta histogram.",
    )
    parser.add_argument(
        "--sched-x",
        type=str,
        default="ub",
        help=(
            "X-axis value for schedulability curve "
            "(e.g., ub, u_n_total, u_o_total, u_avg)."
        ),
    )
    parser.add_argument(
        "--sched-bins",
        type=int,
        default=10,
        help="Bin count for schedulability curve.",
    )
    parser.add_argument(
        "--no-sched",
        action="store_true",
        help="Disable schedulability curve plot.",
    )
    parser.add_argument(
        "--require-ok",
        action="store_true",
        default=True,
        help="Use only rows where federated and cluster_multipath status are ok.",
    )
    parser.add_argument(
        "--no-require-ok",
        action="store_false",
        dest="require_ok",
        help="Include numeric rows even if status is not ok.",
    )
    parser.add_argument(
        "--scatter-methods",
        type=str,
        default="federated,cluster_multipath",
        help="Comma-separated method columns to scatter pairwise.",
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Disable scatter plot generation.",
    )
    parser.add_argument(
        "--scatter-xmin",
        type=float,
        default=30.0,
        help="X-axis min for core-vs-core scatter plots.",
    )
    parser.add_argument(
        "--scatter-xmax",
        type=float,
        default=180.0,
        help="X-axis max for core-vs-core scatter plots.",
    )
    parser.add_argument(
        "--scatter-ymin",
        type=float,
        default=30.0,
        help="Y-axis min for core-vs-core scatter plots.",
    )
    parser.add_argument(
        "--scatter-ymax",
        type=float,
        default=180.0,
        help="Y-axis max for core-vs-core scatter plots.",
    )
    parser.add_argument(
        "--core-ecdf-methods",
        type=str,
        default=None,
        help="Comma-separated method columns for core ECDF (default: scatter methods).",
    )
    parser.add_argument(
        "--no-core-ecdf",
        action="store_true",
        help="Disable core ECDF plot.",
    )
    parser.add_argument(
        "--gap-vs-x",
        type=str,
        default="u_n_total",
        help="Comma-separated x keys for delta/ratio vs x plots.",
    )
    parser.add_argument(
        "--no-gap-vs",
        action="store_true",
        help="Disable gap-vs-x plots for federated vs annealing.",
    )
    parser.add_argument(
        "--boxplot-x",
        type=str,
        default="ub,u_n_total,u_o_total,hi_util_ratio,hi_critical_ratio",
        help="Comma-separated x keys for required-cores boxplots.",
    )
    parser.add_argument(
        "--boxplot-bins",
        type=int,
        default=5,
        help="Bin count for required-cores boxplots.",
    )
    parser.add_argument(
        "--boxplot-methods",
        type=str,
        default="federated,cluster_multipath",
        help="Comma-separated methods for required-cores boxplots.",
    )
    parser.add_argument(
        "--no-boxplot",
        action="store_true",
        help="Disable required-cores boxplots.",
    )
    parser.add_argument(
        "--no-boxplot-logy",
        action="store_false",
        dest="boxplot_logy",
        help="Use linear y-axis for required-cores boxplots.",
    )
    parser.add_argument(
        "--win-tol",
        type=float,
        default=0.0,
        help="Tie tolerance for best/worst/regret plots.",
    )
    parser.add_argument(
        "--best-worst-methods",
        type=str,
        default=None,
        help="Comma-separated methods for best/worst counts (default: scatter methods).",
    )
    parser.add_argument(
        "--no-best-worst",
        action="store_true",
        help="Disable best/worst count plot.",
    )
    parser.add_argument(
        "--regret-methods",
        type=str,
        default=None,
        help="Comma-separated methods for regret boxplot (default: scatter methods).",
    )
    parser.add_argument(
        "--no-regret",
        action="store_true",
        help="Disable regret boxplot.",
    )
    return parser


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


def resolve_input_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = list(path.rglob("ex05_results.csv"))
        if not candidates:
            raise FileNotFoundError(path / "ex05_results.csv")
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return path


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _require_status_ok(row: Dict[str, str], keys: Sequence[str]) -> bool:
    for key in keys:
        if key and row.get(key) != "ok":
            return False
    return True


def _sched_x_value(row: Dict[str, str], key: str) -> Optional[float]:
    if key == "u_avg":
        u_n = _parse_optional_float(row.get("u_n_total"))
        u_o = _parse_optional_float(row.get("u_o_total"))
        if u_n is None or u_o is None:
            return None
        return (u_n + u_o) / 2.0
    if key == "hi_util_ratio":
        hi = _parse_optional_float(row.get("high_util_tasks"))
        total = _parse_optional_float(row.get("tasks"))
        if hi is None or total is None or total <= 0:
            return None
        return hi / total
    if key == "hi_critical_ratio":
        hi = _parse_optional_float(row.get("hi_critical_tasks"))
        total = _parse_optional_float(row.get("tasks"))
        if hi is None or total is None or total <= 0:
            return None
        return hi / total
    return _parse_optional_float(row.get(key))


def _collect_method_value(
    row: Dict[str, str],
    method: str,
    require_ok: bool,
) -> Optional[float]:
    status_key = METHOD_STATUS.get(method)
    if require_ok and status_key and row.get(status_key) != "ok":
        return None
    return _parse_optional_float(row.get(method))


def _collect_method_values(
    row: Dict[str, str],
    methods: Sequence[str],
    require_ok: bool,
) -> Optional[Dict[str, float]]:
    values: Dict[str, float] = {}
    for method in methods:
        value = _collect_method_value(row, method, require_ok)
        if value is None:
            return None
        values[method] = value
    return values


def extract_fed_anneal(
    rows: Sequence[Dict[str, str]],
    require_ok: bool,
) -> List[Dict[str, float]]:
    data: List[Dict[str, float]] = []
    for row in rows:
        if require_ok and not _require_status_ok(
            row, ["status_federated", "status_cluster_multipath"]
        ):
            continue
        fed = _parse_optional_int(row.get("federated"))
        anneal = _parse_optional_int(row.get("cluster_multipath"))
        if fed is None or anneal is None:
            continue
        delta = float(fed - anneal)
        ratio = float(anneal) / float(fed) if fed > 0 else float("nan")
        data.append(
            {
                "fed": float(fed),
                "anneal": float(anneal),
                "delta": delta,
                "ratio": ratio,
            }
        )
    return data


def build_bins(
    data: Sequence[Dict[str, float]],
    bin_size: int,
) -> Dict[int, List[float]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    bins: Dict[int, List[float]] = {}
    for row in data:
        fed = row["fed"]
        bin_start = int(fed // bin_size) * bin_size
        bins.setdefault(bin_start, []).append(row["delta"])
    return bins


def plot_schedulability_curve(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    x_key: str,
    bins: int,
    out_dir: Path,
) -> Optional[Path]:
    values = [v for row in rows if (v := _sched_x_value(row, x_key)) is not None]
    if not values:
        return None
    if bins <= 0:
        bins = 10
    x_min = min(values)
    x_max = max(values)
    if x_max <= x_min:
        x_max = x_min + 1e-9
    width = (x_max - x_min) / bins
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
            key = METHOD_STATUS.get(method)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sched_curve_{x_key}.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_boxplot(
    bins: Dict[int, List[float]],
    bin_size: int,
    out_dir: Path,
) -> Optional[Path]:
    if not bins:
        return None
    sorted_bins = sorted(bins.items(), key=lambda item: item[0])
    data = [values for _, values in sorted_bins]
    labels = [
        f"{start}-{start + bin_size - 1}\\n(n={len(values)})"
        for start, values in sorted_bins
    ]
    width = max(6.0, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Fed-2018 required cores (bin size {bin_size})")
    ax.set_ylabel("Fed-2018 cores - Annealing cores")
    ax.set_title("Fed-2018 core bins vs annealing gap (boxplot)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fed2018_anneal_gap_boxplot.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_bin_stats(
    bins: Dict[int, List[float]],
    bin_size: int,
    out_dir: Path,
) -> Optional[Path]:
    if not bins:
        return None
    sorted_bins = sorted(bins.items(), key=lambda item: item[0])
    centers: List[float] = []
    means: List[float] = []
    medians: List[float] = []
    counts: List[int] = []
    for start, values in sorted_bins:
        centers.append(start + bin_size / 2.0)
        means.append(statistics.fmean(values))
        medians.append(statistics.median(values))
        counts.append(len(values))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, means, marker="o", label="mean")
    ax.plot(centers, medians, marker="s", linestyle="--", label="median")
    for x, y, count in zip(centers, means, counts):
        ax.text(x, y, f"n={count}", fontsize=8, ha="center", va="bottom")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Fed-2018 required cores (bin size {bin_size})")
    ax.set_ylabel("Fed-2018 cores - Annealing cores")
    ax.set_title("Fed-2018 bins vs annealing gap (mean/median)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fed2018_anneal_gap_bin_stats.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_histogram(
    deltas: Sequence[float],
    bins: int,
    out_dir: Path,
) -> Optional[Path]:
    if not deltas:
        return None
    if bins <= 0:
        bins = 20
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(deltas, bins=bins, alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Fed-2018 cores - Annealing cores")
    ax.set_ylabel("Count")
    ax.set_title("Delta distribution (histogram)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fed2018_anneal_gap_hist.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_ecdf(
    deltas: Sequence[float],
    out_dir: Path,
) -> Optional[Path]:
    if not deltas:
        return None
    xs = sorted(deltas)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(xs, ys, where="post")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Fed-2018 cores - Annealing cores")
    ax.set_ylabel("ECDF")
    ax.set_title("Delta distribution (ECDF)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fed2018_anneal_gap_ecdf.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_core_ecdf(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    require_ok: bool,
    out_dir: Path,
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for method in methods:
        values: List[float] = []
        status_key = METHOD_STATUS.get(method)
        for row in rows:
            if require_ok and status_key and row.get(status_key) != "ok":
                continue
            val = _parse_optional_float(row.get(method))
            if val is None:
                continue
            values.append(val)
        if not values:
            continue
        xs = sorted(values)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", label=f"{method} (n={len(xs)})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_xlabel("Required cores")
    ax.set_ylabel("ECDF")
    ax.set_title("Required cores ECDF")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "core_ecdf.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_scatter(
    xs: Sequence[float],
    ys: Sequence[float],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    line: Optional[Tuple[float, float]] = None,
    horizontal: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=18, alpha=0.7)
    if line is not None:
        min_v, max_v = line
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray", linewidth=1)
    if horizontal is not None:
        ax.axhline(horizontal, color="gray", linestyle="--", linewidth=1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)


def plot_fed_anneal_scatter(
    data: Sequence[Dict[str, float]],
    out_dir: Path,
    scatter_xmin: float,
    scatter_xmax: float,
    scatter_ymin: float,
    scatter_ymax: float,
) -> None:
    if not data:
        return
    fed = [row["fed"] for row in data]
    anneal = [row["anneal"] for row in data]
    delta = [row["delta"] for row in data]
    ratio_points = [
        (row["fed"], row["ratio"]) for row in data if math.isfinite(row["ratio"])
    ]

    x_min = scatter_xmin
    x_max = scatter_xmax
    y_min = scatter_ymin
    y_max = scatter_ymax
    line_min = max(x_min, y_min)
    line_max = min(x_max, y_max)
    line = (line_min, line_max) if line_max > line_min else None
    plot_scatter(
        fed,
        anneal,
        out_dir / "scatter_fed2018_vs_anneal.png",
        "Fed-2018 required cores",
        "Annealing required cores",
        f"Fed-2018 vs annealing (n={len(fed)})",
        line=line,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
    )
    plot_scatter(
        fed,
        delta,
        out_dir / "scatter_fed2018_delta_vs_fed.png",
        "Fed-2018 required cores",
        "Fed-2018 cores - Annealing cores",
        f"Delta vs fed-2018 (n={len(fed)})",
        horizontal=0.0,
    )
    if ratio_points:
        ratio_x = [item[0] for item in ratio_points]
        ratio_y = [item[1] for item in ratio_points]
        plot_scatter(
            ratio_x,
            ratio_y,
            out_dir / "scatter_fed2018_ratio_vs_fed.png",
            "Fed-2018 required cores",
            "Annealing / Fed-2018",
            f"Ratio vs fed-2018 (n={len(ratio_points)})",
            horizontal=1.0,
        )


def plot_method_scatter_pairs(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    require_ok: bool,
    out_dir: Path,
    scatter_xmin: float,
    scatter_xmax: float,
    scatter_ymin: float,
    scatter_ymax: float,
) -> None:
    filtered = [m for m in methods if m]
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            m1 = filtered[i]
            m2 = filtered[j]
            xs: List[float] = []
            ys: List[float] = []
            for row in rows:
                status_keys = []
                if require_ok:
                    status_keys = [
                        METHOD_STATUS.get(m1, ""),
                        METHOD_STATUS.get(m2, ""),
                    ]
                    if not _require_status_ok(row, [k for k in status_keys if k]):
                        continue
                x = _parse_optional_float(row.get(m1))
                y = _parse_optional_float(row.get(m2))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if not xs or not ys:
                continue
            x_min = scatter_xmin
            x_max = scatter_xmax
            y_min = scatter_ymin
            y_max = scatter_ymax
            line_min = max(x_min, y_min)
            line_max = min(x_max, y_max)
            line = (line_min, line_max) if line_max > line_min else None
            plot_scatter(
                xs,
                ys,
                out_dir / f"scatter_{m1}_vs_{m2}.png",
                m1,
                m2,
                f"{m1} vs {m2} (n={len(xs)})",
                line=line,
                xlim=(x_min, x_max),
                ylim=(y_min, y_max),
            )


def plot_gap_vs_x(
    rows: Sequence[Dict[str, str]],
    x_key: str,
    require_ok: bool,
    out_dir: Path,
) -> None:
    xs: List[float] = []
    deltas: List[float] = []
    ratios: List[float] = []
    for row in rows:
        if require_ok and not _require_status_ok(
            row, ["status_federated", "status_cluster_multipath"]
        ):
            continue
        x_val = _sched_x_value(row, x_key)
        fed = _parse_optional_float(row.get("federated"))
        anneal = _parse_optional_float(row.get("cluster_multipath"))
        if x_val is None or fed is None or anneal is None:
            continue
        xs.append(x_val)
        deltas.append(fed - anneal)
        ratios.append(anneal / fed if fed > 0 else float("nan"))
    if xs and deltas:
        plot_scatter(
            xs,
            deltas,
            out_dir / f"gap_delta_vs_{x_key}.png",
            x_key,
            "Fed-2018 cores - Annealing cores",
            f"Delta vs {x_key} (n={len(xs)})",
            horizontal=0.0,
        )
    ratio_points = [(x, r) for x, r in zip(xs, ratios) if math.isfinite(r)]
    if ratio_points:
        rx = [item[0] for item in ratio_points]
        ry = [item[1] for item in ratio_points]
        plot_scatter(
            rx,
            ry,
            out_dir / f"gap_ratio_vs_{x_key}.png",
            x_key,
            "Annealing / Fed-2018",
            f"Ratio vs {x_key} (n={len(rx)})",
            horizontal=1.0,
        )


def plot_best_worst_counts(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    tol: float,
    require_ok: bool,
    out_dir: Path,
) -> Optional[Path]:
    best_counts = {method: 0 for method in methods}
    worst_counts = {method: 0 for method in methods}
    total = 0
    for row in rows:
        values = _collect_method_values(row, methods, require_ok)
        if values is None:
            continue
        total += 1
        min_val = min(values.values())
        max_val = max(values.values())
        for method, value in values.items():
            if abs(value - min_val) <= tol:
                best_counts[method] += 1
            if abs(value - max_val) <= tol:
                worst_counts[method] += 1
    if total == 0:
        return None
    labels = list(methods)
    x_pos = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([x - 0.2 for x in x_pos], [best_counts[m] for m in labels], width=0.4, label="best")
    ax.bar([x + 0.2 for x in x_pos], [worst_counts[m] for m in labels], width=0.4, label="worst")
    for x, method in zip(x_pos, labels):
        ax.text(x - 0.2, best_counts[method] + 0.5, str(best_counts[method]), ha="center", va="bottom", fontsize=8)
        ax.text(x + 0.2, worst_counts[method] + 0.5, str(worst_counts[method]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Best/Worst counts (n={total})")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_worst_counts.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_regret_boxplot(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    require_ok: bool,
    out_dir: Path,
) -> Optional[Path]:
    regrets: Dict[str, List[float]] = {method: [] for method in methods}
    for row in rows:
        values = _collect_method_values(row, methods, require_ok)
        if values is None:
            continue
        min_val = min(values.values())
        for method, value in values.items():
            regrets[method].append(value - min_val)
    if not any(regrets.values()):
        return None
    labels = [f"{m}\\n(n={len(regrets[m])})" for m in methods]
    data = [regrets[m] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Regret vs best (cores)")
    ax.set_title("Regret distribution by method")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regret_boxplot.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def plot_core_boxplots_by_bin(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    x_key: str,
    bins: int,
    logy: bool,
    require_ok: bool,
    out_dir: Path,
) -> Optional[Path]:
    samples: List[Tuple[float, Dict[str, float]]] = []
    for row in rows:
        x_val = _sched_x_value(row, x_key)
        if x_val is None:
            continue
        values = _collect_method_values(row, methods, require_ok)
        if values is None:
            continue
        samples.append((x_val, values))
    if not samples:
        return None
    if bins <= 0:
        bins = 5
    x_vals = [sample[0] for sample in samples]
    x_min = min(x_vals)
    x_max = max(x_vals)
    if x_max <= x_min:
        x_max = x_min + 1e-9
    width = (x_max - x_min) / bins

    bin_data = [{method: [] for method in methods} for _ in range(bins)]
    bin_counts = [0 for _ in range(bins)]
    for x_val, values in samples:
        idx = int((x_val - x_min) / width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        bin_counts[idx] += 1
        for method in methods:
            bin_data[idx][method].append(values[method])

    data: List[List[float]] = []
    positions: List[float] = []
    box_methods: List[str] = []
    xticks: List[float] = []
    xticklabels: List[str] = []
    bins_with_data = 0
    spacing = len(methods) + 1

    for idx in range(bins):
        if bin_counts[idx] == 0:
            continue
        bins_with_data += 1
        center = bins_with_data * spacing
        xticks.append(center + (len(methods) - 1) / 2.0)
        bin_start = x_min + idx * width
        bin_end = bin_start + width
        xticklabels.append(f"{bin_start:.3g}-{bin_end:.3g}\\n(n={bin_counts[idx]})")
        for m_idx, method in enumerate(methods):
            values = bin_data[idx][method]
            if not values:
                continue
            data.append(values)
            positions.append(center + m_idx)
            box_methods.append(method)

    if not data:
        return None
    fig_w = max(7.0, bins_with_data * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    for patch, method in zip(bp["boxes"], box_methods):
        patch.set_facecolor(color_map[method])
    handles = [Patch(facecolor=color_map[m], label=m) for m in methods]
    ax.legend(handles=handles)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_xlabel(x_key)
    if logy:
        min_val = min(min(values) for values in data if values)
        if min_val > 0:
            ax.set_yscale("log")
            ax.set_ylabel("Required cores (log scale)")
        else:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylabel("Required cores (symlog)")
    else:
        ax.set_ylabel("Required cores")
    ax.set_title(f"Required cores by {x_key} (binned)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cores_by_{x_key}_boxplot.png"
    fig.tight_layout()
    save_plot(fig, out_path)
    plt.close(fig)
    return out_path


def _parse_method_list(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items


def main() -> None:
    args = build_parser().parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be > 0")
    if args.hist_bins <= 0:
        raise SystemExit("--hist-bins must be > 0")
    if args.sched_bins <= 0:
        raise SystemExit("--sched-bins must be > 0")
    if args.boxplot_bins <= 0:
        raise SystemExit("--boxplot-bins must be > 0")

    input_path = resolve_input_path(args.input)
    try:
        rows = load_rows(input_path)
    except FileNotFoundError:
        raise SystemExit(f"Input CSV not found: {input_path}") from None
    if not rows:
        raise SystemExit("Input CSV has no rows.")

    data = extract_fed_anneal(rows, args.require_ok)
    if not data:
        raise SystemExit("No data matched the filters.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = build_bins(data, args.bin_size)
    plot_boxplot(bins, args.bin_size, out_dir)
    plot_bin_stats(bins, args.bin_size, out_dir)

    deltas = [row["delta"] for row in data]
    plot_histogram(deltas, args.hist_bins, out_dir)
    plot_ecdf(deltas, out_dir)
    plot_fed_anneal_scatter(
        data,
        out_dir,
        args.scatter_xmin,
        args.scatter_xmax,
        args.scatter_ymin,
        args.scatter_ymax,
    )

    scatter_methods = _parse_method_list(args.scatter_methods)
    if not args.no_scatter and scatter_methods:
        plot_method_scatter_pairs(
            rows,
            scatter_methods,
            args.require_ok,
            out_dir,
            args.scatter_xmin,
            args.scatter_xmax,
            args.scatter_ymin,
            args.scatter_ymax,
        )

    if not args.no_sched and scatter_methods:
        plot_schedulability_curve(rows, scatter_methods, args.sched_x, args.sched_bins, out_dir)

    core_ecdf_methods = scatter_methods
    if args.core_ecdf_methods:
        core_ecdf_methods = _parse_method_list(args.core_ecdf_methods)
    if not args.no_core_ecdf and core_ecdf_methods:
        plot_core_ecdf(rows, core_ecdf_methods, args.require_ok, out_dir)

    if not args.no_gap_vs:
        for x_key in _parse_method_list(args.gap_vs_x):
            plot_gap_vs_x(rows, x_key, args.require_ok, out_dir)

    if not args.no_boxplot:
        boxplot_methods = _parse_method_list(args.boxplot_methods)
        for x_key in _parse_method_list(args.boxplot_x):
            plot_core_boxplots_by_bin(
                rows,
                boxplot_methods,
                x_key,
                args.boxplot_bins,
                args.boxplot_logy,
                args.require_ok,
                out_dir,
            )

    best_worst_methods = scatter_methods
    if args.best_worst_methods:
        best_worst_methods = _parse_method_list(args.best_worst_methods)
    if not args.no_best_worst and best_worst_methods:
        plot_best_worst_counts(
            rows,
            best_worst_methods,
            args.win_tol,
            args.require_ok,
            out_dir,
        )

    regret_methods = scatter_methods
    if args.regret_methods:
        regret_methods = _parse_method_list(args.regret_methods)
    if not args.no_regret and regret_methods:
        plot_regret_boxplot(rows, regret_methods, args.require_ok, out_dir)

    print(f"[ex06] points={len(data)} output_dir={out_dir}")


if __name__ == "__main__":
    main()
