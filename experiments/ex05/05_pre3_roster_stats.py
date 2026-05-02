from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.generator.wrapper import RDGenWrapper


DEFAULT_METRICS: Dict[str, str] = {
    "nodes": "Nodes",
    "edges_actual": "Edges",
    "edge_density": "Edge density",
    "workload_hi": "Workload (HI)",
    "critical_hi": "Critical path (HI)",
    "work_per_node": "Work per node (HI)",
    "critical_per_node": "Critical per node (HI)",
    "edges_per_node": "Edges per node",
}

NUMERIC_COLUMNS = (
    "nodes",
    "edges_target",
    "edges_actual",
    "edge_density",
    "workload_hi",
    "workload_lo",
    "critical_hi",
    "cpr",
    "extra_edges_target",
    "extra_edges_added",
    "attempts",
    "period",
    "deadline",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize per-CPR-bin roster characteristics."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/results/ex05_pre/05_pre1"),
        help="ex05_pre run directory that contains rosters/.",
    )
    parser.add_argument(
        "--roster-dir",
        type=Path,
        default=None,
        help="Roster directory (default: <run-dir>/rosters).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/plots).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS.keys()),
        help="Comma-separated metrics to boxplot.",
    )
    parser.add_argument(
        "--no-boxplot",
        action="store_true",
        default=False,
        help="Disable boxplot generation.",
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        default=False,
        help="Disable scatter plots.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        help="Disable summary CSV output.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="rdgen_method",
        help="Category column for colored scatter (default: rdgen_method).",
    )
    parser.add_argument(
        "--split-by-method",
        action="store_true",
        default=False,
        help="Also output separate plots per rdgen_method.",
    )
    parser.add_argument(
        "--no-indegree",
        action="store_true",
        default=False,
        help="Disable indegree distribution plots.",
    )
    return parser


def load_rosters(roster_dir: Path) -> pd.DataFrame:
    catalog = roster_dir / "dag_catalog.csv"
    if catalog.exists():
        df = pd.read_csv(catalog)
        df["source"] = catalog.name
        return df

    files = sorted(roster_dir.glob("dag_roster_cpr_*.csv"))
    if not files:
        raise SystemExit(f"No roster files found in {roster_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        tmp = pd.read_csv(path)
        tmp["source"] = path.name
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_cpr(df: pd.DataFrame) -> pd.DataFrame:
    if "cpr" not in df.columns:
        df["cpr"] = np.nan
    missing = df["cpr"].isna()
    if "critical_hi" in df.columns and "workload_hi" in df.columns:
        denom = df["workload_hi"]
        with np.errstate(divide="ignore", invalid="ignore"):
            calc = df["critical_hi"] / denom
        df.loc[missing, "cpr"] = calc[missing]
    return df


def ensure_edge_density(df: pd.DataFrame) -> pd.DataFrame:
    if "edge_density" not in df.columns:
        df["edge_density"] = np.nan
    if "edges_actual" not in df.columns or "nodes" not in df.columns:
        return df

    mask = df["edge_density"].isna()
    valid = df["nodes"].notna() & (df["nodes"] > 1) & df["edges_actual"].notna()
    mask = mask & valid
    if mask.any():
        max_edges = df.loc[mask, "nodes"] * (df.loc[mask, "nodes"] - 1) / 2.0
        df.loc[mask, "edge_density"] = df.loc[mask, "edges_actual"] / max_edges
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "nodes" in df.columns:
        df["work_per_node"] = df.get("workload_hi") / df["nodes"]
        df["critical_per_node"] = df.get("critical_hi") / df["nodes"]
        df["edges_per_node"] = df.get("edges_actual") / df["nodes"]
    return df


def add_utilization(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, bool]:
    if "workload_hi" not in df.columns:
        return df, "", False

    workload_lo_missing = False
    if "workload_lo" not in df.columns:
        df["workload_lo"] = df["workload_hi"]
        workload_lo_missing = True

    denom = pd.Series(np.nan, index=df.index)
    has_period = "period" in df.columns and (df["period"] > 0).any()
    has_deadline = "deadline" in df.columns and (df["deadline"] > 0).any()

    if "period" in df.columns:
        denom = df["period"].where(df["period"] > 0, np.nan)
    if "deadline" in df.columns:
        denom = denom.fillna(df["deadline"].where(df["deadline"] > 0, np.nan))
    if "critical_hi" in df.columns:
        denom = denom.fillna(df["critical_hi"].where(df["critical_hi"] > 0, np.nan))

    if has_period and has_deadline:
        denom_label = "period>deadline>critical_hi"
    elif has_period:
        denom_label = "period>critical_hi"
    elif has_deadline:
        denom_label = "deadline>critical_hi"
    else:
        denom_label = "critical_hi"

    df["util_denom"] = denom
    with np.errstate(divide="ignore", invalid="ignore"):
        df["util_hi"] = df["workload_hi"] / denom
        df["util_lo"] = df["workload_lo"] / denom

    return df, denom_label, workload_lo_missing


def choose_bin_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("cpr_class", "cpr_class_3"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def add_bin_column(df: pd.DataFrame, bin_col: Optional[str]) -> pd.DataFrame:
    if bin_col is not None:
        df["bin"] = df[bin_col]
        df["bin"] = df["bin"].where(df["bin"].notna(), None)
        return df

    if "cpr" not in df.columns:
        df["bin"] = None
        return df

    bins = np.arange(0.0, 1.01, 0.1)
    labels = [f"{bins[i]:.1f}_{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    df["bin"] = pd.cut(df["cpr"], bins=bins, labels=labels, right=False)
    return df


def sort_bins(labels: Iterable[str]) -> List[str]:
    labels = [str(l) for l in labels if l is not None and str(l) != "nan"]
    if not labels:
        return []
    unique = list(dict.fromkeys(labels))

    if set(unique).issubset({"small", "middle", "big"}):
        order = [label for label in ("small", "middle", "big") if label in unique]
        rest = sorted([label for label in unique if label not in order])
        return order + rest

    def key(label: str) -> Tuple[int, float, str]:
        if "_" in label:
            left = label.split("_")[0]
            try:
                return (0, float(left), label)
            except ValueError:
                return (1, 0.0, label)
        return (2, 0.0, label)

    return sorted(unique, key=key)


def ensure_method_column(df: pd.DataFrame) -> pd.DataFrame:
    if "rdgen_method" in df.columns and df["rdgen_method"].notna().any():
        return df
    if "source" not in df.columns:
        df["rdgen_method"] = "unknown"
        return df
    source = df["source"].astype(str)
    df["rdgen_method"] = np.where(
        source.str.contains("__chain"),
        "chain",
        np.where(source.str.contains("__fan-in"), "fan-in", "unknown"),
    )
    return df


def render_outputs(
    df: pd.DataFrame,
    bins: List[str],
    metrics: List[str],
    out_dir: Path,
    util_denom_label: str,
    workload_lo_missing: bool,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_summary and metrics:
        write_summary(df, metrics, bins, out_dir / "roster_summary_by_cpr.csv")

    plot_bin_counts(df, bins, out_dir)

    if not args.no_boxplot:
        for metric in metrics:
            label = DEFAULT_METRICS.get(metric, metric)
            plot_boxplot(df, bins, metric, label, out_dir)
        plot_utilization_boxplot(df, bins, out_dir, util_denom_label)

    if not args.no_scatter:
        plot_scatter_panels(df, out_dir)
        plot_scatter_by_category(df, "cpr", "nodes", args.category, out_dir)

    if not args.no_indegree:
        indegree_all, indegree_by_method = collect_degree_distributions(df, "in")
        if indegree_all:
            plot_indegree_histogram(
                indegree_all,
                out_dir / "roster_indegree_hist.png",
                "Indegree distribution (all DAGs)",
            )
        if indegree_by_method:
            plot_indegree_panels(indegree_by_method, out_dir / "roster_indegree_by_method.png")

        outdegree_all, outdegree_by_method = collect_degree_distributions(df, "out")
        if outdegree_all:
            plot_indegree_histogram(
                outdegree_all,
                out_dir / "roster_outdegree_hist.png",
                "Outdegree distribution (all DAGs)",
            )
        if outdegree_by_method:
            plot_indegree_panels(outdegree_by_method, out_dir / "roster_outdegree_by_method.png")

    if workload_lo_missing:
        print("Note: workload_lo not found; using workload_hi for LO utilization.")
    print(f"Saved plots to {out_dir}")


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))


def plot_bin_counts(df: pd.DataFrame, bins: Sequence[str], out_dir: Path) -> None:
    counts = [int(df[df["bin"] == label].shape[0]) for label in bins]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(bins)), 4.2))
    ax.bar(bins, counts, color="#4c72b0")
    ax.set_xlabel("CPR bin")
    ax.set_ylabel("Count")
    ax.set_title("Roster counts by CPR bin")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, out_dir / "roster_bin_counts.png")


def plot_boxplot(
    df: pd.DataFrame,
    bins: Sequence[str],
    metric: str,
    label: str,
    out_dir: Path,
) -> None:
    data = []
    labels = []
    for bin_label in bins:
        series = df[df["bin"] == bin_label][metric].dropna()
        if series.empty:
            continue
        data.append(series.values)
        labels.append(bin_label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(labels)), 4.6))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_xlabel("CPR bin")
    ax.set_ylabel(label)
    ax.set_title(f"{label} by CPR bin")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, out_dir / f"roster_boxplot_{metric}.png")


def plot_utilization_boxplot(
    df: pd.DataFrame, bins: Sequence[str], out_dir: Path, denom_label: str
) -> None:
    if "util_lo" not in df.columns or "util_hi" not in df.columns:
        return

    lo_data: List[np.ndarray] = []
    hi_data: List[np.ndarray] = []
    lo_pos: List[float] = []
    hi_pos: List[float] = []

    for idx, bin_label in enumerate(bins):
        subset = df[df["bin"] == bin_label]
        lo = subset["util_lo"].dropna()
        hi = subset["util_hi"].dropna()
        base = idx + 1
        if not lo.empty:
            lo_data.append(lo.values)
            lo_pos.append(base - 0.18)
        if not hi.empty:
            hi_data.append(hi.values)
            hi_pos.append(base + 0.18)

    if not lo_data and not hi_data:
        return

    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(bins)), 4.6))
    if lo_data:
        bp_lo = ax.boxplot(
            lo_data,
            positions=lo_pos,
            widths=0.3,
            patch_artist=True,
            showmeans=True,
        )
        for patch in bp_lo["boxes"]:
            patch.set_facecolor("#4c72b0")
    if hi_data:
        bp_hi = ax.boxplot(
            hi_data,
            positions=hi_pos,
            widths=0.3,
            patch_artist=True,
            showmeans=True,
        )
        for patch in bp_hi["boxes"]:
            patch.set_facecolor("#dd8452")

    ax.set_xticks([i + 1 for i in range(len(bins))])
    ax.set_xticklabels(bins, rotation=45)
    ax.set_xlabel("CPR bin")
    ax.set_ylabel("Utilization")
    title = "Utilization by CPR bin (LO vs HI)"
    if denom_label:
        title += f" [denom={denom_label}]"
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="#4c72b0", label="LO"),
            Patch(facecolor="#dd8452", label="HI"),
        ],
        loc="best",
        frameon=False,
        fontsize=9,
    )
    save_plot(fig, out_dir / "roster_boxplot_utilization_lo_hi.png")


def plot_scatter_panels(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = []
    if "cpr" in df.columns and "nodes" in df.columns:
        pairs.append(("cpr", "nodes", "Nodes"))
    if "cpr" in df.columns and "workload_hi" in df.columns:
        pairs.append(("cpr", "workload_hi", "Workload (HI)"))
    if "cpr" in df.columns and "critical_hi" in df.columns:
        pairs.append(("cpr", "critical_hi", "Critical path (HI)"))
    if "cpr" in df.columns and "edge_density" in df.columns:
        pairs.append(("cpr", "edge_density", "Edge density"))

    if not pairs:
        return

    cols = 2
    rows = int(math.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.8 * rows))
    axes_list = np.array(axes).reshape(-1)

    for ax, (x, y, y_label) in zip(axes_list, pairs):
        subset = df[[x, y]].dropna()
        ax.scatter(subset[x], subset[y], s=18, alpha=0.7, color="#1f77b4")
        corr = subset[x].corr(subset[y]) if len(subset) > 1 else float("nan")
        title = f"{y_label} vs CPR"
        if not np.isnan(corr):
            title += f" (r={corr:.2f})"
        ax.set_title(title)
        ax.set_xlabel("CPR")
        ax.set_ylabel(y_label)

    for ax in axes_list[len(pairs) :]:
        ax.axis("off")

    save_plot(fig, out_dir / "roster_scatter_cpr_panels.png")


def plot_scatter_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    out_dir: Path,
) -> None:
    if category not in df.columns:
        return
    subset = df[[x, y, category]].dropna()
    if subset.empty:
        return

    categories = sorted(subset[category].unique().tolist())
    if not categories:
        return

    cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, cat in enumerate(categories):
        rows = subset[subset[category] == cat]
        ax.scatter(
            rows[x],
            rows[y],
            s=20,
            alpha=0.7,
            color=cmap(idx % cmap.N),
            label=str(cat),
        )

    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(f"{y.replace('_', ' ').title()} vs {x.upper()} by {category}")
    ax.legend(loc="best", fontsize=9, frameon=False)
    save_plot(fig, out_dir / f"roster_scatter_{x}_vs_{y}_by_{category}.png")


def collect_degree_distributions(
    df: pd.DataFrame,
    mode: str,
) -> Tuple[List[int], Dict[str, List[int]]]:
    if mode not in {"in", "out"}:
        raise ValueError(f"mode must be 'in' or 'out' (got {mode})")
    if "xml_path" not in df.columns:
        return [], {}
    paths = df["xml_path"].dropna().astype(str)
    if paths.empty:
        return [], {}

    method_map: Dict[str, str] = {}
    if "rdgen_method" in df.columns:
        for _, row in df.iterrows():
            xml_path = row.get("xml_path")
            if not xml_path:
                continue
            if xml_path in method_map:
                continue
            method = str(row.get("rdgen_method") or "unknown")
            method_map[str(xml_path)] = method

    wrapper = RDGenWrapper(verbose=False)
    indegree_all: List[int] = []
    indegree_by_method: Dict[str, List[int]] = {}
    for xml_path in sorted(set(paths)):
        parsed = wrapper.parse_xml(xml_path)
        if not parsed:
            continue
        dag = parsed[0]
        method = method_map.get(xml_path, "unknown")
        bucket = indegree_by_method.setdefault(method, [])
        for node in dag.nodes.values():
            if len(node.predecessors) == 0:
                continue
            if len(node.successors) == 0:
                continue
            deg = len(node.predecessors) if mode == "in" else len(node.successors)
            indegree_all.append(deg)
            bucket.append(deg)
    return indegree_all, indegree_by_method


def plot_indegree_histogram(values: Sequence[int], out_path: Path, title: str) -> None:
    if not values:
        return
    vmax = max(values)
    bins = np.arange(-0.5, vmax + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    counts, edges, patches = ax.hist(values, bins=bins, color="#4c72b0", alpha=0.85)
    ax.set_xlabel("Indegree")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(list(range(0, vmax + 1)))
    for count, patch in zip(counts, patches):
        if count <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        ax.text(x, y, f"{int(count)}", ha="center", va="bottom", fontsize=5)
    save_plot(fig, out_path)
    plt.close(fig)


def plot_indegree_panels(
    indegree_by_method: Dict[str, List[int]],
    out_path: Path,
) -> None:
    methods = sorted(indegree_by_method.keys())
    if not methods:
        return
    cols = min(3, len(methods))
    rows = int(math.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.0 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for ax, method in zip(axes_list, methods):
        values = indegree_by_method.get(method, [])
        if not values:
            ax.axis("off")
            continue
        vmax = max(values)
        bins = np.arange(-0.5, vmax + 1.5, 1.0)
        counts, _, patches = ax.hist(values, bins=bins, color="#55a868", alpha=0.85)
        ax.set_title(f"Indegree: {method}")
        ax.set_xlabel("Indegree")
        ax.set_ylabel("Count")
        ax.set_xticks(list(range(0, vmax + 1)))
        ax.tick_params(axis="x", labelsize=5)
        for count, patch in zip(counts, patches):
            if count <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2.0
            y = patch.get_height()
            ax.text(x, y, f"{int(count)}", ha="center", va="bottom", fontsize=5)

    for ax in axes_list[len(methods):]:
        ax.axis("off")

    save_plot(fig, out_path)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame, metrics: Sequence[str], bins: Sequence[str], out_path: Path
) -> None:
    if not bins:
        return
    subset = df[df["bin"].isin(bins)].copy()
    if subset.empty:
        return

    summary = (
        subset.groupby("bin")[list(metrics)]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(6)
    )
    summary.to_csv(out_path)


def main() -> None:
    args = build_parser().parse_args()

    roster_dir = args.roster_dir or (args.run_dir / "rosters")
    if not roster_dir.exists():
        raise SystemExit(f"Roster directory not found: {roster_dir}")

    out_dir = args.output_dir or (args.run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rosters(roster_dir)
    df = coerce_numeric(df)
    df = ensure_cpr(df)
    df = ensure_edge_density(df)
    df = add_derived_metrics(df)
    df, util_denom_label, workload_lo_missing = add_utilization(df)
    df = ensure_method_column(df)

    if "status" in df.columns:
        ok_mask = df["status"].fillna("ok") == "ok"
        if ok_mask.any():
            df = df[ok_mask].copy()

    bin_col = choose_bin_column(df)
    df = add_bin_column(df, bin_col)

    bins = sort_bins(df["bin"].dropna().unique())
    if not bins:
        raise SystemExit("No CPR bins found for plotting.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics = [m for m in metrics if m in df.columns]

    render_outputs(df, bins, metrics, out_dir, util_denom_label, workload_lo_missing, args)

    if args.split_by_method and "rdgen_method" in df.columns:
        methods = sorted(
            [m for m in df["rdgen_method"].dropna().unique() if str(m) != "nan"]
        )
        for method in methods:
            df_method = df[df["rdgen_method"] == method].copy()
            if df_method.empty:
                continue
            method_bins = sort_bins(df_method["bin"].dropna().unique())
            if not method_bins:
                continue
            method_out = out_dir / f"by_method_{method}"
            render_outputs(
                df_method,
                method_bins,
                metrics,
                method_out,
                util_denom_label,
                workload_lo_missing,
                args,
            )


if __name__ == "__main__":
    main()
