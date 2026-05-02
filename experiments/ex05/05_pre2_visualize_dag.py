from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx

from src.common.dag_models import DAG
from src.generator.wrapper import RDGenWrapper
from src.utils.visualizer import draw_dag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a DAG from ex05_pre using its generation seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed value recorded in dag_catalog.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/results/ex05_pre/05_pre1"),
        help="ex05_pre run directory that contains rosters/ and dag_tasks/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the visualization (default: run-dir/visualizations).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output format for the visualization.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "chain", "graphviz", "dot"],
        default="auto",
        help="Layout mode (auto uses chain when available).",
    )
    return parser


def _parse_int(value: object) -> Optional[int]:
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


def _parse_float(value: object) -> Optional[float]:
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


def load_catalog(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"Catalog not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                rows.append(dict(row))
    return rows


def find_row_by_seed(rows: List[Dict[str, object]], seed: int) -> Dict[str, object]:
    matches = []
    for row in rows:
        row_seed = _parse_int(row.get("seed"))
        if row_seed == seed:
            matches.append(row)
    if not matches:
        raise SystemExit(f"Seed {seed} not found in catalog.")
    if len(matches) > 1:
        raise SystemExit(f"Seed {seed} appears multiple times in catalog.")
    return matches[0]


def resolve_xml_path(run_dir: Path, row: Dict[str, object]) -> Path:
    raw = row.get("xml_path")
    if raw is None or str(raw).strip() == "":
        raise SystemExit("xml_path is missing in catalog row.")
    xml_path = Path(str(raw))
    candidates = []
    if xml_path.is_absolute():
        candidates.append(xml_path)
    else:
        candidates.append(run_dir / xml_path)
        candidates.append(run_dir / "dag_tasks" / xml_path.name)
        candidates.append(Path(xml_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"XML file not found for seed {row.get('seed')}: {xml_path}")


def count_edges(dag: DAG) -> int:
    return sum(len(node.successors) for node in dag.nodes.values())


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    meta_path = run_dir / "ex05_pre_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Meta file not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    catalog_path = run_dir / "rosters" / "dag_catalog.csv"
    rows = load_catalog(catalog_path)
    row = find_row_by_seed(rows, args.seed)
    if row.get("status") != "ok":
        raise SystemExit(f"Seed {args.seed} is not marked ok in catalog (status={row.get('status')}).")

    xml_path = resolve_xml_path(run_dir, row)
    wrapper = RDGenWrapper(verbose=False)
    dags = wrapper.parse_xml(str(xml_path))
    if not dags:
        raise SystemExit(f"No DAGs parsed from {xml_path}")
    dag = dags[0]

    row_nodes = _parse_int(row.get("nodes"))
    nodes_min = int(meta.get("nodes_min", 1))
    nodes_max = int(meta.get("nodes_max", nodes_min))
    if row_nodes is not None and not (nodes_min <= row_nodes <= nodes_max):
        print(
            f"[warn] nodes out of meta range: nodes={row_nodes} "
            f"range=[{nodes_min}, {nodes_max}]"
        )

    G = nx.DiGraph()
    for node in dag.nodes.values():
        G.add_node(node.id)
    for node in dag.nodes.values():
        for succ in node.successors:
            G.add_edge(node.id, succ)

    node_colors = {node.id: "lightblue" for node in dag.nodes.values()}
    node_labels = {}
    for node in dag.nodes.values():
        node_labels[node.id] = f"{node.id}\n{node.c_lo}"

    dag_id = _parse_int(row.get("dag_id"))
    cpr = _parse_float(row.get("cpr"))
    edge_count = count_edges(dag)

    title_parts = [
        f"DAG {dag_id if dag_id is not None else '?'} seed={args.seed}",
        f"nodes={len(dag.nodes)} edges={edge_count}",
    ]
    if cpr is not None:
        title_parts.append(f"cpr={cpr:.3f}")
    title = " | ".join(title_parts)

    out_dir = args.out_dir if args.out_dir is not None else run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"dag_seed_{args.seed}.{args.format}"
    out_path = out_dir / out_name

    layout_mode = "chain" if str(row.get("rdgen_method", "")).strip().lower() == "chain" else "auto"
    if args.layout != "auto":
        layout_mode = args.layout
    else:
        run_method = str(meta.get("rdgen_method", "")).strip().lower()
        row_method = str(row.get("rdgen_method", "")).strip().lower()
        if row_method == "chain" or run_method == "chain":
            layout_mode = "chain"
    draw_dag(G, dag, title, str(out_path), node_colors, node_labels, layout_mode=layout_mode)
    print(f"[ok] saved {out_path}")


if __name__ == "__main__":
    main()
