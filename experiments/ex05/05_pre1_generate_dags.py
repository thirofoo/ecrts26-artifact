from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import networkx as nx

from src.common.dag_models import DAG
from src.generator.wrapper import RDGenWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a DAG pool for ex05_pre and classify by critical path ratio."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="05_pre1",
        help="Run identifier used for output subdirectory naming.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/ex05_pre"),
        help="Base output directory for the DAG pool.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Total number of DAG IDs to generate (used when --class-cap is 0).",
    )
    parser.add_argument(
        "--class-cap",
        type=int,
        default=100,
        help="Max entries per CPR class roster (0 disables class caps).",
    )
    parser.add_argument(
        "--record-all",
        action="store_true",
        help="Record DAGs even when cpr_class is None.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for DAG generation (1 = sequential).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic generation.",
    )
    parser.add_argument(
        "--nodes-min",
        type=int,
        default=20,
        help="Minimum node count per DAG.",
    )
    parser.add_argument(
        "--nodes-max",
        type=int,
        default=100,
        help="Maximum node count per DAG.",
    )
    parser.add_argument(
        "--edge-density-min",
        type=float,
        default=1.2,
        help="Minimum edge density p for edges = p * nodes.",
    )
    parser.add_argument(
        "--edge-density-max",
        type=float,
        default=4.0,
        help="Maximum edge density p for edges = p * nodes.",
    )
    parser.add_argument(
        "--degree-mode",
        choices=["fixed", "random", "weighted"],
        default="random",
        help="How to set in/out degree (random uses --degree-min/max/step).",
    )
    parser.add_argument(
        "--degree-min",
        type=int,
        default=1,
        help="Minimum in/out degree when --degree-mode=random.",
    )
    parser.add_argument(
        "--degree-max",
        type=int,
        default=2,
        help="Maximum in/out degree when --degree-mode=random.",
    )
    parser.add_argument(
        "--degree-step",
        type=int,
        default=1,
        help="Step for in/out degree when --degree-mode=random.",
    )
    parser.add_argument(
        "--degree-weights",
        type=str,
        default="1,1",
        help="Weights for --degree-min/max/step when --degree-mode=weighted.",
    )
    parser.add_argument(
        "--chi-min",
        type=int,
        default=50,
        help="Minimum C^HI value for each node.",
    )
    parser.add_argument(
        "--chi-max",
        type=int,
        default=500,
        help="Maximum C^HI value for each node.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum attempts per DAG id.",
    )
    parser.add_argument(
        "--rdgen-config",
        type=Path,
        default=Path("src/generator/config_templates/test_config.yaml"),
        help="RD-Gen base config.",
    )
    parser.add_argument(
        "--rdgen-method",
        type=str,
        default="both",
        help="RD-Gen graph generation method (fan-in, chain, or both).",
    )
    parser.add_argument(
        "--chain-count-min",
        type=int,
        default=1,
        help="Minimum number of chains when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-count-max",
        type=int,
        default=1,
        help="Maximum number of chains when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-count-step",
        type=int,
        default=1,
        help="Step for number of chains when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-main-min",
        type=int,
        default=1,
        help="Minimum main sequence length when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-main-max",
        type=int,
        default=30,
        help="Maximum main sequence length when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-main-step",
        type=int,
        default=1,
        help="Step for main sequence length when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-main-sampling",
        choices=["uniform", "log"],
        default="log",
        help="Sampling strategy for main sequence length (uniform range or log-spaced list).",
    )
    parser.add_argument(
        "--chain-main-log-samples",
        type=int,
        default=12,
        help="Number of log-spaced samples when --chain-main-sampling=log.",
    )
    parser.add_argument(
        "--chain-sub-min",
        type=int,
        default=0,
        help="Minimum number of sub sequences when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-sub-max",
        type=int,
        default=10,
        help="Maximum number of sub sequences when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-sub-step",
        type=int,
        default=1,
        help="Step for number of sub sequences when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-extra-edges-min",
        type=int,
        default=0,
        help="Minimum extra random edges after chain generation.",
    )
    parser.add_argument(
        "--chain-extra-edges-max",
        type=int,
        default=0,
        help="Maximum extra random edges after chain generation.",
    )
    parser.add_argument(
        "--chain-extra-edges",
        type=int,
        default=None,
        help="Override extra edges with a fixed value (deprecated).",
    )
    parser.add_argument(
        "--chain-source-nodes",
        type=int,
        default=1,
        help="Number of source nodes when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--chain-sink-nodes",
        type=int,
        default=1,
        help="Number of sink nodes when --rdgen-method=chain.",
    )
    parser.add_argument(
        "--rdgen-tmp-dir",
        type=Path,
        default=Path("data/raw_dags/ex05_pre_tmp"),
        help="Temporary RD-Gen output directory.",
    )
    parser.add_argument(
        "--rdgen-config-out-dir",
        type=Path,
        default=Path("data/results/ex05_pre/rdgen_configs"),
        help="Directory to store generated RD-Gen configs.",
    )
    parser.add_argument(
        "--rdgen-verbose",
        action="store_true",
        help="Show RD-Gen command/output.",
    )
    return parser


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


def _find_top_level_block(lines: list[str], key: str) -> tuple[int, int] | None:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not stripped.startswith(f"{key}:"):
            continue
        if len(line) - len(line.lstrip(" ")) != 0:
            continue
        indent = len(line) - len(line.lstrip(" "))
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            if not next_line.strip():
                j += 1
                continue
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= indent:
                break
            j += 1
        return i, j
    return None


def _replace_top_level_block(lines: list[str], key: str, block: list[str]) -> list[str]:
    found = _find_top_level_block(lines, key)
    if not found:
        if lines and lines[-1].strip():
            return lines + [""] + block
        return lines + block
    start, end = found
    return lines[:start] + block + lines[end:]


def _normalize_rdgen_method(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("fan-in", "fanin", "fan", "fan-in/fan-out", "fan-in/fanout"):
        return "fan-in"
    if normalized in ("chain", "chain-based", "chain_based"):
        return "chain"
    if normalized in ("both", "mix", "mixed"):
        return "both"
    raise SystemExit(f"--rdgen-method must be one of: fan-in, chain, both (got {value})")


def _bool_to_yaml(value: bool) -> str:
    return "True" if value else "False"


def _log_spaced_ints(min_val: int, max_val: int, samples: int) -> List[int]:
    if samples <= 1 or min_val == max_val:
        return [min_val]
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    values: List[int] = []
    for i in range(samples):
        ratio = i / (samples - 1)
        raw = math.exp(log_min + ratio * (log_max - log_min))
        val = int(round(raw))
        if val < min_val:
            val = min_val
        if val > max_val:
            val = max_val
        values.append(val)
    return values


def _topological_order(dag: DAG) -> List[int]:
    nodes = list(dag.nodes.keys())
    if not nodes:
        return []
    in_deg = {node_id: len(dag.nodes[node_id].predecessors) for node_id in nodes}
    queue = sorted([node_id for node_id, deg in in_deg.items() if deg == 0])
    order: List[int] = []
    idx = 0
    while idx < len(queue):
        node_id = queue[idx]
        idx += 1
        order.append(node_id)
        for succ in dag.nodes[node_id].successors:
            if succ not in in_deg:
                continue
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    if len(order) != len(nodes):
        return sorted(nodes)
    return order


def _add_random_edges(dag: DAG, rng: random.Random, extra_edges: int) -> int:
    if extra_edges <= 0:
        return 0
    order = _topological_order(dag)
    if len(order) < 2:
        return 0
    index = {node_id: i for i, node_id in enumerate(order)}
    succ_sets = {node_id: set(dag.nodes[node_id].successors) for node_id in order}
    added = 0
    attempts = 0
    max_attempts = max(50, extra_edges * 20)
    while added < extra_edges and attempts < max_attempts:
        attempts += 1
        src = rng.choice(order[:-1])
        start_idx = index[src] + 1
        if start_idx >= len(order):
            continue
        dst = rng.choice(order[start_idx:])
        if dst in succ_sets[src]:
            continue
        dag.add_edge(src, dst)
        succ_sets[src].add(dst)
        added += 1
    return added


def _export_graphml(dag: DAG, out_path: Path) -> None:
    graph = nx.DiGraph()
    for node in dag.nodes.values():
        graph.add_node(node.id, execution_time=node.c_lo)
    for node in dag.nodes.values():
        for succ in node.successors:
            graph.add_edge(node.id, succ)
    nx.write_graphml_xml(graph, out_path)


def _build_chain_graph_structure(
    chain_count_range: tuple[int, int, int],
    chain_main_range: tuple[int, int, int],
    chain_main_values: Sequence[int] | None,
    chain_sub_range: tuple[int, int, int],
    source_nodes: int,
    sink_nodes: int,
    main_tail: bool,
    sub_tail: bool,
    merge_middle: bool,
    merge_sink: bool,
) -> list[str]:
    chain_min, chain_max, chain_step = chain_count_range
    main_min, main_max, main_step = chain_main_range
    sub_min, sub_max, sub_step = chain_sub_range
    if chain_main_values:
        main_values_text = ", ".join(str(val) for val in chain_main_values)
        main_block = ["  Main sequence length:", f"    Random: [{main_values_text}]"]
    else:
        main_block = [
            "  Main sequence length:",
            f"    Random: ({main_min}, {main_max}, {main_step})",
        ]
    return [
        "Graph structure:",
        '  Generation method: "Chain-based"',
        "  Number of chains:",
        f"    Random: ({chain_min}, {chain_max}, {chain_step})",
        *main_block,
        "  Number of sub sequences:",
        f"    Random: ({sub_min}, {sub_max}, {sub_step})",
        "  Vertically link chains:",
        "    Number of source nodes:",
        f"      Fixed: {source_nodes}",
        f"    Main sequence tail: {_bool_to_yaml(main_tail)}",
        f"    Sub sequence tail: {_bool_to_yaml(sub_tail)}",
        "  Merge chains:",
        "    Number of sink nodes:",
        f"      Fixed: {sink_nodes}",
        f"    Middle of chain: {_bool_to_yaml(merge_middle)}",
        f"    Sink node: {_bool_to_yaml(merge_sink)}",
        "",
    ]


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
    exec_time_range: tuple[int, int] | None,
    degree_mode: str,
    degree_range: tuple[int, int, int] | None,
    degree_list: Sequence[int] | None,
    rdgen_method: str,
    chain_settings: dict[str, object] | None,
) -> None:
    lines = base_config.read_text().splitlines()
    updated = _replace_top_level(lines, "Seed", str(run_seed))
    updated = _replace_top_level(updated, "Number of DAGs", str(num_dags))

    if rdgen_method == "chain":
        if chain_settings is None:
            raise ValueError("chain_settings is required when rdgen_method=chain.")
        updated = _replace_top_level_block(
            updated,
            "Graph structure",
            _build_chain_graph_structure(
                chain_settings["chain_count_range"],
                chain_settings["chain_main_range"],
                chain_settings["chain_main_values"],
                chain_settings["chain_sub_range"],
                chain_settings["chain_source_nodes"],
                chain_settings["chain_sink_nodes"],
                chain_settings["chain_main_tail"],
                chain_settings["chain_sub_tail"],
                chain_settings["chain_merge_middle"],
                chain_settings["chain_merge_sink"],
            ),
        )
    else:
        if nodes is None and edges is not None:
            nodes = _extract_fixed_value(updated, "Number of nodes")
        if edges is not None and nodes is None:
            raise ValueError("--edges requires --nodes or a fixed Number of nodes in the config.")

        if nodes is not None:
            updated = _replace_block(updated, "Number of nodes", [f"Fixed: {nodes}"])

        if degree_list is not None:
            candidates = list(degree_list)
            if nodes is not None and nodes > 1:
                max_degree = max(1, nodes - 1)
                candidates = [val for val in candidates if val <= max_degree]
            if not candidates:
                fallback = 1 if nodes is None else max(1, nodes - 1)
                candidates = [fallback]
            degrees_text = ", ".join(str(val) for val in candidates)
            updated = _replace_block(updated, "In-degree", [f"Random: [{degrees_text}]"])
            updated = _replace_block(updated, "Out-degree", [f"Random: [{degrees_text}]"])
        elif degree_mode == "random" and degree_range is not None:
            deg_min, deg_max, deg_step = degree_range
            if nodes is not None and nodes > 1:
                deg_max = min(deg_max, nodes - 1)
                deg_min = min(deg_min, deg_max)
            updated = _replace_block(
                updated,
                "In-degree",
                [f"Random: ({deg_min}, {deg_max}, {deg_step})"],
            )
            updated = _replace_block(
                updated,
                "Out-degree",
                [f"Random: ({deg_min}, {deg_max}, {deg_step})"],
            )
        elif edges is not None:
            if nodes < 2:
                raise ValueError("--nodes must be >= 2 when using --edges.")
            target_degree = max(1, min(nodes - 1, int(round(edges / nodes))))
            updated = _replace_block(updated, "In-degree", [f"Fixed: {target_degree}"])
            updated = _replace_block(updated, "Out-degree", [f"Fixed: {target_degree}"])
    if exec_time_range is not None:
        chi_min, chi_max = exec_time_range
        updated = _replace_block(
            updated,
            "Execution time",
            [f"Random: ({chi_min}, {chi_max}, 1)"],
        )

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


def count_edges(dag: DAG) -> int:
    return sum(len(node.successors) for node in dag.nodes.values())


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


def _cpr_bin_label(lower: float, upper: float) -> str:
    return f"{lower:.1f}_{upper:.1f}"


CPR_BIN_EDGES = [i / 10.0 for i in range(11)]
CPR_CLASSES_10 = [
    _cpr_bin_label(CPR_BIN_EDGES[i], CPR_BIN_EDGES[i + 1]) for i in range(10)
]
CPR_CLASSES_3 = ("small", "middle", "big")
CPR_EXCLUDED_BINS = {"0.0_0.1", "0.7_0.8", "0.8_0.9", "0.9_1.0"}


def classify_cpr(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    if ratio < 0.0 or ratio > 1.0:
        return None
    idx = int(ratio * 10)
    if idx >= 10:
        idx = 9
    return CPR_CLASSES_10[idx]


def classify_cpr_legacy(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    if 0.3 < ratio <= 0.4:
        return "small"
    if 0.4 < ratio <= 0.5:
        return "middle"
    if 0.5 < ratio <= 0.6:
        return "big"
    return None


def load_existing_catalog(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, object]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            raw_id = row.get("dag_id")
            if raw_id is None:
                continue
            try:
                dag_id = int(float(raw_id))
            except ValueError:
                continue
            rows[dag_id] = _sanitize_row(dict(row))
    return rows


CPR_CLASSES = CPR_CLASSES_10


def classify_cpr_class(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in CPR_CLASSES_10 or text in CPR_CLASSES_3:
        return text
    return None


def is_excluded_cpr_class(value: object) -> bool:
    label = classify_cpr_class(value)
    return label in CPR_EXCLUDED_BINS


def count_cpr_classes(
    rows_by_id: Dict[int, Dict[str, object]],
) -> Dict[str, int]:
    counts = {label: 0 for label in CPR_CLASSES_10}
    for row in rows_by_id.values():
        cpr_class = classify_cpr_class(row.get("cpr_class"))
        if (
            cpr_class is not None
            and cpr_class in counts
            and cpr_class not in CPR_EXCLUDED_BINS
        ):
            counts[cpr_class] += 1
    return counts


def count_cpr_classes_by_method(
    rows_by_id: Dict[int, Dict[str, object]],
    methods: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        method: {label: 0 for label in CPR_CLASSES_10} for method in methods
    }
    for row in rows_by_id.values():
        method = str(row.get("rdgen_method") or "").strip()
        if method not in counts:
            continue
        cpr_class = classify_cpr_class(row.get("cpr_class"))
        if (
            cpr_class is not None
            and cpr_class in counts[method]
            and cpr_class not in CPR_EXCLUDED_BINS
        ):
            counts[method][cpr_class] += 1
    return counts


def classes_full(counts: Dict[str, int], cap: int) -> bool:
    if cap <= 0:
        return False
    return all(
        counts.get(label, 0) >= cap
        for label in CPR_CLASSES_10
        if label not in CPR_EXCLUDED_BINS
    )


def classes_full_by_method(
    counts: Dict[str, Dict[str, int]],
    cap: int,
    methods: Sequence[str],
) -> bool:
    if cap <= 0:
        return False
    for method in methods:
        for label in CPR_CLASSES_10:
            if label in CPR_EXCLUDED_BINS:
                continue
            if counts.get(method, {}).get(label, 0) < cap:
                return False
    return True


def load_progress(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}


def save_progress(path: Path, next_dag_id: int, attempts_total: int) -> None:
    payload = {
        "next_dag_id": next_dag_id,
        "attempts_total": attempts_total,
    }
    path.write_text(json.dumps(payload, indent=2))


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


CATALOG_FIELDS = [
    "dag_id",
    "seed",
    "status",
    "attempts",
    "rdgen_method",
    "nodes",
    "edges_target",
    "edges_actual",
    "edge_density",
    "workload_hi",
    "critical_hi",
    "cpr",
    "cpr_class",
    "cpr_class_3",
    "xml_path",
    "extra_edges_target",
    "extra_edges_added",
    "error",
]


def _sanitize_row(row: Dict[str, object]) -> Dict[str, object]:
    return {key: row.get(key) for key in CATALOG_FIELDS}


def _parse_weights(text: str) -> List[float]:
    raw = text.replace(":", ",")
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("degree weights are empty")
    weights: List[float] = []
    for token in tokens:
        value = float(token)
        if value <= 0:
            raise ValueError("degree weights must be > 0")
        weights.append(value)
    return weights


def _expand_weighted_list(values: Sequence[int], weights: Sequence[float]) -> List[int]:
    expanded: List[int] = []
    for value, weight in zip(values, weights):
        count = int(round(weight))
        if count < 1:
            continue
        expanded.extend([value] * count)
    if not expanded:
        expanded = list(values)
    return expanded


def write_rosters(rows_by_id: Dict[int, Dict[str, object]], out_dir: Path) -> None:
    rows_sorted = [_sanitize_row(rows_by_id[k]) for k in sorted(rows_by_id)]
    write_csv(rows_sorted, out_dir / "dag_catalog.csv")
    for label in CPR_CLASSES_10:
        if label in CPR_EXCLUDED_BINS:
            continue
        bin_rows = [row for row in rows_sorted if row.get("cpr_class") == label]
        write_csv(bin_rows, out_dir / f"dag_roster_cpr_{label}.csv")
    methods = sorted(
        {
            str(row.get("rdgen_method")).strip()
            for row in rows_sorted
            if str(row.get("rdgen_method") or "").strip()
        }
    )
    for method in methods:
        safe_method = method.replace("/", "-").replace(" ", "_")
        for label in CPR_CLASSES_10:
            if label in CPR_EXCLUDED_BINS:
                continue
            bin_rows = [
                row
                for row in rows_sorted
                if row.get("cpr_class") == label and str(row.get("rdgen_method")) == method
            ]
            write_csv(bin_rows, out_dir / f"dag_roster_cpr_{label}__{safe_method}.csv")


def generate_one_dag(
    dag_id: int,
    args: argparse.Namespace,
    wrapper: RDGenWrapper,
    rng_seed: int,
    tmp_dir: Path,
    rdgen_method: str,
) -> Dict[str, object]:
    rng = random.Random(rng_seed)
    nodes_target: int | None = None
    density: float | None = None
    edges: int | None = None
    if rdgen_method != "chain":
        nodes_target = rng.randint(args.nodes_min, args.nodes_max)
        density = rng.uniform(args.edge_density_min, args.edge_density_max)
        edges = max(1, int(round(density * nodes_target)))
    degree_list = None
    if rdgen_method != "chain" and args.degree_mode == "weighted":
        degree_list = getattr(args, "degree_list", None)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    config_out_dir = args.rdgen_config_out_dir
    config_out_dir.mkdir(parents=True, exist_ok=True)
    config_out = config_out_dir / f"rdgen_config_{dag_id:06d}.yaml"
    write_seeded_config(
        args.rdgen_config,
        rng_seed,
        config_out,
        1,
        nodes_target,
        edges,
        exec_time_range=(args.chi_min, args.chi_max),
        degree_mode=args.degree_mode,
        degree_range=(args.degree_min, args.degree_max, args.degree_step),
        degree_list=degree_list,
        rdgen_method=rdgen_method,
        chain_settings=args.chain_settings,
    )

    wrapper.run(str(config_out), str(tmp_dir))
    dags = load_raw_dags(wrapper, tmp_dir)
    if not dags:
        raise RuntimeError("No DAGs generated by RD-Gen.")
    dag = dags[0]

    for node in dag.nodes.values():
        node.c_hi = node.c_lo
        node.sigma_c_hi = node.c_hi
        node.constituent_sigmas = [node.c_hi] if node.c_hi > 0 else []
        node.constituent_probs = []

    nodes_actual = len(dag.nodes)
    extra_edges_target = 0
    extra_edges_added = 0
    if rdgen_method == "chain":
        extra_min = args.chain_extra_edges_min
        extra_max = args.chain_extra_edges_max
        if args.chain_extra_edges is not None:
            extra_min = args.chain_extra_edges
            extra_max = args.chain_extra_edges
        if extra_min > 0 or extra_max > 0:
            extra_edges_target = rng.randint(extra_min, extra_max)
        if extra_edges_target > 0:
            extra_edges_added = _add_random_edges(dag, rng, extra_edges_target)

    workload_hi = compute_workload_hi(dag)
    critical_hi = compute_critical_path_hi(dag)
    cpr = (critical_hi / workload_hi) if workload_hi > 0 else None
    cpr_class = classify_cpr(cpr)
    cpr_class_3 = classify_cpr_legacy(cpr)
    edges_actual = count_edges(dag)

    xml_paths = list(_iter_xml_paths(tmp_dir))
    if not xml_paths:
        raise RuntimeError("Generated XML file not found.")
    xml_path = xml_paths[0]
    if extra_edges_added > 0:
        rewritten = tmp_dir / f"dag_{dag_id:06d}_extra.xml"
        _export_graphml(dag, rewritten)
        xml_path = rewritten

    return {
        "dag": dag,
        "xml_path": xml_path,
        "nodes": nodes_actual,
        "edges_target": edges,
        "edges_actual": edges_actual,
        "edge_density": density,
        "workload_hi": workload_hi,
        "critical_hi": critical_hi,
        "cpr": cpr,
        "cpr_class": cpr_class,
        "cpr_class_3": cpr_class_3,
        "rdgen_method": rdgen_method,
        "extra_edges_target": extra_edges_target,
        "extra_edges_added": extra_edges_added,
    }


def _worker_tmp_dir(base_dir: Path) -> Path:
    return base_dir / f"worker_{os.getpid()}"


def evaluate_dag_worker(
    dag_id: int,
    args: argparse.Namespace,
    dag_dir: Path,
    rdgen_method: str,
) -> Dict[str, object]:
    wrapper = RDGenWrapper(verbose=args.rdgen_verbose)
    tmp_dir = _worker_tmp_dir(args.rdgen_tmp_dir)

    row = {
        "dag_id": dag_id,
        "seed": None,
        "status": "gen_failed",
        "rdgen_method": rdgen_method,
    }
    success = False
    last_seed = None
    attempts_used = 0
    for attempt in range(args.max_attempts):
        run_seed = args.seed + dag_id * 1000 + attempt
        last_seed = run_seed
        attempts_used = attempt + 1
        try:
            result = generate_one_dag(dag_id, args, wrapper, run_seed, tmp_dir, rdgen_method)
        except Exception as exc:
            row["error"] = str(exc)
            continue

        xml_path = result["xml_path"]
        out_xml = dag_dir / f"dag_{dag_id:06d}.xml"
        shutil.copy(xml_path, out_xml)

        row = {
            "dag_id": dag_id,
            "seed": run_seed,
            "status": "ok",
            "attempts": attempts_used,
            "rdgen_method": result.get("rdgen_method", rdgen_method),
            "nodes": result["nodes"],
            "edges_target": result["edges_target"],
            "edges_actual": result["edges_actual"],
            "edge_density": (
                round(result["edge_density"], 6) if result["edge_density"] is not None else None
            ),
            "workload_hi": result["workload_hi"],
            "critical_hi": result["critical_hi"],
            "cpr": round(result["cpr"], 6) if result["cpr"] is not None else None,
            "cpr_class": result["cpr_class"],
            "cpr_class_3": result.get("cpr_class_3"),
            "xml_path": str(out_xml),
            "extra_edges_target": result.get("extra_edges_target"),
            "extra_edges_added": result.get("extra_edges_added"),
        }
        success = True
        break

    if not success:
        row["seed"] = last_seed
        row["attempts"] = attempts_used
    return {
        "dag_id": dag_id,
        "row": _sanitize_row(row),
        "success": success,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.rdgen_method = _normalize_rdgen_method(args.rdgen_method)
    methods_to_generate = (
        ["fan-in", "chain"] if args.rdgen_method == "both" else [args.rdgen_method]
    )
    use_chain = "chain" in methods_to_generate
    use_fanin = "fan-in" in methods_to_generate
    if args.class_cap < 0:
        raise SystemExit("--class-cap must be >= 0.")
    if args.class_cap == 0 and args.target_count < 1:
        raise SystemExit("--target-count must be >= 1 when --class-cap is 0.")
    if args.nodes_min < 1 or args.nodes_max < 1:
        raise SystemExit("--nodes-min/--nodes-max must be >= 1.")
    if args.nodes_min > args.nodes_max:
        raise SystemExit("--nodes-min must be <= --nodes-max.")
    if args.edge_density_min <= 0 or args.edge_density_max <= 0:
        raise SystemExit("--edge-density-min/max must be > 0.")
    if args.edge_density_min > args.edge_density_max:
        raise SystemExit("--edge-density-min must be <= --edge-density-max.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    if use_fanin:
        if args.degree_mode == "random":
            if args.degree_min < 1 or args.degree_max < 1:
                raise SystemExit("--degree-min/--degree-max must be >= 1.")
            if args.degree_min > args.degree_max:
                raise SystemExit("--degree-min must be <= --degree-max.")
            if args.degree_step < 1:
                raise SystemExit("--degree-step must be >= 1.")
        if args.degree_mode == "weighted":
            values = list(range(args.degree_min, args.degree_max + 1, args.degree_step))
            if not values:
                raise SystemExit("degree range is empty for --degree-mode=weighted.")
            try:
                weights = _parse_weights(args.degree_weights)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if len(weights) != len(values):
                raise SystemExit(
                    f"--degree-weights has {len(weights)} values but degree range has {len(values)}."
                )
            args.degree_values = values
            args.degree_weights = weights
            args.degree_list = _expand_weighted_list(values, weights)
    if args.chi_min < 1 or args.chi_max < 1:
        raise SystemExit("--chi-min/--chi-max must be >= 1.")
    if args.chi_min > args.chi_max:
        raise SystemExit("--chi-min must be <= --chi-max.")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1.")
    if use_chain:
        if args.chain_count_min < 1 or args.chain_count_max < 1:
            raise SystemExit("--chain-count-min/--chain-count-max must be >= 1.")
        if args.chain_count_min > args.chain_count_max:
            raise SystemExit("--chain-count-min must be <= --chain-count-max.")
        if args.chain_count_step < 1:
            raise SystemExit("--chain-count-step must be >= 1.")
        if args.chain_main_min < 1 or args.chain_main_max < 1:
            raise SystemExit("--chain-main-min/--chain-main-max must be >= 1.")
        if args.chain_main_min > args.chain_main_max:
            raise SystemExit("--chain-main-min must be <= --chain-main-max.")
        if args.chain_main_step < 1:
            raise SystemExit("--chain-main-step must be >= 1.")
        if args.chain_main_sampling == "log":
            if args.chain_main_min < 1 or args.chain_main_max < 1:
                raise SystemExit("--chain-main-min/--chain-main-max must be >= 1 for log sampling.")
            if args.chain_main_log_samples < 1:
                raise SystemExit("--chain-main-log-samples must be >= 1.")
        if args.chain_sub_min < 0 or args.chain_sub_max < 0:
            raise SystemExit("--chain-sub-min/--chain-sub-max must be >= 0.")
        if args.chain_sub_min > args.chain_sub_max:
            raise SystemExit("--chain-sub-min must be <= --chain-sub-max.")
        if args.chain_sub_step < 1:
            raise SystemExit("--chain-sub-step must be >= 1.")
        if args.chain_extra_edges_min < 0 or args.chain_extra_edges_max < 0:
            raise SystemExit("--chain-extra-edges-min/max must be >= 0.")
        if args.chain_extra_edges_min > args.chain_extra_edges_max:
            raise SystemExit("--chain-extra-edges-min must be <= --chain-extra-edges-max.")
        if args.chain_extra_edges is not None and args.chain_extra_edges < 0:
            raise SystemExit("--chain-extra-edges must be >= 0.")
        if args.chain_source_nodes < 1:
            raise SystemExit("--chain-source-nodes must be >= 1.")
        if args.chain_sink_nodes < 1:
            raise SystemExit("--chain-sink-nodes must be >= 1.")
        if args.chain_source_nodes > args.chain_count_max:
            raise SystemExit("--chain-source-nodes must be <= --chain-count-max.")

    run_dir = args.output_dir / args.run_id
    dag_dir = run_dir / "dag_tasks"
    roster_dir = run_dir / "rosters"
    run_dir.mkdir(parents=True, exist_ok=True)
    dag_dir.mkdir(parents=True, exist_ok=True)
    roster_dir.mkdir(parents=True, exist_ok=True)
    if not args.rdgen_config.exists():
        raise SystemExit(f"RD-Gen config not found: {args.rdgen_config}")
    if args.rdgen_config_out_dir == Path("data/results/ex05_pre/rdgen_configs"):
        args.rdgen_config_out_dir = run_dir / "rdgen_configs"
    if args.rdgen_tmp_dir == Path("data/raw_dags/ex05_pre_tmp"):
        args.rdgen_tmp_dir = run_dir / "rdgen_tmp"

    chain_main_tail = True
    chain_sub_tail = True
    chain_merge_middle = False
    chain_merge_sink = True
    args.chain_settings = None
    if use_chain:
        chain_main_values = None
        if args.chain_main_sampling == "log":
            chain_main_values = _log_spaced_ints(
                args.chain_main_min,
                args.chain_main_max,
                args.chain_main_log_samples,
            )
        args.chain_settings = {
            "chain_count_range": (args.chain_count_min, args.chain_count_max, args.chain_count_step),
            "chain_main_range": (args.chain_main_min, args.chain_main_max, args.chain_main_step),
            "chain_main_values": chain_main_values,
            "chain_sub_range": (args.chain_sub_min, args.chain_sub_max, args.chain_sub_step),
            "chain_source_nodes": args.chain_source_nodes,
            "chain_sink_nodes": args.chain_sink_nodes,
            "chain_main_tail": chain_main_tail,
            "chain_sub_tail": chain_sub_tail,
            "chain_merge_middle": chain_merge_middle,
            "chain_merge_sink": chain_merge_sink,
        }

    meta = {
        "run_id": args.run_id,
        "target_count": args.target_count,
        "class_cap": args.class_cap,
        "record_all": args.record_all,
        "seed": args.seed,
        "workers": args.workers,
        "nodes_min": args.nodes_min,
        "nodes_max": args.nodes_max,
        "edge_density_min": args.edge_density_min,
        "edge_density_max": args.edge_density_max,
        "degree_mode": args.degree_mode,
        "degree_min": args.degree_min,
        "degree_max": args.degree_max,
        "degree_step": args.degree_step,
        "degree_weights": args.degree_weights if args.degree_mode == "weighted" else None,
        "chi_min": args.chi_min,
        "chi_max": args.chi_max,
        "max_attempts": args.max_attempts,
        "rdgen_config": str(args.rdgen_config),
        "rdgen_method": args.rdgen_method,
        "chain_count_min": args.chain_count_min,
        "chain_count_max": args.chain_count_max,
        "chain_count_step": args.chain_count_step,
        "chain_main_min": args.chain_main_min,
        "chain_main_max": args.chain_main_max,
        "chain_main_step": args.chain_main_step,
        "chain_main_sampling": args.chain_main_sampling,
        "chain_main_log_samples": args.chain_main_log_samples,
        "chain_main_values": chain_main_values if use_chain else None,
        "chain_sub_min": args.chain_sub_min,
        "chain_sub_max": args.chain_sub_max,
        "chain_sub_step": args.chain_sub_step,
        "chain_extra_edges_min": args.chain_extra_edges_min,
        "chain_extra_edges_max": args.chain_extra_edges_max,
        "chain_extra_edges": args.chain_extra_edges,
        "chain_source_nodes": args.chain_source_nodes,
        "chain_sink_nodes": args.chain_sink_nodes,
        "chain_main_tail": chain_main_tail,
        "chain_sub_tail": chain_sub_tail,
        "chain_merge_middle": chain_merge_middle,
        "chain_merge_sink": chain_merge_sink,
        "rdgen_tmp_dir": str(args.rdgen_tmp_dir),
        "rdgen_config_out_dir": str(args.rdgen_config_out_dir),
    }
    (run_dir / "ex05_pre_meta.json").write_text(json.dumps(meta, indent=2))

    catalog_path = roster_dir / "dag_catalog.csv"
    rows_by_id = load_existing_catalog(catalog_path)
    record_classified_only = not args.record_all
    if record_classified_only:
        rows_by_id = {
            dag_id: row
            for dag_id, row in rows_by_id.items()
            if row.get("status") == "ok" and classify_cpr_class(row.get("cpr_class")) is not None
        }
        rows_by_id = {
            dag_id: row
            for dag_id, row in rows_by_id.items()
            if not is_excluded_cpr_class(row.get("cpr_class"))
        }
        write_rosters(rows_by_id, roster_dir)

    if args.class_cap > 0:
        class_counts_by_method = count_cpr_classes_by_method(rows_by_id, methods_to_generate)
    else:
        class_counts_by_method = {method: count_cpr_classes(rows_by_id) for method in methods_to_generate}
    progress_path = run_dir / "ex05_pre_progress.json"
    progress = load_progress(progress_path)
    next_dag_id = progress.get("next_dag_id")
    if next_dag_id is None:
        next_dag_id = max(rows_by_id) + 1 if rows_by_id else 1
    else:
        min_next = max(rows_by_id) + 1 if rows_by_id else 1
        next_dag_id = max(next_dag_id, min_next)
    attempts_total = progress.get("attempts_total", 0)

    def cleanup_xml(row: Dict[str, object]) -> None:
        xml_path = row.get("xml_path")
        if not xml_path:
            return
        try:
            Path(str(xml_path)).unlink()
        except FileNotFoundError:
            return

    def handle_result(result: Dict[str, object]) -> bool:
        row = result.get("row")
        if not isinstance(row, dict):
            return False
        dag_id = row.get("dag_id")
        if dag_id is None:
            return False
        if not result.get("success"):
            print(f"[dag {int(dag_id):04d}] failed ({row.get('error')})")
            return False

        nodes = row.get("nodes")
        if nodes is not None:
            try:
                nodes_int = int(nodes)
            except (TypeError, ValueError):
                nodes_int = None
            if nodes_int is not None and not (args.nodes_min <= nodes_int <= args.nodes_max):
                cleanup_xml(row)
                print(
                    f"[dag {int(dag_id):04d}] skip (nodes={nodes_int} "
                    f"range=[{args.nodes_min},{args.nodes_max}])"
                )
                return False

        cpr_class = classify_cpr_class(row.get("cpr_class"))
        method = str(row.get("rdgen_method") or args.rdgen_method)
        if cpr_class in CPR_EXCLUDED_BINS:
            cleanup_xml(row)
            print(f"[dag {int(dag_id):04d}] skip (cpr_class={cpr_class})")
            return False
        if record_classified_only and cpr_class is None:
            cleanup_xml(row)
            print(f"[dag {int(dag_id):04d}] skip (cpr_class=None cpr={row.get('cpr')})")
            return False
        if args.class_cap > 0 and cpr_class is not None:
            if (
                method in class_counts_by_method
                and class_counts_by_method[method].get(cpr_class, 0) >= args.class_cap
            ):
                cleanup_xml(row)
                print(
                    f"[dag {int(dag_id):04d}] skip (class full: {cpr_class} method={method})"
                )
                return False

        rows_by_id[int(dag_id)] = row
        if cpr_class is not None:
            if method in class_counts_by_method:
                class_counts_by_method[method][cpr_class] = (
                    class_counts_by_method[method].get(cpr_class, 0) + 1
                )
        write_rosters(rows_by_id, roster_dir)
        print(
            f"[dag {int(dag_id):04d}] ok nodes={row.get('nodes')} "
            f"edges={row.get('edges_actual')} cpr={row.get('cpr')} "
            f"class={row.get('cpr_class')}"
        )
        return True

    def process_ids(batch_ids: List[int], rdgen_method: str) -> None:
        nonlocal attempts_total
        if args.workers > 1 and len(batch_ids) > 1:
            max_workers = min(args.workers, len(batch_ids))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_dag_worker, dag_id, args, dag_dir, rdgen_method)
                    for dag_id in batch_ids
                ]
                for future in as_completed(futures):
                    attempts_total += 1
                    handle_result(future.result())
                    save_progress(progress_path, next_dag_id, attempts_total)
        else:
            for dag_id in batch_ids:
                attempts_total += 1
                result = evaluate_dag_worker(dag_id, args, dag_dir, rdgen_method)
                handle_result(result)
                save_progress(progress_path, next_dag_id, attempts_total)

    if args.class_cap > 0:
        if classes_full_by_method(class_counts_by_method, args.class_cap, methods_to_generate):
            print("[done] CPR rosters already filled.")
            return
        while not classes_full_by_method(class_counts_by_method, args.class_cap, methods_to_generate):
            for method in methods_to_generate:
                if classes_full(class_counts_by_method.get(method, {}), args.class_cap):
                    continue
                batch_size = max(1, args.workers)
                batch_ids = list(range(next_dag_id, next_dag_id + batch_size))
                next_dag_id += batch_size
                save_progress(progress_path, next_dag_id, attempts_total)
                process_ids(batch_ids, method)
    else:
        pending_ids = [
            dag_id for dag_id in range(1, args.target_count + 1) if dag_id not in rows_by_id
        ]
        if pending_ids:
            if args.workers > 1 and len(pending_ids) > 1:
                max_workers = min(args.workers, len(pending_ids))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            evaluate_dag_worker, dag_id, args, dag_dir, methods_to_generate[0]
                        )
                        for dag_id in pending_ids
                    ]
                    for future in as_completed(futures):
                        handle_result(future.result())
            else:
                for dag_id in pending_ids:
                    result = evaluate_dag_worker(dag_id, args, dag_dir, methods_to_generate[0])
                    handle_result(result)


if __name__ == "__main__":
    main()
