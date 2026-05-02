# src/utils/visualizer.py
import math
import os

import matplotlib.pyplot as plt
import networkx as nx


def _graphviz_layout_cli(G, rankdir="LR"):
    try:
        import shutil
        import subprocess
    except Exception:
        return None
    if shutil.which("dot") is None:
        return None
    nodes = list(G.nodes())
    node_ids = {node: f"n{idx}" for idx, node in enumerate(nodes)}
    id_to_node = {value: key for key, value in node_ids.items()}
    lines = [
        "digraph G {",
        f'  rankdir="{rankdir}";',
        "  graph [splines=true, overlap=false];",
        "  node [shape=circle];",
    ]
    for node in nodes:
        lines.append(f"  {node_ids[node]};")
    for src, dst in G.edges():
        lines.append(f"  {node_ids[src]} -> {node_ids[dst]};")
    lines.append("}")
    dot_input = "\n".join(lines).encode("utf-8")
    try:
        result = subprocess.run(
            ["dot", "-Tplain"],
            input=dot_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    pos = {}
    for line in result.stdout.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("node "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        node_id = parts[1]
        node = id_to_node.get(node_id)
        if node is None:
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            continue
        pos[node] = (x, y)
    return pos if pos else None


def _graphviz_layout(G, rankdir="LR"):
    args = f"-Grankdir={rankdir} -Gnodesep=0.4 -Granksep=0.6"
    try:
        from networkx.drawing.nx_agraph import graphviz_layout as _gv_layout
    except Exception:
        _gv_layout = None
    if _gv_layout is not None:
        try:
            return _gv_layout(G, prog="dot", args=args)
        except Exception:
            return None
    try:
        from networkx.drawing.nx_pydot import graphviz_layout as _pd_layout
    except Exception:
        _pd_layout = None
    if _pd_layout is not None:
        try:
            return _pd_layout(G, prog="dot", args=args)
        except Exception:
            return None
    return _graphviz_layout_cli(G, rankdir=rankdir)

def get_hierarchical_pos(G):
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    node_count = G.number_of_nodes()
    if node_count == 0:
        return {}

    def grid_pos(nodes):
        n = len(nodes)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        x_sep = 2.2
        y_sep = 2.2
        pos = {}
        for idx, node in enumerate(sorted(nodes)):
            col = idx % cols
            row = idx // cols
            pos[node] = (col * x_sep, -row * y_sep)
        return pos

    layers = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            layers[node] = 0
        else:
            layers[node] = max(layers[p] for p in preds) + 1
            
    layer_nodes = {}
    max_layer = 0
    for node, layer in layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
        max_layer = max(max_layer, layer)

    for layer in layer_nodes:
        layer_nodes[layer].sort()

    max_per_layer = max(len(nodes) for nodes in layer_nodes.values())
    if max_layer == 0 or (max_layer <= 1 and max_per_layer >= max(8, int(node_count * 0.7))):
        return grid_pos(G.nodes())
        
    def _index_map(nodes):
        return {node: idx for idx, node in enumerate(nodes)}

    def _barycenter(node, neighbors, neighbor_index, fallback):
        values = [neighbor_index[n] for n in neighbors if n in neighbor_index]
        if not values:
            return fallback
        return sum(values) / len(values)

    sweeps = 4
    for _ in range(sweeps):
        for layer in range(1, max_layer + 1):
            prev_index = _index_map(layer_nodes.get(layer - 1, []))
            current_index = _index_map(layer_nodes.get(layer, []))
            layer_nodes[layer].sort(
                key=lambda n: _barycenter(
                    n,
                    list(G.predecessors(n)),
                    prev_index,
                    current_index.get(n, 0),
                )
            )
        for layer in range(max_layer - 1, -1, -1):
            next_index = _index_map(layer_nodes.get(layer + 1, []))
            current_index = _index_map(layer_nodes.get(layer, []))
            layer_nodes[layer].sort(
                key=lambda n: _barycenter(
                    n,
                    list(G.successors(n)),
                    next_index,
                    current_index.get(n, 0),
                )
            )

    pos = {}
    x_sep = 2.6 + 0.15 * min(max_layer, 8)
    y_sep = 2.1 + 0.18 * max(0, max_per_layer - 4)
    y_sep = min(6.0, y_sep)

    for layer in range(max_layer + 1):
        if layer not in layer_nodes:
            continue
        nodes = layer_nodes[layer]
        n = len(nodes)
        x = layer * x_sep
        for i, node in enumerate(nodes):
            y = (i - (n - 1) / 2) * y_sep
            pos[node] = (x, -y)

    return pos

def _edge_curvatures(G, pos, base_rad=0.18):
    edge_rads = {}
    for src in G.nodes():
        succs = list(G.successors(src))
        if len(succs) <= 1:
            continue
        succs.sort(key=lambda dst: pos.get(dst, (0.0, 0.0))[1])
        mid = (len(succs) - 1) / 2.0
        for i, dst in enumerate(succs):
            edge_rads[(src, dst)] = (i - mid) * base_rad
    for dst in G.nodes():
        preds = list(G.predecessors(dst))
        if len(preds) <= 1:
            continue
        preds.sort(key=lambda src: pos.get(src, (0.0, 0.0))[1])
        mid = (len(preds) - 1) / 2.0
        for i, src in enumerate(preds):
            edge_rads[(src, dst)] = edge_rads.get((src, dst), 0.0) + (i - mid) * base_rad * 0.6
    return edge_rads


def _compress_positions(pos, target_aspect=1.25, min_scale=0.6, rotate_ratio=3.0):
    if not pos:
        return pos
    def _spans(points):
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        return xs, ys, max(xs) - min(xs), max(ys) - min(ys)

    xs, ys, span_x, span_y = _spans(pos)
    if span_x > 0 and span_y > 0 and span_y > span_x * rotate_ratio:
        rotated = {node: (y, -x) for node, (x, y) in pos.items()}
        pos = rotated
        xs, ys, span_x, span_y = _spans(pos)

    if span_x <= 0 or span_y <= 0:
        return pos

    if span_x > span_y * target_aspect:
        scale_x = max(min_scale, (span_y * target_aspect) / span_x)
        cx = (max(xs) + min(xs)) / 2
        return {node: (cx + (x - cx) * scale_x, y) for node, (x, y) in pos.items()}

    if span_y > span_x * target_aspect:
        scale_y = max(min_scale, (span_x * target_aspect) / span_y)
        cy = (max(ys) + min(ys)) / 2
        return {node: (x, cy + (y - cy) * scale_y) for node, (x, y) in pos.items()}

    return pos


def _center_positions(pos):
    if not pos:
        return pos
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    return {node: (x - cx, y - cy) for node, (x, y) in pos.items()}




def _draw_labels(pos, labels, font_size, font_weight, bbox):
    ax = plt.gca()
    for node, label in labels.items():
        if node not in pos:
            continue
        x, y = pos[node]
        try:
            ax.text(
                x, y, label,
                fontsize=font_size,
                fontweight=font_weight,
                ha="center",
                va="center",
                multialignment="center",
                bbox=bbox,
            )
        except TypeError:
            ax.text(
                x, y, label,
                fontsize=font_size,
                fontweight=font_weight,
                ha="center",
                va="center",
                bbox=bbox,
            )


def draw_dag(G, dag_info, title_text, filename, node_colors, node_labels, layout_mode="auto"):
    node_count = len(G.nodes())
    if node_count >= 60:
        node_size = 2000
        font_size = 7
        edge_rad = 0.26
    elif node_count >= 40:
        node_size = 2500
        font_size = 8
        edge_rad = 0.24
    else:
        node_size = 3200
        font_size = 9
        edge_rad = 0.2
    edge_width = 1.6
    edge_alpha = 0.8
    
    draw_nodelist = list(G.nodes())
    draw_colors = [node_colors[n] for n in draw_nodelist]
    
    pos = None
    if layout_mode in {"chain", "graphviz", "dot"}:
        pos = _graphviz_layout(G)
    if pos is None:
        pos = get_hierarchical_pos(G)
    pos = _compress_positions(pos)
    pos = _center_positions(pos)
    
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) if xs else 10.0
    span_y = max(ys) - min(ys) if ys else 8.0
    chain_like = span_y <= 1e-6
    if chain_like:
        node_size = min(node_size, 900)
        font_size = min(font_size, 6)
        edge_rad = min(edge_rad, 0.12)
        edge_width = 1.2
    aspect = span_y / max(span_x, 1e-6)
    if span_x <= 1e-6 or aspect > 4.0:
        edge_rad = 0.0
        edge_width = min(edge_width, 1.2)
        edge_alpha = min(edge_alpha, 0.7)
    fig_w = max(4.0, min(16.0, span_x * 1.05 + 1.5))
    fig_h = max(4.0, min(20.0, span_y * 1.05 + 1.5))
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    
    edge_rads = _edge_curvatures(G, pos, base_rad=edge_rad)
    for src, dst in G.edges():
        rad = edge_rads.get((src, dst), 0.0)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(src, dst)],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="gray",
            width=edge_width,
            alpha=edge_alpha,
            node_size=node_size,
            min_source_margin=18,
            min_target_margin=22,
            connectionstyle=f"arc3,rad={rad}",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=draw_nodelist,
        node_color=draw_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.5,
    )
    
    label_bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65)
    _draw_labels(pos, node_labels, font_size, "bold", label_bbox)
    
    plt.title(title_text, fontsize=14)
    plt.axis('off')

    margin_x = max(0.3, 0.04 * span_x)
    margin_y = max(0.3, 0.04 * span_y)
    ax.set_xlim(-span_x / 2 - margin_x, span_x / 2 + margin_x)
    ax.set_ylim(-span_y / 2 - margin_y, span_y / 2 + margin_y)
    ax.set_anchor("C")
    ax.margins(0)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    # print(f"Plot saved to {filename}")
