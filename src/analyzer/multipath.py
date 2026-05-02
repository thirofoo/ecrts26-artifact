# src/analyzer/multipath.py
from collections import deque
import heapq
from typing import Dict, List, Optional, Sequence, Tuple

from src.common.dag_models import DAG

class MultipathAnalyzer:
    @staticmethod
    def get_workload(dag: DAG, mode: str = "LO") -> int:
        """
        Compute total workload W.

        Task-level MC rule: in HI mode, LO-criticality tasks are dropped, so W=0.
        """
        # LO-criticality tasks do not execute in HI mode.
        if mode == "HI" and dag.criticality == "LO":
            return 0
            
        total_c = 0
        for node in dag.nodes.values():
            # HI mode returns C_HI for each node.
            total_c += node.get_exec_time(mode)
            
        return total_c

    @staticmethod
    def get_critical_path(dag: DAG, mode: str = "LO") -> int:
        """
        Compute critical-path length L.
        """
        # LO-criticality tasks do not execute in HI mode.
        if mode == "HI" and dag.criticality == "LO":
            return 0

        memo = {}

        def _visit(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            
            node = dag.nodes[node_id]
            current_c = node.get_exec_time(mode)
            
            if not node.predecessors:
                memo[node_id] = current_c
                return current_c

            max_pred_path = max(_visit(p_id) for p_id in node.predecessors)
            ret = current_c + max_pred_path
            memo[node_id] = ret
            return ret

        if not dag.nodes:
            return 0
            
        return max(_visit(nid) for nid in dag.nodes)

    @staticmethod
    def calculate_response_time(dag: DAG, m: int) -> dict:
        """Graham bound R = L + (W - L) / m"""
        results = {}
        for mode in ["LO", "HI"]:
            w = MultipathAnalyzer.get_workload(dag, mode)
            l = MultipathAnalyzer.get_critical_path(dag, mode)
            
            if mode == "HI" and dag.criticality == "LO":
                # Dropped tasks have response time 0 in this model.
                r = 0
            else:
                if m > 0:
                    term2 = max(0, w - l) / m
                    r = l + term2
                else:
                    r = float('inf')
                
            results[mode] = {"W": w, "L": l, "R": int(r)}
            
        return results

    @staticmethod
    def calculate_multipath_bounds(dag: DAG, m: int) -> dict:
        """
        Compute the multipath bound (Theorem 1, TCAD'25) for LO and HI.
        The bound ignores task criticality drop and assumes all nodes run in the given mode.
        """
        width = MultipathAnalyzer._compute_width(dag)
        if not dag.nodes or m <= 0 or width == 0:
            return {"LO": 0.0, "HI": 0.0}

        n = min(width, m)
        return {
            "LO": MultipathAnalyzer._calculate_multipath_bound(dag, m, "LO", n),
            "HI": MultipathAnalyzer._calculate_multipath_bound(dag, m, "HI", n),
        }

    @staticmethod
    def min_cores_for_dag(dag: DAG, m_max: int) -> Optional[int]:
        """Return the smallest core count that satisfies the multipath bound."""
        deadline = dag.deadline or dag.period
        if deadline <= 0 or m_max <= 0:
            return None
        mode = "HI" if dag.criticality == "HI" else "LO"
        upper = max(1, m_max)
        if not dag.nodes:
            return 1

        width = MultipathAnalyzer._compute_width(dag)
        if width <= 0:
            return 1

        total = MultipathAnalyzer._total_workload(dag, mode)
        length = MultipathAnalyzer._critical_path_length(dag, mode)
        max_paths = min(width, upper)
        volumes = MultipathAnalyzer._max_volumes_for_cardinalities(dag, mode, max_paths)

        def _bound_for_m(m: int) -> float:
            if m <= 0:
                return float("inf")
            best = float("inf")
            j_max = min(len(volumes), m)
            for j in range(j_max):
                denom = m - j
                if denom <= 0:
                    continue
                bound = length + (total - volumes[j]) / denom
                if bound < best:
                    best = bound
            return best

        if _bound_for_m(upper) > deadline:
            return None

        low = 1
        high = upper
        result = upper
        while low <= high:
            mid = (low + high) // 2
            if _bound_for_m(mid) <= deadline:
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        return result

    @staticmethod
    def min_cores_for_taskset(dags: Sequence[DAG], m_max: int) -> Optional[int]:
        """Return the smallest core count that makes every DAG pass the bound."""
        if not dags:
            return 0
        if m_max <= 0:
            return None
        required = 0
        for dag in dags:
            cores = MultipathAnalyzer.min_cores_for_dag(dag, m_max)
            if cores is None:
                return None
            required = max(required, cores)
        return required

    @staticmethod
    def _calculate_multipath_bound(dag: DAG, m: int, mode: str, n: int) -> float:
        vol = MultipathAnalyzer._total_workload(dag, mode)
        length = MultipathAnalyzer._critical_path_length(dag, mode)
        volumes = MultipathAnalyzer._max_volumes_for_cardinalities(dag, mode, n)
        bounds: List[float] = []

        for j, max_volume in enumerate(volumes):
            denom = m - j
            if denom <= 0:
                continue
            bound = length + (vol - max_volume) / denom
            bounds.append(bound)

        if not bounds:
            return float("inf")
        return min(bounds)

    @staticmethod
    def _total_workload(dag: DAG, mode: str) -> int:
        return sum(node.get_exec_time(mode) for node in dag.nodes.values())

    @staticmethod
    def _critical_path_length(dag: DAG, mode: str) -> int:
        memo: Dict[int, int] = {}

        def _visit(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            node = dag.nodes[node_id]
            current = node.get_exec_time(mode)
            if not node.predecessors:
                memo[node_id] = current
                return current
            best = max(_visit(pred) for pred in node.predecessors)
            memo[node_id] = current + best
            return memo[node_id]

        if not dag.nodes:
            return 0
        return max(_visit(node_id) for node_id in dag.nodes)

    @staticmethod
    def _compute_reachability(dag: DAG) -> Dict[int, set]:
        reachability: Dict[int, set] = {}
        for node_id in dag.nodes:
            visited = set()
            stack = [node_id]
            while stack:
                current = stack.pop()
                for succ in dag.nodes[current].successors:
                    if succ in visited:
                        continue
                    visited.add(succ)
                    stack.append(succ)
            reachability[node_id] = visited
        return reachability

    @staticmethod
    def _compute_width(dag: DAG) -> int:
        """
        Width is the size of the maximum antichain.
        Using Dilworth: width == min chain decomposition size.
        """
        nodes = list(dag.nodes.keys())
        if not nodes:
            return 0
        reachability = MultipathAnalyzer._compute_reachability(dag)
        adj: Dict[int, List[int]] = {u: [] for u in nodes}
        for u in nodes:
            adj[u] = list(reachability[u])

        pair_u: Dict[int, Optional[int]] = {u: None for u in nodes}
        pair_v: Dict[int, Optional[int]] = {v: None for v in nodes}
        dist: Dict[int, int] = {}

        def bfs() -> bool:
            queue = deque()
            for u in nodes:
                if pair_u[u] is None:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float("inf")
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] is None:
                        found = True
                    else:
                        pu = pair_v[v]
                        if dist[pu] == float("inf"):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
            return found

        def dfs(u: int) -> bool:
            for v in adj[u]:
                pu = pair_v[v]
                if pair_v[v] is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float("inf")
            return False

        matching = 0
        while bfs():
            for u in nodes:
                if pair_u[u] is None and dfs(u):
                    matching += 1

        return len(nodes) - matching

    @staticmethod
    def _max_volumes_for_cardinalities(dag: DAG, mode: str, max_paths: int) -> List[int]:
        graph, source, sink = MultipathAnalyzer._build_flow_network(dag, mode)
        return MultipathAnalyzer._min_cost_flow_prefix(graph, source, sink, max_paths)

    @staticmethod
    def _build_flow_network(dag: DAG, mode: str):
        node_ids = list(dag.nodes.keys())
        index = {node_id: i for i, node_id in enumerate(node_ids)}
        n = len(node_ids)
        source = 2 * n
        sink = 2 * n + 1
        graph: List[List[dict]] = [[] for _ in range(2 * n + 2)]

        def add_edge(u: int, v: int, cap: int, cost: int) -> None:
            graph[u].append({"to": v, "cap": cap, "cost": cost, "rev": len(graph[v])})
            graph[v].append({"to": u, "cap": 0, "cost": -cost, "rev": len(graph[u]) - 1})

        reachability = MultipathAnalyzer._compute_reachability(dag)

        for node_id in node_ids:
            i = index[node_id]
            vin = 2 * i
            vout = 2 * i + 1
            cost = -dag.nodes[node_id].get_exec_time(mode)
            add_edge(vin, vout, 1, cost)
            add_edge(source, vin, 1, 0)
            add_edge(vout, sink, 1, 0)

        for u in node_ids:
            for v in reachability[u]:
                uout = 2 * index[u] + 1
                vin = 2 * index[v]
                add_edge(uout, vin, 1, 0)

        return graph, source, sink

    @staticmethod
    def _shortest_path(graph: List[List[dict]], source: int, sink: int):
        n = len(graph)
        dist = [float("inf")] * n
        in_queue = [False] * n
        prev: List[Optional[Tuple[int, int]]] = [None] * n
        queue = deque([source])
        dist[source] = 0
        in_queue[source] = True

        while queue:
            v = queue.popleft()
            in_queue[v] = False
            for i, edge in enumerate(graph[v]):
                if edge["cap"] <= 0:
                    continue
                nd = dist[v] + edge["cost"]
                if nd < dist[edge["to"]]:
                    dist[edge["to"]] = nd
                    prev[edge["to"]] = (v, i)
                    if not in_queue[edge["to"]]:
                        queue.append(edge["to"])
                        in_queue[edge["to"]] = True

        if dist[sink] == float("inf"):
            return None, None
        return dist[sink], prev

    @staticmethod
    def _min_cost_flow_prefix(
        graph: List[List[dict]],
        source: int,
        sink: int,
        max_flow: int,
    ) -> List[int]:
        n = len(graph)
        potential = [0.0] * n
        dist = [float("inf")] * n
        in_queue = [False] * n
        queue = deque([source])
        dist[source] = 0.0
        in_queue[source] = True
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for edge in graph[u]:
                if edge["cap"] <= 0:
                    continue
                v = edge["to"]
                nd = dist[u] + edge["cost"]
                if nd < dist[v]:
                    dist[v] = nd
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        for i, d in enumerate(dist):
            if d < float("inf"):
                potential[i] = d

        total_cost = 0.0
        volumes: List[int] = []
        for _ in range(max_flow):
            dist = [float("inf")] * n
            prev: List[Optional[Tuple[int, int]]] = [None] * n
            dist[source] = 0.0
            heap: List[Tuple[float, int]] = [(0.0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for i, edge in enumerate(graph[u]):
                    if edge["cap"] <= 0:
                        continue
                    v = edge["to"]
                    nd = d + edge["cost"] + potential[u] - potential[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = (u, i)
                        heapq.heappush(heap, (nd, v))
            if dist[sink] == float("inf"):
                break
            for i, d in enumerate(dist):
                if d < float("inf"):
                    potential[i] += d
            v = sink
            while v != source:
                pv, pe = prev[v]
                edge = graph[pv][pe]
                edge["cap"] -= 1
                graph[v][edge["rev"]]["cap"] += 1
                total_cost += edge["cost"]
                v = pv
            volumes.append(int(-total_cost))
        return volumes
