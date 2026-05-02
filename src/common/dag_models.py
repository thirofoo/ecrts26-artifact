import math
from typing import List, Dict, Literal, Optional

Criticality = Literal["HI", "LO"]

class Node:
    def __init__(self, node_id: int, c_lo: int):
        self.id = node_id
        
        # --- Scheduling parameters ---
        self.c_lo: int = c_lo
        self.c_hi: int = c_lo 
        
        # --- Reliability parameters ---
        self.sigma_c_hi: int = c_lo 
        # Optional manual failure probability override (per-hour).
        self.failure_prob: Optional[float] = None
        
        # TCAD Eq.4 parameters used during clustering
        self.constituent_probs: List[float] = []
        # Keep sigma per constituent for easy recomputation when failure_rate changes
        self.constituent_sigmas: List[int] = [c_lo] if c_lo > 0 else []
        
        # Original node IDs merged into this node, used for visualization.
        self.merged_ids: List[int] = [node_id]
        
        # Graph structure.
        self.predecessors: List[int] = []
        self.successors: List[int] = []
        self.original_nodes: List[int] = [node_id]
        self.cluster_id: Optional[int] = None

    def get_exec_time(self, mode: Criticality) -> int:
        return self.c_hi if mode == "HI" else self.c_lo

    def get_failure_prob(self, failure_rate: float) -> float:
        """
        Return the probability that this node exceeds LO-WCET.

        The probability is computed from sigma for this node:
        P = 1 - e^(-lambda * Sigma_C)
        """
        manual_prob = getattr(self, "failure_prob", None)
        if manual_prob is not None:
            return manual_prob
        if self.sigma_c_hi == 0: return 0.0
        return 1.0 - math.exp(-failure_rate * self.sigma_c_hi)

    def initialize_probs(self, failure_rate: float):
        """Refresh the cached probability list for the latest failure rate."""
        manual_prob = getattr(self, "failure_prob", None)
        if manual_prob is not None:
            self.constituent_probs = [manual_prob]
            return
        sigmas = getattr(self, "constituent_sigmas", None)
        if not sigmas:
            base_sigma = getattr(self, "sigma_c_hi", 0)
            if base_sigma > 0:
                self.constituent_sigmas = [base_sigma]
                sigmas = self.constituent_sigmas
            else:
                self.constituent_sigmas = []
                self.constituent_probs = []
                return

        self.constituent_probs = [
            1.0 - math.exp(-failure_rate * sigma)
            for sigma in sigmas
        ]

    @property
    def delta(self) -> int:
        """Additional cost incurred in HI mode."""
        return self.c_hi - self.c_lo

    def __repr__(self):
        return f"Node({self.id}, L:{self.c_lo}/H:{self.c_hi})"

class DAG:
    def __init__(self, dag_id: str):
        self.id = dag_id
        self.nodes: Dict[int, Node] = {}
        self.criticality: Criticality = "LO"
        self.period: int = 0
        self.deadline: int = 0
        self.cluster_groups: Dict[int, List[int]] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, src_id: int, dest_id: int):
        if src_id in self.nodes and dest_id in self.nodes:
            if dest_id not in self.nodes[src_id].successors:
                self.nodes[src_id].successors.append(dest_id)
            if src_id not in self.nodes[dest_id].predecessors:
                self.nodes[dest_id].predecessors.append(src_id)
    
    def get_total_failure_prob(self, failure_rate: float) -> float:
        """Return the union failure probability over all nodes in the DAG."""
        p_survival = 1.0
        for node in self.nodes.values():
            p_node = node.get_failure_prob(failure_rate)
            p_survival *= (1.0 - p_node)
        return 1.0 - p_survival

    def get_max_node_failure_prob(self, failure_rate: float) -> float:
        """
        Return the maximum node or cluster failure probability in the DAG.
        """
        max_p = 0.0
        for node in self.nodes.values():
            p = node.get_failure_prob(failure_rate)
            if p > max_p:
                max_p = p
        return max_p

    def __repr__(self):
        return (f"DAG({self.id}, {self.criticality}, "
                f"N={len(self.nodes)}, T={self.period}, D={self.deadline})")
