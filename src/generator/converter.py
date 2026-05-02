import math
import random
from typing import List, Literal, Optional
from src.common.dag_models import DAG
from src.analyzer.multipath import MultipathAnalyzer

DeadlineType = Literal["implicit", "constrained", "arbitrary"]
Criticality = Literal["HI", "LO"]

class MCConfig:
    def __init__(
        self, 
        p_hi: float = 0.5,           
        force_criticality: Optional[Criticality] = None,
        factor_min: float = 1.5,     
        factor_max: float = 3.5,
        seed: int = 42,
        deadline_type: DeadlineType = "constrained",
        deadline_slack_min: float = 1.2,
        deadline_slack_max: float = 1.5
    ):
        self.p_hi = p_hi
        self.force_criticality = force_criticality
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.seed = seed
        self.deadline_type = deadline_type
        self.deadline_slack_min = deadline_slack_min
        self.deadline_slack_max = deadline_slack_max

class MCDAGConverter:
    def __init__(self, config: MCConfig):
        self.config = config
        random.seed(config.seed)

    def convert(self, dags: List[DAG]) -> List[DAG]:
        self._assign_criticalities(dags)
        for dag in dags:
            self._inject_mc_parameters(dag)
            self._inject_timing_constraints(dag)
        return dags

    def _assign_criticalities(self, dags: List[DAG]):
        n = len(dags)
        if self.config.force_criticality is not None:
            target = self.config.force_criticality
            for dag in dags:
                dag.criticality = target
        else:
            num_hi = int(n * self.config.p_hi)
            if num_hi == 0 and self.config.p_hi > 0 and n > 0:
                num_hi = 1
            indices = list(range(n))
            random.shuffle(indices)
            hi_indices = set(indices[:num_hi])
            for i, dag in enumerate(dags):
                if i in hi_indices:
                    dag.criticality = "HI"
                else:
                    dag.criticality = "LO"

    def _inject_mc_parameters(self, dag: DAG):
        if dag.criticality == "HI":
            for node in dag.nodes.values():
                factor = random.uniform(self.config.factor_min, self.config.factor_max)
                calculated_hi = int(node.c_lo * factor)
                if calculated_hi <= node.c_lo:
                    calculated_hi = node.c_lo + 1
                
                node.c_hi = calculated_hi
                # The node has not been merged yet, so sigma starts as c_hi.
                node.sigma_c_hi = calculated_hi
                node.constituent_sigmas = [calculated_hi]
        else:
            for node in dag.nodes.values():
                node.c_hi = node.c_lo
                node.sigma_c_hi = node.c_lo
                node.constituent_sigmas = [node.c_lo] if node.c_lo > 0 else []

    def _inject_timing_constraints(self, dag: DAG):
        l_lo = MultipathAnalyzer.get_critical_path(dag, "LO")
        l_hi = MultipathAnalyzer.get_critical_path(dag, "HI")
        max_l = max(l_lo, l_hi)

        if dag.deadline > 0:
            if max_l > 0 and dag.deadline <= max_l:
                dag.deadline = max_l + 1
            if dag.period <= 0 or dag.period < dag.deadline:
                dag.period = dag.deadline
            return
        
        slack = random.uniform(self.config.deadline_slack_min, self.config.deadline_slack_max)
        base_deadline = int(math.ceil(max_l * slack))
        if max_l > 0 and base_deadline <= max_l:
            base_deadline = max_l + 1
        
        if self.config.deadline_type == "implicit":
            dag.deadline = base_deadline
            dag.period = base_deadline
        elif self.config.deadline_type == "constrained":
            dag.deadline = base_deadline
            t_factor = random.uniform(1.0, 1.5)
            dag.period = int(base_deadline * t_factor)
        elif self.config.deadline_type == "arbitrary":
            dag.period = base_deadline
            d_factor = random.uniform(0.8, 2.0)
            dag.deadline = int(dag.period * d_factor)
            if dag.deadline <= max_l and max_l > 0:
                dag.deadline = max_l + 1
