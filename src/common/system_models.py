# src/common/system_models.py
from dataclasses import dataclass

@dataclass
class SystemModel:
    """
    Hardware requirements and probabilistic constraints for a target system.
    """
    num_cores: int  # Number of cores (m).
    
    # System-wide allowable failure probability, e.g., 10^-5 per job.
    # Some experiments use looser values than ASIL-D (10^-9) for readability.
    allowable_failure_prob: float = 1e-5
    
    # Failure rate lambda per time unit.
    # Chosen so C ~= 100 gives P ~= 1e-7 in exploratory experiments.
    failure_rate: float = 1e-9

    # Overheads.
    context_switch_overhead: int = 0
    migration_overhead: int = 0

    def failure_budget_per_cluster(self, estimated_cluster_count: int) -> float:
        """Return the maximum risk allowed per cluster under the system-wide constraint."""
        if estimated_cluster_count <= 0:
            return 1.0
        return self.allowable_failure_prob / estimated_cluster_count

    def __repr__(self):
        return (f"SystemModel(m={self.num_cores}, "
                f"lambda={self.failure_rate:.1e}, "
                f"P_allow={self.allowable_failure_prob:.1e})")
