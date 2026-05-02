from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.analyzer.multipath import MultipathAnalyzer
from src.common.dag_models import DAG, Node
from src.common.system_models import SystemModel


@dataclass(frozen=True)
class AssignmentResult:
    reduced_dag: DAG
    cores_required: int
    workload: int
    critical_path: int
    cluster_gms: List[float]
    cluster_count: int
    is_acyclic: bool


@dataclass(frozen=True)
class OptimizationResult:
    reduced_dags: List[DAG]
    core_sum: int
    total_clusters: int
    best_score: float


class ClusterAssignment:
    def __init__(self, dag: DAG, clusters: List[Set[int]], neighbors: Dict[int, Set[int]]):
        self.dag = dag
        self.clusters = [set(cluster) for cluster in clusters]
        self.neighbors = neighbors

    def copy(self) -> "ClusterAssignment":
        return ClusterAssignment(self.dag, [set(cluster) for cluster in self.clusters], self.neighbors)

    def cluster_count(self) -> int:
        return len(self.clusters)


class ProposedClusteringEngine:
    def __init__(
        self,
        system: SystemModel,
        max_exhaustive_split_nodes: int = 12,
        random_split_samples: int = 24,
    ) -> None:
        self.system = system
        # Split candidate generation:
        # - enumerate all connected bipartitions only for small clusters
        # - sample random connected bipartitions for larger clusters
        self.max_exhaustive_split_nodes = max(2, int(max_exhaustive_split_nodes))
        self.random_split_samples = max(1, int(random_split_samples))

    @staticmethod
    def calculate_g_m(prob_list: Sequence[float]) -> float:
        if not prob_list:
            return 0.0

        p_none = 1.0
        for prob in prob_list:
            p_none *= (1.0 - prob)

        p_one = 0.0
        for i, prob in enumerate(prob_list):
            term = prob
            for j, other in enumerate(prob_list):
                if i == j:
                    continue
                term *= (1.0 - other)
            p_one += term

        g_m = 1.0 - p_none - p_one
        return max(0.0, g_m)

    def optimize_taskset(
        self,
        dags: Sequence[DAG],
        time_limit: float = 5.0,
        max_cores: int = 32,
        seed: Optional[int] = None,
        temp_start: float = 1e-1,
        temp_end: float = 1e-3,
        penalty_weight_start: float = 0.5,
        penalty_weight_end: float = 5.0,
        apply_to_low_util: bool = False,
        score_mode: str = "cores",
        score_epsilon: float = 0.0,
        best_log: Optional[List[Dict[str, float]]] = None,
        time_series_log: Optional[List[Dict[str, float]]] = None,
        time_series_interval_ms: int = 100,
        log_accepts: bool = False,
        verbose: bool = True,
    ) -> OptimizationResult:
        allowed_modes = {
            "cores",
            "workload",
            "cores+critical",
            "cores+workload",
            "critical",
        }
        if score_mode not in allowed_modes:
            raise ValueError(f"score_mode must be one of: {', '.join(sorted(allowed_modes))}.")
        if score_epsilon < 0:
            raise ValueError("score_epsilon must be >= 0.")
        rng = random.Random(seed)
        work_dags = [self._clone_dag(dag) for dag in dags]
        for dag in work_dags:
            for node in dag.nodes.values():
                node.initialize_probs(self.system.failure_rate)

        hi_dags = [dag for dag in work_dags if apply_to_low_util or self._is_high_utilization(dag)]
        if not hi_dags:
            return OptimizationResult([], 0, 0, 0.0)

        move_types = ["move", "merge", "split", "merge_split"]
        neighbor_stats = {
            move: {"attempt": 0, "success": 0, "accepted": 0} for move in move_types
        }

        assignments = [
            ClusterAssignment(
                dag,
                [{node_id} for node_id in dag.nodes],
                self._build_neighbors(dag),
            )
            for dag in hi_dags
        ]

        compute_cores = score_mode != "critical"
        results = [
            self._evaluate_assignment(assign, max_cores, compute_cores)
            for assign in assignments
        ]
        if any(not result.is_acyclic for result in results):
            raise RuntimeError("Initial clustering produced a cyclic reduced DAG.")
        core_sum = sum(result.cores_required for result in results)
        workload_sum = sum(result.workload for result in results)
        critical_sum = sum(result.critical_path for result in results)
        total_clusters = sum(result.cluster_count for result in results)
        penalty_weight = self._schedule_penalty_weight(
            penalty_weight_start, penalty_weight_end, 0.0
        )
        penalty_value, per_dag_penalties = self._compute_penalties(results, total_clusters)
        objective_sum = self._objective_value(
            score_mode,
            score_epsilon,
            core_sum,
            workload_sum,
            critical_sum,
        )
        score = objective_sum + penalty_weight * penalty_value
        if verbose:
            print(
                f"Initial score: objective={score_mode} value={objective_sum:.6f} "
                f"cores_sum={core_sum} clusters={total_clusters} "
                f"penalty={penalty_value:.6f} total={score:.6f}"
            )

        best_any_clusters = [assignment.copy().clusters for assignment in assignments]
        best_any_results = [result for result in results]
        best_any_score = score
        best_any_core_sum = core_sum
        best_any_total_clusters = total_clusters

        best_clusters = best_any_clusters
        best_results = best_any_results
        best_score = float("inf")
        best_core_sum = best_any_core_sum
        best_total_clusters = best_any_total_clusters
        best_feasible_found = False
        if penalty_value == 0.0:
            best_clusters = [assignment.copy().clusters for assignment in assignments]
            best_results = [result for result in results]
            best_score = score
            best_core_sum = core_sum
            best_total_clusters = total_clusters
            best_feasible_found = True

        if time_limit <= 0:
            return OptimizationResult(
                [result.reduced_dag for result in best_results],
                best_core_sum,
                best_total_clusters,
                best_score,
            )

        start = time.monotonic()
        next_log_ms = 0.0
        if time_series_interval_ms <= 0:
            time_series_interval_ms = 0
        if best_log is not None:
            temperature0 = self._schedule_temperature(temp_start, temp_end, 0.0)
            penalty_weight0 = self._schedule_penalty_weight(
                penalty_weight_start, penalty_weight_end, 0.0
            )
            objective_sum = self._objective_value(
                score_mode,
                score_epsilon,
                core_sum,
                workload_sum,
                critical_sum,
            )
            self._record_best_update(
                best_log,
                "initial",
                0,
                0.0,
                0.0,
                temperature0,
                penalty_weight0,
                core_sum,
                total_clusters,
                objective_sum,
                penalty_value,
                score,
            )
        if time_series_log is not None and time_series_interval_ms > 0:
            temperature0 = self._schedule_temperature(temp_start, temp_end, 0.0)
            penalty_weight0 = self._schedule_penalty_weight(
                penalty_weight_start, penalty_weight_end, 0.0
            )
            objective_sum = self._objective_value(
                score_mode,
                score_epsilon,
                core_sum,
                workload_sum,
                critical_sum,
            )
            self._record_time_sample(
                time_series_log,
                0.0,
                0.0,
                temperature0,
                penalty_weight0,
                core_sum,
                total_clusters,
                objective_sum,
                penalty_value,
                score,
                best_any_score,
                best_any_core_sum,
                best_score if best_feasible_found else float("nan"),
                best_core_sum if best_feasible_found else float("nan"),
            )
            next_log_ms = float(time_series_interval_ms)
        iteration = 0
        accepted = 0
        while time.monotonic() - start < time_limit:
            iteration += 1
            elapsed = time.monotonic() - start
            progress = min(1.0, elapsed / time_limit)
            temperature = self._schedule_temperature(temp_start, temp_end, progress)
            penalty_weight = self._schedule_penalty_weight(
                penalty_weight_start, penalty_weight_end, progress
            )
            objective_sum = self._objective_value(
                score_mode,
                score_epsilon,
                core_sum,
                workload_sum,
                critical_sum,
            )
            score = objective_sum + penalty_weight * penalty_value

            index = rng.randrange(len(assignments))
            assignment = assignments[index]

            def score_candidate(clusters: List[Set[int]]):
                candidate = ClusterAssignment(assignment.dag, clusters, assignment.neighbors)
                result = self._evaluate_assignment(candidate, max_cores, compute_cores)
                if not result.is_acyclic:
                    return float("inf"), result
                obj = self._objective_value(
                    score_mode,
                    score_epsilon,
                    result.cores_required,
                    result.workload,
                    result.critical_path,
                )
                new_total_clusters = (
                    total_clusters - assignment.cluster_count() + result.cluster_count
                )
                limit = self.system.failure_budget_per_cluster(new_total_clusters)
                penalty = self._penalty_for_gms(result.cluster_gms, limit)
                return obj + penalty_weight * penalty, result

            proposal, move_type, proposal_result = self._propose_neighbor(
                assignment, rng, neighbor_stats, score_candidate
            )
            if proposal is None:
                continue

            if proposal_result is None:
                proposal_result = self._evaluate_assignment(proposal, max_cores, compute_cores)
            if not proposal_result.is_acyclic:
                continue

            new_results = list(results)
            new_results[index] = proposal_result
            new_core_sum = core_sum - results[index].cores_required + proposal_result.cores_required
            new_workload_sum = workload_sum - results[index].workload + proposal_result.workload
            new_critical_sum = critical_sum - results[index].critical_path + proposal_result.critical_path
            new_total_clusters = total_clusters - results[index].cluster_count + proposal_result.cluster_count
            clusters_same = new_total_clusters == total_clusters
            if clusters_same:
                limit = self.system.failure_budget_per_cluster(total_clusters)
                new_dag_penalty = self._penalty_for_gms(proposal_result.cluster_gms, limit)
                new_penalty = penalty_value - per_dag_penalties[index] + new_dag_penalty
                new_per_dag_penalties = None
            else:
                new_penalty, new_per_dag_penalties = self._compute_penalties(
                    new_results, new_total_clusters
                )
                new_dag_penalty = None
            new_objective_sum = self._objective_value(
                score_mode,
                score_epsilon,
                new_core_sum,
                new_workload_sum,
                new_critical_sum,
            )
            new_score = new_objective_sum + penalty_weight * new_penalty

            delta = new_score - score
            is_best = False
            # print(f"temp={math.exp(-delta / max(temperature, 1e-6)):.6f}, delta={delta:.6f}\n")cler
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6)):
                assignments[index] = proposal
                results = new_results
                core_sum = new_core_sum
                workload_sum = new_workload_sum
                critical_sum = new_critical_sum
                total_clusters = new_total_clusters
                score = new_score
                penalty_value = new_penalty
                if clusters_same:
                    per_dag_penalties[index] = new_dag_penalty if new_dag_penalty is not None else 0.0
                else:
                    per_dag_penalties = new_per_dag_penalties or per_dag_penalties
                accepted += 1
                if move_type in neighbor_stats:
                    neighbor_stats[move_type]["accepted"] += 1

                if score < best_any_score:
                    best_any_score = score
                    best_any_core_sum = core_sum
                    best_any_total_clusters = total_clusters
                    best_any_clusters = [assignment.copy().clusters for assignment in assignments]
                    best_any_results = [result for result in results]
                    if best_log is not None:
                        objective_sum = self._objective_value(
                            score_mode,
                            score_epsilon,
                            core_sum,
                            workload_sum,
                            critical_sum,
                        )
                        self._record_best_update(
                            best_log,
                            "best_any",
                            iteration,
                            elapsed * 1000.0,
                            progress,
                            temperature,
                            penalty_weight,
                            core_sum,
                            total_clusters,
                            objective_sum,
                            penalty_value,
                            score,
                        )

                if new_penalty == 0.0 and (not best_feasible_found or score < best_score):
                    best_score = score
                    best_core_sum = core_sum
                    best_total_clusters = total_clusters
                    best_clusters = [assignment.copy().clusters for assignment in assignments]
                    best_results = [result for result in results]
                    best_feasible_found = True
                    is_best = True
                    if best_log is not None:
                        objective_sum = self._objective_value(
                            score_mode,
                            score_epsilon,
                            core_sum,
                            workload_sum,
                            critical_sum,
                        )
                        self._record_best_update(
                            best_log,
                            "best_feasible",
                            iteration,
                            elapsed * 1000.0,
                            progress,
                            temperature,
                            penalty_weight,
                            core_sum,
                            total_clusters,
                            objective_sum,
                            penalty_value,
                            score,
                        )

                if verbose and (log_accepts or delta > 0):
                    tag = " best" if is_best else ""
                    print(
                        f"[improve] iter={iteration} accepted={accepted} "
                        f"score={score:.6f} cores={core_sum} clusters={total_clusters} "
                        f"penalty={new_penalty:.6f} delta={delta:.6f}{tag}"
                    )

            if time_series_log is not None and time_series_interval_ms > 0:
                elapsed_ms = elapsed * 1000.0
                while elapsed_ms >= next_log_ms:
                    objective_sum = self._objective_value(
                        score_mode,
                        score_epsilon,
                        core_sum,
                        workload_sum,
                        critical_sum,
                    )
                    self._record_time_sample(
                        time_series_log,
                        elapsed_ms,
                        progress,
                        temperature,
                        penalty_weight,
                        core_sum,
                        total_clusters,
                        objective_sum,
                        penalty_value,
                        score,
                        best_any_score,
                        best_any_core_sum,
                        best_score if best_feasible_found else float("nan"),
                        best_core_sum if best_feasible_found else float("nan"),
                    )
                    next_log_ms += float(time_series_interval_ms)

        if time_series_log is not None and time_series_interval_ms > 0:
            elapsed_s = time.monotonic() - start
            elapsed_ms = elapsed_s * 1000.0
            last_elapsed = -1.0
            if time_series_log:
                last_elapsed = float(time_series_log[-1].get("elapsed_ms", -1.0))
            if elapsed_ms > last_elapsed + 1e-6:
                progress = min(1.0, elapsed_s / time_limit)
                temperature = self._schedule_temperature(temp_start, temp_end, progress)
                penalty_weight = self._schedule_penalty_weight(
                    penalty_weight_start, penalty_weight_end, progress
                )
                objective_sum = self._objective_value(
                    score_mode,
                    score_epsilon,
                    core_sum,
                    workload_sum,
                    critical_sum,
                )
                self._record_time_sample(
                    time_series_log,
                    elapsed_ms,
                    progress,
                    temperature,
                    penalty_weight,
                    core_sum,
                    total_clusters,
                    objective_sum,
                    penalty_value,
                    score,
                    best_any_score,
                    best_any_core_sum,
                    best_score if best_feasible_found else float("nan"),
                    best_core_sum if best_feasible_found else float("nan"),
                )

        if not best_feasible_found:
            best_score = best_any_score
            best_core_sum = best_any_core_sum
            best_total_clusters = best_any_total_clusters
            best_clusters = best_any_clusters
            best_results = best_any_results

        reduced_best = []
        for dag, clusters in zip(hi_dags, best_clusters):
            reduced, _ = self._build_reduced_dag(dag, clusters)
            reduced_best.append(reduced)

        stats_line = self._format_neighbor_stats(neighbor_stats, move_types)
        if stats_line and verbose:
            print(stats_line)

        if not compute_cores:
            recomputed_core_sum = 0
            for reduced in reduced_best:
                cores_required = MultipathAnalyzer.min_cores_for_dag(reduced, max_cores)
                if cores_required is None:
                    cores_required = max_cores + 1
                recomputed_core_sum += cores_required
            best_core_sum = recomputed_core_sum

        return OptimizationResult(reduced_best, best_core_sum, best_total_clusters, best_score)

    def optimize_dag(
        self,
        dag: DAG,
        time_limit: float = 5.0,
        max_cores: int = 32,
        seed: Optional[int] = None,
        temp_start: float = 100.0,
        temp_end: float = 1.0,
        penalty_weight_start: float = 0.5,
        penalty_weight_end: float = 5.0,
        score_mode: str = "cores",
        score_epsilon: float = 0.0,
        best_log: Optional[List[Dict[str, float]]] = None,
        time_series_log: Optional[List[Dict[str, float]]] = None,
        time_series_interval_ms: int = 100,
        verbose: bool = True,
    ) -> OptimizationResult:
        return self.optimize_taskset(
            [dag],
            time_limit=time_limit,
            max_cores=max_cores,
            seed=seed,
            temp_start=temp_start,
            temp_end=temp_end,
            penalty_weight_start=penalty_weight_start,
            penalty_weight_end=penalty_weight_end,
            apply_to_low_util=True,
            score_mode=score_mode,
            score_epsilon=score_epsilon,
            best_log=best_log,
            time_series_log=time_series_log,
            time_series_interval_ms=time_series_interval_ms,
            verbose=verbose,
        )

    def _score_state(
        self,
        core_sum: int,
        results: Sequence[AssignmentResult],
        total_clusters: int,
        penalty_weight: float,
    ) -> Tuple[float, float]:
        penalty_value, _ = self._compute_penalties(results, total_clusters)
        score = core_sum + penalty_weight * penalty_value
        return score, penalty_value

    def _penalty_for_gms(self, cluster_gms: Sequence[float], limit: float) -> float:
        if limit <= 0:
            return 0.0
        penalty = 0.0
        for g_m in cluster_gms:
            if g_m <= limit:
                continue
            penalty += (g_m - limit) / max(limit, 1e-12)
        return penalty

    @staticmethod
    def _objective_value(
        score_mode: str,
        score_epsilon: float,
        core_sum: int,
        workload_sum: int,
        critical_sum: int,
    ) -> float:
        if score_mode == "cores":
            return float(core_sum)
        if score_mode == "workload":
            return float(workload_sum)
        if score_mode == "critical":
            return score_epsilon * float(critical_sum)
        if score_mode == "cores+critical":
            return float(core_sum) + score_epsilon * float(critical_sum)
        if score_mode == "cores+workload":
            return float(core_sum) + score_epsilon * float(workload_sum)
        return float(core_sum)

    def _compute_penalties(
        self,
        results: Sequence[AssignmentResult],
        total_clusters: int,
    ) -> Tuple[float, List[float]]:
        if total_clusters <= 0:
            return 0.0, [0.0 for _ in results]
        limit = self.system.failure_budget_per_cluster(total_clusters)
        per_dag: List[float] = []
        penalty_value = 0.0
        for result in results:
            dag_penalty = self._penalty_for_gms(result.cluster_gms, limit)
            per_dag.append(dag_penalty)
            penalty_value += dag_penalty
        return penalty_value, per_dag

    def _schedule_temperature(self, start: float, end: float, progress: float) -> float:
        if progress <= 0.0:
            return start
        if progress >= 1.0:
            return end
        if end <= 0:
            return max(0.0, start * (1.0 - progress))
        if start <= 0:
            return end
        ratio = end / start
        return start * (ratio ** progress)

    def _schedule_penalty_weight(
        self,
        start: float,
        end: float,
        progress: float,
    ) -> float:
        if progress <= 0.0:
            return start
        if progress >= 1.0:
            return end
        return start + (end - start) * progress

    @staticmethod
    def _record_best_update(
        best_log: List[Dict[str, float]],
        kind: str,
        iteration: int,
        elapsed_ms: float,
        progress: float,
        temperature: float,
        penalty_weight: float,
        core_sum: int,
        total_clusters: int,
        objective_sum: float,
        penalty_value: float,
        score: float,
    ) -> None:
        best_log.append(
            {
                "kind": kind,
                "iteration": float(iteration),
                "elapsed_ms": elapsed_ms,
                "progress": progress,
                "temperature": temperature,
                "penalty_weight": penalty_weight,
                "core_sum": float(core_sum),
                "total_clusters": float(total_clusters),
                "objective": objective_sum,
                "penalty": penalty_value,
                "score": score,
            }
        )

    @staticmethod
    def _record_time_sample(
        time_series_log: List[Dict[str, float]],
        elapsed_ms: float,
        progress: float,
        temperature: float,
        penalty_weight: float,
        core_sum: int,
        total_clusters: int,
        objective_sum: float,
        penalty_value: float,
        score: float,
        best_any_score: float,
        best_any_core_sum: int,
        best_feasible_score: float,
        best_feasible_core_sum: float,
    ) -> None:
        time_series_log.append(
            {
                "elapsed_ms": elapsed_ms,
                "progress": progress,
                "temperature": temperature,
                "penalty_weight": penalty_weight,
                "core_sum": float(core_sum),
                "total_clusters": float(total_clusters),
                "objective": objective_sum,
                "penalty": penalty_value,
                "score_current": score,
                "score_best_any": best_any_score,
                "cores_best_any": float(best_any_core_sum),
                "score_best_feasible": best_feasible_score,
                "cores_best_feasible": best_feasible_core_sum,
            }
        )

    def _evaluate_assignment(
        self,
        assignment: ClusterAssignment,
        max_cores: int,
        compute_cores: bool = True,
    ) -> AssignmentResult:
        reduced, cluster_gms = self._build_reduced_dag(assignment.dag, assignment.clusters)
        is_acyclic = self._is_acyclic(reduced)
        if not is_acyclic:
            workload = self._workload_for_dag(reduced)
            return AssignmentResult(
                reduced,
                max_cores + 1,
                workload,
                0,
                cluster_gms,
                len(assignment.clusters),
                False,
            )
        workload = self._workload_for_dag(reduced)
        critical_path = self._critical_path_for_dag(reduced)
        if compute_cores:
            cores_required = MultipathAnalyzer.min_cores_for_dag(reduced, max_cores)
            if cores_required is None:
                cores_required = max_cores + 1
        else:
            cores_required = 0
        return AssignmentResult(
            reduced,
            cores_required,
            workload,
            critical_path,
            cluster_gms,
            len(assignment.clusters),
            True,
        )

    @staticmethod
    def _workload_for_dag(dag: DAG) -> int:
        mode = "HI" if dag.criticality == "HI" else "LO"
        return MultipathAnalyzer.get_workload(dag, mode)

    @staticmethod
    def _critical_path_for_dag(dag: DAG) -> int:
        mode = "HI" if dag.criticality == "HI" else "LO"
        return MultipathAnalyzer.get_critical_path(dag, mode)

    def _propose_neighbor(
        self,
        assignment: ClusterAssignment,
        rng: random.Random,
        stats: Optional[Dict[str, Dict[str, int]]] = None,
        scorer=None,
    ) -> Tuple[Optional[ClusterAssignment], Optional[str], Optional[AssignmentResult]]:
        for _ in range(12):
            move_type = rng.choice(["move", "merge", "split", "merge_split"])
            if stats is not None and move_type in stats:
                stats[move_type]["attempt"] += 1
            if move_type == "move":
                proposal, result = self._propose_move(assignment, rng, scorer)
            elif move_type == "merge":
                proposal = self._propose_merge(assignment, rng)
                result = None
            elif move_type == "split":
                proposal, result = self._propose_split(assignment, rng, scorer)
            else:
                proposal, result = self._propose_merge_split(assignment, rng, scorer)

            if proposal is not None:
                if stats is not None and move_type in stats:
                    stats[move_type]["success"] += 1
                return proposal, move_type, result
        return None, None, None

    @staticmethod
    def _format_neighbor_stats(
        stats: Dict[str, Dict[str, int]],
        order: Sequence[str],
    ) -> str:
        if not stats:
            return ""
        parts = []
        for key in order:
            item = stats.get(key)
            if not item:
                continue
            parts.append(
                f"{key}=succ:{item['success']} acc:{item['accepted']} try:{item['attempt']}"
            )
        if not parts:
            return ""
        return "[neighbor stats] " + " | ".join(parts)

    def _propose_move(
        self,
        assignment: ClusterAssignment,
        rng: random.Random,
        scorer=None,
    ) -> Tuple[Optional[ClusterAssignment], Optional[AssignmentResult]]:
        if assignment.cluster_count() < 2:
            return None, None
        clusters = [set(cluster) for cluster in assignment.clusters]
        indices = list(range(len(clusters)))
        rng.shuffle(indices)

        for source_idx in indices:
            source_cluster = clusters[source_idx]
            if not source_cluster:
                continue
            adjacent = [
                idx
                for idx, cluster in enumerate(clusters)
                if idx != source_idx
                and self._clusters_adjacent(source_cluster, cluster, assignment.neighbors)
            ]
            if not adjacent:
                continue
            target_idx = rng.choice(adjacent)
            target_cluster = clusters[target_idx]

            candidates = [
                node_id
                for node_id in source_cluster
                if self._is_adjacent_to_cluster(node_id, target_cluster, assignment.neighbors)
            ]
            if not candidates:
                continue

            best_score = float("inf")
            best_choices: List[Tuple[List[Set[int]], AssignmentResult]] = []

            for node_id in candidates:
                new_clusters: List[Set[int]] = []
                new_source = set(source_cluster)
                new_source.remove(node_id)
                if new_source and not self._is_connected(new_source, assignment.neighbors):
                    continue
                new_target = set(target_cluster)
                new_target.add(node_id)
                if not self._is_connected(new_target, assignment.neighbors):
                    continue

                for idx, cluster in enumerate(clusters):
                    if idx == source_idx:
                        if new_source:
                            new_clusters.append(new_source)
                        continue
                    if idx == target_idx:
                        new_clusters.append(new_target)
                    else:
                        new_clusters.append(set(cluster))

                if scorer is None:
                    return ClusterAssignment(assignment.dag, new_clusters, assignment.neighbors), None
                score, result = scorer(new_clusters)
                if score < best_score - 1e-9:
                    best_score = score
                    best_choices = [(new_clusters, result)]
                elif abs(score - best_score) <= 1e-9:
                    best_choices.append((new_clusters, result))

            if best_choices:
                chosen_clusters, result = rng.choice(best_choices)
                return (
                    ClusterAssignment(assignment.dag, chosen_clusters, assignment.neighbors),
                    result,
                )

        return None, None

    def _propose_merge(self, assignment: ClusterAssignment, rng: random.Random) -> Optional[ClusterAssignment]:
        if assignment.cluster_count() < 2:
            return None
        clusters = [set(cluster) for cluster in assignment.clusters]
        for _ in range(8):
            idx_a, idx_b = rng.sample(range(len(clusters)), 2)
            cluster_a = clusters[idx_a]
            cluster_b = clusters[idx_b]
            if not self._clusters_adjacent(cluster_a, cluster_b, assignment.neighbors):
                continue
            merged = cluster_a | cluster_b
            for idx in sorted([idx_a, idx_b], reverse=True):
                clusters.pop(idx)
            clusters.append(merged)
            return ClusterAssignment(assignment.dag, clusters, assignment.neighbors)
        return None

    def _propose_split(
        self,
        assignment: ClusterAssignment,
        rng: random.Random,
        scorer=None,
    ) -> Tuple[Optional[ClusterAssignment], Optional[AssignmentResult]]:
        clusters = [set(cluster) for cluster in assignment.clusters]
        candidates = [i for i, cluster in enumerate(clusters) if len(cluster) > 1]
        if not candidates:
            return None, None
        idx = rng.choice(candidates)
        cluster_nodes = clusters[idx]

        best_score = float("inf")
        best_choices: List[Tuple[List[Set[int]], AssignmentResult]] = []

        for left, right in self._iter_split_candidates(
            cluster_nodes, assignment.neighbors, rng
        ):
            new_clusters = [set(cluster) for cluster in clusters]
            new_clusters.pop(idx)
            new_clusters.append(left)
            new_clusters.append(right)

            if scorer is None:
                return ClusterAssignment(assignment.dag, new_clusters, assignment.neighbors), None
            score, result = scorer(new_clusters)
            if score < best_score - 1e-9:
                best_score = score
                best_choices = [(new_clusters, result)]
            elif abs(score - best_score) <= 1e-9:
                best_choices.append((new_clusters, result))

        if best_choices:
            chosen_clusters, result = rng.choice(best_choices)
            return ClusterAssignment(assignment.dag, chosen_clusters, assignment.neighbors), result
        return None, None

    def _propose_merge_split(
        self,
        assignment: ClusterAssignment,
        rng: random.Random,
        scorer=None,
    ) -> Tuple[Optional[ClusterAssignment], Optional[AssignmentResult]]:
        if assignment.cluster_count() < 2:
            return None, None
        clusters = [set(cluster) for cluster in assignment.clusters]
        for _ in range(8):
            idx_a, idx_b = rng.sample(range(len(clusters)), 2)
            cluster_a = clusters[idx_a]
            cluster_b = clusters[idx_b]
            if not self._clusters_adjacent(cluster_a, cluster_b, assignment.neighbors):
                continue
            merged = cluster_a | cluster_b

            best_score = float("inf")
            best_choices: List[Tuple[List[Set[int]], AssignmentResult]] = []

            for left, right in self._iter_split_candidates(
                merged, assignment.neighbors, rng
            ):
                new_clusters = [set(cluster) for cluster in clusters]
                for idx in sorted([idx_a, idx_b], reverse=True):
                    new_clusters.pop(idx)
                new_clusters.append(left)
                new_clusters.append(right)

                if scorer is None:
                    return ClusterAssignment(assignment.dag, new_clusters, assignment.neighbors), None
                score, result = scorer(new_clusters)
                if score < best_score - 1e-9:
                    best_score = score
                    best_choices = [(new_clusters, result)]
                elif abs(score - best_score) <= 1e-9:
                    best_choices.append((new_clusters, result))

            if best_choices:
                chosen_clusters, result = rng.choice(best_choices)
                return ClusterAssignment(assignment.dag, chosen_clusters, assignment.neighbors), result

        return None, None

    def _iter_split_candidates(
        self,
        node_ids: Set[int],
        neighbors: Dict[int, Set[int]],
        rng: random.Random,
    ) -> Iterable[Tuple[Set[int], Set[int]]]:
        if len(node_ids) <= self.max_exhaustive_split_nodes:
            yield from self._iter_connected_splits(node_ids, neighbors)
            return

        seen: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
        sample_target = self.random_split_samples
        attempts = 0
        max_attempts = max(sample_target * 10, 32)
        while len(seen) < sample_target and attempts < max_attempts:
            attempts += 1
            split = self._random_connected_split(node_ids, neighbors, rng)
            if split is None:
                continue
            left, right = split
            left_key = tuple(sorted(left))
            right_key = tuple(sorted(right))
            if left_key > right_key:
                left, right = right, left
                left_key, right_key = right_key, left_key
            key = (left_key, right_key)
            if key in seen:
                continue
            seen.add(key)
            yield left, right

        # If random sampling cannot find any valid split, fallback.
        if not seen:
            yield from self._iter_connected_splits(node_ids, neighbors)

    def _iter_connected_splits(
        self,
        node_ids: Set[int],
        neighbors: Dict[int, Set[int]],
    ) -> Iterable[Tuple[Set[int], Set[int]]]:
        if len(node_ids) <= 1:
            return
        nodes = sorted(node_ids)
        anchor = nodes[0]
        rest = nodes[1:]
        rest_count = len(rest)
        total_masks = 1 << rest_count
        node_set = set(nodes)

        for mask in range(total_masks):
            left = {anchor}
            for i in range(rest_count):
                if mask & (1 << i):
                    left.add(rest[i])
            if len(left) == len(node_set):
                continue
            right = node_set - left
            if not self._is_connected(left, neighbors):
                continue
            if not self._is_connected(right, neighbors):
                continue
            yield left, right

    def _random_connected_split(
        self,
        node_ids: Set[int],
        neighbors: Dict[int, Set[int]],
        rng: random.Random,
    ) -> Optional[Tuple[Set[int], Set[int]]]:
        if len(node_ids) <= 1:
            return None
        node_list = list(node_ids)
        for _ in range(8):
            target_size = rng.randint(1, len(node_list) - 1)
            seed = rng.choice(node_list)
            subset = self._grow_connected_subset(seed, target_size, node_ids, neighbors, rng)
            if subset is None:
                continue
            remainder = node_ids - subset
            if not remainder:
                continue
            if not self._is_connected(remainder, neighbors):
                continue
            return subset, remainder
        return None

    def _grow_connected_subset(
        self,
        seed: int,
        target_size: int,
        node_ids: Set[int],
        neighbors: Dict[int, Set[int]],
        rng: random.Random,
    ) -> Optional[Set[int]]:
        subset = {seed}
        frontier = [seed]
        while frontier and len(subset) < target_size:
            current = rng.choice(frontier)
            choices = [
                nid
                for nid in neighbors.get(current, set())
                if nid in node_ids and nid not in subset
            ]
            if choices:
                nxt = rng.choice(choices)
                subset.add(nxt)
                frontier.append(nxt)
            else:
                frontier.remove(current)
        if len(subset) != target_size:
            return None
        return subset

    def _build_reduced_dag(self, dag: DAG, clusters: List[Set[int]]) -> Tuple[DAG, List[float]]:
        reduced = DAG(f"{dag.id}_proposed")
        reduced.criticality = dag.criticality
        reduced.period = dag.period
        reduced.deadline = dag.deadline
        reduced.cluster_groups = {}

        cluster_gms: List[float] = []
        node_to_cluster: Dict[int, int] = {}

        for idx, node_ids in enumerate(clusters):
            reduced.cluster_groups[idx] = sorted(node_ids)
            for node_id in node_ids:
                node_to_cluster[node_id] = idx

            probs: List[float] = []
            sigmas: List[int] = []
            sum_c_lo = 0
            max_delta = 0
            merged_ids: List[int] = []

            for node_id in node_ids:
                node = dag.nodes[node_id]
                sum_c_lo += int(node.c_lo)
                max_delta = max(max_delta, int(node.c_hi - node.c_lo))
                merged_ids.extend(node.merged_ids or [node_id])

                if node.constituent_probs:
                    probs.extend(node.constituent_probs)
                else:
                    probs.append(node.get_failure_prob(self.system.failure_rate))

                node_sigmas = list(getattr(node, "constituent_sigmas", []))
                if node_sigmas:
                    sigmas.extend(node_sigmas)
                else:
                    sigmas.append(int(node.sigma_c_hi))

            g_m = 0.0
            if len(probs) > 1:
                g_m = self.calculate_g_m(probs)
            cluster_gms.append(g_m)

            new_node = Node(idx, sum_c_lo)
            new_node.c_hi = sum_c_lo + max_delta
            new_node.sigma_c_hi = sum(sigmas)
            new_node.constituent_probs = list(probs)
            new_node.constituent_sigmas = list(sigmas)
            new_node.merged_ids = sorted(merged_ids)
            new_node.original_nodes = sorted(merged_ids)
            new_node.cluster_id = idx
            reduced.add_node(new_node)

        for src_id, src_node in dag.nodes.items():
            for dst_id in src_node.successors:
                src_cluster = node_to_cluster[src_id]
                dst_cluster = node_to_cluster[dst_id]
                if src_cluster == dst_cluster:
                    continue
                reduced.add_edge(src_cluster, dst_cluster)

        return reduced, cluster_gms

    def _is_acyclic(self, dag: DAG) -> bool:
        indegree = {node_id: len(node.predecessors) for node_id, node in dag.nodes.items()}
        queue = [node_id for node_id, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for succ in dag.nodes[current].successors:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        return visited == len(dag.nodes)

    def _is_adjacent_to_cluster(
        self,
        node_id: int,
        cluster: Set[int],
        neighbors: Dict[int, Set[int]],
    ) -> bool:
        return bool(neighbors.get(node_id, set()) & cluster)

    def _clusters_adjacent(
        self,
        cluster_a: Set[int],
        cluster_b: Set[int],
        neighbors: Dict[int, Set[int]],
    ) -> bool:
        if not cluster_a or not cluster_b:
            return False
        for node_id in cluster_a:
            if neighbors.get(node_id, set()) & cluster_b:
                return True
        return False

    def _is_connected(self, node_ids: Set[int], neighbors: Dict[int, Set[int]]) -> bool:
        if len(node_ids) <= 1:
            return True
        start = next(iter(node_ids))
        stack = [start]
        visited = {start}
        while stack:
            current = stack.pop()
            for nxt in neighbors.get(current, set()):
                if nxt not in node_ids or nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return visited == node_ids

    def _build_neighbors(self, dag: DAG) -> Dict[int, Set[int]]:
        neighbors: Dict[int, Set[int]] = {}
        for node_id, node in dag.nodes.items():
            neighbors[node_id] = set(node.predecessors) | set(node.successors)
        return neighbors

    def _is_high_utilization(self, dag: DAG) -> bool:
        period = dag.period or dag.deadline
        if period <= 0:
            return True
        mode = "HI" if dag.criticality == "HI" else "LO"
        total = sum(node.get_exec_time(mode) for node in dag.nodes.values())
        return (total / period) > 1.0

    def _clone_dag(self, dag: DAG) -> DAG:
        new_dag = DAG(f"{dag.id}_copy")
        new_dag.criticality = dag.criticality
        new_dag.period = dag.period
        new_dag.deadline = dag.deadline

        for node in dag.nodes.values():
            cloned = Node(node.id, node.c_lo)
            cloned.c_hi = node.c_hi
            cloned.sigma_c_hi = node.sigma_c_hi
            cloned.constituent_probs = list(node.constituent_probs)
            cloned.constituent_sigmas = list(getattr(node, "constituent_sigmas", []))
            cloned.failure_prob = getattr(node, "failure_prob", None)
            cloned.merged_ids = list(node.merged_ids)
            cloned.original_nodes = list(node.original_nodes)
            cloned.cluster_id = getattr(node, "cluster_id", None)
            new_dag.add_node(cloned)

        for src_id, src_node in dag.nodes.items():
            for dst_id in src_node.successors:
                new_dag.add_edge(src_id, dst_id)

        return new_dag
