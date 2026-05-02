from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
import time
from typing import Dict, List, Optional, Sequence, Tuple

from src.analyzer.multipath import MultipathAnalyzer
from src.common.dag_models import DAG

Criticality = str


@dataclass(frozen=True)
class TaskParams:
    task_id: str
    criticality: Criticality
    c_n: float
    c_o: float
    l_n: float
    l_o: float
    period: int
    deadline: int

    @property
    def u_n(self) -> float:
        if self.deadline <= 0:
            return float("inf")
        return self.c_n / self.deadline

    @property
    def u_o(self) -> float:
        if self.deadline <= 0:
            return float("inf")
        return self.c_o / self.deadline

    @property
    def u_lo(self) -> float:
        return self.u_n

    @property
    def u_hi(self) -> float:
        return self.u_o if self.criticality == "HI" else 0.0

    @property
    def u_hi_lo(self) -> float:
        return self.u_n if self.criticality == "HI" else 0.0


@dataclass
class MCFQResult:
    schedulable: bool
    cores: int
    typical_used: Optional[int]
    critical_used: Optional[int]
    low_util_cores: int
    lh_cores: int
    hh_cores: Optional[int]
    details: Dict[str, int]



class MCFQAnalyzer:
    """
    MCFQ schedulability test (ECRTS 2018) for MC DAG tasks.
    Low-util tasks are assigned by MC-Partition-UT-0.75 (RTS 2013).
    """

    @dataclass
    class _ProcessorState:
        hi_hi: float = 0.0
        hi_lo: float = 0.0
        lo_lo: float = 0.0
        hi_only: bool = False

    @staticmethod
    @lru_cache(maxsize=1)
    def _gurobi_available() -> bool:
        try:
            import gurobipy  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _select_lh_tasks(
        requirements: Sequence[int],
        idle_cores: int,
        use_ilp: bool,
    ) -> Tuple[int, int, bool]:
        if idle_cores <= 0 or not requirements:
            return 0, 0, False
        if use_ilp:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                model = gp.Model("lh_keep", env=env)
                model.Params.LogToConsole = 0
                x = model.addVars(len(requirements), vtype=GRB.BINARY, name="x")
                model.setObjective(gp.quicksum(x[i] for i in range(len(requirements))), GRB.MAXIMIZE)
                model.addConstr(
                    gp.quicksum(requirements[i] * x[i] for i in range(len(requirements))) <= idle_cores
                )
                model.optimize()
                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    kept = [i for i in range(len(requirements)) if x[i].X > 0.5]
                    kept_cores = sum(requirements[i] for i in kept)
                    return len(kept), kept_cores, True
            except Exception:
                pass

        kept_cores = 0
        kept_count = 0
        for cores in sorted(requirements):
            if kept_cores + cores > idle_cores:
                continue
            kept_cores += cores
            kept_count += 1
        return kept_count, kept_cores, False

    @staticmethod
    def from_dags(dags: Sequence[DAG]) -> List[TaskParams]:
        tasks: List[TaskParams] = []
        for dag in dags:
            c_n = MultipathAnalyzer.get_workload(dag, "LO")
            l_n = MultipathAnalyzer.get_critical_path(dag, "LO")
            if dag.criticality == "HI":
                c_o = MultipathAnalyzer.get_workload(dag, "HI")
                l_o = MultipathAnalyzer.get_critical_path(dag, "HI")
            else:
                c_o = c_n
                l_o = l_n

            deadline = dag.deadline or dag.period
            period = dag.period or dag.deadline
            tasks.append(
                TaskParams(
                    task_id=str(dag.id),
                    criticality=dag.criticality,
                    c_n=float(c_n),
                    c_o=float(c_o),
                    l_n=float(l_n),
                    l_o=float(l_o),
                    period=int(period or 0),
                    deadline=int(deadline or 0),
                )
            )
        return tasks

    @staticmethod
    def _classify(tasks: Sequence[TaskParams]) -> Tuple[List[TaskParams], ...]:
        hh: List[TaskParams] = []
        hl: List[TaskParams] = []
        lh: List[TaskParams] = []
        ll: List[TaskParams] = []
        for task in tasks:
            if task.u_o > 1:
                if task.criticality == "HI":
                    hh.append(task)
                else:
                    lh.append(task)
            else:
                if task.criticality == "HI":
                    hl.append(task)
                else:
                    ll.append(task)
        return hh, hl, lh, ll

    def _schh(self, task: TaskParams, mu_n: int, mu_o: int) -> bool:
        if task.deadline <= 0 or mu_n <= 0 or mu_o <= 0 or mu_o < mu_n:
            return False
        omega = (task.c_o - task.c_n) - (task.l_o - task.l_n)
        omega_stats = getattr(self, "_omega_stats", None)
        if omega_stats is not None:
            omega_stats["total"] += 1
        if omega < 0:
            if omega_stats is not None:
                omega_stats["negative"] += 1
            omega = 0.0

        term1 = (task.c_n - task.l_n) / mu_n
        term2 = omega / mu_o
        term3 = task.l_o
        min_term = min(task.l_n, omega / mu_n) if mu_n > 0 else 0.0
        term4 = min_term * (1.0 - mu_n / mu_o)
        return task.deadline >= term1 + term2 + term3 + term4

    def _hh_pairs(self, task: TaskParams, m: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for mu_n in range(1, m + 1):
            for mu_o in range(mu_n, m + 1):
                if not self._schh(task, mu_n, mu_o):
                    continue
                if mu_o == mu_n or not self._schh(task, mu_n, mu_o - 1):
                    pairs.append((mu_n, mu_o))
        return pairs

    @staticmethod
    def _lh_required_processors(task: TaskParams) -> Optional[int]:
        if task.deadline <= 0 or task.deadline <= task.l_n:
            return None
        denom = task.deadline - task.l_n
        numer = task.c_n - task.l_n
        if denom <= 0:
            return None
        return max(1, ceil(numer / denom))

    @staticmethod
    def _lo_bound(hi_hi: float, hi_lo: float) -> float:
        denom = 1.0 - (hi_hi - hi_lo)
        if denom <= 0.0:
            return -1.0
        return (1.0 - hi_hi) / denom

    @classmethod
    def _can_partition_ut_075(cls, tasks: Sequence[TaskParams], m: int) -> bool:
        if not tasks:
            return True

        hi_tasks = [task for task in tasks if task.criticality == "HI"]
        lo_tasks = [task for task in tasks if task.criticality == "LO"]

        heavy_hi = [task for task in hi_tasks if task.u_hi > 0.75]
        if len(heavy_hi) > m:
            return False

        processors = [cls._ProcessorState() for _ in range(m)]
        for idx, task in enumerate(heavy_hi):
            proc = processors[idx]
            proc.hi_only = True
            proc.hi_hi = task.u_hi
            proc.hi_lo = task.u_hi_lo

        regular_hi = [task for task in hi_tasks if task.u_hi <= 0.75]
        for task in regular_hi:
            placed = False
            for proc in processors:
                limit = 1.0 if proc.hi_only else 0.75
                if proc.hi_hi + task.u_hi <= limit:
                    proc.hi_hi += task.u_hi
                    proc.hi_lo += task.u_hi_lo
                    placed = True
                    break
            if not placed:
                return False

        for task in lo_tasks:
            placed = False
            for proc in processors:
                if proc.hi_only:
                    continue
                bound = cls._lo_bound(proc.hi_hi, proc.hi_lo)
                if bound < 0.0:
                    continue
                if proc.lo_lo + task.u_lo <= bound + 1e-12:
                    proc.lo_lo += task.u_lo
                    placed = True
                    break
            if not placed:
                return False

        return True

    @classmethod
    def _min_partition_cores_ut_075(cls, tasks: Sequence[TaskParams], m_max: int) -> Optional[int]:
        if not tasks:
            return 0
        low = 1
        high = max(1, m_max)
        result: Optional[int] = None
        while low <= high:
            mid = (low + high) // 2
            if cls._can_partition_ut_075(tasks, mid):
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        return result

    @staticmethod
    def _combine_hh_pairs(pairs_list: Sequence[List[Tuple[int, int]]], m: int) -> Dict[int, int]:
        dp: Dict[int, int] = {0: 0}
        for pairs in pairs_list:
            next_dp: Dict[int, int] = {}
            for typ_sum, crit_sum in dp.items():
                for mu_n, mu_o in pairs:
                    new_typ = typ_sum + mu_n
                    new_crit = crit_sum + mu_o
                    if new_typ > m or new_crit > m:
                        continue
                    if new_typ not in next_dp or new_crit < next_dp[new_typ]:
                        next_dp[new_typ] = new_crit
            dp = next_dp
            if not dp:
                break
        return dp

    def min_cores(
        self,
        dags: Sequence[DAG],
        m_max: int,
        use_ilp: bool | None = None,
        debug: bool = False,
        debug_every: int = 10,
    ) -> Optional[int]:
        """Return the smallest core count that is schedulable for the task set."""
        if not dags:
            return 0
        upper = max(1, m_max)
        low = 1
        high = upper
        best: Optional[int] = None
        step = 0
        while low <= high:
            mid = (low + high) // 2
            step += 1
            if debug and (step == 1 or step % max(1, debug_every) == 0 or low == high):
                print(f"[mcfq] bs step={step} m={mid} range=[{low},{high}]", flush=True)
            start = time.perf_counter() if debug else 0.0
            result = self.analyze(dags, mid, use_ilp=use_ilp, debug=debug)
            if debug and (step == 1 or step % max(1, debug_every) == 0 or low == high):
                elapsed = time.perf_counter() - start
                print(
                    f"[mcfq] m={mid} done in {elapsed:.3f}s sched={result.schedulable}",
                    flush=True,
                )
            if result.schedulable:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        return best

    def analyze(
        self,
        dags: Sequence[DAG],
        m: int,
        use_ilp: bool | None = None,
        debug: bool = False,
    ) -> MCFQResult:
        if use_ilp is None:
            use_ilp = self._gurobi_available()
        omega_stats = {"total": 0, "negative": 0}
        self._omega_stats = omega_stats
        try:
            tasks = self.from_dags(dags)
            hh, hl, lh, ll = self._classify(tasks)
            if debug:
                print(
                    f"[mcfq] analyze m={m} tasks={len(tasks)} "
                    f"hh={len(hh)} hl={len(hl)} lh={len(lh)} ll={len(ll)}",
                    flush=True,
                )
            implicit_mismatch = sum(
                1
                for task in tasks
                if task.deadline > 0 and task.period > 0 and task.deadline != task.period
            )
            if any(task.deadline <= 0 for task in tasks):
                return MCFQResult(
                    schedulable=False,
                    cores=m,
                    typical_used=None,
                    critical_used=None,
                    low_util_cores=0,
                    lh_cores=0,
                    hh_cores=None,
                    details={
                        "invalid_deadline": 1,
                        "implicit_mismatch": implicit_mismatch,
                        "omega_negative": omega_stats["negative"],
                        "omega_total": omega_stats["total"],
                    },
                )

            low_util_tasks = [*hl, *ll]
            low_util_cores = self._min_partition_cores_ut_075(low_util_tasks, m)
            if debug:
                print(
                    f"[mcfq] m={m} low_util_cores={low_util_cores} "
                    f"lh_count={len(lh)}",
                    flush=True,
                )
            if low_util_cores is None or low_util_cores > m:
                return MCFQResult(
                    schedulable=False,
                    cores=m,
                    typical_used=None,
                    critical_used=None,
                    low_util_cores=low_util_cores or 0,
                    lh_cores=0,
                    hh_cores=None,
                    details={
                        "low_util_unschedulable": 1,
                        "implicit_mismatch": implicit_mismatch,
                        "omega_negative": omega_stats["negative"],
                        "omega_total": omega_stats["total"],
                    },
                )

            lh_cores = 0
            lh_requirements: List[int] = []
            for task in lh:
                required = self._lh_required_processors(task)
                if required is None:
                    return MCFQResult(
                        schedulable=False,
                        cores=m,
                        typical_used=None,
                        critical_used=None,
                        low_util_cores=low_util_cores,
                        lh_cores=0,
                        hh_cores=None,
                        details={
                            "lh_unschedulable": 1,
                            "implicit_mismatch": implicit_mismatch,
                            "omega_negative": omega_stats["negative"],
                            "omega_total": omega_stats["total"],
                        },
                    )
                lh_cores += required
                lh_requirements.append(required)

            pairs_list: List[List[Tuple[int, int]]] = []
            hh_log_every = max(1, len(hh) // 5) if debug and hh else 1
            for idx, task in enumerate(hh):
                start = time.perf_counter() if debug else 0.0
                pairs = self._hh_pairs(task, m)
                if debug and (idx % hh_log_every == 0 or idx == len(hh) - 1):
                    elapsed = time.perf_counter() - start
                    print(
                        f"[mcfq] m={m} hh[{idx + 1}/{len(hh)}] "
                        f"pairs={len(pairs)} time={elapsed:.3f}s",
                        flush=True,
                    )
                if not pairs:
                    return MCFQResult(
                        schedulable=False,
                        cores=m,
                        typical_used=None,
                        critical_used=None,
                        low_util_cores=low_util_cores,
                        lh_cores=lh_cores,
                        hh_cores=None,
                        details={
                            "hh_pairs_empty": 1,
                            "implicit_mismatch": implicit_mismatch,
                            "omega_negative": omega_stats["negative"],
                            "omega_total": omega_stats["total"],
                        },
                    )
                pairs_list.append(pairs)

            dp_start = time.perf_counter() if debug else 0.0
            dp = self._combine_hh_pairs(pairs_list, m)
            if debug:
                elapsed = time.perf_counter() - dp_start
                print(f"[mcfq] m={m} dp_size={len(dp)} time={elapsed:.3f}s", flush=True)
            if not dp:
                return MCFQResult(
                    schedulable=False,
                    cores=m,
                    typical_used=None,
                    critical_used=None,
                    low_util_cores=low_util_cores,
                    lh_cores=lh_cores,
                    hh_cores=None,
                    details={
                        "hh_dp_empty": 1,
                        "implicit_mismatch": implicit_mismatch,
                        "omega_negative": omega_stats["negative"],
                        "omega_total": omega_stats["total"],
                    },
                )

            best_typical: Optional[int] = None
            best_critical: Optional[int] = None
            best_hh_typical: Optional[int] = None
            best_hh_critical: Optional[int] = None
            for typ_sum, crit_sum in dp.items():
                if typ_sum + lh_cores + low_util_cores > m:
                    continue
                if crit_sum + low_util_cores > m:
                    continue
                if best_hh_critical is None or crit_sum < best_hh_critical:
                    best_hh_typical = typ_sum
                    best_hh_critical = crit_sum
                elif crit_sum == best_hh_critical and best_hh_typical is not None:
                    if typ_sum < best_hh_typical:
                        best_hh_typical = typ_sum
                        best_hh_critical = crit_sum

            if best_hh_typical is not None and best_hh_critical is not None:
                best_typical = best_hh_typical + lh_cores + low_util_cores
                best_critical = best_hh_critical + low_util_cores

            schedulable = best_typical is not None and best_critical is not None

            kept_lh_cores = 0
            kept_lh_count = 0
            lh_ilp_used = False
            idle_cores = 0
            if schedulable and best_hh_critical is not None:
                idle_cores = max(0, m - (best_hh_critical + low_util_cores))
                kept_lh_count, kept_lh_cores, lh_ilp_used = self._select_lh_tasks(
                    lh_requirements,
                    idle_cores,
                    use_ilp,
                )
                best_critical = best_hh_critical + low_util_cores + kept_lh_cores

            return MCFQResult(
                schedulable=schedulable,
                cores=m,
                typical_used=best_typical,
                critical_used=best_critical,
                low_util_cores=low_util_cores,
                lh_cores=lh_cores,
                hh_cores=best_hh_typical,
                details={
                    "hh": len(hh),
                    "hl": len(hl),
                    "lh": len(lh),
                    "ll": len(ll),
                    "hh_typical": best_hh_typical or 0,
                    "hh_critical": best_hh_critical or 0,
                    "lh_kept": kept_lh_count,
                    "lh_kept_cores": kept_lh_cores,
                    "lh_ilp_used": 1 if lh_ilp_used else 0,
                    "idle_cores": idle_cores,
                    "implicit_mismatch": implicit_mismatch,
                    "omega_negative": omega_stats["negative"],
                    "omega_total": omega_stats["total"],
                },
            )
        finally:
            self._omega_stats = None

