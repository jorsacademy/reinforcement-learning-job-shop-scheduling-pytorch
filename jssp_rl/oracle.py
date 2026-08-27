from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .env import schedule_from_dispatch_sequence
from .instance import JSSPInstance


@dataclass(frozen=True)
class ExactJSSPResult:
    makespan: int
    status: str
    starts: np.ndarray | None
    best_bound: float | None = None


def exact_serial_dispatch_oracle(instance: JSSPInstance) -> ExactJSSPResult:
    """Exhaustive exact oracle for tiny fixtures.

    Enumerates every precedence-feasible job dispatch sequence. This is only
    tractable for small total operation counts and is used as an independent
    local regression oracle.
    """
    if instance.n_operations > 10:
        raise ValueError("tiny exhaustive oracle is limited to <=10 operations")

    counts = [instance.n_machines] * instance.n_jobs
    best = [10**18]

    def recurse(prefix, remaining):
        if len(prefix) == instance.n_operations:
            makespan, _, _ = schedule_from_dispatch_sequence(instance, prefix)
            if makespan < best[0]:
                best[0] = makespan
            return
        for j in range(instance.n_jobs):
            if remaining[j] > 0:
                remaining[j] -= 1
                prefix.append(j)
                recurse(prefix, remaining)
                prefix.pop()
                remaining[j] += 1

    recurse([], counts[:])
    return ExactJSSPResult(int(best[0]), "OPTIMAL_EXHAUSTIVE", None, float(best[0]))


def solve_cpsat_exact(
    instance: JSSPInstance,
    *,
    time_limit: float = 10.0,
    workers: int = 1,
) -> ExactJSSPResult:
    """Independent OR-Tools CP-SAT formulation with interval/no-overlap constraints."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("OR-Tools is required for CP-SAT exact evaluation") from exc

    model = cp_model.CpModel()
    horizon = instance.horizon
    starts, ends = {}, {}
    machine_intervals = {m: [] for m in range(instance.n_machines)}

    for j in range(instance.n_jobs):
        for k in range(instance.n_machines):
            p = int(instance.durations[j, k])
            m = int(instance.machines[j, k])
            s = model.new_int_var(0, horizon, f"s_{j}_{k}")
            e = model.new_int_var(0, horizon, f"e_{j}_{k}")
            interval = model.new_interval_var(s, p, e, f"I_{j}_{k}")
            starts[j, k] = s
            ends[j, k] = e
            machine_intervals[m].append(interval)

    for m in range(instance.n_machines):
        model.add_no_overlap(machine_intervals[m])

    for j in range(instance.n_jobs):
        for k in range(instance.n_machines - 1):
            model.add(starts[j, k+1] >= ends[j, k])

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        makespan,
        [ends[j, instance.n_machines-1] for j in range(instance.n_jobs)],
    )
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 0
    status_code = solver.solve(model)

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactJSSPResult(
            makespan=horizon,
            status="NO_SOLUTION",
            starts=None,
            best_bound=float(solver.best_objective_bound),
        )

    start_values = np.zeros(
        (instance.n_jobs, instance.n_machines), dtype=np.int64
    )
    for j in range(instance.n_jobs):
        for k in range(instance.n_machines):
            start_values[j, k] = int(solver.value(starts[j, k]))

    status = "OPTIMAL" if status_code == cp_model.OPTIMAL else "FEASIBLE"
    return ExactJSSPResult(
        makespan=int(solver.value(makespan)),
        status=status,
        starts=start_values,
        best_bound=float(solver.best_objective_bound),
    )
