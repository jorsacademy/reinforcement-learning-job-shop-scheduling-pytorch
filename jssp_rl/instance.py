from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JSSPInstance:
    """Rectangular classical job-shop instance.

    machines[j, k] is the machine used by job j's kth operation.
    durations[j, k] is the corresponding integer processing time.
    """

    machines: np.ndarray
    durations: np.ndarray

    def __post_init__(self):
        machines = np.asarray(self.machines, dtype=np.int64)
        durations = np.asarray(self.durations, dtype=np.int64)
        if machines.ndim != 2 or durations.shape != machines.shape:
            raise ValueError("machines and durations must have the same [job,operation] shape")
        if machines.shape[0] < 2 or machines.shape[1] < 2:
            raise ValueError("benchmark expects at least 2 jobs and 2 operations per job")
        if np.any(durations <= 0):
            raise ValueError("processing times must be positive")
        n_machines = machines.shape[1]
        if np.any(machines < 0) or np.any(machines >= n_machines):
            raise ValueError("machine ids must lie in [0,n_machines)")
        for row in machines:
            if len(np.unique(row)) != n_machines:
                raise ValueError("each job route must visit every machine exactly once")

    @property
    def n_jobs(self) -> int:
        return int(self.machines.shape[0])

    @property
    def n_machines(self) -> int:
        return int(self.machines.shape[1])

    @property
    def n_operations(self) -> int:
        return self.n_jobs * self.n_machines

    @property
    def horizon(self) -> int:
        return int(self.durations.sum())


def generate_jssp_instance(
    *,
    seed: int,
    n_jobs: int = 4,
    n_machines: int = 4,
    min_duration: int = 1,
    max_duration: int = 9,
) -> JSSPInstance:
    if n_jobs < 2 or n_machines < 2:
        raise ValueError("n_jobs and n_machines must be >= 2")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("invalid duration bounds")

    rng = np.random.default_rng(seed)
    machines = np.stack(
        [rng.permutation(n_machines) for _ in range(n_jobs)],
        axis=0,
    )
    durations = rng.integers(
        min_duration,
        max_duration + 1,
        size=(n_jobs, n_machines),
        dtype=np.int64,
    )
    return JSSPInstance(machines=machines, durations=durations)


def generate_instance_batch(
    n_instances: int,
    *,
    seed: int,
    n_jobs: int,
    n_machines: int,
    min_duration: int = 1,
    max_duration: int = 9,
):
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    return tuple(
        generate_jssp_instance(
            seed=seed + 104729 * i,
            n_jobs=n_jobs,
            n_machines=n_machines,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        for i in range(n_instances)
    )
