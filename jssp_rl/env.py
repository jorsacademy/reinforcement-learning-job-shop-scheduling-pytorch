from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GymBase = gym.Env
except ImportError:  # local source can still be tested without Gymnasium installed
    gym = None
    spaces = None
    class _GymBase:  # minimal fallback; CI tests the real Gymnasium API
        pass

from .instance import JSSPInstance


@dataclass(frozen=True)
class ScheduledOperation:
    job: int
    operation: int
    machine: int
    start: int
    end: int


class JobShopEnv(_GymBase):
    """Serial schedule-generation environment for classical JSSP.

    Action = choose a job whose next operation should be appended to its
    required machine. Every unfinished job is a legal action.

    Dense reward is potential-based:
        r_t = makespan_before - makespan_after
    Hence, with gamma=1, episode return telescopes exactly to -final_makespan.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, instance: JSSPInstance):
        super().__init__()
        self.instance = instance
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.job_feature_dim = 7 + 3 * self.n_machines
        self.global_feature_dim = 4

        if spaces is not None:
            self.action_space = spaces.Discrete(self.n_jobs)
            self.observation_space = spaces.Dict({
                "job_features": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(self.n_jobs, self.job_feature_dim),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.n_jobs),
            })

        self._reset_state()

    def _reset_state(self):
        self.next_operation = np.zeros(self.n_jobs, dtype=np.int64)
        self.job_ready = np.zeros(self.n_jobs, dtype=np.int64)
        self.machine_ready = np.zeros(self.n_machines, dtype=np.int64)
        self.current_makespan = 0
        self.scheduled = []

    def reset(self, *, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)
        self._reset_state()
        return self._observation(), {"makespan": 0}

    def action_mask(self) -> np.ndarray:
        return (self.next_operation < self.n_machines).astype(np.int8)

    def _observation(self):
        H = max(float(self.instance.horizon), 1.0)
        max_p = max(float(self.instance.durations.max()), 1.0)
        features = np.zeros((self.n_jobs, self.job_feature_dim), dtype=np.float32)

        for j in range(self.n_jobs):
            k = int(self.next_operation[j])
            finished = k >= self.n_machines
            if finished:
                next_machine = 0
                next_p = 0.0
                earliest = float(self.job_ready[j])
                remaining_work = 0.0
                ops_remaining = 0.0
            else:
                next_machine = int(self.instance.machines[j, k])
                next_p = float(self.instance.durations[j, k])
                earliest = float(max(self.job_ready[j], self.machine_ready[next_machine]))
                remaining_work = float(self.instance.durations[j, k:].sum())
                ops_remaining = float(self.n_machines - k)

            base = [
                next_p / max_p,
                float(self.job_ready[j]) / H,
                float(self.machine_ready[next_machine]) / H if not finished else 0.0,
                earliest / H,
                remaining_work / H,
                ops_remaining / self.n_machines,
                1.0 if finished else 0.0,
            ]
            features[j, :7] = np.asarray(base, dtype=np.float32)
            if not finished:
                # Immediate routing requirement.
                features[j, 7 + next_machine] = 1.0

                # Full remaining route summary: processing time and relative
                # future position for every machine. This gives the policy
                # critical-path/routing information without exposing a label.
                dur_offset = 7 + self.n_machines
                pos_offset = 7 + 2 * self.n_machines
                features[j, pos_offset:pos_offset + self.n_machines] = -1.0
                for rel, kk in enumerate(range(k, self.n_machines)):
                    mm = int(self.instance.machines[j, kk])
                    pp = float(self.instance.durations[j, kk])
                    features[j, dur_offset + mm] = pp / max_p
                    features[j, pos_offset + mm] = rel / max(self.n_machines - 1, 1)

        completed_ops = int(self.next_operation.sum())
        global_features = np.asarray([
            float(self.current_makespan) / H,
            completed_ops / self.instance.n_operations,
            float(self.machine_ready.mean()) / H,
            float(self.machine_ready.max()) / H,
        ], dtype=np.float32)

        return {
            "job_features": features,
            "global_features": global_features,
            "action_mask": self.action_mask(),
        }

    def step(self, action):
        j = int(action)
        if not 0 <= j < self.n_jobs:
            raise ValueError("action outside job range")
        k = int(self.next_operation[j])
        if k >= self.n_machines:
            raise ValueError(f"job {j} is already complete")

        machine = int(self.instance.machines[j, k])
        duration = int(self.instance.durations[j, k])
        start = int(max(self.job_ready[j], self.machine_ready[machine]))
        end = start + duration

        previous_makespan = int(self.current_makespan)
        self.job_ready[j] = end
        self.machine_ready[machine] = end
        self.next_operation[j] += 1
        self.current_makespan = max(self.current_makespan, end)
        self.scheduled.append(
            ScheduledOperation(j, k, machine, start, end)
        )

        reward = float(previous_makespan - self.current_makespan)
        terminated = bool(np.all(self.next_operation >= self.n_machines))
        info = {
            "makespan": int(self.current_makespan),
            "scheduled_operation": self.scheduled[-1],
        }
        return self._observation(), reward, terminated, False, info

    def render(self):
        lines = []
        for m in range(self.n_machines):
            ops = sorted(
                (op for op in self.scheduled if op.machine == m),
                key=lambda x: x.start,
            )
            body = " ".join(
                f"J{op.job}O{op.operation}[{op.start},{op.end})"
                for op in ops
            )
            lines.append(f"M{m}: {body}")
        return "\n".join(lines)


def schedule_from_dispatch_sequence(instance: JSSPInstance, sequence):
    env = JobShopEnv(instance)
    env.reset()
    total_reward = 0.0
    for action in sequence:
        _, reward, terminated, _, _ = env.step(int(action))
        total_reward += reward
    if not terminated:
        raise ValueError("dispatch sequence did not complete all operations")
    return int(env.current_makespan), tuple(env.scheduled), float(total_reward)


def audit_schedule(instance: JSSPInstance, operations) -> float:
    """Return maximum precedence/machine-overlap violation; zero means feasible."""
    by_job = {j: [] for j in range(instance.n_jobs)}
    by_machine = {m: [] for m in range(instance.n_machines)}
    violation = 0.0
    if len(operations) != instance.n_operations:
        violation = max(violation, 1.0)

    seen = set()
    for op in operations:
        if not (0 <= op.job < instance.n_jobs and 0 <= op.operation < instance.n_machines):
            violation = max(violation, 1.0)
            continue
        expected_machine = int(instance.machines[op.job, op.operation])
        expected_duration = int(instance.durations[op.job, op.operation])
        violation = max(
            violation,
            float(op.machine != expected_machine),
            float(op.start < 0),
            float(abs((op.end - op.start) - expected_duration)),
        )
        key = (op.job, op.operation)
        if key in seen:
            violation = max(violation, 1.0)
        seen.add(key)
        by_job[op.job].append(op)
        by_machine[op.machine].append(op)
    for j, ops in by_job.items():
        ops = sorted(ops, key=lambda x: x.operation)
        if len(ops) != instance.n_machines:
            violation = max(violation, 1.0)
            continue
        for a, b in zip(ops, ops[1:]):
            violation = max(violation, float(max(a.end - b.start, 0)))

    for m, ops in by_machine.items():
        ops = sorted(ops, key=lambda x: x.start)
        for a, b in zip(ops, ops[1:]):
            violation = max(violation, float(max(a.end - b.start, 0)))

    return float(violation)
