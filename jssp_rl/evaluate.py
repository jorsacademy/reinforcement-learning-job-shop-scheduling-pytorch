from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .baselines import spt, mwkr, earliest_start, random_feasible
from .env import JobShopEnv, audit_schedule
from .oracle import solve_cpsat_exact


@dataclass(frozen=True)
class MethodSummary:
    name: str
    mean_makespan: float
    mean_gap_pct: float | None
    feasible_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    methods: tuple
    exact_optima: tuple
    cp_sat_optimal_count: int


@torch.no_grad()
def rollout_policy(model, instance, *, device="cpu", greedy=True):
    env = JobShopEnv(instance)
    obs, _ = env.reset()
    done = False
    while not done:
        j = torch.tensor(obs["job_features"][None], dtype=torch.float32, device=device)
        g = torch.tensor(obs["global_features"][None], dtype=torch.float32, device=device)
        mask = torch.tensor(obs["action_mask"][None], dtype=torch.bool, device=device)
        action, _, _ = model.act(j, g, mask, greedy=greedy)
        obs, _, done, _, _ = env.step(int(action.item()))
    return env.current_makespan, tuple(env.scheduled)


def evaluate_against_cpsat(
    model,
    instances,
    *,
    device="cpu",
    cp_sat_time_limit=10.0,
):
    exact = [
        solve_cpsat_exact(x, time_limit=cp_sat_time_limit, workers=1)
        for x in instances
    ]
    optimal_mask = np.asarray([x.status == "OPTIMAL" for x in exact], dtype=bool)
    optima = np.asarray([x.makespan for x in exact], dtype=float)

    records = {
        "PPO greedy": [],
        "SPT": [],
        "MWKR": [],
        "Earliest-start": [],
        "Random": [],
    }
    feasible = {k: [] for k in records}

    for idx, instance in enumerate(instances):
        outputs = {
            "PPO greedy": rollout_policy(model, instance, device=device),
            "SPT": spt(instance),
            "MWKR": mwkr(instance),
            "Earliest-start": earliest_start(instance),
            "Random": random_feasible(instance, seed=90_000 + idx),
        }
        for name, (makespan, operations) in outputs.items():
            records[name].append(float(makespan))
            feasible[name].append(audit_schedule(instance, operations) <= 1e-9)

    summaries = []
    for name, values in records.items():
        values = np.asarray(values, dtype=float)
        if np.any(optimal_mask):
            gaps = 100.0 * (
                values[optimal_mask] - optima[optimal_mask]
            ) / np.maximum(optima[optimal_mask], 1e-9)
            mean_gap = float(gaps.mean())
        else:
            mean_gap = None
        summaries.append(MethodSummary(
            name=name,
            mean_makespan=float(values.mean()),
            mean_gap_pct=mean_gap,
            feasible_rate=float(np.mean(feasible[name])),
        ))

    return EvaluationResult(
        methods=tuple(summaries),
        exact_optima=tuple(exact),
        cp_sat_optimal_count=int(optimal_mask.sum()),
    )
