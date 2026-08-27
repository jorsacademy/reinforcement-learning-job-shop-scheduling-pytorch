from __future__ import annotations

import argparse
import numpy as np
import torch

from jssp_rl import (
    JSSPInstance,
    JobShopEnv,
    PPOConfig,
    audit_schedule,
    exact_serial_dispatch_oracle,
    generate_instance_batch,
    train_ppo,
)
from jssp_rl.baselines import earliest_start, mwkr, spt
from jssp_rl.evaluate import evaluate_against_cpsat, rollout_policy


def self_test():
    instance = JSSPInstance(
        machines=np.array([[0, 1], [1, 0]], dtype=np.int64),
        durations=np.array([[3, 2], [2, 4]], dtype=np.int64),
    )
    env = JobShopEnv(instance)
    obs, _ = env.reset()
    assert int(obs["action_mask"].sum()) == 2

    total_reward = 0.0
    for action in (0, 1, 0, 1):
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    assert done
    assert abs(total_reward + env.current_makespan) <= 1e-9
    assert audit_schedule(instance, env.scheduled) == 0.0

    exact = exact_serial_dispatch_oracle(instance)
    assert exact.status == "OPTIMAL_EXHAUSTIVE"
    assert exact.makespan <= env.current_makespan
    print(
        "JSSP environment self-test: OK "
        f"(dispatch makespan={env.current_makespan}, tiny exact={exact.makespan})"
    )


def run_training(args):
    device = args.device
    config = PPOConfig(
        seed=args.seed,
        n_jobs=args.jobs,
        n_machines=args.machines,
        rollout_instances=args.rollout_instances,
        epochs=args.epochs,
        ppo_updates=args.ppo_updates,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gae_lambda=args.gae_lambda,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
    )
    result = train_ppo(config, device=device)
    best_validation = min(
        row["validation_greedy_makespan"] for row in result.history
    )
    print(f"best validation greedy makespan : {best_validation:.3f}")

    test_instances = generate_instance_batch(
        args.test_instances,
        seed=args.seed + 9_000_000,
        n_jobs=args.jobs,
        n_machines=args.machines,
    )

    print("=" * 92)
    print("HELD-OUT DISPATCHING BENCHMARK")
    print("=" * 92)
    methods = {"PPO greedy": [], "SPT": [], "MWKR": [], "Earliest-start": []}
    for instance in test_instances:
        methods["PPO greedy"].append(rollout_policy(
            result.model, instance, device=device, greedy=True
        )[0])
        methods["SPT"].append(spt(instance)[0])
        methods["MWKR"].append(mwkr(instance)[0])
        methods["Earliest-start"].append(earliest_start(instance)[0])

    for name, values in methods.items():
        a = np.asarray(values, dtype=float)
        print(f"{name:<18} mean makespan={a.mean():8.3f} median={np.median(a):8.3f}")

    if args.cpsat:
        evaluation = evaluate_against_cpsat(
            result.model,
            test_instances[:args.cpsat_instances],
            device=device,
            cp_sat_time_limit=args.cpsat_time_limit,
        )
        print()
        print(
            f"CP-SAT exact statuses: "
            f"{evaluation.cp_sat_optimal_count}/{len(evaluation.exact_optima)} OPTIMAL"
        )
        for m in evaluation.methods:
            gap = "n/a" if m.mean_gap_pct is None else f"{m.mean_gap_pct:.3f}%"
            print(
                f"{m.name:<18} mean={m.mean_makespan:8.3f} "
                f"exact-gap={gap:<10} feasible={m.feasible_rate:.3f}"
            )

    if args.checkpoint:
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "config": config.__dict__,
            },
            args.checkpoint,
        )
        print(f"checkpoint: {args.checkpoint}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--machines", type=int, default=4)
    p.add_argument("--rollout-instances", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--ppo-updates", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=192)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--test-instances", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--cpsat", action="store_true")
    p.add_argument("--cpsat-instances", type=int, default=8)
    p.add_argument("--cpsat-time-limit", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_training(args)
