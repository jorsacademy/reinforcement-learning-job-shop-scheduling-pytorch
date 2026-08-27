import unittest
import numpy as np
import torch

from jssp_rl import (
    JSSPInstance,
    JobShopEnv,
    PPOConfig,
    audit_schedule,
    exact_serial_dispatch_oracle,
    generate_jssp_instance,
    schedule_from_dispatch_sequence,
    train_ppo,
)
from jssp_rl.baselines import earliest_start, mwkr, random_feasible, spt
from jssp_rl.policy import JobTransformerActorCritic


class JSSPCoreTests(unittest.TestCase):
    def tiny(self):
        return JSSPInstance(
            machines=np.array([[0, 1], [1, 0]], dtype=np.int64),
            durations=np.array([[3, 2], [2, 4]], dtype=np.int64),
        )

    def test_instance_generation_reproducible(self):
        a = generate_jssp_instance(seed=11, n_jobs=4, n_machines=4)
        b = generate_jssp_instance(seed=11, n_jobs=4, n_machines=4)
        np.testing.assert_array_equal(a.machines, b.machines)
        np.testing.assert_array_equal(a.durations, b.durations)
        for row in a.machines:
            np.testing.assert_array_equal(np.sort(row), np.arange(4))

    def test_reward_telescopes_exactly_to_negative_makespan(self):
        instance = self.tiny()
        makespan, operations, total_reward = schedule_from_dispatch_sequence(
            instance, [0, 1, 0, 1]
        )
        self.assertEqual(total_reward, -makespan)
        self.assertEqual(audit_schedule(instance, operations), 0.0)

    def test_invalid_completed_job_action_rejected(self):
        env = JobShopEnv(self.tiny())
        env.reset()
        env.step(0)
        env.step(0)
        with self.assertRaises(ValueError):
            env.step(0)

    def test_tiny_exhaustive_oracle_dominates_dispatch_rules(self):
        instance = self.tiny()
        exact = exact_serial_dispatch_oracle(instance)
        self.assertEqual(exact.status, "OPTIMAL_EXHAUSTIVE")
        for method in (spt, mwkr, earliest_start):
            makespan, operations = method(instance)
            self.assertGreaterEqual(makespan, exact.makespan)
            self.assertEqual(audit_schedule(instance, operations), 0.0)

    def test_all_baselines_produce_feasible_schedules(self):
        instance = generate_jssp_instance(seed=12, n_jobs=4, n_machines=4)
        outputs = [
            spt(instance),
            mwkr(instance),
            earliest_start(instance),
            random_feasible(instance, seed=4),
        ]
        for _, operations in outputs:
            self.assertEqual(audit_schedule(instance, operations), 0.0)

    def test_policy_mask_blocks_completed_jobs_and_gradients_flow(self):
        instance = generate_jssp_instance(seed=13, n_jobs=4, n_machines=4)
        env = JobShopEnv(instance)
        obs, _ = env.reset()
        env.step(0)
        env.step(0)
        env.step(0)
        obs, _, _, _, _ = env.step(0)
        self.assertEqual(obs["action_mask"][0], 0)

        model = JobTransformerActorCritic(
            job_feature_dim=env.job_feature_dim,
            global_feature_dim=env.global_feature_dim,
            hidden_dim=32,
            heads=4,
            layers=1,
        )
        j = torch.tensor(obs["job_features"][None], dtype=torch.float32)
        g = torch.tensor(obs["global_features"][None], dtype=torch.float32)
        mask = torch.tensor(obs["action_mask"][None], dtype=torch.bool)
        logits, value = model(j, g, mask)
        self.assertTrue(torch.isneginf(logits[0, 0]))
        loss = -torch.log_softmax(logits, dim=-1)[0, 1] + 0.1 * value.square().mean()
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ))

    def test_short_real_ppo_training_updates_parameters(self):
        config = PPOConfig(
            seed=14,
            n_jobs=3,
            n_machines=3,
            rollout_instances=8,
            epochs=2,
            ppo_updates=2,
            minibatch_size=36,
            hidden_dim=32,
            heads=4,
            layers=1,
        )
        torch.manual_seed(config.seed)
        probe_env = JobShopEnv(generate_jssp_instance(
            seed=config.seed,
            n_jobs=3,
            n_machines=3,
        ))
        initial = JobTransformerActorCritic(
            job_feature_dim=probe_env.job_feature_dim,
            global_feature_dim=probe_env.global_feature_dim,
            hidden_dim=32,
            heads=4,
            layers=1,
        )
        before = [p.detach().clone() for p in initial.parameters()]

        # train_ppo constructs the same seeded initial architecture.
        trained = train_ppo(config, device="cpu").model
        after = list(trained.parameters())
        self.assertTrue(any(
            not torch.allclose(a, b)
            for a, b in zip(before, after)
        ))

    def test_gamma_other_than_one_is_rejected(self):
        config = PPOConfig(
            n_jobs=3,
            n_machines=3,
            rollout_instances=4,
            epochs=1,
            gamma=0.99,
            hidden_dim=32,
            heads=4,
            layers=1,
        )
        with self.assertRaises(ValueError):
            train_ppo(config)


if __name__ == "__main__":
    unittest.main()
