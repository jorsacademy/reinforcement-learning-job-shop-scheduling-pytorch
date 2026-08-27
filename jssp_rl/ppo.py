from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .env import JobShopEnv
from .instance import generate_instance_batch
from .policy import JobTransformerActorCritic


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42
    n_jobs: int = 4
    n_machines: int = 4
    rollout_instances: int = 64
    epochs: int = 60
    ppo_updates: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    reward_scale: float = 0.02
    validation_instances: int = 32
    hidden_dim: int = 96
    heads: int = 4
    layers: int = 2


@dataclass(frozen=True)
class TrainResult:
    model: JobTransformerActorCritic
    history: tuple


def _stack_observations(observations, device):
    job = torch.tensor(
        np.stack([x["job_features"] for x in observations]),
        dtype=torch.float32,
        device=device,
    )
    global_features = torch.tensor(
        np.stack([x["global_features"] for x in observations]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.tensor(
        np.stack([x["action_mask"] for x in observations]),
        dtype=torch.bool,
        device=device,
    )
    return job, global_features, mask


def collect_rollout(model, instances, *, device, gae_lambda=0.95, reward_scale=1.0):
    envs = [JobShopEnv(x) for x in instances]
    observations = [env.reset()[0] for env in envs]
    B = len(envs)
    T = instances[0].n_operations

    jobs, globals_, masks = [], [], []
    actions, old_logprobs, values, rewards, dones = [], [], [], [], []

    for _ in range(T):
        jfeat, gfeat, mask = _stack_observations(observations, device)
        with torch.no_grad():
            action, logprob, value = model.act(
                jfeat, gfeat, mask, greedy=False
            )

        next_obs, step_rewards, step_dones = [], [], []
        for i, env in enumerate(envs):
            obs, reward, done, _, _ = env.step(int(action[i].item()))
            next_obs.append(obs)
            step_rewards.append(reward)
            step_dones.append(done)

        jobs.append(jfeat.cpu())
        globals_.append(gfeat.cpu())
        masks.append(mask.cpu())
        actions.append(action.cpu())
        old_logprobs.append(logprob.cpu())
        values.append(value.cpu())
        rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
        dones.append(torch.tensor(step_dones, dtype=torch.float32))
        observations = next_obs

    raw_reward_t = torch.stack(rewards)   # [T,B], exact environment reward
    value_t = torch.stack(values)          # [T,B]
    done_t = torch.stack(dones)           # [T,B]

    episode_returns = raw_reward_t.sum(dim=0).numpy()
    makespans = np.asarray([env.current_makespan for env in envs], dtype=float)
    if not np.allclose(episode_returns, -makespans, atol=1e-6):
        raise AssertionError("dense reward no longer telescopes to -makespan")

    # Positive constant scaling leaves the scheduling objective unchanged but
    # makes critic targets numerically better conditioned.
    reward_t = raw_reward_t * float(reward_scale)
    advantages = torch.zeros_like(reward_t)
    gae = torch.zeros(B, dtype=torch.float32)
    next_value = torch.zeros(B, dtype=torch.float32)
    for t in reversed(range(T)):
        not_done = 1.0 - done_t[t]
        delta = reward_t[t] + next_value * not_done - value_t[t]
        gae = delta + float(gae_lambda) * not_done * gae
        advantages[t] = gae
        next_value = value_t[t]
    returns = advantages + value_t

    batch = {
        "job_features": torch.stack(jobs).reshape(T*B, *jobs[0].shape[1:]),
        "global_features": torch.stack(globals_).reshape(T*B, -1),
        "action_mask": torch.stack(masks).reshape(T*B, -1),
        "actions": torch.stack(actions).reshape(T*B),
        "old_logprobs": torch.stack(old_logprobs).reshape(T*B),
        "returns": returns.reshape(T*B),
        "advantages": advantages.reshape(T*B),
    }
    return batch, makespans


def ppo_update(model, optimizer, batch, config: PPOConfig, *, device, rng):
    n = len(batch["actions"])
    advantages = batch["advantages"].clone()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = []
    for _ in range(config.ppo_updates):
        order = rng.permutation(n)
        for start in range(0, n, config.minibatch_size):
            idx_np = order[start:start + config.minibatch_size]
            idx = torch.tensor(idx_np, dtype=torch.long)

            j = batch["job_features"][idx].to(device)
            g = batch["global_features"][idx].to(device)
            mask = batch["action_mask"][idx].to(device)
            action = batch["actions"][idx].to(device)
            old_logprob = batch["old_logprobs"][idx].to(device)
            ret = batch["returns"][idx].to(device)
            adv = advantages[idx].to(device)

            dist, value = model.distribution_and_value(j, g, mask)
            logprob = dist.log_prob(action)
            ratio = torch.exp(logprob - old_logprob)
            unclipped = ratio * adv
            clipped = torch.clamp(
                ratio, 1.0-config.clip_ratio, 1.0+config.clip_ratio
            ) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(value, ret)
            entropy = dist.entropy().mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append((
                float(policy_loss.item()),
                float(value_loss.item()),
                float(entropy.item()),
            ))
    return np.mean(metrics, axis=0)


@torch.no_grad()
def _greedy_mean_makespan(model, instances, *, device):
    makespans = []
    for instance in instances:
        env = JobShopEnv(instance)
        obs, _ = env.reset()
        done = False
        while not done:
            j, g, mask = _stack_observations([obs], device)
            action, _, _ = model.act(j, g, mask, greedy=True)
            obs, _, done, _, _ = env.step(int(action.item()))
        makespans.append(env.current_makespan)
    return float(np.mean(makespans))


def train_ppo(config: PPOConfig, *, device="cpu") -> TrainResult:
    if abs(config.gamma - 1.0) > 1e-12:
        raise ValueError("gamma must remain 1.0 so return exactly matches makespan objective")
    if not (0.0 <= config.gae_lambda <= 1.0):
        raise ValueError("gae_lambda must be in [0,1]")
    if config.reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    if config.rollout_instances < 1 or config.validation_instances < 1 or config.epochs < 1:
        raise ValueError("rollout/validation instances and epochs must be positive")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    probe = JobShopEnv(generate_instance_batch(
        1,
        seed=config.seed,
        n_jobs=config.n_jobs,
        n_machines=config.n_machines,
    )[0])
    model = JobTransformerActorCritic(
        job_feature_dim=probe.job_feature_dim,
        global_feature_dim=probe.global_feature_dim,
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        layers=config.layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    validation = generate_instance_batch(
        config.validation_instances,
        seed=config.seed + 77_777_777,
        n_jobs=config.n_jobs,
        n_machines=config.n_machines,
    )
    best_validation = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history = []
    for epoch in range(1, config.epochs + 1):
        instances = generate_instance_batch(
            config.rollout_instances,
            seed=config.seed + 1_000_003 * epoch,
            n_jobs=config.n_jobs,
            n_machines=config.n_machines,
        )
        batch, makespans = collect_rollout(
            model,
            instances,
            device=device,
            gae_lambda=config.gae_lambda,
            reward_scale=config.reward_scale,
        )
        policy_loss, value_loss, entropy = ppo_update(
            model, optimizer, batch, config, device=device, rng=rng
        )
        validation_makespan = _greedy_mean_makespan(
            model, validation, device=device
        )
        if validation_makespan < best_validation - 1e-12:
            best_validation = validation_makespan
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        row = {
            "epoch": epoch,
            "sampled_mean_makespan": float(makespans.mean()),
            "validation_greedy_makespan": float(validation_makespan),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        }
        history.append(row)
        if epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs//6) == 0:
            print(
                f"epoch={epoch:03d} sampled_makespan={row['sampled_mean_makespan']:.3f} "
                f"validation_greedy={validation_makespan:.3f} "
                f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} entropy={entropy:.4f}"
            )

    model.load_state_dict(best_state)
    return TrainResult(model=model, history=tuple(history))
