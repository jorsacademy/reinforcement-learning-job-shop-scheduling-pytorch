from __future__ import annotations
import numpy as np
from .env import JobShopEnv
from .instance import JSSPInstance


def _rollout(instance: JSSPInstance, chooser):
    env = JobShopEnv(instance)
    obs, _ = env.reset()
    done = False
    while not done:
        valid = np.flatnonzero(obs["action_mask"])
        action = int(chooser(env, valid))
        obs, _, done, _, _ = env.step(action)
    return env.current_makespan, tuple(env.scheduled)


def spt(instance: JSSPInstance):
    """Shortest next processing time."""
    def choose(env, valid):
        keys = []
        for j in valid:
            k = env.next_operation[j]
            p = env.instance.durations[j, k]
            keys.append((int(p), int(j)))
        return min(keys)[1]
    return _rollout(instance, choose)


def mwkr(instance: JSSPInstance):
    """Most work remaining."""
    def choose(env, valid):
        keys = []
        for j in valid:
            k = env.next_operation[j]
            rem = env.instance.durations[j, k:].sum()
            keys.append((-int(rem), int(j)))
        return min(keys)[1]
    return _rollout(instance, choose)


def earliest_start(instance: JSSPInstance):
    """Choose next operation with smallest current earliest start time."""
    def choose(env, valid):
        keys = []
        for j in valid:
            k = env.next_operation[j]
            m = env.instance.machines[j, k]
            est = max(env.job_ready[j], env.machine_ready[m])
            keys.append((int(est), int(j)))
        return min(keys)[1]
    return _rollout(instance, choose)


def random_feasible(instance: JSSPInstance, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return _rollout(instance, lambda env, valid: rng.choice(valid))
