"""
Module for evaluating and comparing policies in a CVD reactor environment.
"""

from typing import Callable, List

import numpy as np

from cvd_env.reactor_env import CVDReactorEnv


def evaluate_policy(
    policy_fn: Callable[[CVDReactorEnv, np.ndarray], int],
    env: CVDReactorEnv,
    n_episodes: int = 10,
) -> List[float]:
    """Evaluate a policy over multiple episodes."""
    errors = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(env, obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        errors.append(info["thickness_error"])
    return errors
