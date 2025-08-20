"""
Module for evaluating and comparing policies in a CVD reactor environment.
Metrics: error, effort, overshoot, reversals, steps.
"""

from typing import Callable, Dict, List, TypedDict

import numpy as np

from cvd_env.reactor_env import CVDReactorEnv


class EpisodeMetrics(TypedDict):
    """
    Metrics for a single episode of interaction with the environment.
    """

    error: float
    effort: float
    overshoot: float
    reversals: float
    steps: int


def _count_reversals(xs: List[float]) -> int:
    """
    Count direction reversals in a sequence of deltas.
    Zeros are ignored (no direction), so they don't create fake reversals.
    """
    if not xs:
        return 0
    s = np.sign(np.asarray(xs, dtype=float))
    s = s[s != 0]  # drop zeros
    if s.size < 2:
        return 0
    # sign flip: +1 -> -1 or -1 -> +1 => diff has magnitude 2
    return int(np.sum(np.abs(np.diff(s)) == 2))


def evaluate_policy(
    policy_fn: Callable[[CVDReactorEnv, np.ndarray], int],
    env: CVDReactorEnv,
    n_episodes: int = 10,
) -> List[EpisodeMetrics]:
    """
    Run n_episodes and return a list of per-episode metrics dicts.

    Metrics:
      - error: final |h - h_target|
      - effort: Σ(|ΔT| + |ΔF|)
      - overshoot: max(0, max(h) - h_target)
      - reversals: sign-change count in ΔT and ΔF after reaching 0.5*h_target
      - steps: episode length
    """
    metrics: List[EpisodeMetrics] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        effort = 0.0
        deltas_T: List[float] = []
        deltas_F: List[float] = []
        h_traj: List[float] = [float(obs[0])]

        while not done:
            action = int(policy_fn(env, obs))
            dT, dF = env.action_map[action]
            deltas_T.append(float(dT))
            deltas_F.append(float(dF))
            effort += abs(dT) + abs(dF)

            obs, _, terminated, truncated, info = env.step(action)
            h_traj.append(float(obs[0]))
            done = bool(terminated or truncated)

        error = float(info["thickness_error"])
        overshoot = float(max(0.0, max(h_traj) - env.h_target))

        # Index where thickness first reaches half target (state index).
        # Actions/deltas are between states, so align by subtracting 1.
        halfway_idx_state = next(
            (i for i, h in enumerate(h_traj) if h >= 0.5 * env.h_target),
            len(h_traj) - 1,
        )
        start_idx = max(0, halfway_idx_state - 1)
        reversals_T = _count_reversals(deltas_T[start_idx:])
        reversals_F = _count_reversals(deltas_F[start_idx:])
        reversals = float(reversals_T + reversals_F)

        metrics.append(
            EpisodeMetrics(
                error=error,
                effort=float(effort),
                overshoot=overshoot,
                reversals=reversals,
                steps=len(h_traj) - 1,
            )
        )

    return metrics


def summarize_metrics(all_metrics: List[EpisodeMetrics]) -> Dict[str, Dict[str, float]]:
    """
    Convenience: compute mean/std for each metric over a list of episodes.
    Returns: {metric: {"mean": ..., "std": ...}, ...}
    """
    keys = ["error", "effort", "overshoot", "reversals", "steps"]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out
