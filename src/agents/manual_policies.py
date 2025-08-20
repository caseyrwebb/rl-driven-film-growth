"""
Manual policies for the CVD reactor environment. These policies can be used for testing
or as baselines.
"""

import numpy as np
from numpy.typing import NDArray

from cvd_env.reactor_env import CVDReactorEnv
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def always_max_policy(env: CVDReactorEnv, state: NDArray[np.float32]) -> int:
    """
    Policy: Always take the action that increases both T and F as much as possible.
    """
    _, T, F = state
    best_idx = None
    best_score = -np.inf
    for i, (dT, dF) in enumerate(env.action_map):
        T_new = np.clip(T + dT, env.T_min, env.T_max)
        F_new = np.clip(F + dF, env.F_min, env.F_max)
        score = (T_new - T) + (F_new - F)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx is None:
        raise RuntimeError("No valid action found in always_max_policy")
    return best_idx


def random_policy(env: CVDReactorEnv) -> int:
    """
    Policy: Take a random action at each step.
    """
    return int(env.action_space.sample())


def proportional_policy(
    env: CVDReactorEnv,
    state: NDArray[np.float32],
    large_thresh: float = 20.0,
    moderate_thresh: float = 5.0,
    small_thresh: float = 1.0,
) -> int:
    """
    Policy: Adjust T and F proportionally to the remaining error in thickness.
    - If far from target, increase T and F.
    - If close, hold or decrease.

    We mainly change T because it has a much larger impact on deposition rate than F,
    due to the exponential sensitivity of rate to T in the Arrhenius equation.
    (F increases rate only linearly or sub-linearly, but T can speed up growth by orders
    of magnitude.)
    """
    h, _, _ = state
    error = env.h_target - h
    if error > large_thresh:
        action_goal = (10, 5)
    elif error > moderate_thresh:
        action_goal = (10, 0)
    elif error > small_thresh:
        action_goal = (0, 0)
    else:
        action_goal = (-10, -5)
    best_idx = np.argmin(
        [
            (dT - action_goal[0]) ** 2 + (dF - action_goal[1]) ** 2
            for (dT, dF) in env.action_map
        ]
    )
    return int(best_idx)


def _nearest_action_index(env: CVDReactorEnv, goal_dT: float, goal_dF: float) -> int:
    """
    Pick the discrete action (ΔT, ΔF) closest to a desired 'goal' move.
    """
    return int(
        np.argmin(
            [(dT - goal_dT) ** 2 + (dF - goal_dF) ** 2 for (dT, dF) in env.action_map]
        )
    )


def step_ramp_policy(
    env: CVDReactorEnv,
    state: NDArray[np.float32],
    t_phase: float = 0.80,
    f_phase: float = 0.95,
    coast_band: float = 1.0,
) -> int:
    """
    Step-ramp baseline:
      1) Ramp temperature early to accelerate growth (goal ΔT=+10, ΔF=0)
      2) Hold T and nudge flow as we approach target (goal ΔT=0, ΔF=+5)
      3) Coast near the target (goal ΔT=0, ΔF=0)
      4) If we overshoot or are extremely close, gently back off (goal ΔT=-10, ΔF=-5)

    This mirrors a common "heat first, then trim with flow, then hold" heuristic.
    """
    h, _, _ = state
    target = env.h_target
    error = float(target - h)

    if h < t_phase * target:
        goal_dT, goal_dF = 10.0, 0.0  # ramp T
    elif h < f_phase * target:
        goal_dT, goal_dF = 0.0, 5.0  # feather with F
    elif abs(error) <= coast_band:
        goal_dT, goal_dF = 0.0, 0.0  # coast near target
    elif error < 0:  # overshoot
        goal_dT, goal_dF = -10.0, -5.0  # back off
    else:
        # Slightly below target but past F-phase: keep holding; let dynamics finish
        goal_dT, goal_dF = 0.0, 0.0

    return _nearest_action_index(env, goal_dT, goal_dF)


if __name__ == "__main__":
    setup_logging()
    logger.info("=== Manual Policies Demo ===")
    _env = CVDReactorEnv(
        target_thickness=100.0,
        max_steps=20,
        k0=1e14,
        Ea=250000,
        alpha=1.0,
        mode="real",
        dt=0.01,
    )

    policies = {
        "Always Max": always_max_policy,
        "Proportional": proportional_policy,
        "Step Ramp": step_ramp_policy,
        "Random": lambda env, _: random_policy(env),
    }

    summary = []

    for name, policy_fn in policies.items():
        logger.info("\n--- Demo: %s Policy ---", name)
        _state, _ = _env.reset()
        for step in range(_env.max_steps):
            action = policy_fn(_env, _state)
            delta_T, delta_F = _env.action_map[action]
            logger.info(
                (
                    "Step %d: Action %d (ΔT=%.1f, ΔF=%.1f) | "
                    "State: [thick=%.1f, T=%.1f, F=%.1f]"
                ),
                step + 1,
                action,
                delta_T,
                delta_F,
                _state[0],
                _state[1],
                _state[2],
            )
            _state, reward, terminated, truncated, info = _env.step(action)
            if terminated or truncated:
                summary.append(
                    {
                        "Policy": name,
                        "Steps": step + 1,
                        "Final Thickness": _state[0],
                        "Error": info["thickness_error"],
                        "Terminated": terminated,
                    }
                )
                break
        else:
            summary.append(
                {
                    "Policy": name,
                    "Steps": _env.max_steps,
                    "Final Thickness": _state[0],
                    "Error": info["thickness_error"],
                    "Terminated": False,
                }
            )

    logger.info("\n=== Summary Table ===")
    header = (
        f"{'Policy':>15} | {'Steps':>5} | {'Final Thick':>12} | "
        f"{'Error':>8} | {'Terminated':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary:
        logger.info(
            "%15s | %5d | %12.2f | %8.2f | %10s",
            row["Policy"],
            row["Steps"],
            row["Final Thickness"],
            row["Error"],
            str(row["Terminated"]),
        )
