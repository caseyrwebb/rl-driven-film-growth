"""Chemical Vapor Deposition (CVD) reactor environment for RL optimization.

This module defines a Gymnasium environment simulating a simplified CVD reactor
using an Arrhenius-based model. The environment allows an RL agent to control
temperature and flow rate to achieve a target film thickness.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from utils.plots import plot_trajectories


class CVDReactorEnv(gym.Env[NDArray[np.float32], np.ndarray]):
    """A Gymnasium environment for a simplified CVD reactor.

    The environment models film deposition using an Arrhenius equation, with
    temperature (T) and flow rate (F) as control variables. The goal is to reach
    a target film thickness by adjusting T and F via discrete actions.
    """

    def __init__(self, target_thickness: float = 100.0, max_steps: int = 50):
        """Initialize the CVD reactor environment.

        Args:
            target_thickness (float): Target film thickness in nm (default: 100.0).
            max_steps (int): Maximum steps per episode (default: 50).
        """
        super().__init__()

        self.k0 = 1e3  # Pre-exponential factor
        self.Ea = 50000  # Activation energy (J/mol)
        self.R = 8.314  # Gas constant (J/mol·K)
        self.alpha = 0.5  # Flow rate exponent
        self.dt = 1.0  # Time step (s)

        # Process constraints
        self.T_min, self.T_max = 500.0, 1000.0  # K
        self.F_min, self.F_max = 10.0, 100.0  # sccm
        self.h_target = float(target_thickness)  # nm
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=np.array([0.0, self.T_min, self.F_min], dtype=np.float32),
            high=np.array([np.inf, self.T_max, self.F_max], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(9)
        self.action_map = [
            (-10.0, -5.0),
            (-10.0, 0.0),
            (-10.0, 5.0),
            (0.0, -5.0),
            (0.0, 0.0),
            (0.0, 5.0),
            (10.0, -5.0),
            (10.0, 0.0),
            (10.0, 5.0),
        ]

        self.state: Optional[NDArray[np.float32]] = None
        self.step_count: int = 0
        self.thickness_history: list[float] = []
        self.T_history: list[float] = []
        self.F_history: list[float] = []

    def _get_deposition_rate(self, T: float, F: float) -> float:
        """Calculate deposition rate based on temperature and flow rate.

        Args:
            T (float): Temperature in Kelvin.
            F (float): Flow rate in sccm.

        Returns:
            float: Deposition rate in nm/s.
        """
        rate = self.k0 * (F**self.alpha) * np.exp(-self.Ea / (self.R * T))
        return float(rate)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional reset options.

        Returns:
            Tuple[NDArray[np.float32], Dict[str, Any]]: Initial state and
            info dictionary.
        """
        super().reset(seed=seed, options=options)
        self.state = np.array([0.0, 750.0, 50.0], dtype=np.float32)
        self.step_count = 0
        self.thickness_history = [0.0]
        self.T_history = [750.0]
        self.F_history = [50.0]
        return self.state, {}

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment.

        Args:
            action (int): Discrete action index (0-8).

        Returns:
            Tuple containing:
                - NDArray[np.float32]: New state [thickness, T, F].
                - float: Reward (negative thickness error).
                - bool: Whether the episode terminated (reached target thickness).
                - bool: Whether the episode truncated (reached max steps).
                - Dict[str, Any]: Info dictionary with thickness error.
        """
        assert self.state is not None, "Call reset() before step()"
        h, T, F = self.state
        delta_T, delta_F = self.action_map[action]

        T = float(np.clip(T + delta_T, self.T_min, self.T_max))
        F = float(np.clip(F + delta_F, self.F_min, self.F_max))

        r = self._get_deposition_rate(T, F)
        h += r * self.dt

        self.state = np.array([h, T, F], dtype=np.float32)
        self.thickness_history.append(float(h))
        self.T_history.append(float(T))
        self.F_history.append(float(F))
        self.step_count += 1

        error = float(np.abs(h - self.h_target))
        reward = float(-error)
        terminated = bool(h >= self.h_target)
        truncated = bool(self.step_count >= self.max_steps)

        info = {"thickness_error": float(error)}
        print(
            f"Step {self.step_count}: h={h:.2f} nm, Reward={reward:.2f}, "
            f"Terminated={terminated}, Type={type(terminated)}"
        )

        return self.state, reward, terminated, truncated, info

    def render(self) -> None:
        """Print the current state for debugging."""
        assert self.state is not None, "Call reset() before render()"
        h, T, F = self.state
        print(
            f"Step {self.step_count}: Thickness={h:.2f} nm, "
            f"T={T:.2f} K, F={F:.2f} sccm"
        )

    def plot(self) -> None:
        """Plot thickness, temperature, and flow rate trajectories."""
        plot_trajectories(
            self.thickness_history,
            self.T_history,
            self.F_history,
            self.h_target,
            title="CVD Reactor Trajectories",
        )

    def close(self) -> None:
        """Clean up resources (placeholder)."""
        pass


if __name__ == "__main__":
    env = CVDReactorEnv(target_thickness=100.0, max_steps=50)
    test_obs, _ = env.reset()
    test_done = False
    while not test_done:
        test_action = int(env.action_space.sample())
        test_obs, test_reward, test_terminated, test_truncated, test_info = env.step(
            test_action
        )
        env.render()
        test_done = test_terminated or test_truncated
    print(
        f"Final thickness: {test_obs[0]:.2f} nm, "
        f"Error: {test_info['thickness_error']:.2f} nm"
    )
    env.plot()
