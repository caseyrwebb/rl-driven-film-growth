"""Chemical Vapor Deposition (CVD) reactor environment for RL optimization.

This module defines a Gymnasium environment simulating a simplified CVD reactor
using an Arrhenius-based model. The environment allows an RL agent to control
temperature and flow rate to achieve a target film thickness.

Kinetics model:

    rate = k0 * exp(-Ea / (R * T)) * [SiH4]

Parameters (k0, Ea) taken from:
    Newman, C.G.; O'Neal, H.E.; Ring, M.A.; Leska, F.; Shipley, N.
    Kinetics and mechanism of the silane decomposition,
    Int. J. Chem. Kinet. 11, 1167 (1979).
    NIST Chemical Kinetics Database:
    https://kinetics.nist.gov/kinetics/Detail?id=1979NEW/ONE1167:4

Note: This is a first-order gas-phase model. If model or parameterization changes,
update both this comment and the source citation.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from utils.logger import get_logger, setup_logging
from utils.plots import plot_trajectories

logger = get_logger(__name__)


class CVDReactorEnv(gym.Env[NDArray[np.float32], np.int64]):
    """A Gymnasium environment for a simplified CVD reactor.

    The environment models film deposition using an Arrhenius equation, with
    temperature (T) and flow rate (F) as control variables. The goal is to reach
    a target film thickness by adjusting T and F via discrete actions.
    """

    def __init__(
        self,
        target_thickness: float = 100.0,
        max_steps: int = 50,
        k0: float = 1e3,
        Ea: float = 50000,
        R: float = 8.314,
        alpha: float = 0.5,
        dt: float = 1.0,
        P: float = 1e5,
        V: float = 1e-3,
        mode: str = "real",
    ):
        """Initialize the CVD reactor environment.

        Args:
            target_thickness (float): Target film thickness in nm (default: 100.0).
            max_steps (int): Maximum steps per episode (default: 50).
            k0 (float): Pre-exponential factor (default: 1e3).
            Ea (float): Activation energy in J/mol (default: 50000).
            R (float): Gas constant in J/(mol·K) (default: 8.314).
            alpha (float): Flow rate exponent (default: 0.5).
            dt (float): Time step in seconds (default: 1.0).
            P (float): Pressure in Pa (default: 1 bar).
            V (float): Volume in m^3 (default: 1 liter).
            mode (str): "toy" for simplified model, "real" for more complex
        """
        super().__init__()

        self.mode = mode
        if self.mode not in ["toy", "real"]:
            raise ValueError("Mode must be 'toy' or 'real'")

        self.k0 = k0
        self.Ea = Ea
        self.R = R
        self.alpha = alpha
        self.dt = dt
        self.P = P
        self.V = V

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
        """
        Calculate deposition rate based on temperature and flow rate.
        Args:
            T (float): Temperature in Kelvin.
            F (float): Flow rate in sccm (standard cm^3/min).
        Returns:
            float: Deposition rate in nm/s.
        """
        if self.mode == "toy":
            return self.k0 * (F**self.alpha) * np.exp(-self.Ea / (self.R * T))
        if self.mode == "real":
            P = self.P  # Pa
            V = self.V  # m^3
            k = self.k0 * np.exp(-self.Ea / (self.R * T))  # s^-1
            F_mol = (F * P) / (self.R * T * 60)  # mol/s
            conc = F_mol / V  # mol/m^3
            return k * conc
        raise ValueError(f"Unknown mode {self.mode!r} in _get_deposition_rate")

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
        logger.debug(
            "Step %d: h=%.2f nm, Reward=%.2f, Terminated=%s, Type=%s",
            self.step_count,
            h,
            reward,
            terminated,
            type(terminated),
        )

        return self.state, reward, terminated, truncated, info

    def render(self) -> None:
        """Log the current state for debugging."""
        assert self.state is not None, "Call reset() before render()"
        h, T, F = self.state
        logger.info(
            "Step %d: Thickness=%.2f nm, T=%.2f K, F=%.2f sccm",
            self.step_count,
            h,
            T,
            F,
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
    setup_logging()
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
    logger.info(
        "Final thickness: %.2f nm, Error: %.2f nm",
        test_obs[0],
        test_info["thickness_error"],
    )
    env.plot()
