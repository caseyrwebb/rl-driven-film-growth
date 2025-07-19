"""Train and evaluate a DQN agent for the CVD reactor environment.

This module provides functions to train a DQN agent using Stable-Baselines3
on the CVDReactorEnv and evaluate its performance by running episodes and
plotting trajectories.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from cvd_env.reactor_env import CVDReactorEnv
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def train_dqn(
    env: CVDReactorEnv, total_timesteps: int = 5000, learning_rate: float = 1e-3
) -> DQN:
    """Train a DQN agent on the CVD environment.

    Args:
        env (CVDReactorEnv): The CVD reactor environment.
        total_timesteps (int): Number of training timesteps (default: 5000).
        learning_rate (float): Learning rate for the DQN optimizer (default: 1e-3).

    Returns:
        DQN: Trained DQN model.
    """
    check_env(env)
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_dqn(model: DQN, env: CVDReactorEnv, n_episodes: int = 1) -> None:
    """Evaluate the DQN agent and plot trajectories.

    Args:
        model (DQN): Trained DQN model.
        env (CVDReactorEnv): The CVD reactor environment.
        n_episodes (int): Number of episodes to evaluate (default: 1).
    """
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, info = env.step(int(action))
            env.render()
            done = terminated or truncated
        logger.info(
            "Episode %d: Final thickness: %.2f nm, Error: %.2f nm",
            episode + 1,
            obs[0],
            info["thickness_error"],
        )
        env.plot()


if __name__ == "__main__":
    test_env = CVDReactorEnv(target_thickness=100.0, max_steps=50)
    test_model = train_dqn(test_env, total_timesteps=5000)
    evaluate_dqn(test_model, test_env, n_episodes=1)
