# pylint: skip-file
import numpy as np
import pytest

from cvd_env.reactor_env import CVDReactorEnv


@pytest.fixture
def env():
    return CVDReactorEnv(target_thickness=100.0, max_steps=50)


def test_initialization(env):
    assert env.h_target == 100.0
    assert env.max_steps == 50
    assert env.k0 == 1e3
    assert env.Ea == 50000
    assert env.alpha == 0.5


def test_reset(env):
    state, _ = env.reset()
    assert np.allclose(state, [0.0, 750.0, 50.0])
    assert len(env.thickness_history) == 1
    assert len(env.T_history) == 1
    assert len(env.F_history) == 1


def test_step_bounds(env):
    env.reset()
    action = 4
    state, _, _, _, _ = env.step(action)
    assert env.T_min <= state[1] <= env.T_max
    assert env.F_min <= state[2] <= env.F_max


def test_deposition_rate(env):
    r = env._get_deposition_rate(750.0, 50.0)
    expected = env.k0 * (50.0**env.alpha) * np.exp(-env.Ea / (env.R * 750.0))
    assert np.isclose(r, expected)


def test_episode_runs(env):
    env.reset()
    done = False
    steps = 0
    while not done and steps < env.max_steps:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 0
