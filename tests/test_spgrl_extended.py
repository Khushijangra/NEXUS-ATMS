import pytest
import numpy as np
import torch
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.core.state_types import SPGRLState

@pytest.fixture(scope="module")
def env():
    return SPGRLEnv()

# Generate 15 queue permutation tests
@pytest.mark.parametrize("q_n, q_s, q_e, q_w", [
    (0, 0, 0, 0),
    (100, 100, 100, 100),
    (50, 0, 0, 0),
    (0, 50, 0, 0),
    (0, 0, 50, 0),
    (0, 0, 0, 50),
    (20, 20, 5, 5),
    (5, 5, 20, 20),
    (0, 0, 200, 200),
    (200, 200, 0, 0),
    (1, 2, 3, 4),
    (4, 3, 2, 1),
    (10, 10, 10, 10),
    (30, 30, 30, 30),
    (80, 80, 80, 80),
])
def test_queue_dynamics(env, q_n, q_s, q_e, q_w):
    env.reset()
    env.queue = np.array([q_n, q_s, q_e, q_w], dtype=np.float32)
    zt, r, d, i = env.step(0, compute_zt=False)
    assert env.queue.shape == (4,)
    assert (env.queue >= 0).all()

# Generate 15 reward permutation tests
@pytest.mark.parametrize("action, emergency_active", [
    (0, False), (1, False), (2, False), (3, False),
    (0, True), (1, True), (2, True), (3, True),
    (0, False), (1, False), (2, False), (3, False),
    (0, True), (1, True), (2, True)
])
def test_reward_bounds(env, action, emergency_active):
    env.reset()
    env.emergency = emergency_active
    zt, r, d, i = env.step(action, compute_zt=False)
    assert isinstance(r, float)
    assert r <= 2.0  # Max possible reward is 2.0 (Emergency cleared)
    assert r >= -10.0 # Bounded negative penalty

def test_carbon_accumulation(env):
    env.reset()
    initial_carbon = env.carbon
    env.step(0, compute_zt=False)
    assert env.carbon > initial_carbon

def test_terminal_condition(env):
    env.reset()
    env.current_step = 199
    zt, r, d, i = env.step(0, compute_zt=False)
    assert d is True
