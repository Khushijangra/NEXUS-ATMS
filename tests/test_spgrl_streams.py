import pytest
import torch
import numpy as np
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.core.stream_interfaces import (
    get_semantic_state, get_prediction_state, get_emergency_state,
    get_behavioral_state, get_graph_state, get_carbon_state
)
from v2.rl.spgrl_environment import SPGRLEnv
from v2.safety.safety_wrapper import SafetyWrapper
from v2.core.unified_state import UnifiedStateBuilder
from v2.core.state_types import SPGRLState
from v2.rl.ppo_agent import PPOAgent, RolloutBuffer

def test_semantic():
    As = get_semantic_state()
    assert As is not None
    assert As.shape == (1, 1)
    assert not torch.isnan(As).any()

def test_behavior():
    Ab = get_behavioral_state()
    assert Ab is not None
    assert Ab.shape == (1, 1)
    assert not torch.isnan(Ab).any()

def test_prediction():
    features = torch.zeros(1, 30, 5)
    Ft, Cf = get_prediction_state(features)
    assert Ft is not None
    assert Ft.shape == (1, 50)
    assert not torch.isnan(Ft).any()

def test_confidence():
    features = torch.zeros(1, 30, 5)
    Ft, Cf = get_prediction_state(features)
    assert Cf is not None
    assert Cf.shape == (1, 50)
    assert not torch.isnan(Cf).any()
    assert (Cf >= 0).all() # Variance must be non-negative

def test_graph():
    Gt = get_graph_state()
    assert Gt is not None
    assert Gt.shape == (1, 64)
    assert not torch.isnan(Gt).any()

def test_carbon():
    Ct = get_carbon_state()
    assert Ct is not None
    assert Ct.shape == (1, 1)
    assert not torch.isnan(Ct).any()

def test_emergency():
    Et = get_emergency_state(is_active=True)
    assert Et is not None
    assert Et.shape == (1, 1)
    assert Et[0, 0].item() == 1.0

def test_safety():
    env = SafetyWrapper(SPGRLEnv())
    env.reset()
    env.env.emergency = True
    zt, r, d, i = env.step(0) # Propose unsafe action
    assert env.unsafe_proposed > 0
    assert env.overrides_performed > 0
    assert i.get('override', False) is True

def test_unified_state():
    env = SPGRLEnv()
    zt = env.reset()
    assert zt.shape == (168,)
    assert not np.isnan(zt).any()
    assert not np.isinf(zt).any()

def test_environment():
    env = SPGRLEnv()
    env.reset()
    zt, r, d, i = env.step(0)
    assert zt.shape == (168,)
    assert isinstance(r, float)
    assert isinstance(d, bool)
    assert 'reward_components' in i

def test_ppo():
    agent = PPOAgent(state_dim=168, action_dim=4)
    buffer = RolloutBuffer()
    zt = np.zeros(168, dtype=np.float32)
    action = agent.act(zt, buffer)
    assert action in [0, 1, 2, 3]
    assert len(buffer.states) == 1
    assert len(buffer.actions) == 1
