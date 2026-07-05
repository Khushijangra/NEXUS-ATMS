import sys
import numpy as np
import torch
from pathlib import Path

# Need to ensure v2 modules can be imported
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.core.unified_state import UnifiedStateBuilder
from v2.core.state_types import SPGRLState
from v2.core.stream_interfaces import (
    get_semantic_state,
    get_prediction_state,
    get_emergency_state,
    get_behavioral_state,
    get_graph_state,
    get_carbon_state
)

class SPGRLEnv:
    def __init__(self):
        # Observation is 168D
        self.observation_space_dim = 168
        # Discrete Action Space:
        # 0: North-South Green
        # 1: East-West Green
        # 2: Emergency Override
        # 3: Adaptive Timing
        self.action_space_dim = 4
        
        self.state_builder = UnifiedStateBuilder()
        
        # Internal Environment State (Mock SUMO for now)
        self.queue = np.zeros(4, dtype=np.float32) # N, S, E, W
        self.wait = np.zeros(4, dtype=np.float32)
        self.emergency = False
        self.carbon = 0.0
        self.anomaly_score = 0.0
        
        import collections
        self.history_buffer = collections.deque(maxlen=30)
        
        # Try to load prediction scaler
        import pickle
        scaler_path = root / "v2" / "prediction" / "lstm" / "dataset" / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = {"mean": np.zeros(5), "std": np.ones(5)}
        
        self.max_steps = 200
        self.current_step = 0
        self.prev_reward = 0.0
        
    def _build_zt(self):
        """Construct the 168D Zt state using the real interfaces."""
        # 1. Fetch real and dummy streams observing the environment
        As = get_semantic_state()
        
        # Prepare real historical features for LSTM
        if len(self.history_buffer) == 30:
            hist_np = np.array(self.history_buffer) # (30, 5)
            # Scale features
            hist_np = (hist_np - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)
            feat_tensor = torch.FloatTensor(hist_np).unsqueeze(0) # (1, 30, 5)
        else:
            feat_tensor = torch.zeros(1, 30, 5)
            
        Ft, Cf = get_prediction_state(features=feat_tensor)
        Et = get_emergency_state(is_active=self.emergency)
        
        Ab = get_behavioral_state()
        Gt = get_graph_state()
        Ct = get_carbon_state(queue=np.sum(self.queue), delay=np.sum(self.wait))
        
        state = SPGRLState(
            As=As,
            Ab=Ab,
            Ft=Ft,
            Cf=Cf,
            Gt=Gt,
            Ct=Ct,
            Et=Et
        )
        zt_tensor = self.state_builder.build(state, batch_size=1)
        
        # For environment internal tracking (from Semantic Stream)
        if As is not None and As.numel() > 0:
            self.anomaly_score = float(As[0, 0].item())
        else:
            self.anomaly_score = 0.0
            
        # We no longer overwrite self.emergency from Et.
        # The environment defines the emergency; the stream (Et) simply observes it.
            
        zt_flat = zt_tensor.detach().cpu().numpy().flatten()
        
        # VERY IMPORTANT: Normalize Zt to prevent NN explosion
        zt_flat = np.clip(zt_flat, -1000.0, 1000.0) 
        zt_flat = np.tanh(zt_flat / 100.0) # Keeps state within [-1, 1]
        
        return zt_flat
        
    def reset(self):
        self.queue = np.random.uniform(0, 30, size=4).astype(np.float32)
        self.wait = np.random.uniform(0, 60, size=4).astype(np.float32)
        self.carbon = np.random.uniform(10, 50)
        self.emergency = np.random.rand() < 0.1 # 10% chance to start with emergency
        self.current_step = 0
        self.prev_reward = 0.0
        
        # Fill history buffer with zeros or initial state
        self.history_buffer.clear()
        initial_feats = [np.sum(self.queue), np.sum(self.wait), 10.0, np.sum(self.queue)/200.0, 10.0]
        for _ in range(30):
            self.history_buffer.append(initial_feats)
        
        # Build the real initial state through the neural network pipeline
        self.cached_zt = self._build_zt()
        return self.cached_zt
        
    def step(self, action, compute_zt=True):
        """
        Action semantics:
        0: N-S green (reduces N,S queue/wait; increases E,W)
        1: E-W green (reduces E,W queue/wait; increases N,S)
        2: Emergency (forces priority route clearance)
        3: Adaptive (balanced reduction but higher carbon overhead)
        """
        # 1. Traffic physics update
        inflow = np.random.uniform(2, 8, size=4)
        self.queue += inflow
        self.wait += self.queue * 0.5
        
        # 1.5 Randomly spawn emergency if one is not active (2% chance per step)
        if not self.emergency and np.random.rand() < 0.02:
            self.emergency = True
            
        # Action effects
        if action == 0:
            self.queue[[0, 1]] = np.maximum(0, self.queue[[0, 1]] - 15)
            self.wait[[0, 1]] = np.maximum(0, self.wait[[0, 1]] - 20)
            self.carbon += 5.0
        elif action == 1:
            self.queue[[2, 3]] = np.maximum(0, self.queue[[2, 3]] - 15)
            self.wait[[2, 3]] = np.maximum(0, self.wait[[2, 3]] - 20)
            self.carbon += 5.0
        elif action == 2:
            if self.emergency:
                self.queue = np.maximum(0, self.queue - 25) # Massive clearance
                self.emergency = False # Cleared
            self.wait = np.maximum(0, self.wait - 5)
            self.carbon += 15.0 # Fast vehicles emit more
        elif action == 3:
            self.queue = np.maximum(0, self.queue - 8)
            self.wait = np.maximum(0, self.wait - 10)
            self.carbon += 8.0
            
        # 1.8 Append to history buffer
        total_queue = np.sum(self.queue)
        new_feats = [total_queue, np.sum(self.wait), 10.0, total_queue/200.0, 15.0 - (total_queue/20.0)]
        self.history_buffer.append(new_feats)
            
        # 2. Recompute Zt dynamically every step
        # This properly updates Ct (Carbon) and Et (Emergency) in the agent's observation
        if compute_zt:
            zt = self._build_zt()
        else:
            zt = None
        
        # 3. Compute Normalized Reward
        Q_max = 200.0
        D_max = 1000.0
        C_max = 1000.0
        
        congestion_penalty = np.sum(self.queue) / Q_max
        delay_penalty = np.sum(self.wait) / D_max
        carbon_penalty = self.carbon / C_max
        
        emergency_reward = 2.0 if (action == 2 and self.emergency) else 0.0
        if self.emergency and action != 2:
            emergency_reward = -2.0 # Missing emergency penalty
            
        anomaly_norm = np.tanh(abs(self.anomaly_score) / 100.0)
        
        raw_reward = emergency_reward - congestion_penalty - delay_penalty - carbon_penalty - anomaly_norm
        
        # Reward Smoothing Buffer
        smoothed_reward = 0.9 * raw_reward + 0.1 * self.prev_reward
        self.prev_reward = smoothed_reward
        
        # 4. Check terminal
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'reward_components': {
                'emergency_term': emergency_reward,
                'queue_term': -congestion_penalty,
                'delay_term': -delay_penalty,
                'carbon_term': -carbon_penalty,
                'semantic_term': -anomaly_norm,
                'raw_reward': raw_reward,
                'smoothed_reward': smoothed_reward
            }
        }
        
        return zt, float(smoothed_reward), done, info
