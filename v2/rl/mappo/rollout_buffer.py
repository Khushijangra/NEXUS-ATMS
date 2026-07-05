import torch
import numpy as np

class MAPPORolloutBuffer:
    """
    Stores rollout data for all N agents over T steps.
    Incorporates Global State, Topology ID, Communication Messages, and Safety Overrides.
    """
    def __init__(self, num_agents: int, buffer_size: int, state_dim: int = 168, action_dim: int = 4, msg_dim: int = 16):
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        
        # Local agent properties
        self.obs = torch.zeros((buffer_size, num_agents, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, num_agents, action_dim), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        
        # Global CTDE properties
        self.global_metrics = torch.zeros((buffer_size, 5), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.adjacency = torch.zeros((buffer_size, num_agents, num_agents), dtype=torch.float32)
        
        # Diagnostics
        self.communications = torch.zeros((buffer_size, num_agents, msg_dim), dtype=torch.float32)
        self.safety_overrides = torch.zeros((buffer_size, num_agents, 1), dtype=torch.float32)
        
        self.step = 0
        
    def add(self, obs, action, log_prob, reward, done, value, global_metric, adj, comms, safety):
        if self.step >= self.buffer_size:
            raise RuntimeError("RolloutBuffer is full.")
            
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.global_metrics[self.step] = global_metric
        self.adjacency[self.step] = adj
        self.communications[self.step] = comms
        self.safety_overrides[self.step] = safety
        
        self.step += 1
        
    def reset(self):
        self.step = 0
