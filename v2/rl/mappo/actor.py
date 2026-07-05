import torch
import torch.nn as nn
from typing import Tuple

class SharedActor(nn.Module):
    """
    Shared Actor for MAPPO.
    All N agents use these exact same weights to map their local observation (168D)
    into a local action (4D), maximizing sample efficiency.
    """
    def __init__(self, state_dim: int = 168, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        
        # Softplus to ensure positive std
        self.action_std_param = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, local_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        local_obs: [B, N, 168] or [B, 168]
        returns: mean, std
        """
        features = self.net(local_obs)
        mean = torch.tanh(self.action_mean(features))
        std = torch.nn.functional.softplus(self.action_std_param).expand_as(mean)
        return mean, std
        
    def get_action(self, local_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(local_obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        # Clip action for safety
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob
