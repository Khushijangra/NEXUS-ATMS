import torch
import torch.nn as nn
from v2.rl.mappo.actor import SharedActor
from v2.rl.mappo.critic import CentralizedCritic

class MAPPOAgent(nn.Module):
    """
    High-level API wrapping the Shared Actor and Centralized Critic.
    Maintains the CTDE architecture required for multi-agent training.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.state_dim = config.get("state_dim", 168)
        self.action_dim = config.get("action_dim", 4)
        self.encoder_type = config.get("critic_type", "gcn")
        
        self.actor = SharedActor(self.state_dim, self.action_dim)
        self.critic = CentralizedCritic(self.state_dim, self.encoder_type)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 3e-4))
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))
        
    def act(self, local_obs: torch.Tensor) -> tuple:
        """
        local_obs: [B, N, 168]
        Uses SharedActor to generate N actions.
        """
        with torch.no_grad():
            action, log_prob = self.actor.get_action(local_obs)
        return action, log_prob
        
    def evaluate_value(self, states: torch.Tensor, adj: torch.Tensor, global_metrics: torch.Tensor) -> torch.Tensor:
        """
        states: [B, N, 168]
        adj: [B, N, N]
        global_metrics: [B, 5]
        """
        with torch.no_grad():
            value = self.critic(states, adj, global_metrics)
        return value
