import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from v2.rl.mappo.mappo_agent import MAPPOAgent
from v2.rl.mappo.losses import compute_actor_loss, compute_critic_loss, compute_entropy_bonus

class MAPPOLearner:
    """
    Executes the joint PPO gradient updates over the rollout buffer.
    Supports (B, N, D) batching for N-agent parameter sharing.
    """
    def __init__(self, agent: MAPPOAgent, config: dict):
        self.agent = agent
        self.clip_epsilon = config.get("ppo_clip", 0.2)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.epochs = config.get("epochs", 10)
        
    def update(self, buffer, advantages, returns):
        """
        buffer: MAPPORolloutBuffer
        advantages: [T, N, 1]
        returns: [T, N, 1]
        """
        # Normalize advantages across batch and agents
        flat_adv = advantages.view(-1)
        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        
        # Build Dataset
        # Shapes:
        # obs: [T, N, 168]
        # actions: [T, N, 4]
        # log_probs: [T, N, 1]
        # values: [T, 1]
        # global_metrics: [T, 5]
        # adjacency: [T, N, N]
        dataset = TensorDataset(
            buffer.obs[:buffer.step],
            buffer.actions[:buffer.step],
            buffer.log_probs[:buffer.step],
            buffer.values[:buffer.step],
            buffer.global_metrics[:buffer.step],
            buffer.adjacency[:buffer.step],
            advantages,
            returns
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_actor_loss = 0
        total_value_loss = 0
        
        for _ in range(self.epochs):
            for batch in loader:
                b_obs, b_act, b_logprob, b_val, b_global, b_adj, b_adv, b_ret = batch
                # b_obs: [B, N, 168]
                
                # 1. Forward Pass (Actor)
                # Since the actor is shared, we can flatten B and N for the forward pass 
                # to maximize parallelization, or just rely on PyTorch's native broadcasting.
                mean, std = self.agent.actor(b_obs)
                dist = torch.distributions.Normal(mean, std)
                new_logprob = dist.log_prob(b_act).sum(dim=-1, keepdim=True)
                entropy = compute_entropy_bonus(std)
                
                # 2. Forward Pass (Critic)
                new_value = self.agent.critic(b_obs, b_adj, b_global) # [B, 1]
                
                # 3. Compute Losses
                actor_loss = compute_actor_loss(new_logprob, b_logprob, b_adv, self.clip_epsilon)
                value_loss = compute_critic_loss(new_value, b_ret, b_val, self.clip_epsilon)
                
                # Total loss
                loss = actor_loss + 0.5 * value_loss - self.entropy_coeff * entropy
                
                # 4. Backprop
                self.agent.optimizer_actor.zero_grad()
                self.agent.optimizer_critic.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
                
                self.agent.optimizer_actor.step()
                self.agent.optimizer_critic.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                
        num_updates = self.epochs * len(loader)
        return total_actor_loss / num_updates, total_value_loss / num_updates
