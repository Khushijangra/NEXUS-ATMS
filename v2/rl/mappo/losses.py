import torch
import torch.nn.functional as F

def compute_actor_loss(action_log_probs, old_action_log_probs, advantages, clip_epsilon=0.2):
    """
    Computes PPO surrogate clipping loss for actors.
    Shapes:
        action_log_probs: [B, N, 1]
        old_action_log_probs: [B, N, 1]
        advantages: [B, N, 1]
    """
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    return actor_loss

def compute_critic_loss(values, returns, old_values=None, clip_epsilon=0.2):
    """
    Computes Centralized Critic value loss.
    Shapes:
        values: [B, 1]
        returns: [B, N, 1] -> averaged to [B, 1]
    """
    # Average the returns across N agents to match the Centralized Critic's scalar target
    mean_returns = returns.mean(dim=1)
    
    if old_values is not None:
        value_clipped = old_values + torch.clamp(values - old_values, -clip_epsilon, clip_epsilon)
        loss1 = (values - mean_returns).pow(2)
        loss2 = (value_clipped - mean_returns).pow(2)
        value_loss = 0.5 * torch.max(loss1, loss2).mean()
    else:
        value_loss = 0.5 * (values - mean_returns).pow(2).mean()
        
    return value_loss

def compute_entropy_bonus(action_stds):
    """
    Shapes:
        action_stds: [B, N, A]
    """
    entropy = 0.5 + 0.5 * torch.log(2.0 * torch.pi * action_stds.pow(2))
    return entropy.mean()
