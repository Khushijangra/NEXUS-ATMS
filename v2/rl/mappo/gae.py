import torch

def compute_mappo_gae(rewards: torch.Tensor, 
                      values: torch.Tensor, 
                      dones: torch.Tensor, 
                      next_value: torch.Tensor,
                      gamma: float = 0.99, 
                      lambda_gae: float = 0.95):
    """
    Computes Generalized Advantage Estimation (GAE) for MAPPO.
    
    rewards: [T, N, 1]
    values: [T, 1] (Centralized Critic value of Global State)
    dones: [T, N, 1]
    next_value: [1] (Centralized Critic value of next Global State)
    
    returns: advantages [T, N, 1], returns [T, N, 1]
    """
    T, N, _ = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Expand values to [T, N, 1] so each agent gets a copy of the centralized baseline
    expanded_values = values.unsqueeze(1).expand(-1, N, -1)
    expanded_next_value = next_value.unsqueeze(0).expand(N, -1)
    
    last_gae = torch.zeros((N, 1), device=rewards.device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = expanded_next_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_val = expanded_values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            
        # delta^i_t = r^i_t + gamma * V(S_{t+1}) - V(S_t)
        delta = rewards[t] + gamma * next_val * next_non_terminal - expanded_values[t]
        
        # A^i_t = delta^i_t + gamma * lambda * A^i_{t+1}
        advantages[t] = delta + gamma * lambda_gae * next_non_terminal * last_gae
        last_gae = advantages[t]
        
    returns = advantages + expanded_values
    return advantages, returns
