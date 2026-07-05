import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim=168, action_dim=4):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim=168):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim=168, action_dim=4, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = Actor(state_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def act(self, state, buffer):
        """Select an action during interaction and store to buffer."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if state.ndim == 1:
                state = state.unsqueeze(0)
                
            action_probs = self.policy_old(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        buffer.states.append(state.squeeze(0))
        buffer.actions.append(action.squeeze(0))
        buffer.logprobs.append(action_logprob.squeeze(0))
            
        return action.item()
        
    def evaluate(self, state, action):
        """Evaluate actions during PPO update."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy
        
    def update(self, buffer):
        """PPO update using collected trajectories."""
        # Convert lists to tensors
        old_states = torch.stack(buffer.states, dim=0).detach()
        old_actions = torch.stack(buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(buffer.logprobs, dim=0).detach()
        rewards = buffer.rewards
        is_terminals = buffer.is_terminals
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        # Normalizing the returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        avg_loss = 0
        avg_entropy = 0
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            avg_loss += loss.mean().item()
            avg_entropy += dist_entropy.mean().item()
            
        avg_loss /= self.K_epochs
        avg_entropy /= self.K_epochs
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        # Clear buffer
        buffer.clear()
        
        return avg_loss, avg_entropy
