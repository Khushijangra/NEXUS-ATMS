import torch
import numpy as np
import time
from v2.rl.mappo.multi_intersection_env import MultiIntersectionEnv
from v2.rl.mappo.mappo_agent import MAPPOAgent
from v2.rl.mappo.rollout_buffer import MAPPORolloutBuffer
from v2.rl.mappo.communication import MAPPOCommunication
from v2.rl.mappo.gae import compute_mappo_gae
from v2.rl.mappo.learner import MAPPOLearner

class MAPPOTrainer:
    def __init__(self, env: MultiIntersectionEnv, agent: MAPPOAgent, config: dict, manager=None):
        self.env = env
        self.agent = agent
        self.config = config
        self.manager = manager
        
        self.num_agents = env.num_agents
        self.buffer_size = config.get("buffer_size", 200)
        self.buffer = MAPPORolloutBuffer(self.num_agents, self.buffer_size)
        
        self.comm = MAPPOCommunication()
        self.learner = MAPPOLearner(self.agent, self.config)
        
        self.gamma = config.get("gamma", 0.99)
        self.lambda_gae = config.get("lambda_gae", 0.95)
        
    def train(self, total_episodes: int):
        best_reward = -float('inf')
        
        for episode in range(1, total_episodes + 1):
            obs_list = self.env.reset()
            # Convert list of N [168] arrays to [1, N, 168]
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)
            
            episode_reward = 0
            dones = [False] * self.num_agents
            step = 0
            
            start_time = time.time()
            
            while not all(dones) and step < self.buffer_size:
                adj_tensor = torch.tensor(self.env.get_adjacency(), dtype=torch.float32).unsqueeze(0)
                global_metrics_tensor = torch.tensor(self.env.get_global_metrics(), dtype=torch.float32).unsqueeze(0)
                
                # Actor Forward
                action, log_prob = self.agent.act(obs_tensor)
                
                # Critic Forward
                value = self.agent.evaluate_value(obs_tensor, adj_tensor, global_metrics_tensor)
                
                # Communication payload
                msgs = self.comm(obs_tensor, adj_tensor)
                
                # Environment Step
                # Convert action [1, N, 4] to list of numpy arrays
                action_np = action.squeeze(0).cpu().numpy()
                action_list = [action_np[i] for i in range(self.num_agents)]
                
                next_obs_list, rewards, dones, infos = self.env.step(action_list)
                
                reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(1, self.num_agents, 1)
                done_tensor = torch.tensor(dones, dtype=torch.float32).view(1, self.num_agents, 1)
                
                safety_overrides = sum([info.get('safety_overrides', 0) for info in infos])
                safety_tensor = torch.tensor([safety_overrides] * self.num_agents, dtype=torch.float32).view(1, self.num_agents, 1)
                
                self.buffer.add(
                    obs_tensor.squeeze(0), 
                    action.squeeze(0), 
                    log_prob.squeeze(0), 
                    reward_tensor.squeeze(0), 
                    done_tensor.squeeze(0), 
                    value.squeeze(0), 
                    global_metrics_tensor.squeeze(0), 
                    adj_tensor.squeeze(0),
                    msgs.squeeze(0),
                    safety_tensor.squeeze(0)
                )
                
                obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).unsqueeze(0)
                episode_reward += sum(rewards)
                step += 1
                
            # Compute next value for GAE
            adj_tensor = torch.tensor(self.env.get_adjacency(), dtype=torch.float32).unsqueeze(0)
            global_metrics_tensor = torch.tensor(self.env.get_global_metrics(), dtype=torch.float32).unsqueeze(0)
            next_value = self.agent.evaluate_value(obs_tensor, adj_tensor, global_metrics_tensor).squeeze(0)
            
            # GAE
            advantages, returns = compute_mappo_gae(
                self.buffer.rewards[:step],
                self.buffer.values[:step],
                self.buffer.dones[:step],
                next_value,
                self.gamma,
                self.lambda_gae
            )
            
            # Learn
            actor_loss, value_loss = self.learner.update(self.buffer, advantages, returns)
            self.buffer.reset()
            
            elapsed = time.time() - start_time
            avg_reward = episode_reward / self.num_agents
            
            print(f"Ep {episode:03d} | Avg Reward: {avg_reward:7.1f} | Actor Loss: {actor_loss:6.4f} | Value Loss: {value_loss:6.4f} | Time: {elapsed:5.1f}s")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                if self.manager:
                    # Save checkpoint logic
                    pass
