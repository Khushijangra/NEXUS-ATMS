import torch
import numpy as np
import time
from v2.rl.mappo.multi_intersection_env import MultiIntersectionEnv
from v2.rl.mappo.mappo_agent import MAPPOAgent

class MAPPOEvaluator:
    def __init__(self, env: MultiIntersectionEnv, agent: MAPPOAgent):
        self.env = env
        self.agent = agent
        
    def evaluate(self, num_episodes: int = 10):
        self.agent.eval()
        
        total_rewards = []
        total_queues = []
        total_delays = []
        total_carbons = []
        
        for ep in range(1, num_episodes + 1):
            obs_list = self.env.reset()
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)
            
            episode_reward = 0
            dones = [False] * self.env.num_agents
            
            while not all(dones):
                with torch.no_grad():
                    # Deterministic action selection for evaluation
                    mean, std = self.agent.actor(obs_tensor)
                    action = torch.clamp(mean, -1.0, 1.0)
                    
                action_np = action.squeeze(0).cpu().numpy()
                action_list = [action_np[i] for i in range(self.env.num_agents)]
                
                next_obs_list, rewards, dones, infos = self.env.step(action_list)
                
                obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).unsqueeze(0)
                episode_reward += sum(rewards)
                
            metrics = self.env.get_global_metrics()
            
            total_rewards.append(episode_reward)
            total_queues.append(metrics[0])
            total_delays.append(metrics[1])
            total_carbons.append(metrics[2])
            
            print(f"Eval Ep {ep:03d} | Reward: {episode_reward:7.1f} | Queue: {metrics[0]:6.1f} | Delay: {metrics[1]:6.1f}")
            
        self.agent.train()
        return {
            "mean_reward": np.mean(total_rewards),
            "mean_queue": np.mean(total_queues),
            "mean_delay": np.mean(total_delays),
            "mean_carbon": np.mean(total_carbons)
        }
