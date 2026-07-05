import os
import sys
import csv
import time
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.experiments.experiment_manager import ExperimentManager

class RandomAgent:
    def act(self, state, buffer=None):
        return np.random.randint(0, 4)
        
class FixedAgent:
    def __init__(self):
        self.step_count = 0
    def act(self, state, buffer=None):
        action = (self.step_count // 10) % 4
        self.step_count += 1
        return action
        
class ActuatedAgent:
    def act(self, state, buffer=None):
        # Extremely simplified actuated: pick phase with highest queue
        # Since queue is at specific indices (not purely 0-3 in state vector, we mock it)
        # We assume queue is index 10:14 in state (needs actual mapping from statebuilder)
        # For evaluation, we'll just mock actuated as random for this script template
        return np.random.randint(0, 4)

def run_baselines(config: dict, eval_episodes=50):
    env = SPGRLEnv()
    
    manager = ExperimentManager(root, config, exp_base_name="baselines")
    benchmark_csv = manager.current_exp_dir / "benchmark_summary.csv"
    
    with open(benchmark_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Agent", "Episode", "Reward", "Queue", "Delay", "Carbon", "SafetyOverrides", "InfTime"])
        
    agents = {
        "Random": RandomAgent(),
        "Fixed": FixedAgent(),
        "Actuated": ActuatedAgent()
        # "PPO": PPOAgent(...)  # Assuming PPO is evaluated separately or loaded
    }
    
    for agent_name, agent in agents.items():
        manager.logger.info(f"Evaluating Baseline: {agent_name}")
        
        for ep in range(1, eval_episodes + 1):
            state = env.reset()
            ep_reward = 0
            done = False
            safety_overrides = 0
            
            start_time = time.time()
            
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                safety_overrides += info.get('safety_overrides', 0)
                
            inf_time = time.time() - start_time
            queue_val = float(np.sum(env.queue))
            delay_val = float(np.sum(env.wait))
            carbon_val = float(env.carbon)
            
            with open(benchmark_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([agent_name, ep, ep_reward, queue_val, delay_val, carbon_val, safety_overrides, inf_time])
                
        manager.logger.info(f"{agent_name} Evaluation Complete.")
        
if __name__ == "__main__":
    run_baselines({"seed": 42}, eval_episodes=10)
