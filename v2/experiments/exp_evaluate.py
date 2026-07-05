import os
import sys
import csv
import time
import numpy as np
import torch
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.rl.ppo_agent import PPOAgent

def run_evaluate(exp_dir: Path, eval_episodes: int = 10):
    if not exp_dir.exists():
        print(f"Error: Experiment dir {exp_dir} not found.")
        return
        
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"Error: Checkpoints dir {ckpt_dir} not found.")
        return
        
    eval_csv = exp_dir / "evaluation_metrics.csv"
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "Episode", "Reward", "Queue", "Delay", "Carbon", "SafetyOverrides", "Time"])
        
    env = SPGRLEnv()
    agent = PPOAgent(state_dim=168, action_dim=4)
    
    ckpts = list(ckpt_dir.glob("*.pth"))
    for ckpt_path in ckpts:
        print(f"\nEvaluating checkpoint: {ckpt_path.name}")
        agent.actor.load_state_dict(torch.load(ckpt_path, weights_only=True))
        agent.actor.eval()
        
        for ep in range(1, eval_episodes + 1):
            state = env.reset()
            ep_reward = 0
            done = False
            safety_overrides = 0
            
            start_time = time.time()
            
            while not done:
                action = agent.act(state, None)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                safety_overrides += info.get('safety_overrides', 0)
                
            inf_time = time.time() - start_time
            queue_val = float(np.sum(env.queue))
            delay_val = float(np.sum(env.wait))
            carbon_val = float(env.carbon)
            
            with open(eval_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ckpt_path.name, ep, ep_reward, queue_val, delay_val, carbon_val, safety_overrides, inf_time])
                
            print(f"[{ckpt_path.name}] Eval Ep {ep:03d} | Reward: {ep_reward:7.1f} | Queue: {queue_val:6.1f}")
            
    print(f"\nEvaluation complete. Results saved to {eval_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory (e.g. experiments/2026...)")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    
    run_evaluate(Path(args.exp_dir), args.episodes)
