import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv

def evaluate_controller(env, controller_type, num_episodes=10):
    total_rewards = []
    final_queues = []
    final_delays = []
    final_carbons = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        ep_reward = 0
        
        while not done:
            if controller_type == "random":
                action = np.random.randint(0, env.action_space_dim)
            elif controller_type == "fixed":
                action = step % env.action_space_dim
            elif controller_type == "actuated":
                # N/S is indices 0,1. E/W is indices 2,3
                q_ns = np.sum(env.queue[[0, 1]])
                q_ew = np.sum(env.queue[[2, 3]])
                if env.emergency:
                    action = 2 # Emergency override prioritised
                elif q_ns > q_ew:
                    action = 0
                elif q_ew > q_ns:
                    action = 1
                else:
                    action = 3 # Adaptive/balanced
            
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            step += 1
            
        total_rewards.append(ep_reward)
        final_queues.append(np.sum(env.queue))
        final_delays.append(np.sum(env.wait))
        final_carbons.append(env.carbon)
        
    return {
        "reward": np.mean(total_rewards),
        "queue": np.mean(final_queues),
        "delay": np.mean(final_delays),
        "carbon": np.mean(final_carbons)
    }

def main():
    print("Benchmarking Baselines...")
    env = SPGRLEnv()
    
    results = {}
    for controller in ["random", "fixed", "actuated"]:
        print(f"Evaluating {controller.capitalize()} Controller...")
        res = evaluate_controller(env, controller, num_episodes=10)
        results[controller] = res
        
    print("\n--- Benchmark Results ---")
    print(f"{'Controller':<15} | {'Avg Reward':<12} | {'Final Queue':<12} | {'Final Delay':<12} | {'Final Carbon':<12}")
    print("-" * 75)
    for ctrl, res in results.items():
        print(f"{ctrl.capitalize():<15} | {res['reward']:<12.2f} | {res['queue']:<12.2f} | {res['delay']:<12.2f} | {res['carbon']:<12.2f}")

if __name__ == "__main__":
    main()
