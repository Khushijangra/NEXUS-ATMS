import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv

def main():
    print("Running SPGRL Environment Stability Test...")
    env = SPGRLEnv()
    state = env.reset()
    
    queues = []
    delays = []
    carbons = []
    rewards = []
    
    done = False
    step = 0
    while not done:
        action = np.random.randint(0, env.action_space_dim)
        state, reward, done, _ = env.step(action)
        
        queues.append(np.sum(env.queue))
        delays.append(np.sum(env.wait))
        carbons.append(env.carbon)
        rewards.append(reward)
        step += 1
        
    print(f"Episode finished after {step} steps.")
    print(f"Final Queue: {queues[-1]:.2f}")
    print(f"Final Delay: {delays[-1]:.2f}")
    print(f"Final Carbon: {carbons[-1]:.2f}")
    print(f"Average Reward: {np.mean(rewards):.4f}")
    print(f"Min Reward: {np.min(rewards):.4f} | Max Reward: {np.max(rewards):.4f}")
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SPGRL Environment Stability Test (200 Random Steps)')
    
    axs[0, 0].plot(queues, color='blue')
    axs[0, 0].set_title('Queue vs Time')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Total Vehicles')
    
    axs[0, 1].plot(delays, color='red')
    axs[0, 1].set_title('Delay vs Time')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Total Delay')
    
    axs[1, 0].plot(carbons, color='green')
    axs[1, 0].set_title('Carbon vs Time')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Carbon Emissions')
    
    axs[1, 1].plot(rewards, color='purple')
    axs[1, 1].set_title('Reward vs Time')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('Reward')
    
    plt.tight_layout()
    output_path = root / "v2" / "evaluate" / "env_stability.png"
    plt.savefig(output_path)
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    main()
