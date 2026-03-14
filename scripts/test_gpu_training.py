"""Quick test: Stable-Baselines3 training on GPU."""
from stable_baselines3 import DQN, PPO
import gymnasium as gym
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Test DQN on GPU
print("--- Testing DQN on CUDA ---")
env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, device="cuda", verbose=0)
print(f"DQN device: {model.device}")
model.learn(total_timesteps=1000)
print("DQN training on GPU: OK")

# Test PPO on GPU
print("\n--- Testing PPO on CUDA ---")
model = PPO("MlpPolicy", env, device="cuda", verbose=0)
print(f"PPO device: {model.device}")
model.learn(total_timesteps=1000)
print("PPO training on GPU: OK")

env.close()
print("\n✅ All SB3 agents confirmed running on GPU!")
