"""
DQN Agent for Traffic Signal Control
Deep Q-Network implementation using Stable-Baselines3
"""

import os
from typing import Dict, Optional, Callable
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


def _setup_gpu() -> str:
    """Configure GPU if available, fall back to CPU otherwise."""
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available — using CPU")
        return "cpu"
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] {gpu_name} | {vram_gb:.1f} GB VRAM | CUDA {torch.version.cuda}")
    return "cuda"


class DQNAgent:
    """
    Deep Q-Network Agent for traffic signal control.

    Uses Stable-Baselines3's DQN implementation with custom
    hyperparameters optimized for traffic signal control.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Dict,
        log_dir: str = "logs",
        model_dir: str = "models",
    ):
        """
        Initialize DQN Agent.

        Args:
            env: Gymnasium environment.
            config: Configuration dictionary.
            log_dir: Directory for tensorboard logs.
            model_dir: Directory for saving models.
        """
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.model_dir = model_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        dqn_cfg = config.get("agent", {}).get("dqn", {})
        device = _setup_gpu()

        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get("agent", {}).get("learning_rate", 3e-4),
            buffer_size=dqn_cfg.get("buffer_size", 100000),
            learning_starts=1000,
            batch_size=dqn_cfg.get("batch_size", 64),
            tau=1.0,
            gamma=config.get("agent", {}).get("gamma", 0.99),
            train_freq=4,
            gradient_steps=1,
            target_update_interval=dqn_cfg.get("target_update_freq", 1000),
            exploration_fraction=0.1,
            exploration_initial_eps=dqn_cfg.get("epsilon_start", 1.0),
            exploration_final_eps=dqn_cfg.get("epsilon_end", 0.05),
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )

        self.eval_env = None
        self._setup_eval_env()

    def _setup_eval_env(self) -> None:
        """Set up evaluation environment."""
        self.eval_env = Monitor(self.env)

    def train(
        self,
        total_timesteps: int = 500000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 50000,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Train the DQN agent.

        Args:
            total_timesteps: Total training timesteps.
            eval_freq: Evaluate every n timesteps.
            n_eval_episodes: Episodes per evaluation.
            save_freq: Save model every n timesteps.
            callback: Optional additional callback.
        """
        callbacks = []

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best"),
            log_path=self.log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(self.model_dir, "checkpoints"),
            name_prefix="dqn_traffic",
        )
        callbacks.append(checkpoint_callback)

        if callback:
            callbacks.append(callback)

        print(f"[DQN] Starting training for {total_timesteps:,} timesteps...")
        print(f"[DQN] Device: {self.model.device}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
        )

        final_path = os.path.join(self.model_dir, "dqn_final.zip")
        self.model.save(final_path)
        print(f"[DQN] Training complete. Final model saved to {final_path}")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Get action from trained model."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def load(self, path: str) -> None:
        """Load a pre-trained model."""
        self.model = DQN.load(path, env=self.env)
        print(f"[DQN] Model loaded from {path}")

    def save(self, path: str) -> None:
        """Save current model."""
        self.model.save(path)
        print(f"[DQN] Model saved to {path}")

    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate the agent over multiple episodes.

        Args:
            n_episodes: Number of evaluation episodes.
            render: Whether to render.

        Returns:
            Dictionary of evaluation metrics.
        """
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                if render:
                    self.env.render()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            if "metrics" in info:
                episode_metrics.append(info["metrics"])

            print(f"  Episode {ep + 1}/{n_episodes}: Reward = {total_reward:.2f}")

        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episodes": n_episodes,
        }

        if episode_metrics:
            results["avg_waiting_time"] = float(
                np.mean([m.get("avg_waiting_time", 0) for m in episode_metrics])
            )
            results["avg_queue_length"] = float(
                np.mean([m.get("avg_queue_length", 0) for m in episode_metrics])
            )
            results["avg_throughput"] = float(
                np.mean([m.get("throughput", 0) for m in episode_metrics])
            )

        return results
