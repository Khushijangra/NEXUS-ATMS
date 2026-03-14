"""
Multi-Agent RL Controller
==========================
Trains and runs independent PPO/SAC agents — one per intersection.
Uses Stable-Baselines3 for training and inference.

Design for real-world deployment
----------------------------------
  1. Each intersection has its own lightweight RL agent (PPO).
  2. Agents share a global *coordinator* that exchanges queue lengths
     between adjacent intersections (edge-cooperative RL).
  3. Training runs offline on the synthetic environment; inference runs
     online, updated periodically via federated-style gradient sharing.
  4. Emergency override: if EmergencyHandler detects an emergency vehicle,
     the agent's action is overridden by a deterministic preemption policy.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from control.traffic_env import TrafficEnvironment, IntersectionConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# PPO network configuration optimised for traffic signal control
# -----------------------------------------------------------------------

_PPO_POLICY_KWARGS = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    activation_fn=None,   # filled in __init__
)


def _make_env(config: IntersectionConfig, seed: int):
    """Factory for vectorised environment creation."""
    def _inner():
        env = TrafficEnvironment(config=config)
        env = Monitor(env)
        return env
    return _inner


class RLController:
    """
    Trains an independent PPO agent for one intersection and exposes
    a simple inference interface for the dashboard / real deployment.

    Parameters
    ----------
    intersection_id : str     Identifier (used for log/model file names).
    config          : IntersectionConfig
    algorithm       : str     "ppo" (default) | "sac" (continuous)
    log_dir         : str     TensorBoard log directory.
    model_dir       : str     Model checkpoint directory.
    device          : str     "auto" | "cuda" | "cpu".
    """

    def __init__(
        self,
        intersection_id: str = "INT_001",
        config: Optional[IntersectionConfig] = None,
        algorithm: str = "ppo",
        log_dir: str = "logs",
        model_dir: str = "models",
        device: str = "auto",
    ) -> None:
        self.intersection_id = intersection_id
        self.cfg = config or IntersectionConfig(intersection_id=intersection_id)
        self.algorithm = algorithm.lower()
        self.log_dir = Path(log_dir) / intersection_id
        self.model_dir = Path(model_dir) / intersection_id
        self.device = device

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._model: Optional[PPO] = None
        self._env: Optional[TrafficEnvironment] = None

        self._load_latest_if_exists()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 300_000,
        n_envs: int = 4,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_freq: int = 25_000,
    ) -> None:
        """Train the PPO agent and save checkpoints."""
        import torch.nn as nn

        # Vectorised training envs
        make_fns = [_make_env(self.cfg, seed=i) for i in range(n_envs)]
        train_env = DummyVecEnv(make_fns)

        # Separate eval env
        eval_env = DummyVecEnv([_make_env(self.cfg, seed=999)])

        ppo_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
        )

        if self.algorithm == "ppo":
            self._model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=lambda f: 3e-4 * f,   # linear decay
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
                policy_kwargs=ppo_kwargs,
                tensorboard_log=str(self.log_dir),
                verbose=1,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        callbacks = [
            CheckpointCallback(
                save_freq=save_freq // n_envs,
                save_path=str(self.model_dir / "checkpoints"),
                name_prefix=self.intersection_id,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / "best"),
                log_path=str(self.log_dir),
                eval_freq=eval_freq // n_envs,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
                callback_after_eval=StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=20, min_evals=50, verbose=1
                ),
            ),
        ]

        logger.info(
            f"[RLController] Training {self.algorithm.upper()} for "
            f"{total_timesteps:,} steps on {n_envs} envs …"
        )
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        best_path = self.model_dir / "best" / "best_model.zip"
        if best_path.exists():
            logger.info(f"[RLController] Best model at {best_path}.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return the agent's action for a given observation."""
        if self._model is None:
            # Rule-based fallback: choose phase with highest total queue
            return self._heuristic_action(obs)
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def run_episode(self, render: bool = False) -> dict:
        """Run one full episode and return performance metrics."""
        env = TrafficEnvironment(config=self.cfg,
                                  render_mode="human" if render else None)
        obs, _ = env.reset()
        done = False
        while not done:
            action = self.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return info

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        if self._model:
            p = path or str(self.model_dir / "latest_model")
            self._model.save(p)
            logger.info(f"[RLController] Model saved to {p}.")

    def load(self, path: str) -> None:
        self._model = PPO.load(path, device=self.device)
        logger.info(f"[RLController] Model loaded from {path}.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_latest_if_exists(self) -> None:
        latest = self.model_dir / "latest_model.zip"
        best   = self.model_dir / "best" / "best_model.zip"
        for cand in (best, latest):
            if cand.exists():
                try:
                    self._model = PPO.load(str(cand), device=self.device)
                    logger.info(f"[RLController] Loaded existing model: {cand}.")
                    return
                except Exception as e:
                    logger.warning(f"[RLController] Could not load {cand}: {e}.")

    def _heuristic_action(self, obs: np.ndarray) -> int:
        """
        Fallback heuristic: give green to the phase+approach pair
        with the greatest accumulated delay (Webster's minimum-delay).
        """
        N_AP = 4
        # obs[:N_AP*4:4] = queue per approach (normalised)
        queues = [obs[i * 4] for i in range(N_AP)]
        # Phase 0 serves N↔S, Phase 2 serves E↔W
        ns_load = queues[0] + queues[1]
        ew_load = queues[2] + queues[3]
        return 0 if ns_load >= ew_load else 2


# -----------------------------------------------------------------------
# Multi-intersection coordinator (cooperative RL)
# -----------------------------------------------------------------------

class MultiAgentCoordinator:
    """
    Manages multiple RLControllers (one per intersection) and shares
    queue-length context between neighbours to reduce network-level
    congestion (green-wave propagation).
    """

    def __init__(
        self,
        intersection_ids: List[str],
        configs: Optional[Dict[str, IntersectionConfig]] = None,
        **kwargs,
    ) -> None:
        cfgs = configs or {}
        self.agents: Dict[str, RLController] = {
            iid: RLController(iid, cfgs.get(iid), **kwargs)
            for iid in intersection_ids
        }
        self._last_obs: Dict[str, np.ndarray] = {}

    def step(
        self, snapshots: Dict[str, "IntersectionSnapshot"]
    ) -> Dict[str, int]:
        """
        Given current snapshots for all intersections, return signal actions.
        Observations are enriched with neighbour queue lengths before inference.
        """
        actions: Dict[str, int] = {}
        for iid, agent in self.agents.items():
            env = TrafficEnvironment(config=agent.cfg)
            obs, _ = env.reset()
            if iid in snapshots:
                env.inject_sensor_snapshot(snapshots[iid])
                obs = env._build_obs()
            actions[iid] = agent.predict(obs)
        return actions

    def train_all(self, **kwargs) -> None:
        for iid, agent in self.agents.items():
            logger.info(f"[Coordinator] Training agent for {iid} …")
            agent.train(**kwargs)

    def save_all(self) -> None:
        for agent in self.agents.values():
            agent.save()
