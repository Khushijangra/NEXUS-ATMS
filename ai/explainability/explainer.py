"""
NEXUS-ATMS  —  Explainable AI (XAI) Module
=============================================
Provides transparency into RL traffic-signal decisions:

1. **Feature Importance via Permutation**
   Shuffles individual state dimensions and measures reward drop.
2. **SHAP-style Q-Value Decomposition**
   Uses KernelSHAP (model-agnostic) to attribute per-feature contribution
   to the DQN Q-value for each action.
3. **State Sensitivity Analysis**
   Gradient-based saliency (∂Q/∂state) for neural-network agents.
4. **Decision Explanation Generator**
   Human-readable explanations for signal-switch decisions.

Usage:
    python -m ai.explainability.explainer --model models/dqn_20260226_014406/best/best_model.zip
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Feature names matching SumoEnvironment 13-dim observation
SINGLE_INTERSECTION_FEATURES = [
    "queue_north", "queue_south", "queue_east", "queue_west",
    "wait_north", "wait_south", "wait_east", "wait_west",
    "phase_NS", "phase_EW", "phase_yellow", "phase_transition",
    "time_since_change",
]

# 21-dim per-junction features for multi-agent env
MULTI_AGENT_FEATURES = [
    "queue_N", "queue_S", "queue_E", "queue_W",
    "wait_N", "wait_S", "wait_E", "wait_W",
    "phase_0", "phase_1", "phase_2", "phase_3",
    "time_since_change",
    "neighbor_pressure_N", "neighbor_pressure_S",
    "neighbor_pressure_E", "neighbor_pressure_W",
    "hour_sin", "hour_cos",
    "global_congestion", "emergency_flag",
]


class TrafficXAI:
    """Explainable AI for traffic signal RL agents."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        n_background: int = 200,
    ):
        self.model_path = model_path
        self.feature_names = feature_names or SINGLE_INTERSECTION_FEATURES
        self.n_background = n_background
        self.sb3_model = None
        self._q_network = None

        if model_path and os.path.isfile(model_path):
            self._load_model(model_path)

    # ------------------------------------------------------------------ #
    #  Model loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_model(self, path: str) -> None:
        """Load SB3 DQN/PPO model for analysis."""
        try:
            from stable_baselines3 import DQN, PPO, A2C
        except ImportError:
            logger.error("stable-baselines3 required for XAI analysis")
            return

        for cls in (DQN, PPO, A2C):
            try:
                self.sb3_model = cls.load(path)
                self._q_network = self.sb3_model.policy
                logger.info(f"[XAI] Loaded {cls.__name__} from {path}")
                return
            except Exception:
                continue
        logger.error(f"[XAI] Could not load model from {path}")

    # ------------------------------------------------------------------ #
    #  1. Permutation Feature Importance                                  #
    # ------------------------------------------------------------------ #

    def permutation_importance(
        self,
        observations: np.ndarray,
        n_repeats: int = 30,
    ) -> Dict[str, float]:
        """
        Compute feature importance by permuting each feature and measuring
        the change in predicted Q-value spread (action confidence).

        Parameters
        ----------
        observations : np.ndarray  shape [N, state_dim]
        n_repeats    : shuffling repeats per feature

        Returns
        -------
        dict : feature_name → importance score (mean Q-value drop)
        """
        if self.sb3_model is None:
            return self._random_importance()

        import torch

        base_qvals = self._get_q_values(observations)  # [N, n_actions]
        base_confidence = np.std(base_qvals, axis=1).mean()

        importance = {}
        n_features = observations.shape[1]
        rng = np.random.RandomState(42)

        for i in range(n_features):
            drops = []
            for _ in range(n_repeats):
                obs_perm = observations.copy()
                rng.shuffle(obs_perm[:, i])
                perm_qvals = self._get_q_values(obs_perm)
                perm_conf = np.std(perm_qvals, axis=1).mean()
                drops.append(base_confidence - perm_conf)
            name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
            importance[name] = float(np.mean(drops))

        # Normalize to [0, 1]
        max_imp = max(abs(v) for v in importance.values()) or 1.0
        importance = {k: abs(v) / max_imp for k, v in importance.items()}
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------ #
    #  2. SHAP-style Q-Value Attribution                                  #
    # ------------------------------------------------------------------ #

    def shap_explain(
        self,
        observation: np.ndarray,
        n_samples: int = 100,
        background: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Approximate SHAP values for each action's Q-value using Kernel SHAP.

        Parameters
        ----------
        observation : np.ndarray  shape [state_dim]
        n_samples   : number of coalition samples
        background  : reference distribution [N, state_dim]

        Returns
        -------
        dict with keys:
          "shap_values" : [n_actions, state_dim] — per-action attributions
          "base_values" : [n_actions] — expected Q-value
          "feature_names" : list of str
        """
        if self.sb3_model is None:
            return {"shap_values": [], "base_values": [], "feature_names": self.feature_names}

        if background is None:
            background = self._generate_background(observation)

        nd = len(observation)
        rng = np.random.RandomState(0)

        # Base Q-values (all features from background mean)
        bg_mean = background.mean(axis=0)
        base_q = self._get_q_values(bg_mean[np.newaxis])[0]  # [n_actions]

        # Full Q-values (all features from observation)
        full_q = self._get_q_values(observation[np.newaxis])[0]

        # Accumulate marginal contributions via random coalitions
        shap_accum = np.zeros((len(base_q), nd))
        counts = np.zeros(nd)

        for _ in range(n_samples):
            perm = rng.permutation(nd)
            coalition = bg_mean.copy()
            prev_q = base_q.copy()
            for feat_idx in perm:
                coalition[feat_idx] = observation[feat_idx]
                cur_q = self._get_q_values(coalition[np.newaxis])[0]
                contribution = cur_q - prev_q
                shap_accum[:, feat_idx] += contribution
                counts[feat_idx] += 1
                prev_q = cur_q

        shap_values = shap_accum / np.maximum(counts, 1)

        return {
            "shap_values": shap_values.tolist(),
            "base_values": base_q.tolist(),
            "feature_names": self.feature_names[:nd],
        }

    # ------------------------------------------------------------------ #
    #  3. Gradient Saliency                                               #
    # ------------------------------------------------------------------ #

    def gradient_saliency(
        self,
        observation: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute |∂Q/∂state| for the chosen action — shows which state
        dimensions the network is most sensitive to.
        """
        if self.sb3_model is None:
            return {}

        import torch

        obs_t = torch.FloatTensor(observation).unsqueeze(0).requires_grad_(True)
        device = next(self._q_network.parameters()).device
        obs_t = obs_t.to(device)

        # Forward through Q-network
        try:
            q_values = self._q_network.q_net(
                self._q_network.extract_features(obs_t, self._q_network.features_extractor)
            )
        except AttributeError:
            # PPO/A2C - use value function instead
            try:
                _, q_values, _ = self._q_network.forward(obs_t)
            except Exception:
                return {}

        chosen_action = q_values.argmax(dim=1)
        chosen_q = q_values[0, chosen_action]
        chosen_q.backward()

        grads = obs_t.grad.detach().cpu().numpy()[0]
        saliency = np.abs(grads)
        saliency = saliency / (saliency.max() + 1e-10)

        nd = len(saliency)
        return {
            self.feature_names[i] if i < len(self.feature_names) else f"f{i}":
            float(saliency[i])
            for i in range(nd)
        }

    # ------------------------------------------------------------------ #
    #  4. Decision Explanation                                            #
    # ------------------------------------------------------------------ #

    def explain_decision(
        self,
        observation: np.ndarray,
        action: int,
        q_values: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate a human-readable explanation for a signal decision.

        Returns dict with:
         - action_label: "KEEP" or "SWITCH"
         - reason: human-readable string
         - confidence: float [0, 1]
         - top_factors: list of (feature, contribution) tuples
         - state_summary: dict of readable state metrics
        """
        obs = np.asarray(observation, dtype=np.float32)
        nd = len(obs)

        # Parse state
        queues = obs[:4] if nd >= 4 else obs
        waits = obs[4:8] if nd >= 8 else np.zeros(4)

        q_margin = 0.0
        if q_values is not None:
            q_margin = float(q_values[action] - q_values[1 - action])
        elif self.sb3_model is not None:
            qv = self._get_q_values(obs[np.newaxis])[0]
            q_margin = float(qv[action] - qv[1 - action])

        confidence = min(1.0, abs(q_margin) / (abs(q_margin) + 0.5))

        # Determine reason
        directions = ["North", "South", "East", "West"]
        max_queue_idx = int(np.argmax(queues))
        max_wait_idx = int(np.argmax(waits))

        if action == 0:
            action_label = "KEEP"
            reason = (
                f"Maintaining current phase. "
                f"Current flow is adequate (max queue: {queues[max_queue_idx]:.1f} "
                f"on {directions[max_queue_idx]})."
            )
        else:
            action_label = "SWITCH"
            reason = (
                f"Switching signal phase. "
                f"High pressure on {directions[max_queue_idx]} approach "
                f"(queue={queues[max_queue_idx]:.1f}, "
                f"wait={waits[max_wait_idx]:.1f}s)."
            )

        # Top factors from gradient saliency
        top_factors = []
        saliency = self.gradient_saliency(obs)
        if saliency:
            sorted_s = sorted(saliency.items(), key=lambda x: -x[1])[:5]
            top_factors = [(name, float(val)) for name, val in sorted_s]

        return {
            "action": action,
            "action_label": action_label,
            "reason": reason,
            "confidence": confidence,
            "q_margin": q_margin,
            "top_factors": top_factors,
            "state_summary": {
                "queues": {d: float(queues[i]) for i, d in enumerate(directions)},
                "waits": {d: float(waits[i]) for i, d in enumerate(directions)},
                "max_queue_direction": directions[max_queue_idx],
                "max_wait_direction": directions[max_wait_idx],
            },
        }

    # ------------------------------------------------------------------ #
    #  5. Batch Analysis + Report                                        #
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        observations: np.ndarray,
        output_dir: str = "results/xai",
    ) -> Dict:
        """
        Full XAI analysis report with plots.

        Parameters
        ----------
        observations : [N, state_dim]  — a sample of states
        output_dir   : where to save plots and JSON

        Returns
        -------
        dict with all analysis results
        """
        os.makedirs(output_dir, exist_ok=True)

        print("[XAI] Computing permutation feature importance...")
        importance = self.permutation_importance(observations)

        print("[XAI] Computing SHAP values for sample states...")
        shap_results = []
        sample_indices = np.linspace(0, len(observations) - 1, min(10, len(observations))).astype(int)
        for idx in sample_indices:
            sr = self.shap_explain(observations[idx], n_samples=50)
            shap_results.append(sr)

        print("[XAI] Computing gradient saliency over dataset...")
        saliencies = []
        for idx in sample_indices:
            sal = self.gradient_saliency(observations[idx])
            if sal:
                saliencies.append(sal)

        # Aggregate saliency
        avg_saliency = {}
        if saliencies:
            for key in saliencies[0]:
                avg_saliency[key] = float(np.mean([s.get(key, 0) for s in saliencies]))

        report = {
            "permutation_importance": importance,
            "avg_gradient_saliency": avg_saliency,
            "n_samples_analyzed": len(observations),
            "feature_names": self.feature_names,
            "shap_samples": len(shap_results),
        }

        # Save JSON
        json_path = os.path.join(output_dir, "xai_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[XAI] Report saved to {json_path}")

        # Generate plots
        self._plot_importance(importance, output_dir)
        if shap_results and shap_results[0]["shap_values"]:
            self._plot_shap_summary(shap_results, output_dir)

        return report

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_q_values(self, observations: np.ndarray) -> np.ndarray:
        """Get Q-values from SB3 model for a batch of observations."""
        import torch

        obs_t = torch.FloatTensor(observations)
        device = next(self._q_network.parameters()).device
        obs_t = obs_t.to(device)

        with torch.no_grad():
            try:
                # DQN
                features = self._q_network.extract_features(
                    obs_t, self._q_network.features_extractor)
                q_vals = self._q_network.q_net(features)
            except AttributeError:
                # PPO/A2C — use action distribution logits as proxy
                _, q_vals, _ = self._q_network.forward(obs_t)

        return q_vals.cpu().numpy()

    def _generate_background(self, obs: np.ndarray) -> np.ndarray:
        """Generate reference distribution around observation."""
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 0.1, size=(self.n_background, len(obs)))
        bg = np.clip(obs + noise, 0, 1).astype(np.float32)
        return bg

    def _random_importance(self) -> Dict[str, float]:
        """Placeholder importance when no model is loaded."""
        return {name: 0.0 for name in self.feature_names}

    # ------------------------------------------------------------------ #
    #  Plots                                                              #
    # ------------------------------------------------------------------ #

    def _plot_importance(self, importance: Dict[str, float], output_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        names = list(importance.keys())
        values = list(importance.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))  # type: ignore
        bars = ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance (normalized)")
        ax.set_title("NEXUS-ATMS — Feature Importance for Signal Decisions")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance plot → {path}")

    def _plot_shap_summary(self, shap_results: List[Dict], output_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Aggregate SHAP values across samples for action 0
        all_shap = []
        for sr in shap_results:
            if sr["shap_values"]:
                all_shap.append(sr["shap_values"][0])  # action 0

        if not all_shap:
            return

        shap_arr = np.array(all_shap)  # [n_samples, n_features]
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        names = shap_results[0]["feature_names"]

        # Sort by mean |SHAP|
        order = np.argsort(mean_abs)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(order)), mean_abs[order], color="#1f77b4")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("NEXUS-ATMS — SHAP Feature Attribution (Keep Phase)")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP summary plot → {path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run XAI analysis on trained RL agent")
    parser.add_argument("--model", required=True, help="Path to SB3 model .zip")
    parser.add_argument("--n-samples", type=int, default=500, help="State samples to generate")
    parser.add_argument("--output", default="results/xai")
    args = parser.parse_args()

    print("=" * 60)
    print("  NEXUS-ATMS — Explainable AI Analysis")
    print("=" * 60)

    xai = TrafficXAI(model_path=args.model)

    # Generate random observations for analysis
    n_feats = len(xai.feature_names)
    rng = np.random.RandomState(42)
    observations = rng.uniform(0, 1, size=(args.n_samples, n_feats)).astype(np.float32)

    report = xai.generate_report(observations, output_dir=args.output)

    # Show a sample explanation
    print("\n--- Sample Decision Explanation ---")
    sample_obs = observations[0]
    explanation = xai.explain_decision(sample_obs, action=1)
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
