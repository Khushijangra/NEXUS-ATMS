"""
Metrics Tracker for Smart Traffic Management System
Centralised tracking of training and evaluation metrics.
"""

import json
import os
import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class MetricsTracker:
    """
    Tracks and records traffic simulation metrics across episodes.

    Records per-episode results and computes running averages for
    waiting time, queue length, throughput, reward, and phase changes.
    """

    def __init__(self, save_dir: str = "results"):
        """
        Args:
            save_dir: Directory to persist metric logs.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.episode_metrics: List[Dict] = []
        self.step_metrics: List[Dict] = []
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_episode(self, metrics: Dict) -> None:
        """Append one episode's metrics to the log."""
        metrics["timestamp"] = time.time() - self._start_time
        metrics["episode"] = len(self.episode_metrics) + 1
        self.episode_metrics.append(metrics)

    def record_step(self, metrics: Dict) -> None:
        """Append a single-step snapshot (used for live dashboards)."""
        metrics["timestamp"] = time.time() - self._start_time
        self.step_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return summary statistics across all recorded episodes."""
        if not self.episode_metrics:
            return {}

        keys = [
            "total_waiting_time",
            "avg_waiting_time",
            "avg_queue_length",
            "throughput",
            "total_reward",
            "phase_changes",
        ]
        summary: Dict = {"total_episodes": len(self.episode_metrics)}

        for key in keys:
            values = [m.get(key, 0) for m in self.episode_metrics if key in m]
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))
                summary[f"min_{key}"] = float(np.min(values))
                summary[f"max_{key}"] = float(np.max(values))

        return summary

    def get_learning_curve(self, key: str = "total_reward", window: int = 10) -> Dict:
        """Return raw and smoothed values for a given metric key."""
        raw = [m.get(key, 0) for m in self.episode_metrics]
        smoothed = []
        for i in range(len(raw)):
            start = max(0, i - window + 1)
            smoothed.append(float(np.mean(raw[start : i + 1])))
        return {"raw": raw, "smoothed": smoothed}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = "metrics.json") -> str:
        """Save all metrics to a JSON file and return the path."""
        path = os.path.join(self.save_dir, filename)
        data = {
            "summary": self.get_summary(),
            "episodes": self.episode_metrics,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        """Load a MetricsTracker from a previously saved JSON file."""
        tracker = cls(save_dir=os.path.dirname(path))
        with open(path, "r") as f:
            data = json.load(f)
        tracker.episode_metrics = data.get("episodes", [])
        return tracker
