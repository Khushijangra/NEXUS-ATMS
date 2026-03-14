"""
Demo Data Generator for Dashboard
Generates realistic simulated traffic data without requiring SUMO.
"""

import math
import random
import time
from typing import Dict, List


class DemoDataGenerator:
    """
    Generates realistic-looking traffic simulation data for the dashboard
    when SUMO is not available. Useful for presentations and demos.
    """

    def __init__(self, mode: str = "rl"):
        """
        Args:
            mode: "baseline" for fixed-timing, "rl" for RL agent demo.
        """
        self.mode = mode
        self._tick = 0
        self._hour = 8  # start at 8 AM

    def _time_factor(self) -> float:
        """Time-of-day multiplier for traffic volume."""
        # Peak at 8-9 AM and 5-6 PM
        if 8 <= self._hour <= 9 or 17 <= self._hour <= 18:
            return 1.0
        elif 7 <= self._hour <= 10 or 16 <= self._hour <= 19:
            return 0.75
        elif 12 <= self._hour <= 13:
            return 0.65
        elif 22 <= self._hour or self._hour <= 5:
            return 0.15
        else:
            return 0.5

    def get_snapshot(self) -> Dict:
        """
        Return one time-step snapshot of traffic metrics.

        Returns:
            Dictionary containing queue lengths, waiting times,
            throughput, phase info, and performance indicators.
        """
        self._tick += 1
        # Advance simulated clock
        if self._tick % 60 == 0:
            self._hour = (self._hour + 1) % 24

        tf = self._time_factor()
        noise = random.gauss(0, 0.05)

        # RL agent gradually improves performance
        rl_improvement = 1.0
        if self.mode == "rl":
            rl_improvement = max(0.55, 1.0 - self._tick * 0.001)

        # Queue lengths
        base_queue = tf * 15 + noise * 5
        queues = {
            d: max(0, base_queue * random.uniform(0.6, 1.4) * rl_improvement)
            for d in ["north", "south", "east", "west"]
        }

        # Waiting times
        base_wait = tf * 35 + noise * 10
        waits = {
            d: max(0, base_wait * random.uniform(0.7, 1.3) * rl_improvement)
            for d in ["north", "south", "east", "west"]
        }

        # Throughput (higher is better — inversely affected by congestion)
        throughput = max(0, (900 * tf / rl_improvement) + random.gauss(0, 30))

        # Phase
        cycle_pos = self._tick % 72
        if cycle_pos < 33:
            phase = "NS-Green"
        elif cycle_pos < 36:
            phase = "NS-Yellow"
        elif cycle_pos < 69:
            phase = "EW-Green"
        else:
            phase = "EW-Yellow"

        return {
            "tick": self._tick,
            "hour": self._hour,
            "mode": self.mode,
            "queues": queues,
            "waiting_times": waits,
            "total_queue": sum(queues.values()),
            "avg_waiting_time": sum(waits.values()) / 4,
            "throughput": throughput,
            "phase": phase,
            "time_factor": tf,
        }

    def get_history(self, n: int = 100) -> List[Dict]:
        """Generate n snapshots in sequence."""
        return [self.get_snapshot() for _ in range(n)]
