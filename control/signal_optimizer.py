"""
Green-Wave Signal Optimiser
============================
Computes phase offsets along an arterial corridor so that vehicles
travelling at a target speed encounter consecutive green lights.

Used by the MultiAgentCoordinator to seed initial signal plans
before RL fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GreenWaveConfig:
    """Parameters for one arterial corridor."""
    corridor_junctions: List[str]
    target_speed_mps: float = 13.9          # ~50 km/h
    spacing_m: float = 200.0                # distance between junctions
    cycle_length_s: float = 90.0
    green_ratio: float = 0.55               # fraction of cycle that is green


class GreenWaveOptimizer:
    """
    Calculate progressive signal offsets for a single corridor.

    The offset Δ_i for junction *i* is computed as::

        Δ_i = (distance_i / target_speed) mod cycle_length

    This ensures a platoon released at junction 0 arrives at each
    successive junction exactly when the signal turns green.
    """

    def __init__(self, config: GreenWaveConfig | None = None) -> None:
        self.cfg = config or GreenWaveConfig(corridor_junctions=[])

    def compute_offsets(self) -> Dict[str, float]:
        """Return ``{junction_id: offset_seconds}`` for the corridor."""
        offsets: Dict[str, float] = {}
        travel_time_per_link = (
            self.cfg.spacing_m / self.cfg.target_speed_mps
            if self.cfg.target_speed_mps > 0
            else 0.0
        )
        for idx, jid in enumerate(self.cfg.corridor_junctions):
            raw_offset = idx * travel_time_per_link
            offsets[jid] = round(raw_offset % self.cfg.cycle_length_s, 2)
        logger.info(
            "[GreenWave] Computed offsets for %d junctions.", len(offsets)
        )
        return offsets

    def apply_offsets(
        self,
        offsets: Dict[str, float],
        phase_starts: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Shift each junction's green-phase start by the computed offset.

        Parameters
        ----------
        offsets : dict   from :meth:`compute_offsets`
        phase_starts : dict   current green-phase start time per junction

        Returns
        -------
        dict  adjusted green-phase start times
        """
        adjusted = {}
        for jid, start in phase_starts.items():
            delta = offsets.get(jid, 0.0)
            adjusted[jid] = (start + delta) % self.cfg.cycle_length_s
        return adjusted
