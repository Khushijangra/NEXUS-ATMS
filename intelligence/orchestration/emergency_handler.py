"""
Emergency Handler for RL Controller
====================================
Thin wrapper that integrates the EmergencyCorridorEngine
with the RL signal-control loop.  When an active corridor
exists, it overrides the agent's action for affected junctions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class EmergencyHandler:
    """Preempts RL actions at junctions where an emergency corridor is active."""

    PREEMPT_GREEN_PHASE = 0  # phase index that gives green to the corridor

    def __init__(self) -> None:
        self._corridor_junctions: Set[str] = set()

    # ------------------------------------------------------------------
    # Corridor state updates (called by the dashboard / backend)
    # ------------------------------------------------------------------

    def activate_corridor(self, junction_ids: list[str]) -> None:
        """Mark *junction_ids* as needing emergency preemption."""
        self._corridor_junctions.update(junction_ids)
        logger.info(
            "[EmergencyHandler] Corridor active at %d junctions.", len(junction_ids)
        )

    def deactivate_corridor(self, junction_ids: list[str] | None = None) -> None:
        """Clear preemption — all junctions if *junction_ids* is None."""
        if junction_ids is None:
            self._corridor_junctions.clear()
        else:
            self._corridor_junctions -= set(junction_ids)
        logger.info("[EmergencyHandler] Corridor cleared.")

    # ------------------------------------------------------------------
    # Action override
    # ------------------------------------------------------------------

    def override_action(
        self,
        junction_id: str,
        agent_action: int,
    ) -> int:
        """
        Return the preemption phase if the junction is in an active corridor,
        otherwise return the agent's original action unchanged.
        """
        if junction_id in self._corridor_junctions:
            logger.debug(
                "[EmergencyHandler] Overriding junction %s → phase %d",
                junction_id,
                self.PREEMPT_GREEN_PHASE,
            )
            return self.PREEMPT_GREEN_PHASE
        return agent_action

    @property
    def active(self) -> bool:
        return len(self._corridor_junctions) > 0

    @property
    def active_junctions(self) -> list[str]:
        return sorted(self._corridor_junctions)
