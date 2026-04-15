"""Build graph snapshots from per-junction traffic state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_ORDER = (
    "queue_length",
    "waiting_time",
    "current_phase",
    "phase_time",
    "predicted_inflow",
    "emergency_flag",
)


class GraphStateBuilder:
    """Construct node features and adjacency matrix for intersection graphs."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _extract_node_features(self, payload: Dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                self._as_float(payload.get("queue_length", 0.0)),
                self._as_float(payload.get("waiting_time", 0.0)),
                self._as_float(payload.get("current_phase", 0.0)),
                self._as_float(payload.get("phase_time", 0.0)),
                self._as_float(payload.get("predicted_inflow", 0.0)),
                1.0 if bool(payload.get("emergency_flag", False)) else 0.0,
            ],
            dtype=np.float32,
        )

    def build(
        self,
        node_state_map: Dict[str, Dict[str, Any]],
        neighbor_map: Dict[str, Dict[str, Optional[str]]],
        node_order: Optional[Sequence[str]] = None,
        self_loop: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build `(node_features, adjacency_matrix, node_ids)`.

        `neighbor_map[node_id]` is expected to contain optional keys `north`, `south`, `east`, `west`.
        Missing neighbors are ignored safely.
        """
        if not node_state_map:
            return np.zeros((0, len(FEATURE_ORDER)), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

        node_ids = list(node_order) if node_order else sorted(node_state_map.keys())
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        features = np.zeros((len(node_ids), len(FEATURE_ORDER)), dtype=np.float32)
        adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)

        for nid in node_ids:
            idx = id_to_idx[nid]
            payload = node_state_map.get(nid, {}) or {}
            features[idx] = self._extract_node_features(payload)

            if self_loop:
                adjacency[idx, idx] = 1.0

            neighbors = neighbor_map.get(nid, {}) or {}
            for direction in ("north", "south", "east", "west"):
                nb = neighbors.get(direction)
                if not nb:
                    continue
                nb_idx = id_to_idx.get(nb)
                if nb_idx is None:
                    continue
                adjacency[idx, nb_idx] = 1.0

        if self.debug:
            logger.info(
                "[GraphStateBuilder] node_features=%s adjacency=%s nodes=%d",
                tuple(features.shape),
                tuple(adjacency.shape),
                len(node_ids),
            )

        return features, adjacency, node_ids

    def build_from_flat_observation(
        self,
        observation: np.ndarray,
        node_ids: Sequence[str],
        local_state_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract graph node features from flattened multi-junction observations.

        The method assumes per-junction layout from `MultiAgentSumoEnv` local state:
        queue[0:4], wait[4:8], phase one-hot[8:12], phase_time[12], ...
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        n_nodes = len(node_ids)
        expected = n_nodes * local_state_dim
        if obs.size < expected:
            raise ValueError(f"Observation too small for graph extraction: got {obs.size}, expected >= {expected}")

        node_state_map: Dict[str, Dict[str, Any]] = {}
        for i, nid in enumerate(node_ids):
            start = i * local_state_dim
            node_obs = obs[start : start + local_state_dim]

            queue_length = float(np.sum(node_obs[0:4])) if node_obs.size >= 4 else 0.0
            waiting_time = float(np.sum(node_obs[4:8])) if node_obs.size >= 8 else 0.0
            phase_slice = node_obs[8:12] if node_obs.size >= 12 else np.zeros((4,), dtype=np.float32)
            current_phase = float(int(np.argmax(phase_slice))) if phase_slice.size else 0.0
            phase_time = float(node_obs[12]) if node_obs.size >= 13 else 0.0
            predicted_inflow = float(np.mean(node_obs[13:17])) if node_obs.size >= 17 else 0.0
            emergency_flag = bool(node_obs[-1] > 0.5) if node_obs.size else False

            node_state_map[nid] = {
                "queue_length": queue_length,
                "waiting_time": waiting_time,
                "current_phase": current_phase,
                "phase_time": phase_time,
                "predicted_inflow": predicted_inflow,
                "emergency_flag": emergency_flag,
            }

        # Adjacency cannot be inferred robustly from flat observation alone.
        return self.build(node_state_map=node_state_map, neighbor_map={}, node_order=node_ids)
