"""Lightweight graph message passing for junction coordination."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GraphCoordinator:
    """Mean-neighbor message passing with safe fallback behavior."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    def enhance(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Return enhanced per-node states: concat(local, mean(neighbors))."""
        x = np.asarray(node_features, dtype=np.float32)
        adj = np.asarray(adjacency_matrix, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"node_features must be rank-2, got {x.ndim}")
        if adj.ndim != 2:
            raise ValueError(f"adjacency_matrix must be rank-2, got {adj.ndim}")
        if adj.shape[0] != x.shape[0] or adj.shape[1] != x.shape[0]:
            raise ValueError(
                f"adjacency shape {adj.shape} incompatible with node_features {x.shape}"
            )

        denom = np.sum(adj, axis=1, keepdims=True)
        neighbor_sum = adj @ x

        # Fallback if a node has no valid neighbors: use local state as message.
        neighbor_mean = np.divide(neighbor_sum, np.maximum(denom, 1.0))
        no_neighbor_mask = (denom <= 0.0).reshape(-1, 1)
        neighbor_mean = np.where(no_neighbor_mask, x, neighbor_mean)

        enhanced = np.concatenate([x, neighbor_mean], axis=1)

        if self.debug:
            logger.info(
                "[GraphCoordinator] local=%s adjacency=%s enhanced=%s",
                tuple(x.shape),
                tuple(adj.shape),
                tuple(enhanced.shape),
            )

        return enhanced

    def enhance_batch(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Batch variant for `(B, N, F)` + `(B, N, N)` arrays."""
        x = np.asarray(node_features, dtype=np.float32)
        adj = np.asarray(adjacency_matrix, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(f"node_features batch must be rank-3, got {x.ndim}")
        if adj.ndim != 3:
            raise ValueError(f"adjacency_matrix batch must be rank-3, got {adj.ndim}")
        if adj.shape[0] != x.shape[0] or adj.shape[1] != x.shape[1] or adj.shape[2] != x.shape[1]:
            raise ValueError(
                f"adjacency shape {adj.shape} incompatible with node_features {x.shape}"
            )

        out = [self.enhance(node_features=x[b], adjacency_matrix=adj[b]) for b in range(x.shape[0])]
        return np.stack(out, axis=0)
