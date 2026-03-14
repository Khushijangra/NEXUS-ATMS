"""
Optical-Flow Speed Estimator
=============================
Estimates vehicle speeds from track history using a pixel-to-metre
calibration factor.  For a real deployment the calibration is computed
once from a known road-marking distance in the camera's FOV.

Also provides a sparse Lucas-Kanade optical flow helper for
frame-level motion analysis (useful when no detector is running).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.tracker import Track

logger = logging.getLogger(__name__)

# Default calibration: 1 pixel ≈ 0.05 metres (typical overhead camera ~10m high)
DEFAULT_PIX_PER_METRE = 20.0   # pixels / metre


class SpeedEstimator:
    """
    Estimates vehicle speed (km/h) from the change in track centre positions.

    Parameters
    ----------
    pixels_per_metre : float
        Camera calibration — how many pixels equal one metre on the road plane.
    fps              : float
        Expected video frame rate used to convert pixel-displacement to m/s.
    history_len      : int
        Number of past positions used for smoothed speed estimate.
    """

    def __init__(
        self,
        pixels_per_metre: float = DEFAULT_PIX_PER_METRE,
        fps: float = 25.0,
        history_len: int = 8,
    ) -> None:
        self.ppm = pixels_per_metre
        self.fps = fps
        self.history_len = history_len
        self._speed_cache: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, tracks: List[Track]) -> Dict[int, float]:
        """
        Compute speed for every active track.

        Returns
        -------
        dict  {track_id: speed_kmh}
        """
        result: Dict[int, float] = {}
        for trk in tracks:
            h = trk.history[-self.history_len:]
            if len(h) < 2:
                result[trk.track_id] = 0.0
                continue

            # Cumulative Euclidean distance over history window
            total_px = sum(
                math.hypot(h[i][0] - h[i - 1][0], h[i][1] - h[i - 1][1])
                for i in range(1, len(h))
            )
            time_s = (len(h) - 1) / self.fps
            speed_ms = (total_px / self.ppm) / time_s
            speed_kmh = round(speed_ms * 3.6, 1)
            result[trk.track_id] = speed_kmh
            self._speed_cache[trk.track_id] = speed_kmh

        return result

    def average_speed(self, tracks: List[Track]) -> float:
        """Return mean speed across all active tracks (km/h)."""
        speeds = self.estimate(tracks)
        if not speeds:
            return 0.0
        return round(sum(speeds.values()) / len(speeds), 1)

    # ------------------------------------------------------------------
    # Lucas-Kanade Optical Flow helper (no tracker needed)
    # ------------------------------------------------------------------

    @staticmethod
    def lk_flow(
        prev_gray: np.ndarray, curr_gray: np.ndarray, max_corners: int = 200
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Sparse optical flow using Lucas-Kanade.

        Returns (good_old, good_new) point arrays, or (None, None) if
        opencv-python is not installed.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return None, None

        feat_params = dict(maxCorners=max_corners, qualityLevel=0.2,
                           minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                   10, 0.03))

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feat_params)
        if p0 is None:
            return None, None

        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0,
                                                  None, **lk_params)
        good_old = p0[status == 1]
        good_new = p1[status == 1]
        return good_old, good_new
