"""
Multi-Object Vehicle Tracker — Simple IoU-based SORT-lite
==========================================================
Assigns persistent track IDs to detections frame-over-frame.
Keeps only the last *max_age* frames before pruning a lost track.
No extra dependencies required (pure NumPy).

Swap `update()` with any drop-in tracker (ByteTrack, DeepSORT) that follows
the same interface without changing downstream consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from vision.detector import Detection

logger = logging.getLogger(__name__)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union for two bboxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    label: str
    age: int = 0          # frames since last matched
    hits: int = 1         # total matched frames
    history: List[Tuple[int, int]] = field(default_factory=list)  # centre history


class VehicleTracker:
    """
    Simple greedy IOU tracker.

    Parameters
    ----------
    iou_threshold : float   Minimum IOU to associate detection with track.
    max_age       : int     Frames a track survives without a match.
    min_hits      : int     Hits before a track is reported to consumers.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_age: int = 5,
        min_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: List[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Match *detections* to existing tracks, update positions,
        and return detections with ``track_id`` filled in.
        """
        # ------ 1. Build IOU cost matrix ------
        n_trk = len(self._tracks)
        n_det = len(detections)

        if n_trk == 0:
            for det in detections:
                det.track_id = self._register(det)
            self._age_tracks(matched_ids=set(d.track_id for d in detections))
            return detections

        iou_mat = np.zeros((n_trk, n_det), dtype=float)
        for ti, trk in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou_mat[ti, di] = _iou(trk.bbox, det.bbox)

        # ------ 2. Greedy matching ------
        matched_trk: set[int] = set()
        matched_det: set[int] = set()

        # Sort by IOU descending
        pairs = sorted(
            ((iou_mat[ti, di], ti, di) for ti in range(n_trk) for di in range(n_det)),
            reverse=True,
        )
        for iou_val, ti, di in pairs:
            if iou_val < self.iou_threshold:
                break
            if ti in matched_trk or di in matched_det:
                continue
            trk = self._tracks[ti]
            det = detections[di]
            trk.bbox = det.bbox
            trk.age = 0
            trk.hits += 1
            trk.history.append(det.center)
            if len(trk.history) > 30:
                trk.history.pop(0)
            det.track_id = trk.track_id
            matched_trk.add(ti)
            matched_det.add(di)

        # ------ 3. Register new detections ------
        for di, det in enumerate(detections):
            if di not in matched_det:
                det.track_id = self._register(det)

        # ------ 4. Age unmatched tracks ------
        self._age_tracks(matched_ids=set(
            self._tracks[ti].track_id for ti in matched_trk
        ))

        # ------ 5. Filter by min_hits ------
        return [
            d for d in detections
            if self._get_track(d.track_id) and
               self._get_track(d.track_id).hits >= self.min_hits
        ]

    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.age <= self.max_age]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register(self, det: Detection) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks.append(Track(
            track_id=tid,
            bbox=det.bbox,
            class_id=det.class_id,
            label=det.label,
            history=[det.center],
        ))
        return tid

    def _age_tracks(self, matched_ids: set) -> None:
        for trk in self._tracks:
            if trk.track_id not in matched_ids:
                trk.age += 1
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

    def _get_track(self, tid: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == tid:
                return t
        return None
