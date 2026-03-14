"""
Zone-Based Vehicle Counter
===========================
Counts vehicles crossing virtual lines (stop-line, counting-line)
and tracks queue length inside waiting zones.

Each intersection approach is described by a *CountingZone*, which
specifies a polygon and a direction.  When a tracked vehicle's centre
enters the zone it is counted once per unique track_id.

Throughput = vehicles that cross the EXIT line of the zone per cycle.
Queue      = vehicles currently inside the WAITING polygon.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.detector import Detection

logger = logging.getLogger(__name__)

# Approach labels
APPROACHES = ("north", "south", "east", "west")


@dataclass
class CountingZone:
    """A quadrilateral waiting zone + a counting line."""
    name: str
    # polygon corners (x,y) clockwise — waiting area
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    # counting line: two points [(x1,y1),(x2,y2)]  — tracks crossing this line
    counting_line: Optional[List[Tuple[int, int]]] = None
    # per-cycle counts
    queue: int = 0
    throughput: int = 0
    total_counted: int = 0
    _seen_ids: set = field(default_factory=set)
    _in_zone_ids: set = field(default_factory=set)


class ZoneCounter:
    """
    Manages multiple counting zones for a single intersection camera.

    Parameters
    ----------
    zones : dict
        Mapping of zone_name → CountingZone definition.
        Pass {} to use auto-generated zones for a 720×1280 frame.
    """

    def __init__(
        self,
        zones: Optional[Dict[str, CountingZone]] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> None:
        if zones is None:
            zones = self._default_zones(frame_shape)
        self.zones: Dict[str, CountingZone] = zones
        self._cycle_start = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> Dict[str, dict]:
        """
        Update counts from a new frame's detections.

        Returns
        -------
        dict  {zone_name: {queue, throughput, total}}
        """
        for zone in self.zones.values():
            zone.queue = 0

        for det in detections:
            cx, cy = det.center
            for zone in self.zones.values():
                if self._point_in_polygon(cx, cy, zone.polygon):
                    zone.queue += 1
                    if det.track_id != -1 and det.track_id not in zone._seen_ids:
                        zone.throughput += 1
                        zone.total_counted += 1
                        zone._seen_ids.add(det.track_id)

        # Trim seen_ids to avoid unbounded growth
        for zone in self.zones.values():
            if len(zone._seen_ids) > 10_000:
                zone._seen_ids = set(list(zone._seen_ids)[-5_000:])

        return {
            name: {
                "queue": z.queue,
                "throughput": z.throughput,
                "total": z.total_counted,
            }
            for name, z in self.zones.items()
        }

    def reset_cycle_counts(self) -> None:
        """Reset per-cycle throughput counters (call at each signal cycle start)."""
        for zone in self.zones.values():
            zone.throughput = 0
            zone._seen_ids.clear()
        self._cycle_start = time.time()

    def get_queue_vector(self, approaches: Tuple = APPROACHES) -> np.ndarray:
        """Return queue lengths as a fixed-length numpy vector."""
        return np.array(
            [self.zones.get(a, CountingZone(a)).queue for a in approaches],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_polygon(px: int, py: int, polygon: List[Tuple[int, int]]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside

    def _default_zones(
        self, shape: Tuple[int, int]
    ) -> Dict[str, CountingZone]:
        """Auto-generate four approach zones for a standard overhead camera."""
        H, W = shape
        cx, cy = W // 2, H // 2
        margin = 40
        return {
            "north": CountingZone(
                "north",
                polygon=[(cx - 80, 0), (cx + 80, 0), (cx + 80, cy - margin),
                         (cx - 80, cy - margin)],
                counting_line=[(cx - 80, cy - margin), (cx + 80, cy - margin)],
            ),
            "south": CountingZone(
                "south",
                polygon=[(cx - 80, cy + margin), (cx + 80, cy + margin),
                         (cx + 80, H), (cx - 80, H)],
                counting_line=[(cx - 80, cy + margin), (cx + 80, cy + margin)],
            ),
            "east": CountingZone(
                "east",
                polygon=[(cx + margin, cy - 60), (W, cy - 60),
                         (W, cy + 60), (cx + margin, cy + 60)],
                counting_line=[(cx + margin, cy - 60), (cx + margin, cy + 60)],
            ),
            "west": CountingZone(
                "west",
                polygon=[(0, cy - 60), (cx - margin, cy - 60),
                         (cx - margin, cy + 60), (0, cy + 60)],
                counting_line=[(cx - margin, cy - 60), (cx - margin, cy + 60)],
            ),
        }
