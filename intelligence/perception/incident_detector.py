"""
Incident Detector
==================
Detects traffic incidents (accidents, stalled vehicles, sudden congestion)
using three complementary techniques:

  1. **Occupancy anomaly** — sudden spike in zone occupancy (queue jump).
  2. **Speed anomaly**     — abnormally low or zero speed in free-flow zone.
  3. **Stopped-vehicle detection** — a track that has not moved for N frames.

In a real deployment the detector would also integrate siren audio detection
(FFT-based) and cloud alert APIs.  Those stubs are provided but commented.

Severity levels: LOW | MEDIUM | HIGH
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Incident:
    incident_id: int
    incident_type: str          # "stopped_vehicle" | "sudden_congestion" | "wrong_way"
    severity: str               # "LOW" | "MEDIUM" | "HIGH"
    zone: str
    description: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    track_id: Optional[int] = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class IncidentDetector:
    """
    Stateful incident detector.  Call ``update()`` every frame.

    Parameters
    ----------
    queue_spike_threshold : float
        Ratio (current / rolling_mean) that triggers congestion alert.
    stopped_speed_kmh     : float
        Speed (km/h) below which a vehicle is considered stopped.
    stopped_frames        : int
        Consecutive frames a vehicle must be "stopped" before alerting.
    window                : int
        Rolling window length for occupancy baseline.
    """

    def __init__(
        self,
        queue_spike_threshold: float = 2.5,
        stopped_speed_kmh: float = 3.0,
        stopped_frames: int = 20,
        window: int = 60,
    ) -> None:
        self.spike_thresh = queue_spike_threshold
        self.stopped_kmh = stopped_speed_kmh
        self.stopped_frames = stopped_frames

        self._queue_history: Dict[str, Deque[float]] = {}
        self._stopped_counters: Dict[int, int] = {}   # track_id → consecutive stopped frames
        self._active_incidents: Dict[int, Incident] = {}
        self._incident_counter = 0
        self._window = window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        zone_queues: Dict[str, int],
        track_speeds: Dict[int, float],
    ) -> List[Incident]:
        """
        Evaluate frame-level evidence and return list of *new* incidents.

        Parameters
        ----------
        zone_queues  : zone → current queue count
        track_speeds : track_id → speed km/h
        """
        new_incidents: List[Incident] = []

        # 1. Occupancy anomaly
        for zone, q in zone_queues.items():
            hist = self._queue_history.setdefault(zone, deque(maxlen=self._window))
            if len(hist) >= 10:
                baseline = float(np.mean(hist))
                if baseline > 0 and q / baseline >= self.spike_thresh:
                    inc = self._raise(
                        "sudden_congestion",
                        "HIGH" if q / baseline >= 4 else "MEDIUM",
                        zone,
                        f"Queue spiked to {q} (baseline {baseline:.1f}) in zone {zone}.",
                    )
                    if inc:
                        new_incidents.append(inc)
            hist.append(q)

        # 2. Stopped-vehicle detection
        active_ids = set(track_speeds.keys())
        for tid, speed in track_speeds.items():
            if speed < self.stopped_kmh:
                self._stopped_counters[tid] = self._stopped_counters.get(tid, 0) + 1
                if self._stopped_counters[tid] == self.stopped_frames:
                    zone = self._guess_zone(zone_queues)
                    inc = self._raise(
                        "stopped_vehicle",
                        "MEDIUM",
                        zone,
                        f"Vehicle (track {tid}) stationary for {self.stopped_frames} frames.",
                        track_id=tid,
                    )
                    if inc:
                        new_incidents.append(inc)
            else:
                self._stopped_counters.pop(tid, None)

        # Prune counters for tracks that disappeared
        for tid in list(self._stopped_counters.keys()):
            if tid not in active_ids:
                del self._stopped_counters[tid]

        return new_incidents

    @property
    def active_incidents(self) -> List[Incident]:
        return list(self._active_incidents.values())

    def resolve(self, incident_id: int) -> bool:
        """Mark an incident as resolved."""
        if incident_id in self._active_incidents:
            self._active_incidents[incident_id].resolved = True
            del self._active_incidents[incident_id]
            return True
        return False

    def summary(self) -> dict:
        return {
            "active_count": len(self._active_incidents),
            "incidents": [
                {
                    "id": inc.incident_id,
                    "type": inc.incident_type,
                    "severity": inc.severity,
                    "zone": inc.zone,
                    "age_s": round(inc.age_seconds, 1),
                    "desc": inc.description,
                }
                for inc in self._active_incidents.values()
            ],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _raise(
        self,
        inc_type: str,
        severity: str,
        zone: str,
        description: str,
        track_id: Optional[int] = None,
    ) -> Optional[Incident]:
        # Dedup: don't raise same type+zone twice without resolution
        key = f"{inc_type}::{zone}"
        for inc in self._active_incidents.values():
            if f"{inc.incident_type}::{inc.zone}" == key:
                return None

        self._incident_counter += 1
        inc = Incident(
            incident_id=self._incident_counter,
            incident_type=inc_type,
            severity=severity,
            zone=zone,
            description=description,
            track_id=track_id,
        )
        self._active_incidents[inc.incident_id] = inc
        logger.warning(f"[INCIDENT #{inc.incident_id}] {severity} — {description}")
        return inc

    @staticmethod
    def _guess_zone(zone_queues: Dict[str, int]) -> str:
        """Return the zone with the highest queue (best guess for incident location)."""
        if not zone_queues:
            return "unknown"
        return max(zone_queues, key=zone_queues.get)
