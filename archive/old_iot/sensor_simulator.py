"""
IoT Sensor Simulator
=====================
Simulates the real-time data that would arrive from:
  • Inductive loop detectors (buried in road — count + occupancy)
  • Radar / LiDAR speed sensors
  • Environmental sensors (air quality, weather)
  • Pedestrian push-button detectors
  • Emergency vehicle transponders

Data is published to an in-process event bus OR to a real MQTT broker.
The simulator honours time-of-day traffic patterns (rush hour, night, etc.)
and injects fault events to test resilience.

Design note:
  This code is intentionally written so that swapping ``SensorSimulator``
  for a real sensor driver (e.g. a RS-485 / NTCIP interface or a REST poll
  of a cloud IoT platform) requires only replacing this one class.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Sensor reading data model
# -----------------------------------------------------------------------

class SensorType(str, Enum):
    LOOP     = "inductive_loop"
    RADAR    = "radar"
    ENVIRON  = "environmental"
    PEDEST   = "pedestrian"
    EMERGENCY = "emergency_transponder"


@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    intersection_id: str
    approach: str           # "north" | "south" | "east" | "west" | "all"
    timestamp: float = field(default_factory=time.time)

    # Loop detector fields
    vehicle_count: int = 0
    occupancy_pct: float = 0.0       # 0–100 %
    headway_s: float = 0.0           # average time gap between vehicles (s)

    # Radar fields
    speed_kmh: float = 0.0
    speed_85th_kmh: float = 0.0      # 85th percentile speed

    # Environmental
    visibility_m: float = 1000.0
    rainfall_mm_h: float = 0.0
    temp_c: float = 20.0
    aqi: float = 50.0                 # Air Quality Index

    # Pedestrian
    ped_waiting: int = 0
    ped_cross_request: bool = False

    # Emergency
    emergency_active: bool = False
    emergency_type: str = ""         # "ambulance" | "fire" | "police"

    # Fault flag (set when sensor malfunctions)
    is_fault: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sensor_type"] = self.sensor_type.value
        return d


# -----------------------------------------------------------------------
# Traffic demand models (real-world patterns)
# -----------------------------------------------------------------------

def _time_demand_factor(hour: float) -> float:
    """
    Returns a traffic demand multiplier [0, 1] for a given hour of day.
    Calibrated from typical urban traffic surveys.
    """
    # AM peak 7:30–9:00, PM peak 17:00–18:30, midday shoulder
    am = math.exp(-((hour - 8.25) ** 2) / 0.8)
    pm = math.exp(-((hour - 17.75) ** 2) / 0.6)
    shoulder = 0.35 * math.exp(-((hour - 12.5) ** 2) / 4.0)
    night_base = 0.08
    return min(1.0, max(night_base, am * 0.95 + pm * 0.9 + shoulder))


def _weather_speed_factor(rainfall_mm_h: float, visibility_m: float) -> float:
    """Speed reduction due to weather (0.6 – 1.0)."""
    rain_factor = 1.0 - min(0.25, rainfall_mm_h / 40.0)
    vis_factor = 1.0 if visibility_m >= 500 else max(0.6, visibility_m / 500.0)
    return rain_factor * vis_factor


# -----------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------

class SensorSimulator:
    """
    High-fidelity sensor data simulator for a network of intersections.

    Parameters
    ----------
    intersection_ids : list[str]
        Identifiers for each simulated intersection.
    approaches       : list[str]
        Approach directions to simulate (default: N, S, E, W).
    speed_kmh        : float
        Free-flow speed on each approach.
    fault_rate       : float
        Probability (0–1) that a sensor reading is faulted per tick.
    real_time        : bool
        If True, simulator uses real wall-clock time for hour-of-day.
        If False, simulated time advances by ``sim_step_s`` each tick.
    sim_start_hour   : float
        Initial hour (0–24).  Only used when real_time=False.
    sim_step_s       : float
        Seconds of simulated time advanced per tick (real_time=False mode).
    callback         : callable(SensorReading)
        Optional function called with every reading (for MQTT publish etc.)
    """

    APPROACHES = ("north", "south", "east", "west")

    def __init__(
        self,
        intersection_ids: List[str] = ("INT_001",),
        approaches: Optional[List[str]] = None,
        speed_kmh: float = 50.0,
        fault_rate: float = 0.005,
        real_time: bool = False,
        sim_start_hour: float = 7.5,
        sim_step_s: float = 5.0,
        callback: Optional[Callable[[SensorReading], None]] = None,
    ) -> None:
        self.intersection_ids = list(intersection_ids)
        self.approaches = list(approaches or self.APPROACHES)
        self.free_flow_speed = speed_kmh
        self.fault_rate = fault_rate
        self.real_time = real_time
        self._sim_hour = sim_start_hour
        self._sim_step_s = sim_step_s
        self._callback = callback

        # Emergency vehicle state
        self._emergency: Dict[str, dict] = {}   # int_id → {active, type, ttl}

        # Random seed for reproducibility
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> List[SensorReading]:
        """
        Advance simulation by one timestep and return all sensor readings.
        """
        hour = self._current_hour()
        demand = _time_demand_factor(hour)

        readings: List[SensorReading] = []

        for int_id in self.intersection_ids:
            # --- Environmental sensor (one per intersection) ---
            readings.append(self._env_reading(int_id, hour))

            # --- Emergency transponder ---
            readings.append(self._emergency_reading(int_id))

            for approach in self.approaches:
                sid = f"{int_id}_{approach}"

                # Loop detector
                readings.append(self._loop_reading(sid, int_id, approach, demand))

                # Radar sensor
                readings.append(self._radar_reading(sid, int_id, approach, demand, hour))

                # Pedestrian button
                readings.append(self._ped_reading(sid, int_id, approach, demand))

        if not self.real_time:
            self._sim_hour = (self._sim_hour + self._sim_step_s / 3600.0) % 24.0

        if self._callback:
            for r in readings:
                self._callback(r)

        return readings

    def inject_emergency(self, intersection_id: str, etype: str = "ambulance",
                         duration_ticks: int = 20) -> None:
        """Inject an emergency vehicle event at a specific intersection."""
        self._emergency[intersection_id] = {
            "active": True, "type": etype, "ttl": duration_ticks
        }
        logger.info(f"[IoT] Emergency {etype} injected at {intersection_id}.")

    def inject_incident(self, intersection_id: str, approach: str,
                        duration_ticks: int = 30) -> None:
        """Suddenly increase demand on an approach to simulate an incident."""
        key = (intersection_id, approach)
        self._incidents = getattr(self, '_incidents', {})
        self._incidents[key] = duration_ticks
        logger.info(f"[IoT] Incident injected at {intersection_id}/{approach} for {duration_ticks} ticks.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_hour(self) -> float:
        if self.real_time:
            t = time.localtime()
            return t.tm_hour + t.tm_min / 60.0
        return self._sim_hour

    def _fault(self) -> bool:
        return self._rng.random() < self.fault_rate

    def _loop_reading(
        self, sid: str, int_id: str, approach: str, demand: float
    ) -> SensorReading:
        fault = self._fault()
        noise = self._rng.gauss(0, 0.06)
        occ = max(0.0, min(100.0, (demand + noise) * 85))
        count = max(0, int((demand + self._rng.gauss(0, 0.08)) * 18))
        headway = (3600.0 / max(count, 1)) if count else 99.0
        return SensorReading(
            sensor_id=f"{sid}_loop",
            sensor_type=SensorType.LOOP,
            intersection_id=int_id,
            approach=approach,
            vehicle_count=0 if fault else count,
            occupancy_pct=0.0 if fault else round(occ, 1),
            headway_s=0.0 if fault else round(headway, 1),
            is_fault=fault,
        )

    def _radar_reading(
        self, sid: str, int_id: str, approach: str, demand: float, hour: float
    ) -> SensorReading:
        fault = self._fault()
        rainfall = self._rng.gauss(0, 2) if hour in range(14, 18) else 0.0
        rainfall = max(0.0, rainfall)
        wf = _weather_speed_factor(rainfall, 1000.0)
        congestion_factor = 1.0 - demand * 0.5
        speed = max(5.0, self.free_flow_speed * wf * congestion_factor
                    + self._rng.gauss(0, 3))
        return SensorReading(
            sensor_id=f"{sid}_radar",
            sensor_type=SensorType.RADAR,
            intersection_id=int_id,
            approach=approach,
            speed_kmh=0.0 if fault else round(speed, 1),
            speed_85th_kmh=0.0 if fault else round(speed * 1.15, 1),
            is_fault=fault,
        )

    def _env_reading(self, int_id: str, hour: float) -> SensorReading:
        rain = max(0.0, self._rng.gauss(0, 1.5))
        vis = max(100.0, 1000.0 - rain * 20 + self._rng.gauss(0, 30))
        temp = 18 + 5 * math.sin((hour - 6) * math.pi / 12) + self._rng.gauss(0, 1)
        aqi = 60 + 40 * _time_demand_factor(hour) + self._rng.gauss(0, 5)
        return SensorReading(
            sensor_id=f"{int_id}_env",
            sensor_type=SensorType.ENVIRON,
            intersection_id=int_id,
            approach="all",
            visibility_m=round(vis, 0),
            rainfall_mm_h=round(rain, 1),
            temp_c=round(temp, 1),
            aqi=round(max(0, aqi), 1),
        )

    def _emergency_reading(self, int_id: str) -> SensorReading:
        em = self._emergency.get(int_id, {})
        active = em.get("active", False)
        if active:
            em["ttl"] -= 1
            if em["ttl"] <= 0:
                em["active"] = False
        return SensorReading(
            sensor_id=f"{int_id}_emg",
            sensor_type=SensorType.EMERGENCY,
            intersection_id=int_id,
            approach="all",
            emergency_active=active,
            emergency_type=em.get("type", "") if active else "",
        )

    def _ped_reading(
        self, sid: str, int_id: str, approach: str, demand: float
    ) -> SensorReading:
        waiting = max(0, int(demand * 6 + self._rng.gauss(0, 1)))
        request = self._rng.random() < demand * 0.3
        return SensorReading(
            sensor_id=f"{sid}_ped",
            sensor_type=SensorType.PEDEST,
            intersection_id=int_id,
            approach=approach,
            ped_waiting=waiting,
            ped_cross_request=request,
        )


def demand_aqi(hour: float) -> float:
    """AQI contribution from traffic (peaks with AM/PM rush)."""
    return 20 * _time_demand_factor(hour)
