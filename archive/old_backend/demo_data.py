"""
Demo Data Generator for Dashboard
=================================
Produces realistic, stochastic traffic snapshots without requiring SUMO.

Design goals:
  - Peak-hour congestion and off-peak relief
  - Uneven junction loads across a 4x4 city grid
  - Random arrivals using Poisson-like sampling
  - Accumulated waiting time per junction and per approach
  - RL mode with reduced waiting times relative to baseline
  - Emergency green-corridor events with delay-before / delay-after outputs
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


JUNCTION_IDS = [f"J{r}_{c}" for r in range(4) for c in range(4)]
APPROACHES = ("north", "south", "east", "west")


@dataclass
class CorridorEvent:
    event_id: str
    vehicle_id: str
    vehicle_type: str
    origin: str
    destination: str
    path: List[str]
    activated_tick: int
    delay_before_s: float
    delay_after_s: float
    active: bool = True
    created_at: float = field(default_factory=time.time)


class DemoDataGenerator:
    """Generates realistic traffic simulation data for demos and reporting."""

    def __init__(self, mode: str = "rl"):
        self.mode = mode.strip().lower() if mode else "rl"
        if self.mode not in ("baseline", "rl"):
            self.mode = "rl"

        self._tick = 0
        self._hour = 8.0
        self._rng = random.Random(42)
        self._np_rng = np.random.default_rng(42)

        # Static but uneven junction demand weights so each junction behaves differently.
        self._junction_weights: Dict[str, float] = {}
        self._junction_lanes: Dict[str, Dict[str, float]] = {}
        self._junction_state: Dict[str, Dict[str, float]] = {}
        for idx, jid in enumerate(JUNCTION_IDS):
            row, col = divmod(idx, 4)
            center_bias = 1.0 + 0.18 * (1.5 - abs(1.5 - row)) + 0.15 * (1.5 - abs(1.5 - col))
            noise_bias = 0.90 + 0.25 * self._rng.random()
            self._junction_weights[jid] = round(center_bias * noise_bias, 3)
            self._junction_lanes[jid] = self._build_lane_bias(row, col)
            self._junction_state[jid] = {
                "wait": self._rng.uniform(6.0, 14.0),
                "queue": self._rng.uniform(2.0, 7.0),
                "throughput": self._rng.uniform(15.0, 22.0),
                "phase_age": self._rng.randint(0, 60),
            }

        self._corridor_event: Optional[CorridorEvent] = None
        self._corridor_history: List[CorridorEvent] = []

    def _build_lane_bias(self, row: int, col: int) -> Dict[str, float]:
        north = 0.9 + 0.15 * (3 - row)
        south = 0.8 + 0.12 * row
        east = 0.85 + 0.10 * col
        west = 0.82 + 0.08 * (3 - col)
        return {"north": north, "south": south, "east": east, "west": west}

    def _time_factor(self) -> float:
        """Time-of-day multiplier for traffic demand."""
        if 8 <= self._hour <= 9 or 17 <= self._hour <= 18:
            return 1.00
        if 7 <= self._hour < 8 or 9 < self._hour <= 10 or 16 <= self._hour < 17 or 18 < self._hour <= 19:
            return 0.80
        if 12 <= self._hour <= 13:
            return 0.65
        if 22 <= self._hour or self._hour <= 5:
            return 0.18
        return 0.48

    def _control_factor(self) -> float:
        """Lower is better. RL gradually improves; baseline remains harsher."""
        if self.mode == "rl":
            return max(0.68, 0.88 - self._tick * 0.0025)
        return 1.34 + min(0.18, self._tick * 0.0007)

    def _simulate_phase(self) -> str:
        cycle_pos = self._tick % 72
        if cycle_pos < 33:
            return "NS-Green"
        if cycle_pos < 36:
            return "NS-Yellow"
        if cycle_pos < 69:
            return "EW-Green"
        return "EW-Yellow"

    def _maybe_advance_time(self) -> None:
        if self._tick % 12 == 0:
            self._hour = (self._hour + 0.25) % 24.0

    def activate_green_corridor(
        self,
        vehicle_id: str = "AMB_01",
        vehicle_type: str = "ambulance",
        origin: str = "J0_0",
        destination: str = "J3_3",
    ) -> Dict:
        """Create a realistic green-corridor event over a grid path."""
        if origin not in JUNCTION_IDS or destination not in JUNCTION_IDS:
            raise ValueError("origin/destination must be valid junction IDs")

        origin_idx = JUNCTION_IDS.index(origin)
        dest_idx = JUNCTION_IDS.index(destination)
        orow, ocol = divmod(origin_idx, 4)
        drow, dcol = divmod(dest_idx, 4)

        path = [origin]
        r, c = orow, ocol
        while c != dcol:
            c += 1 if dcol > c else -1
            path.append(f"J{r}_{c}")
        while r != drow:
            r += 1 if drow > r else -1
            path.append(f"J{r}_{c}")

        base_delay = self._estimate_crowd_delay(path)
        corridor_gain = 0.45 if self.mode == "rl" else 0.32
        after_delay = max(0.0, base_delay * (1.0 - corridor_gain))

        event = CorridorEvent(
            event_id=f"EC-{int(time.time())}-{self._tick:03d}",
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            origin=origin,
            destination=destination,
            path=path,
            activated_tick=self._tick,
            delay_before_s=round(base_delay, 2),
            delay_after_s=round(after_delay, 2),
            active=True,
        )
        self._corridor_event = event
        self._corridor_history.append(event)
        return {
            "event_id": event.event_id,
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "origin": origin,
            "destination": destination,
            "path": path,
            "delay_before_s": event.delay_before_s,
            "delay_after_s": event.delay_after_s,
            "active": True,
        }

    def _estimate_crowd_delay(self, path: List[str]) -> float:
        if not path:
            return 0.0
        avg_wait = 0.0
        for jid in path:
            avg_wait += self._junction_state[jid]["wait"]
        return avg_wait / len(path) * 1.5

    def _update_corridor(self, avg_wait: float) -> Optional[Dict]:
        if not self._corridor_event:
            # Trigger one corridor automatically for the demo so the dataset is not empty.
            if self._tick in (4, 5, 6):
                return self.activate_green_corridor()
            return None

        event = self._corridor_event
        active_window = 6
        event.active = (self._tick - event.activated_tick) <= active_window
        if not event.active:
            self._corridor_event = None
        return {
            "event_id": event.event_id,
            "vehicle_id": event.vehicle_id,
            "vehicle_type": event.vehicle_type,
            "origin": event.origin,
            "destination": event.destination,
            "path": event.path,
            "delay_before_s": event.delay_before_s,
            "delay_after_s": event.delay_after_s,
            "active": event.active,
        }

    def get_snapshot(self) -> Dict:
        """Return one time-step snapshot of realistic traffic metrics."""
        self._tick += 1
        self._maybe_advance_time()

        demand_factor = self._time_factor()
        control_factor = self._control_factor()
        phase = self._simulate_phase()

        queues: Dict[str, float] = {a: 0.0 for a in APPROACHES}
        waits: Dict[str, float] = {a: 0.0 for a in APPROACHES}
        total_queue = 0.0
        total_wait = 0.0
        total_throughput = 0.0

        junctions: Dict[str, Dict] = {}

        corridor = self._update_corridor(0.0)
        corridor_active_path = set(corridor["path"]) if corridor and corridor.get("active") else set()

        for idx, jid in enumerate(JUNCTION_IDS):
            row, col = divmod(idx, 4)
            state = self._junction_state[jid]
            weight = self._junction_weights[jid]
            lane_bias = self._junction_lanes[jid]

            # Random arrivals follow a Poisson-like process with uneven lane pressure.
            base_lambda = 2.6 + 5.2 * demand_factor * weight
            lane_arrivals = {
                lane: int(self._np_rng.poisson(max(0.15, base_lambda * bias * (0.85 + 0.25 * self._rng.random()))))
                for lane, bias in lane_bias.items()
            }

            # Different lane loads and ongoing queue accumulation.
            prev_queue = float(state["queue"])
            arrival_total = sum(lane_arrivals.values())
            queue_decay = 0.56 if self.mode == "rl" else 0.38
            new_queue = prev_queue * queue_decay + arrival_total * (0.85 if self.mode == "rl" else 1.10)

            imbalance = abs(lane_arrivals["north"] + lane_arrivals["south"] - lane_arrivals["east"] - lane_arrivals["west"])
            corridor_bonus = 0.65 if jid in corridor_active_path else 1.0

            # Waiting time is accumulated per junction, then softened by RL control.
            base_wait = (
                2.5
                + 1.8 * demand_factor
                + 0.18 * new_queue
                + 0.14 * imbalance
                + self._rng.uniform(-0.6, 1.4)
            )
            if self.mode == "rl":
                wait = prev_queue * 0.05 + base_wait * control_factor * corridor_bonus
            else:
                wait = prev_queue * 0.10 + base_wait * control_factor * 1.08

            # Keep values strictly non-zero and variable.
            wait = max(0.8, wait)
            new_queue = max(0.5, new_queue)

            # Throughput is higher when congestion is lower.
            depart_factor = 1.30 if self.mode == "rl" else 0.92
            throughput = max(0.0, (12.0 + 5.0 * demand_factor) * depart_factor * (1.0 - min(0.7, new_queue / 80.0)))

            # Update state for the next tick.
            state["wait"] = wait
            state["queue"] = new_queue * (0.90 if self.mode == "rl" else 0.97)
            state["throughput"] = throughput
            state["phase_age"] = (state["phase_age"] + 5) % 60

            junctions[jid] = {
                "junction_id": jid,
                "row": row,
                "col": col,
                "phase": phase,
                "mode": self.mode,
                "waiting_time": round(wait, 2),
                "queue_length": round(new_queue, 2),
                "throughput": round(throughput, 2),
                "lane_waiting": {
                    "north": round(wait * (0.95 + 0.05 * lane_bias["north"]), 2),
                    "south": round(wait * (0.92 + 0.05 * lane_bias["south"]), 2),
                    "east": round(wait * (0.90 + 0.05 * lane_bias["east"]), 2),
                    "west": round(wait * (0.88 + 0.05 * lane_bias["west"]), 2),
                },
                "vehicle_count": int(round(new_queue + arrival_total * 0.35)),
                "is_corridor": jid in corridor_active_path,
                "active_lane": "north_south" if phase.startswith("NS") else "east_west",
            }

            for lane in APPROACHES:
                queues[lane] += lane_arrivals[lane] + new_queue * 0.25
                waits[lane] += junctions[jid]["lane_waiting"][lane]

            total_queue += new_queue
            total_wait += wait
            total_throughput += throughput

        avg_wait = total_wait / len(JUNCTION_IDS)

        # Corridor event can now be refreshed with a meaningful delay reduction.
        corridor = self._update_corridor(avg_wait)

        # Slightly re-compute corridor-aware waits for active path entries.
        if corridor and corridor.get("active"):
            for jid in corridor.get("path", []):
                if jid in junctions:
                    before = junctions[jid]["waiting_time"]
                    after = max(0.5, before * (0.62 if self.mode == "rl" else 0.75))
                    junctions[jid]["waiting_time"] = round(after, 2)
                    total_wait += after - before

        # Advance queue/throughput aggregates after corridor adjustment.
        avg_wait = total_wait / len(JUNCTION_IDS)
        total_queue = sum(j["queue_length"] for j in junctions.values())
        total_throughput = sum(j["throughput"] for j in junctions.values())

        throughput = max(0.0, total_throughput)

        return {
            "tick": self._tick,
            "timestamp": time.time(),
            "hour": round(self._hour, 2),
            "mode": self.mode,
            "phase": phase,
            "time_factor": round(demand_factor, 3),
            "control_factor": round(control_factor, 3),
            "queues": queues,
            "waiting_times": waits,
            "total_queue": round(total_queue, 2),
            "avg_waiting_time": round(avg_wait, 2),
            "throughput": round(throughput, 2),
            "junctions": junctions,
            "corridor": corridor or {"active": False},
        }

    def get_history(self, n: int = 100) -> List[Dict]:
        """Generate n snapshots in sequence."""
        return [self.get_snapshot() for _ in range(n)]
