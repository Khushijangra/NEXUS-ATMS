"""
NEXUS-ATMS Emergency Corridor Engine
=====================================
When an emergency vehicle is detected, this module:
1. Finds the optimal path to destination using A*
2. Pre-clears all signals along the route to GREEN
3. Tracks the vehicle's progress along the corridor
4. Auto-reverts signals after the vehicle passes
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class EmergencyEvent:
    """Tracks a single emergency corridor activation."""
    event_id: str
    vehicle_type: str           # ambulance, fire_truck, police
    vehicle_id: str
    origin: str                 # source junction ID
    destination: str            # target junction ID
    path: List[str] = field(default_factory=list)      # junction IDs in order
    activated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    cleared_junctions: List[str] = field(default_factory=list)
    status: str = "active"      # active | completed | timeout
    estimated_time_s: float = 0.0
    actual_time_s: float = 0.0

    @property
    def time_saved_s(self) -> float:
        if self.actual_time_s > 0 and self.estimated_time_s > 0:
            return max(0.0, self.estimated_time_s * 1.6 - self.actual_time_s)
        return 0.0


class EmergencyCorridorEngine:
    """
    Autonomous emergency corridor generator using graph-based path planning.

    The city's road network is modelled as a weighted directed graph.
    Edge weights reflect real-time congestion levels so the corridor
    picks the fastest path, not just the shortest.
    """

    def __init__(
        self,
        cascade_lookahead: int = 4,
        corridor_timeout: int = 300,
    ) -> None:
        self.cascade_lookahead = cascade_lookahead
        self.corridor_timeout = corridor_timeout

        # City road graph: nodes = junctions, edges = road segments
        self.graph: nx.DiGraph = nx.DiGraph()
        self._active_events: Dict[str, EmergencyEvent] = {}
        self._completed_events: List[EmergencyEvent] = []
        self._event_counter = 0

        # Signal overrides: junction_id -> original phase
        self._overridden_signals: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------

    def build_grid_graph(self, rows: int = 4, cols: int = 4, spacing: float = 200.0):
        """
        Build a 4×4 grid graph matching the SUMO network.
        Edge weights = distance (updated with congestion later).
        """
        self.graph.clear()
        for r in range(rows):
            for c in range(cols):
                jid = f"J{r}_{c}"
                self.graph.add_node(jid, row=r, col=c, x=c * spacing, y=r * spacing)

        # Horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                a, b = f"J{r}_{c}", f"J{r}_{c+1}"
                self.graph.add_edge(a, b, weight=spacing, base_weight=spacing)
                self.graph.add_edge(b, a, weight=spacing, base_weight=spacing)

        # Vertical edges
        for c in range(cols):
            for r in range(rows - 1):
                a, b = f"J{r}_{c}", f"J{r+1}_{c}"
                self.graph.add_edge(a, b, weight=spacing, base_weight=spacing)
                self.graph.add_edge(b, a, weight=spacing, base_weight=spacing)

        logger.info(f"[Emergency] Built {rows}x{cols} city graph: "
                     f"{self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")

    def update_edge_weights(self, congestion_map: Dict[str, float]):
        """
        Update edge weights based on real-time congestion.

        Parameters
        ----------
        congestion_map : dict
            Maps "J0_0->J0_1" style keys to congestion factor (1.0 = free, 3.0 = jammed).
        """
        for edge_key, factor in congestion_map.items():
            parts = edge_key.split("->")
            if len(parts) == 2 and self.graph.has_edge(parts[0], parts[1]):
                base = self.graph[parts[0]][parts[1]]["base_weight"]
                self.graph[parts[0]][parts[1]]["weight"] = base * max(1.0, factor)

    # ------------------------------------------------------------------
    # Corridor Activation
    # ------------------------------------------------------------------

    def activate_corridor(
        self,
        vehicle_id: str,
        vehicle_type: str,
        origin: str,
        destination: str,
    ) -> Optional[EmergencyEvent]:
        """
        Activate an emergency corridor from origin to destination.

        Returns the EmergencyEvent with the planned path, or None if
        no path exists.
        """
        if origin not in self.graph or destination not in self.graph:
            logger.error(f"[Emergency] Invalid origin/destination: {origin} -> {destination}")
            return None

        try:
            path = nx.astar_path(
                self.graph, origin, destination,
                heuristic=self._heuristic,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            logger.error(f"[Emergency] No path found: {origin} -> {destination}")
            return None

        self._event_counter += 1
        event = EmergencyEvent(
            event_id=f"EC-{int(time.time())}-{self._event_counter:03d}",
            vehicle_type=vehicle_type,
            vehicle_id=vehicle_id,
            origin=origin,
            destination=destination,
            path=path,
            estimated_time_s=self._estimate_travel_time(path),
        )

        self._active_events[event.event_id] = event
        logger.info(f"[Emergency] Corridor {event.event_id} ACTIVATED: "
                     f"{vehicle_type} | {origin} -> {destination} | "
                     f"Path: {' -> '.join(path)} | ETA: {event.estimated_time_s:.0f}s")

        # Pre-clear signals along the corridor
        self._cascade_clear(event)
        return event

    def _cascade_clear(self, event: EmergencyEvent):
        """Pre-clear ahead junctions to green for the corridor direction."""
        for jid in event.path[:self.cascade_lookahead]:
            if jid not in self._overridden_signals:
                self._overridden_signals[jid] = -1  # Store original phase later
            event.cleared_junctions.append(jid)
            logger.info(f"  [Emergency] Junction {jid} -> GREEN for corridor")

    def get_corridor_signal_overrides(self) -> Dict[str, str]:
        """
        Get current signal overrides for all active corridors.

        Returns
        -------
        dict : junction_id -> "green_direction" (e.g., "NS" or "EW")
        """
        overrides = {}
        for event in self._active_events.values():
            if event.status != "active":
                continue
            for i, jid in enumerate(event.path[:-1]):
                next_jid = event.path[i + 1]
                direction = self._infer_direction(jid, next_jid)
                overrides[jid] = direction
        return overrides

    # ------------------------------------------------------------------
    # Vehicle Progress Tracking
    # ------------------------------------------------------------------

    def update_vehicle_position(self, event_id: str, current_junction: str):
        """Update the position of the emergency vehicle along the corridor."""
        event = self._active_events.get(event_id)
        if not event or event.status != "active":
            return

        if current_junction in event.path:
            idx = event.path.index(current_junction)
            # Revert signals behind the vehicle
            for jid in event.cleared_junctions[:]:
                if event.path.index(jid) < idx:
                    self._revert_signal(jid)
                    event.cleared_junctions.remove(jid)

            # Pre-clear more junctions ahead
            ahead_start = idx + 1
            ahead_end = min(idx + 1 + self.cascade_lookahead, len(event.path))
            for jid in event.path[ahead_start:ahead_end]:
                if jid not in event.cleared_junctions:
                    event.cleared_junctions.append(jid)
                    self._overridden_signals[jid] = -1
                    logger.info(f"  [Emergency] Pre-clearing {jid}")

            # Check if vehicle reached destination
            if current_junction == event.destination:
                self._complete_event(event)

    def _complete_event(self, event: EmergencyEvent):
        """Mark corridor as completed and revert all signals."""
        event.status = "completed"
        event.completed_at = time.time()
        event.actual_time_s = event.completed_at - event.activated_at

        # Revert all cleared signals
        for jid in event.cleared_junctions:
            self._revert_signal(jid)
        event.cleared_junctions.clear()

        del self._active_events[event.event_id]
        self._completed_events.append(event)

        logger.info(f"[Emergency] Corridor {event.event_id} COMPLETED: "
                     f"actual={event.actual_time_s:.0f}s | "
                     f"saved≈{event.time_saved_s:.0f}s")

    def check_timeouts(self):
        """Check and auto-revert timed-out corridors."""
        now = time.time()
        for eid, event in list(self._active_events.items()):
            if now - event.activated_at > self.corridor_timeout:
                event.status = "timeout"
                event.completed_at = now
                event.actual_time_s = now - event.activated_at
                for jid in event.cleared_junctions:
                    self._revert_signal(jid)
                event.cleared_junctions.clear()
                del self._active_events[eid]
                self._completed_events.append(event)
                logger.warning(f"[Emergency] Corridor {eid} TIMED OUT after {self.corridor_timeout}s")

    def _revert_signal(self, junction_id: str):
        """Revert a junction's signal to normal AI control."""
        if junction_id in self._overridden_signals:
            del self._overridden_signals[junction_id]
            logger.info(f"  [Emergency] Junction {junction_id} -> reverted to AI control")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _heuristic(self, u: str, v: str) -> float:
        """A* heuristic: Euclidean distance between junctions."""
        n1 = self.graph.nodes[u]
        n2 = self.graph.nodes[v]
        return ((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2) ** 0.5

    def _infer_direction(self, from_jid: str, to_jid: str) -> str:
        """Infer corridor direction from junction pair."""
        fn = self.graph.nodes[from_jid]
        tn = self.graph.nodes[to_jid]
        dx = tn.get("col", 0) - fn.get("col", 0)
        dy = tn.get("row", 0) - fn.get("row", 0)
        if abs(dy) > abs(dx):
            return "NS"
        return "EW"

    def _estimate_travel_time(self, path: List[str]) -> float:
        """Estimate travel time for the path (seconds)."""
        total = 0.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i + 1]):
                total += self.graph[path[i]][path[i + 1]]["weight"]
        # Convert distance to time: assume emergency speed ~60 km/h = 16.67 m/s
        return total / 16.67

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_active_events(self) -> List[Dict]:
        """Get all active corridor events as dicts."""
        return [
            {
                "event_id": e.event_id,
                "vehicle_type": e.vehicle_type,
                "vehicle_id": e.vehicle_id,
                "origin": e.origin,
                "destination": e.destination,
                "path": e.path,
                "cleared_junctions": e.cleared_junctions,
                "elapsed_s": time.time() - e.activated_at,
                "estimated_time_s": e.estimated_time_s,
                "status": e.status,
            }
            for e in self._active_events.values()
        ]

    def get_completed_events(self) -> List[Dict]:
        """Get all completed corridor events as dicts."""
        return [
            {
                "event_id": e.event_id,
                "vehicle_type": e.vehicle_type,
                "origin": e.origin,
                "destination": e.destination,
                "actual_time_s": e.actual_time_s,
                "time_saved_s": e.time_saved_s,
                "status": e.status,
            }
            for e in self._completed_events
        ]

    def get_stats(self) -> Dict:
        """Get overall emergency corridor statistics."""
        completed = self._completed_events
        total_saved = sum(e.time_saved_s for e in completed)
        return {
            "active_corridors": len(self._active_events),
            "completed_today": len(completed),
            "total_time_saved_s": total_saved,
            "avg_response_time_s": (
                sum(e.actual_time_s for e in completed) / len(completed)
                if completed else 0.0
            ),
            "overridden_junctions": list(self._overridden_signals.keys()),
        }
