"""
NEXUS-ATMS Road Maintenance AI
================================
Detects potential road damage (potholes) by analysing patterns of
abnormal vehicle braking at the same location, then generates
maintenance work orders with GPS coordinates.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BrakingEvent:
    """A single hard-braking event."""
    vehicle_id: str
    timestamp: float
    location: Tuple[float, float]   # (x, y) metres or (lat, lon)
    deceleration: float             # m/s² (negative = braking)
    speed_before: float             # m/s


@dataclass
class MaintenanceWorkOrder:
    """Auto-generated road maintenance work order."""
    order_id: str
    created_at: float
    location: Tuple[float, float]
    confidence: float               # 0-1 confidence of road damage
    num_events: int
    avg_deceleration: float
    severity: str                   # LOW, MEDIUM, HIGH
    status: str = "open"            # open, dispatched, resolved
    description: str = ""
    geojson: Optional[Dict] = None


class RoadMaintenanceAI:
    """
    Analyses vehicle braking patterns to predict road damage locations.

    When multiple vehicles brake hard at the same spot repeatedly,
    the system flags it as a potential pothole/damage and generates
    a maintenance work order.
    """

    def __init__(
        self,
        hard_braking_threshold: float = -4.0,   # m/s²
        cluster_radius: float = 20.0,            # metres
        min_events_for_flag: int = 5,
        days_lookback: int = 7,
    ):
        self.threshold = hard_braking_threshold
        self.cluster_radius = cluster_radius
        self.min_events = min_events_for_flag
        self.lookback_days = days_lookback

        self._braking_events: List[BrakingEvent] = []
        self._work_orders: List[MaintenanceWorkOrder] = []
        self._order_counter = 0

        # Spatial index: grid cell -> list of events
        self._grid_cell_size = cluster_radius
        self._spatial_grid: Dict[Tuple[int, int], List[BrakingEvent]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Event Recording
    # ------------------------------------------------------------------

    def record_braking(
        self,
        vehicle_id: str,
        location: Tuple[float, float],
        deceleration: float,
        speed_before: float,
    ):
        """Record a braking event if it exceeds the hard-braking threshold."""
        if deceleration > self.threshold:  # Not hard enough
            return

        event = BrakingEvent(
            vehicle_id=vehicle_id,
            timestamp=time.time(),
            location=location,
            deceleration=deceleration,
            speed_before=speed_before,
        )
        self._braking_events.append(event)

        # Add to spatial grid
        cell = self._location_to_cell(location)
        self._spatial_grid[cell].append(event)

        # Check if this cluster now exceeds threshold
        self._check_cluster(cell)

    def _location_to_cell(self, loc: Tuple[float, float]) -> Tuple[int, int]:
        """Convert (x, y) to grid cell index."""
        return (
            int(loc[0] // self._grid_cell_size),
            int(loc[1] // self._grid_cell_size),
        )

    # ------------------------------------------------------------------
    # Cluster Analysis
    # ------------------------------------------------------------------

    def _check_cluster(self, cell: Tuple[int, int]):
        """Check if the events in a grid cell warrant a work order."""
        cutoff = time.time() - self.lookback_days * 86400
        events = [e for e in self._spatial_grid[cell] if e.timestamp > cutoff]

        if len(events) < self.min_events:
            return

        # Check if we already have an open work order for this cell
        cell_center = (
            (cell[0] + 0.5) * self._grid_cell_size,
            (cell[1] + 0.5) * self._grid_cell_size,
        )
        for wo in self._work_orders:
            if wo.status == "open" and self._distance(wo.location, cell_center) < self.cluster_radius:
                return  # Already flagged

        # Generate work order
        avg_decel = sum(e.deceleration for e in events) / len(events)
        confidence = min(1.0, len(events) / (self.min_events * 3))

        if avg_decel < -6.0:
            severity = "HIGH"
        elif avg_decel < -5.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        self._order_counter += 1
        order = MaintenanceWorkOrder(
            order_id=f"WO-{datetime.now().strftime('%Y')}-{self._order_counter:04d}",
            created_at=time.time(),
            location=cell_center,
            confidence=round(confidence, 2),
            num_events=len(events),
            avg_deceleration=round(avg_decel, 2),
            severity=severity,
            description=(
                f"Repeated hard braking detected at ({cell_center[0]:.0f}, "
                f"{cell_center[1]:.0f}). {len(events)} events in "
                f"{self.lookback_days} days. Possible road damage."
            ),
            geojson=self._to_geojson(cell_center, severity, len(events)),
        )
        self._work_orders.append(order)
        logger.info(f"[Maintenance] Work order {order.order_id} created: "
                     f"{severity} | {len(events)} events | "
                     f"location=({cell_center[0]:.0f}, {cell_center[1]:.0f})")

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def _to_geojson(
        self,
        location: Tuple[float, float],
        severity: str,
        num_events: int,
    ) -> Dict:
        """Convert a flagged location to GeoJSON format."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [location[0], location[1]],
            },
            "properties": {
                "severity": severity,
                "num_events": num_events,
                "type": "road_damage_predicted",
            },
        }

    # ------------------------------------------------------------------
    # SUMO Integration
    # ------------------------------------------------------------------

    def process_sumo_vehicles(self, vehicle_data: List[Dict]):
        """
        Process vehicle data from SUMO simulation.

        Each vehicle dict should have: id, x, y, speed, acceleration
        """
        for v in vehicle_data:
            accel = v.get("acceleration", 0.0)
            if accel < self.threshold:
                self.record_braking(
                    vehicle_id=v["id"],
                    location=(v["x"], v["y"]),
                    deceleration=accel,
                    speed_before=v.get("speed", 0.0),
                )

    # ------------------------------------------------------------------
    # Work Order Management
    # ------------------------------------------------------------------

    def get_open_orders(self) -> List[Dict]:
        """Get all open maintenance work orders."""
        return [
            {
                "order_id": wo.order_id,
                "created_at": wo.created_at,
                "location": wo.location,
                "confidence": wo.confidence,
                "num_events": wo.num_events,
                "avg_deceleration": wo.avg_deceleration,
                "severity": wo.severity,
                "status": wo.status,
                "description": wo.description,
            }
            for wo in self._work_orders if wo.status == "open"
        ]

    def resolve_order(self, order_id: str):
        """Mark a work order as resolved."""
        for wo in self._work_orders:
            if wo.order_id == order_id:
                wo.status = "resolved"
                logger.info(f"[Maintenance] Work order {order_id} RESOLVED")
                return True
        return False

    def export_geojson(self, output_path: str = "reports/road_damage.geojson") -> str:
        """Export all flagged locations as a GeoJSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        features = [wo.geojson for wo in self._work_orders if wo.geojson]
        collection = {
            "type": "FeatureCollection",
            "features": features,
        }
        with open(output_path, "w") as f:
            json.dump(collection, f, indent=2)
        logger.info(f"[Maintenance] GeoJSON exported: {output_path} ({len(features)} points)")
        return output_path

    def get_stats(self) -> Dict:
        """Get road maintenance statistics."""
        return {
            "total_braking_events": len(self._braking_events),
            "total_work_orders": len(self._work_orders),
            "open_orders": sum(1 for wo in self._work_orders if wo.status == "open"),
            "resolved_orders": sum(1 for wo in self._work_orders if wo.status == "resolved"),
            "high_severity": sum(1 for wo in self._work_orders if wo.severity == "HIGH"),
            "monitored_cells": len(self._spatial_grid),
        }
