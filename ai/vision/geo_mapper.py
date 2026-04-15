"""
Geo Mapper for Camera Detections
================================
Maps pixel coordinates from camera frames to map coordinates using a
homography transform with CPU-only OpenCV.

If calibration points are not provided, it falls back to a deterministic
normalized mapping around the configured city center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RoadState:
    road_id: str
    vehicle_count: int
    density: float
    congestion: str


class RoadGeoMapper:
    """Convert frame points into latitude/longitude and road-level metrics."""

    def __init__(
        self,
        city_center_lat: float,
        city_center_lon: float,
        frame_width: int = 1280,
        frame_height: int = 720,
        span_lat: float = 0.012,
        span_lon: float = 0.016,
        image_points: Optional[Sequence[Sequence[float]]] = None,
        geo_points: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.city_center_lat = float(city_center_lat)
        self.city_center_lon = float(city_center_lon)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.span_lat = float(span_lat)
        self.span_lon = float(span_lon)

        self._H = None
        if image_points and geo_points and len(image_points) >= 4 and len(geo_points) >= 4:
            try:
                import cv2  # type: ignore

                src = np.array(image_points[:4], dtype=np.float32)
                dst = np.array([[gp[1], gp[0]] for gp in geo_points[:4]], dtype=np.float32)
                H, _ = cv2.findHomography(src, dst, method=0)
                self._H = H
            except Exception:
                self._H = None

    def map_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Return (lat, lon) for an image pixel."""
        if self._H is not None:
            p = np.array([x, y, 1.0], dtype=np.float64)
            q = self._H.dot(p)
            if abs(q[2]) > 1e-6:
                lon = float(q[0] / q[2])
                lat = float(q[1] / q[2])
                return lat, lon

        # Fallback normalized mapping over a local bounding box.
        nx = max(0.0, min(1.0, float(x) / max(1.0, self.frame_width)))
        ny = max(0.0, min(1.0, float(y) / max(1.0, self.frame_height)))
        lon = self.city_center_lon + (nx - 0.5) * self.span_lon
        # invert Y so top is north
        lat = self.city_center_lat + (0.5 - ny) * self.span_lat
        return lat, lon

    def lane_name(self, x: float) -> str:
        nx = max(0.0, min(1.0, float(x) / max(1.0, self.frame_width)))
        if nx < 0.25:
            return "west"
        if nx < 0.5:
            return "south"
        if nx < 0.75:
            return "north"
        return "east"

    def summarize_roads(self, lane_counts: Dict[str, int], lane_capacity: int = 18) -> List[RoadState]:
        roads: List[RoadState] = []
        for lane in ("north", "south", "east", "west"):
            count = int(lane_counts.get(lane, 0))
            density = min(1.0, count / float(max(1, lane_capacity)))
            if density < 0.33:
                congestion = "low"
            elif density < 0.66:
                congestion = "medium"
            else:
                congestion = "high"
            roads.append(
                RoadState(
                    road_id=f"{lane}_arterial",
                    vehicle_count=count,
                    density=density,
                    congestion=congestion,
                )
            )
        return roads
