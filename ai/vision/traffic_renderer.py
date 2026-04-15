"""
Synthetic Traffic Scene Renderer
=================================
Generates realistic-looking road/intersection frames with animated vehicles
so the dashboard has something to show when no real traffic camera is available.

Returns (frame_bgr, detections) tuples compatible with VehicleDetector output.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vehicle colours (BGR)
# ---------------------------------------------------------------------------
_CAR_COLORS = [
    (60, 60, 180),    # red
    (40, 120, 220),   # orange
    (20, 200, 200),   # yellow
    (180, 180, 180),  # silver
    (40, 40, 40),     # black
    (220, 220, 220),  # white
    (100, 160, 40),   # green
    (200, 110, 50),   # blue
    (80, 80, 160),    # dark-red
    (140, 200, 240),  # cream
]

_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class VehicleSim:
    """One simulated vehicle moving along a lane."""
    x: float
    y: float
    speed: float             # pixels per tick
    angle: float             # radians (0 = right, pi/2 = down)
    vw: int                  # vehicle width  (on screen)
    vh: int                  # vehicle height (on screen)
    color: Tuple[int, int, int]
    roof_color: Tuple[int, int, int]
    class_id: int            # 2=car, 5=bus, 7=truck
    label: str
    track_id: int
    conf: float
    lane_id: int
    spawn_min: Tuple[float, float]
    spawn_max: Tuple[float, float]

    def move(self) -> None:
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

    def out_of_bounds(self, w: int, h: int, margin: int = 60) -> bool:
        return (
            self.x < -margin or self.x > w + margin
            or self.y < -margin or self.y > h + margin
        )

    def bbox(self) -> Tuple[int, int, int, int]:
        cx, cy = int(self.x), int(self.y)
        hx, hy = self.vw // 2, self.vh // 2
        return (cx - hx, cy - hy, cx + hx, cy + hy)


class TrafficRenderer:
    """
    Renders a top-down intersection with 4 approach lanes and animated vehicles.

    Usage
    -----
    renderer = TrafficRenderer(width=1280, height=720)
    frame, detections = renderer.render()   # call every ~100 ms
    """

    # Road geometry
    ROAD_W = 120       # half-width of each road arm (pixels)
    BOX_W = 180        # intersection box half-width

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 10.0,
        seed: int = 42,
    ) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise RuntimeError("opencv-python is required for TrafficRenderer")

        self.W = width
        self.H = height
        self.fps = fps
        self.cx = width // 2
        self.cy = height // 2
        self._rng = random.Random(seed)
        self._tick = 0
        self._vehicles: List[VehicleSim] = []
        self._next_id = 1

        self._phase = "NS_GREEN"   # current signal phase
        self._phase_ticks = 0

        self._bg = self._build_background()
        self._spawn_initial()

    # ------------------------------------------------------------------
    # Background (built once)
    # ------------------------------------------------------------------

    def _build_background(self) -> np.ndarray:
        import cv2

        bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Grass / ground
        bg[:] = (45, 80, 45)

        cx, cy = self.cx, self.cy
        rw = self.ROAD_W
        bx = self.BOX_W

        # Roads (dark asphalt)
        road_color = (50, 50, 50)
        cv2.rectangle(bg, (cx - rw, 0),            (cx + rw, self.H),  road_color, -1)
        cv2.rectangle(bg, (0, cy - rw),             (self.W, cy + rw),  road_color, -1)
        cv2.rectangle(bg, (cx - bx, cy - bx),      (cx + bx, cy + bx), road_color, -1)

        # Centre-line dashes (white)
        dash_color = (200, 200, 200)
        dash_len, gap = 28, 18
        # Vertical road
        y = 0
        while y < cy - bx:
            cv2.line(bg, (cx, y), (cx, y + dash_len), dash_color, 2)
            y += dash_len + gap
        y = cy + bx
        while y < self.H:
            cv2.line(bg, (cx, y), (cx, y + dash_len), dash_color, 2)
            y += dash_len + gap
        # Horizontal road
        x = 0
        while x < cx - bx:
            cv2.line(bg, (x, cy), (x + dash_len, cy), dash_color, 2)
            x += dash_len + gap
        x = cx + bx
        while x < self.W:
            cv2.line(bg, (x, cy), (x + dash_len, cy), dash_color, 2)
            x += dash_len + gap

        # Stop lines
        stop_color = (255, 255, 255)
        cv2.line(bg, (cx - rw, cy - bx), (cx - 4, cy - bx), stop_color, 4)    # south-facing
        cv2.line(bg, (cx + 4, cy + bx), (cx + rw, cy + bx), stop_color, 4)    # north-facing
        cv2.line(bg, (cx - bx, cy + 4), (cx - bx, cy + rw), stop_color, 4)    # east-facing
        cv2.line(bg, (cx + bx, cy - rw), (cx + bx, cy - 4), stop_color, 4)    # west-facing

        # Pavement edges (lighter grey border on road)
        kerb = (100, 100, 100)
        cv2.rectangle(bg, (cx - rw, 0),  (cx - rw + 4, self.H), kerb, -1)
        cv2.rectangle(bg, (cx + rw - 4, 0), (cx + rw, self.H), kerb, -1)
        cv2.rectangle(bg, (0, cy - rw),  (self.W, cy - rw + 4), kerb, -1)
        cv2.rectangle(bg, (0, cy + rw - 4), (self.W, cy + rw), kerb, -1)

        # Building silhouettes (corners of intersection)
        self._draw_buildings(bg, cx, cy, bx, rw)

        return bg

    def _draw_buildings(self, bg, cx, cy, bx, rw):
        import cv2
        corners = [
            (0, 0, cx - rw, cy - rw),
            (cx + rw, 0, self.W, cy - rw),
            (0, cy + rw, cx - rw, self.H),
            (cx + rw, cy + rw, self.W, self.H),
        ]
        bld_colors = [(60, 70, 80), (55, 65, 75), (65, 75, 85), (58, 68, 78)]
        for (x1, y1, x2, y2), bc in zip(corners, bld_colors):
            rng = random.Random(x1 + y1)
            # fill block
            cv2.rectangle(bg, (x1, y1), (x2, y2), bc, -1)
            # window grid
            for wy in range(y1 + 10, y2 - 5, 20):
                for wx in range(x1 + 10, x2 - 5, 18):
                    if rng.random() > 0.3:
                        win_c = (200, 230, 255) if rng.random() > 0.4 else (20, 30, 10)
                        cv2.rectangle(bg, (wx, wy), (wx + 10, wy + 12), win_c, -1)

    # ------------------------------------------------------------------
    # Lane definitions (cx, cy relative; angle in radians)
    # ------------------------------------------------------------------

    def _lanes(self):
        cx, cy = self.cx, self.cy
        rw = self.ROAD_W
        bx = self.BOX_W
        offset = rw // 2  # lane offset from centre line

        # (spawn_x, spawn_y, dest_x, dest_y, angle, direction)
        return [
            # North arm → going south (downward)
            dict(sx=cx - offset, sy=-50,         angle=math.pi / 2,   lid=0),
            # South arm → going north (upward)
            dict(sx=cx + offset, sy=self.H + 50, angle=-math.pi / 2,  lid=1),
            # West arm → going east (rightward)
            dict(sx=-50,          sy=cy - offset, angle=0.0,           lid=2),
            # East arm → going west (leftward)
            dict(sx=self.W + 50,  sy=cy + offset, angle=math.pi,       lid=3),
        ]

    # ------------------------------------------------------------------
    # Vehicle spawning
    # ------------------------------------------------------------------

    def _make_vehicle(self, lane: dict) -> VehicleSim:
        rng = self._rng
        cls_id = rng.choices([2, 5, 7], weights=[0.75, 0.15, 0.10])[0]
        if cls_id == 2:    # car
            vw, vh = rng.randint(38, 52), rng.randint(22, 32)
            speed = rng.uniform(2.5, 4.5)
        elif cls_id == 5:  # bus
            vw, vh = rng.randint(60, 80), rng.randint(28, 36)
            speed = rng.uniform(1.5, 2.5)
        else:              # truck
            vw, vh = rng.randint(55, 70), rng.randint(28, 34)
            speed = rng.uniform(1.8, 3.0)

        color = rng.choice(_CAR_COLORS)
        roof_r = max(0, color[0] - 40)
        roof_g = max(0, color[1] - 40)
        roof_b = max(0, color[2] - 40)

        tid = self._next_id
        self._next_id += 1

        return VehicleSim(
            x=float(lane["sx"]),
            y=float(lane["sy"]),
            speed=speed,
            angle=lane["angle"],
            vw=vw, vh=vh,
            color=color,
            roof_color=(roof_r, roof_g, roof_b),
            class_id=cls_id,
            label={2: "car", 5: "bus", 7: "truck"}[cls_id],
            track_id=tid,
            conf=round(rng.uniform(0.72, 0.98), 2),
            lane_id=lane["lid"],
            spawn_min=(float(lane["sx"]) - 20, float(lane["sy"]) - 20),
            spawn_max=(float(lane["sx"]) + 20, float(lane["sy"]) + 20),
        )

    def _spawn_initial(self) -> None:
        lanes = self._lanes()
        for lane in lanes:
            # Stagger initial positions along each lane
            for offset_steps in range(0, 5):
                v = self._make_vehicle(lane)
                v.x += offset_steps * 90 * math.cos(lane["angle"])
                v.y += offset_steps * 90 * math.sin(lane["angle"])
                self._vehicles.append(v)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def render(self):
        """Return (annotated_frame_bgr:np.ndarray, detections:list)."""
        import cv2
        from ai.vision.detector import Detection   # type: ignore

        self._tick += 1

        # Update signal phase roughly every 5 seconds
        self._phase_ticks += 1
        if self._phase_ticks > int(self.fps * 5):
            self._phase_ticks = 0
            phases = ["NS_GREEN", "EW_GREEN", "YELLOW", "ALL_RED"]
            idx = (phases.index(self._phase) + 1) % len(phases)
            self._phase = phases[idx]

        # Move vehicles
        for v in self._vehicles:
            # Simple stop-at-red logic
            stopping = self._should_stop(v)
            if not stopping:
                v.move()

        # Remove out-of-bounds vehicles and spawn replacements
        lanes = self._lanes()
        alive = [v for v in self._vehicles if not v.out_of_bounds(self.W, self.H)]
        dead_lanes = set(range(len(lanes))) - {v.lane_id for v in alive}
        for lid in dead_lanes:
            alive.append(self._make_vehicle(lanes[lid]))
        # Also stochastically spawn extra to keep density up
        if self._rng.random() < 0.12 and len(alive) < 20:
            alive.append(self._make_vehicle(self._rng.choice(lanes)))
        self._vehicles = alive

        # Draw frame
        frame = self._bg.copy()
        self._draw_signals(frame)

        detections = []
        for v in self._vehicles:
            x1, y1, x2, y2 = v.bbox()
            # Skip vehicles fully inside intersection box (too close to stop-line)
            if (self.cx - self.BOX_W < (x1 + x2) // 2 < self.cx + self.BOX_W and
                    self.cy - self.BOX_W < (y1 + y2) // 2 < self.cy + self.BOX_W):
                continue
            self._draw_vehicle(frame, v)
            detections.append(Detection(
                track_id=v.track_id,
                class_id=v.class_id,
                label=v.label,
                confidence=v.conf,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))

        # HUD overlay
        self._draw_hud(frame, detections)

        return frame, detections

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _should_stop(self, v: VehicleSim) -> bool:
        """Return True if vehicle should stop at red light."""
        cx, cy = self.cx, self.cy
        bx = self.BOX_W + 10
        # NS lanes (lid 0,1)
        if v.lane_id == 0:  # coming from north, going south
            if self._phase not in ("NS_GREEN",) and (cy - bx - 40) < v.y < (cy - bx + 5):
                return True
        elif v.lane_id == 1:  # coming from south, going north
            if self._phase not in ("NS_GREEN",) and (cy + bx - 5) < v.y < (cy + bx + 40):
                return True
        elif v.lane_id == 2:  # from west, going east
            if self._phase not in ("EW_GREEN",) and (cx - bx - 40) < v.x < (cx - bx + 5):
                return True
        elif v.lane_id == 3:  # from east, going west
            if self._phase not in ("EW_GREEN",) and (cx + bx - 5) < v.x < (cx + bx + 40):
                return True
        return False

    def _draw_vehicle(self, frame: np.ndarray, v: VehicleSim) -> None:
        import cv2
        x1, y1, x2, y2 = v.bbox()
        # Clamp to frame
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(self.W - 1, x2), min(self.H - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            return

        # Body
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), v.color, -1)
        # Roof / cabin (central 50%)
        rw = (x2c - x1c)
        rh = (y2c - y1c)
        rx1 = x1c + rw // 4
        rx2 = x2c - rw // 4
        ry1 = y1c + rh // 5
        ry2 = y2c - rh // 5
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), v.roof_color, -1)
        # Windscreen glare
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry1 + (ry2 - ry1) // 3), (200, 220, 240), -1)
        # Vehicle outline
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (20, 20, 20), 1)
        # Headlights / tail-lights
        hl_color = (255, 255, 230) if v.lane_id in (0, 2) else (30, 30, 200)
        hl_size = 3
        if v.lane_id == 0:    # going south: headlights at bottom
            cv2.circle(frame, (x1c + 4, y2c - 3), hl_size, hl_color, -1)
            cv2.circle(frame, (x2c - 4, y2c - 3), hl_size, hl_color, -1)
        elif v.lane_id == 1:  # going north: headlights at top
            cv2.circle(frame, (x1c + 4, y1c + 3), hl_size, hl_color, -1)
            cv2.circle(frame, (x2c - 4, y1c + 3), hl_size, hl_color, -1)
        elif v.lane_id == 2:  # going east: headlights at right
            cv2.circle(frame, (x2c - 3, y1c + 4), hl_size, hl_color, -1)
            cv2.circle(frame, (x2c - 3, y2c - 4), hl_size, hl_color, -1)
        elif v.lane_id == 3:  # going west: headlights at left
            cv2.circle(frame, (x1c + 3, y1c + 4), hl_size, hl_color, -1)
            cv2.circle(frame, (x1c + 3, y2c - 4), hl_size, hl_color, -1)

    def _draw_signals(self, frame: np.ndarray) -> None:
        import cv2
        cx, cy = self.cx, self.cy
        bx = self.BOX_W

        phase = self._phase
        ns_green = phase == "NS_GREEN"
        ew_green = phase == "EW_GREEN"
        yellow = phase == "YELLOW"

        def _sig(pos, green):
            x, y = pos
            cv2.circle(frame, (x, y), 8, (30, 30, 30), -1)
            if yellow:
                col = (0, 200, 255)
            elif green:
                col = (0, 220, 50)
            else:
                col = (30, 30, 220)
            cv2.circle(frame, (x, y), 6, col, -1)

        _sig((cx - bx - 10, cy - bx - 10), ns_green)   # NW corner → N/S
        _sig((cx + bx + 10, cy + bx + 10), ns_green)   # SE corner → N/S
        _sig((cx - bx - 10, cy + bx + 10), ew_green)   # SW corner → E/W
        _sig((cx + bx + 10, cy - bx - 10), ew_green)   # NE corner → E/W

    def _draw_hud(self, frame: np.ndarray, detections: list) -> None:
        import cv2
        # Annotate each detection
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 60), 2)
            label_txt = f"ID{d.track_id} {d.label} {d.confidence:.2f}"
            cv2.putText(frame, label_txt, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)

        n = len(detections)
        ts = time.strftime("%H:%M:%S")
        cv2.rectangle(frame, (0, 0), (340, 44), (0, 0, 0), -1)
        cv2.putText(frame, f"NEXUS-ATMS  |  {ts}  |  Vehicles: {n}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1)
        cv2.putText(frame, f"YOLO  Phase: {self._phase}  Detecting...",
                    (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 60), 1)
