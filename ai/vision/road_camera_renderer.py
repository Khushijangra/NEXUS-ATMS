"""
Road Camera Renderer — NEXUS-ATMS
===================================
Generates realistic CCTV traffic camera footage: perspective road view with
animated mixed traffic (cars, buses, trucks, autos, motorbikes) and YOLO-style
detection overlays.  Designed to look like actual Indian road monitoring.

Usage:
    renderer = RoadCameraRenderer(direction="north", junction_id="J1_1")
    frame, detections = renderer.render()   # call at 10 fps
"""

from __future__ import annotations

import math
import random
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Frame dimensions
# ---------------------------------------------------------------------------
W, H = 640, 480

# ---------------------------------------------------------------------------
# Perspective geometry  (camera mounted ~6 m high, looking down the approach)
# ---------------------------------------------------------------------------
VP_X = W // 2           # vanishing-point x (always screen centre)
VP_Y = int(H * 0.39)    # vanishing-point y = horizon line

# Road edges at horizon  (narrow)
RD_TOP_L = VP_X - 88
RD_TOP_R = VP_X + 88

# Road edges at bottom of frame  (full width)
RD_BOT_L = 0
RD_BOT_R = W

# Shoulder colour (outside road edges)
SHOULDER_W = 0.10        # 10 % of road half-width at that depth

# ---------------------------------------------------------------------------
# Palette (BGR)
# ---------------------------------------------------------------------------
SKY_TOP    = (140, 150, 120)
SKY_BOT    = (185, 200, 175)
ROAD_COL   = (50, 53, 50)
SHOULDER_C = (95, 100, 95)
MARKING_C  = (230, 230, 230)
DIVIDER_C  = (0, 200, 230)      # yellow centre divider
RAIL_C     = (80, 80, 80)       # guard-rail

# Vehicle body palettes (BGR)
CAR_COLS = [
    (240, 240, 240),  # white
    (200, 200, 200),  # silver
    (60,  60, 170),   # red
    (170, 60,  40),   # blue
    (50,  50,  50),   # black
    (30, 210, 240),   # orange-yellow (taxi)
    (60, 140,  60),   # green
    (200,  80,  40),  # deep-blue
    (50, 100, 160),   # dark-red
    (100, 120, 180),  # violet-grey
]
BUS_COLS   = [(40, 130, 40), (30, 100, 180), (20, 20, 80)]
TRUCK_COLS = [(200, 200, 200), (60, 80, 160), (40, 40, 40)]
AUTO_COLS  = [(0, 195, 255), (0, 215, 245)]   # yellow tuk-tuk

# Vehicle base dimensions at full scale (depth = 1.0)
VEH_BASE: Dict[str, Dict] = {
    "car":       {"w": 115, "h": 70,  "cls": 2,  "label": "car"},
    "bus":       {"w": 145, "h": 95,  "cls": 5,  "label": "bus"},
    "truck":     {"w": 130, "h": 90,  "cls": 7,  "label": "truck"},
    "motorbike": {"w":  48, "h": 30,  "cls": 3,  "label": "motorbike"},
    "auto":      {"w":  72, "h": 52,  "cls": 2,  "label": "auto"},
}

SPAWN_WEIGHTS = [0.52, 0.14, 0.12, 0.12, 0.10]   # car, bus, truck, motorbike, auto
VEH_TYPES     = ["car", "bus", "truck", "motorbike", "auto"]

# Label colours for YOLO boxes (per class)
BOX_COLORS = {
    "car":       (0, 200,  50),
    "bus":       (0, 180, 255),
    "truck":     (0, 120, 255),
    "motorbike": (100, 220,  0),
    "auto":      (0, 230, 200),
}


# ---------------------------------------------------------------------------
# Perspective helpers
# ---------------------------------------------------------------------------

def _persp_y(depth: float) -> int:
    """Screen y for a vehicle at depth (0 = far/horizon, 1 = near/bottom)."""
    return int(VP_Y + depth * (H - VP_Y))


def _persp_scale(depth: float) -> float:
    """Size scale factor.  0 at horizon → 1 at camera."""
    return max(0.0, depth) ** 0.72


def _road_bounds(y: int) -> Tuple[int, int]:
    """Return (left_edge, right_edge) of road at screen row *y*."""
    if y <= VP_Y:
        return RD_TOP_L, RD_TOP_R
    t = (y - VP_Y) / (H - VP_Y)
    left  = int(RD_TOP_L + t * (RD_BOT_L - RD_TOP_L))
    right = int(RD_TOP_R + t * (RD_BOT_R - RD_TOP_R))
    return left, right


def _lane_cx(depth: float, lane: int) -> int:
    """
    x-centre for *lane* (0 = left lane, 1 = right lane) at *depth*.
    Lanes: both carry traffic coming towards the camera.
    """
    y = _persp_y(depth)
    rl, rr = _road_bounds(y)
    # Small clearance from edge
    margin = int((rr - rl) * 0.08)
    inner_l = rl + margin
    inner_r = rr - margin
    half = (inner_r - inner_l) // 2
    if lane == 0:
        return inner_l + half // 2
    else:
        return inner_l + half + half // 2


# ---------------------------------------------------------------------------
# Vehicle dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Veh:
    vtype:    str
    lane:     int       # 0 or 1
    depth:    float     # 0 = far, 1 = near camera
    speed:    float     # depth units per tick
    color:    Tuple[int, int, int]
    conf:     float
    track_id: int
    label:    str
    cls_id:   int

    @property
    def scale(self) -> float:
        return _persp_scale(self.depth)

    @property
    def screen_y(self) -> int:
        return _persp_y(self.depth)

    @property
    def screen_x(self) -> int:
        return _lane_cx(self.depth, self.lane)

    def bbox(self) -> Tuple[int, int, int, int]:
        b = VEH_BASE[self.vtype]
        s = self.scale
        hw = int(b["w"] * s * 0.5)
        hh = int(b["h"] * s * 0.5)
        cx, cy = self.screen_x, self.screen_y
        return (max(0, cx - hw), max(0, cy - hh),
                min(W - 1, cx + hw), min(H - 1, cy + hh))

    def out_of_frame(self) -> bool:
        return self.depth > 1.05 or self.depth < -0.02


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class RoadCameraRenderer:
    """
    Simulates a traffic CCTV camera for one approach direction.

    Parameters
    ----------
    direction   : "north" | "south" | "east" | "west"
    junction_id : e.g. "J1_1"
    fps         : target render framerate (controls vehicle speeds)
    source_url  : if set, tries to read real frames via OpenCV VideoCapture
                  (RTSP, MJPEG URL, or YouTube URL extracted by yt-dlp).
                  Falls back to rendered mode if unavailable unless strict_source=True.
    strict_source : if True, do not produce synthetic frames when camera feed
                    is missing or unavailable.
    """

    def __init__(
        self,
        direction: str = "north",
        junction_id: str = "J1_1",
        fps: float = 10.0,
        source_url: Optional[str] = None,
        strict_source: bool = False,
    ) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise RuntimeError("opencv-python required for RoadCameraRenderer")

        self.direction   = direction.upper()
        self.junction_id = junction_id
        self.fps         = fps
        self._tick       = 0
        self._phase      = "NS_GREEN"
        self._phase_ticks = 0
        self._strict_source = strict_source

        self._rng   = random.Random(hash(direction + junction_id) & 0xFFFFFFFF)
        self._vehicles: List[_Veh] = []
        self._next_id = 1
        self._render_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_detections: List[Dict] = []
        self._last_render_at = 0.0

        # External stream (optional)
        self._cap = None
        self._yolo_model = None
        self._yolo_model_error: Optional[str] = None
        self._yolo_model_loading = False
        self._bg_subtractor = None
        if source_url:
            self._cap = self._open_source(source_url)
            if self._cap is None or not self._cap.isOpened():
                self._cap = None
        elif self._strict_source:
            raise RuntimeError("strict_source=True requires a camera source_url")

        # Pre-render static background
        self._bg = self._build_bg()

        # Spawn initial traffic
        self._spawn_initial()

    # -----------------------------------------------------------------------
    # External source
    # -----------------------------------------------------------------------

    def _open_source(self, url: str):
        """Try to open a video/stream URL.  Returns cv2.VideoCapture or None."""
        try:
            import cv2  # type: ignore
            # Try yt-dlp to resolve YouTube URLs
            if "youtube.com" in url or "youtu.be" in url:
                try:
                    import yt_dlp  # type: ignore
                    ydl_opts = {"format": "best[height<=480]", "quiet": True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        real_url = info.get("url")
                        if real_url:
                            cap = cv2.VideoCapture(real_url)
                            if cap.isOpened():
                                return cap
                except Exception:
                    pass
            source = int(url) if str(url).isdigit() else url
            cap = cv2.VideoCapture(source)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap if cap.isOpened() else None
        except Exception:
            return None

    def close(self) -> None:
        """Release external camera resources if they are open."""
        with self._render_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    # -----------------------------------------------------------------------
    # Background construction (called once)
    # -----------------------------------------------------------------------

    def _build_bg(self) -> np.ndarray:
        import cv2  # type: ignore

        bg = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Sky gradient ──────────────────────────────────────────────────
        for y in range(VP_Y + 20):
            t = y / (VP_Y + 20)
            r = int(SKY_TOP[2] + t * (SKY_BOT[2] - SKY_TOP[2]))
            g = int(SKY_TOP[1] + t * (SKY_BOT[1] - SKY_TOP[1]))
            b = int(SKY_TOP[0] + t * (SKY_BOT[0] - SKY_TOP[0]))
            bg[y, :] = (b, g, r)

        # ── Background buildings / treeline at horizon ────────────────────
        self._draw_skyline(bg)

        # ── Road surface (trapezoid) ──────────────────────────────────────
        road_pts = np.array([
            [RD_TOP_L, VP_Y],
            [RD_TOP_R, VP_Y],
            [RD_BOT_R, H],
            [RD_BOT_L, H],
        ], dtype=np.int32)
        cv2.fillPoly(bg, [road_pts], ROAD_COL)

        # ── Shoulder areas ────────────────────────────────────────────────
        left_shoulder = np.array([
            [RD_TOP_L - 20, VP_Y], [RD_TOP_L, VP_Y],
            [RD_BOT_L, H], [RD_BOT_L - 60, H],
        ], dtype=np.int32)
        cv2.fillPoly(bg, [left_shoulder], SHOULDER_C)

        right_shoulder = np.array([
            [RD_TOP_R, VP_Y], [RD_TOP_R + 20, VP_Y],
            [RD_BOT_R + 60, H], [RD_BOT_R, H],
        ], dtype=np.int32)
        cv2.fillPoly(bg, [right_shoulder], SHOULDER_C)

        # ── Guard rails ───────────────────────────────────────────────────
        cv2.line(bg, (RD_TOP_L, VP_Y), (RD_BOT_L, H), RAIL_C, 3)
        cv2.line(bg, (RD_TOP_R, VP_Y), (RD_BOT_R, H), RAIL_C, 3)

        # ── Road edge lines ───────────────────────────────────────────────
        cv2.line(bg, (RD_TOP_L, VP_Y), (RD_BOT_L, H), MARKING_C, 2)
        cv2.line(bg, (RD_TOP_R, VP_Y), (RD_BOT_R, H), MARKING_C, 2)

        # ── Centre divider (solid yellow) ─────────────────────────────────
        cv2.line(bg, (VP_X, VP_Y), (VP_X, H), DIVIDER_C, 3)

        # ── Lane dashes (left lane) ───────────────────────────────────────
        self._draw_lane_dashes(bg, 0)

        # ── Side trees / poles ────────────────────────────────────────────
        self._draw_side_elements(bg)

        # ── Asphalt texture (subtle noise) ────────────────────────────────
        noise = np.random.RandomState(42).randint(-6, 6, (H, W, 1), dtype=np.int16)
        road_mask = np.zeros((H, W), dtype=bool)
        for y in range(VP_Y, H):
            rl, rr = _road_bounds(y)
            if rl < rr:
                road_mask[y, max(0, rl):min(W, rr)] = True
        bg_float = bg.astype(np.int16)
        bg_float[road_mask] += noise[road_mask, 0:1]
        bg = np.clip(bg_float, 0, 255).astype(np.uint8)

        return bg

    def _draw_skyline(self, bg: np.ndarray) -> None:
        """Draw simplified city skyline / treeline at horizon."""
        import cv2  # type: ignore
        rng = random.Random(99)
        x = 0
        while x < W:
            kind = rng.choice(["bld", "bld", "tree", "gap"])
            w = rng.randint(20, 55)
            if kind == "bld":
                h_bld = rng.randint(12, 50)
                y_top = VP_Y - h_bld
                col = (rng.randint(55, 80), rng.randint(60, 85), rng.randint(55, 80))
                cv2.rectangle(bg, (x, y_top), (x + w, VP_Y), col, -1)
                # Windows
                for wy in range(y_top + 4, VP_Y - 4, 8):
                    for wx in range(x + 3, x + w - 3, 10):
                        win_c = (220, 240, 255) if rng.random() > 0.35 else (15, 20, 10)
                        cv2.rectangle(bg, (wx, wy), (wx + 6, wy + 5), win_c, -1)
            elif kind == "tree":
                tx = x + w // 2
                cv2.ellipse(bg, (tx, VP_Y - 14), (w // 2, 18), 0, 0, 360, (30, 100, 30), -1)
                cv2.rectangle(bg, (tx - 3, VP_Y - 10), (tx + 3, VP_Y), (25, 50, 55), -1)
            x += w + rng.randint(0, 8)

    def _draw_lane_dashes(self, bg: np.ndarray, lane_gap_idx: int) -> None:
        """Draw dashed white lane separator between left and right lane."""
        import cv2  # type: ignore
        # The separator runs from VP to bottom, offset to left of centre
        # For 2 lanes each side → 1 dashed line between them (left of centre divider)
        # Actually for single-direction 2 lanes: dashed line between them
        steps = 28
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 0.55) / steps   # dash length
            # left lane separator (between left and right lane)
            for t in [t0]:
                y0 = int(VP_Y + t0 * (H - VP_Y))
                y1 = int(VP_Y + t1 * (H - VP_Y))
                # x of separator at these depths
                depth0 = t0
                depth1 = t1
                x0 = int((_lane_cx(depth0, 0) + _lane_cx(depth0, 1)) / 2)
                x1 = int((_lane_cx(depth1, 0) + _lane_cx(depth1, 1)) / 2)
                thick = max(1, int(2 * (0.1 + 0.9 * t0)))
                cv2.line(bg, (x0, y0), (x1, y1), MARKING_C, thick)

    def _draw_side_elements(self, bg: np.ndarray) -> None:
        """Draw trees, poles, and barriers along roadsides."""
        import cv2  # type: ignore
        rng = random.Random(7)
        # Left side trees
        num_trees = 9
        for i in range(num_trees):
            depth = 0.05 + i / num_trees * 0.95
            y = _persp_y(depth)
            scale = _persp_scale(depth)
            rl, _ = _road_bounds(y)
            tx = rl - int(25 * scale)
            r_canopy = max(3, int(16 * scale))
            cv2.ellipse(bg, (tx, y - int(8 * scale)), (r_canopy, int(r_canopy * 1.3)),
                        0, 0, 360, (35, 110, 35), -1)
            cv2.line(bg, (tx, y - int(4 * scale)), (tx, y + int(5 * scale)),
                     (30, 55, 55), max(1, int(2 * scale)))
        # Right side trees
        for i in range(num_trees):
            depth = 0.05 + i / num_trees * 0.95
            y = _persp_y(depth)
            scale = _persp_scale(depth)
            _, rr = _road_bounds(y)
            tx = rr + int(25 * scale)
            r_canopy = max(3, int(16 * scale))
            cv2.ellipse(bg, (tx, y - int(8 * scale)), (r_canopy, int(r_canopy * 1.3)),
                        0, 0, 360, (35, 110, 35), -1)
            cv2.line(bg, (tx, y - int(4 * scale)), (tx, y + int(5 * scale)),
                     (30, 55, 55), max(1, int(2 * scale)))
        # Signal pole at horizon edge
        cv2.line(bg, (VP_X - 2, VP_Y - 30), (VP_X - 2, VP_Y + 15), (60, 60, 60), 3)
        cv2.line(bg, (VP_X - 2, VP_Y - 30), (VP_X + 12, VP_Y - 30), (60, 60, 60), 2)

    # -----------------------------------------------------------------------
    # Vehicle management
    # -----------------------------------------------------------------------

    def _spawn_one(self, start_depth: float = 0.02) -> _Veh:
        rng = self._rng
        vtype = rng.choices(VEH_TYPES, weights=SPAWN_WEIGHTS)[0]
        lane = rng.randint(0, 1)

        # Pick colour
        if vtype == "car":
            color = rng.choice(CAR_COLS)
        elif vtype == "bus":
            color = rng.choice(BUS_COLS)
        elif vtype == "truck":
            color = rng.choice(TRUCK_COLS)
        elif vtype == "auto":
            color = rng.choice(AUTO_COLS)
        else:  # motorbike
            color = (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80))

        # Speed in depth-units/tick (smaller = faster apparent motion from horizon)
        base_speed = {
            "car":       rng.uniform(0.009, 0.016),
            "bus":       rng.uniform(0.006, 0.010),
            "truck":     rng.uniform(0.007, 0.011),
            "motorbike": rng.uniform(0.012, 0.020),
            "auto":      rng.uniform(0.008, 0.013),
        }[vtype]

        tid = self._next_id
        self._next_id += 1

        return _Veh(
            vtype=vtype,
            lane=lane,
            depth=float(start_depth),
            speed=base_speed,
            color=color,
            conf=round(rng.uniform(0.72, 0.97), 2),
            track_id=tid,
            label=VEH_BASE[vtype]["label"],
            cls_id=VEH_BASE[vtype]["cls"],
        )

    def _spawn_initial(self) -> None:
        for lane in range(2):
            for i in range(5):
                v = self._spawn_one(start_depth=0.05 + i * 0.18)
                v.lane = lane
                self._vehicles.append(v)

    # -----------------------------------------------------------------------
    # Phase / signal
    # -----------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def _is_green(self) -> bool:
        """Return True if this approach has a green signal."""
        d = self.direction
        if d in ("NORTH", "SOUTH"):
            return self._phase == "NS_GREEN"
        else:
            return self._phase == "EW_GREEN"

    # -----------------------------------------------------------------------
    # Per-tick vehicle update
    # -----------------------------------------------------------------------

    def _update_vehicles(self) -> None:
        green = self._is_green()
        stop_depth = 0.62   # stop-line depth when signal is red

        updated = []
        for v in self._vehicles:
            # Red-light stopping behaviour
            if not green and abs(v.depth - stop_depth) < 0.04 and v.depth < stop_depth + 0.05:
                # Hold position — decelerate
                pass
            else:
                v.depth += v.speed

            if not v.out_of_frame():
                updated.append(v)

        self._vehicles = updated

        # Spawn replacements + maintain density
        per_lane = [sum(1 for v in self._vehicles if v.lane == l) for l in range(2)]
        for lane in range(2):
            if per_lane[lane] < 4:
                nv = self._spawn_one(0.01 + self._rng.uniform(0, 0.05))
                nv.lane = lane
                self._vehicles.append(nv)

        # Occasional extra vehicle for density bursts
        if self._rng.random() < 0.08 and len(self._vehicles) < 14:
            nv = self._spawn_one(0.01)
            self._vehicles.append(nv)

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _draw_vehicle(self, frame: np.ndarray, v: _Veh) -> None:
        """Draw one vehicle with perspective scaling."""
        import cv2  # type: ignore

        x1, y1, x2, y2 = v.bbox()
        if x2 <= x1 or y2 <= y1 or y1 >= H or y2 <= 0:
            return

        bw, bh = x2 - x1, y2 - y1
        s = v.scale

        # ── Body ─────────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), v.color, -1)

        # ── Roof / cabin ─────────────────────────────────────────────────
        roof_c = tuple(max(0, c - 30) for c in v.color)
        rx1 = x1 + bw // 4
        rx2 = x2 - bw // 4
        ry1 = y1 + bh // 5
        ry2 = y2 - bh // 5
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roof_c, -1)

        # ── Windscreen (lighter strip, vehicles coming towards camera → at BOTTOM of body)
        wsc_h = max(2, bh // 5)
        wsc_c = tuple(min(255, c + 60) for c in v.color)
        cv2.rectangle(frame, (x1 + bw // 7, y2 - wsc_h - 2),
                      (x2 - bw // 7, y2 - 2), wsc_c, -1)
        # tint
        cv2.rectangle(frame, (x1 + bw // 7, y2 - wsc_h - 2),
                      (x2 - bw // 7, y2 - 2), (200, 220, 240), 1)

        # ── Headlights (front = bottom when coming towards camera) ────────
        if bw > 12:
            hl = max(2, int(3 * s))
            cv2.circle(frame, (x1 + max(2, bw // 5), y2 - max(2, bh // 5)), hl, (255, 255, 220), -1)
            cv2.circle(frame, (x2 - max(2, bw // 5), y2 - max(2, bh // 5)), hl, (255, 255, 220), -1)

        # ── Bus / truck extra details ─────────────────────────────────────
        if v.vtype == "bus" and bw > 30:
            # side windows row
            for wx in range(x1 + bw // 4, x2 - bw // 8, max(5, bw // 5)):
                cv2.rectangle(frame, (wx, y1 + bh // 4), (wx + max(3, bw // 8), y1 + bh // 2),
                              (180, 200, 220), -1)
        if v.vtype == "truck" and bw > 25:
            # cargo box behind cab
            cab_x2 = x1 + bw // 3
            cv2.rectangle(frame, (x1, y1), (cab_x2, y2), roof_c, -1)
            cv2.line(frame, (cab_x2, y1), (cab_x2, y2), (20, 20, 20), 1)

        # ── Auto-rickshaw shape (3-corner canopy) ─────────────────────────
        if v.vtype == "auto" and bw > 16:
            # Dark canopy across top
            cv2.rectangle(frame, (x1, y1), (x2, y1 + bh // 3), (20, 20, 20), -1)
            # Lighter passenger area
            cv2.rectangle(frame, (x1 + bw // 5, y1 + bh // 3), (x2 - bw // 5, y2 - bh // 4),
                          (180, 200, 220), -1)

        # ── Outline ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 15, 15), 1)

    def _draw_yolo_box(self, frame: np.ndarray, v: _Veh) -> None:
        """Draw YOLO-style detection box + label."""
        import cv2  # type: ignore

        x1, y1, x2, y2 = v.bbox()
        if x2 <= x1 or y2 <= y1:
            return

        col = BOX_COLORS.get(v.label, (0, 200, 50))
        thick = max(1, int(2 * v.scale))

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

        label_txt = f"ID{v.track_id} {v.label} {v.conf:.2f}"
        fs = max(0.28, v.scale * 0.45)
        ft = 1
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        ly = max(th + 2, y1)
        cv2.rectangle(frame, (x1, ly - th - 3), (x1 + tw + 4, ly + 1), col, -1)
        cv2.putText(frame, label_txt, (x1 + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft)

    def _draw_signal_state(self, frame: np.ndarray) -> None:
        """Draw traffic signal indicator at road vanishing point."""
        import cv2  # type: ignore

        green = self._is_green()
        yellow = self._phase == "YELLOW"
        # Signal housing
        sx, sy = VP_X + 14, VP_Y - 30
        cv2.rectangle(frame, (sx - 5, sy - 16), (sx + 15, sy + 32), (20, 20, 20), -1)
        # Red
        rc = (30, 30, 200) if not (not green and not yellow) else (30, 30, 200)
        if not green and not yellow:
            rc = (50, 50, 255)
        cv2.circle(frame, (sx + 5, sy - 8), 5, rc, -1)
        # Yellow
        yc = (0, 200, 240) if yellow else (20, 70, 70)
        cv2.circle(frame, (sx + 5, sy + 4), 5, yc, -1)
        # Green
        gc = (0, 220, 50) if green else (20, 70, 20)
        cv2.circle(frame, (sx + 5, sy + 16), 5, gc, -1)

    def _draw_hud(self, frame: np.ndarray, n_vehicles: int) -> None:
        """Draw CCTV-style HUD overlay (corner info boxes)."""
        import cv2  # type: ignore

        ts = time.strftime("%d/%m/%Y  %H:%M:%S")
        green = self._is_green()
        sig_txt = "GREEN" if green else ("YELLOW" if self._phase == "YELLOW" else "RED")
        sig_col = (0, 220, 50) if green else ((0, 200, 240) if self._phase == "YELLOW" else (50, 50, 255))

        # ── Top-left: system label ────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (225, 58), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (225, 58), (0, 80, 0), 1)
        cv2.putText(frame, "NEXUS-ATMS  |  TRAFFIC CAM",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 50), 1)
        cv2.putText(frame, f"Junction: {self.junction_id}  [{self.direction}]",
                    (6, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
        cv2.putText(frame, ts,
                    (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # ── Top-right: signal priority ────────────────────────────────────
        priority = "HIGH" if n_vehicles > 10 else ("MEDIUM" if n_vehicles > 5 else "LOW")
        pri_col  = (50, 50, 255) if priority == "HIGH" else ((0, 200, 240) if priority == "MEDIUM" else (0, 220, 50))
        cv2.rectangle(frame, (W - 200, 0), (W, 52), (0, 0, 0), -1)
        cv2.rectangle(frame, (W - 200, 0), (W, 52), sig_col, 1)
        cv2.putText(frame, f"Signal: {sig_txt}",
                    (W - 194, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sig_col, 1)
        cv2.putText(frame, f"Priority: {priority}",
                    (W - 194, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pri_col, 1)
        cv2.putText(frame, f"Vehicles: {n_vehicles}",
                    (W - 194, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # ── Bottom-left: YOLO + LIVE badge ────────────────────────────────
        cv2.rectangle(frame, (0, H - 26), (200, H), (0, 0, 0), -1)
        cv2.putText(frame, "YOLOv8  |  LIVE DETECTION",
                    (6, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1)
        # LIVE blinking dot
        if (self._tick // 5) % 2 == 0:
            cv2.circle(frame, (W - 12, H - 12), 6, (0, 50, 255), -1)
            cv2.putText(frame, "REC", (W - 45, H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 50, 255), 1)

    # -----------------------------------------------------------------------
    # Public render API
    # -----------------------------------------------------------------------

    def render(self):
        """
        Returns (annotated_bgr_frame: np.ndarray, detections: list).
        'detections' is a list of Detection-compatible dicts with keys:
            track_id, class_id, label, confidence, bbox, center
        """
        import cv2  # type: ignore

        with self._render_lock:
            now = time.time()
            min_interval = 1.0 / max(float(self.fps), 1.0)
            if self._last_frame is not None and (now - self._last_render_at) < (min_interval * 0.75):
                return self._last_frame.copy(), list(self._last_detections)

            # If we have a real source, try to use it.
            if self._cap is not None:
                try:
                    self._cap.grab()
                except Exception:
                    pass
                ok, real_frame = self._cap.read()
                if ok and real_frame is not None:
                    frame, detections = self._process_real_frame(real_frame)
                    self._last_frame = frame.copy()
                    self._last_detections = list(detections)
                    self._last_render_at = now
                    return frame, detections
                if self._strict_source:
                    raise RuntimeError("Live camera frame unavailable")

            if self._strict_source:
                raise RuntimeError("No live camera source available")

            self._tick += 1
            self._phase_ticks += 1

            # Phase cycling (for isolated camera view)
            if self._phase_ticks > int(self.fps * 6):
                self._phase_ticks = 0
                phases = ["NS_GREEN", "NS_GREEN", "EW_GREEN", "EW_GREEN", "YELLOW"]
                idx = sum(1 for p in phases[:phases.index(self._phase) + 1]) % len(phases)
                self._phase = phases[idx]

            self._update_vehicles()

            # Sort by depth ascending so far vehicles drawn first (behind near ones)
            sorted_vehs = sorted(self._vehicles, key=lambda v: v.depth)

            frame = self._bg.copy()
            self._draw_signal_state(frame)

            visible_vehs = [v for v in sorted_vehs if 0.05 < v.depth < 0.98]

            for v in visible_vehs:
                self._draw_vehicle(frame, v)
            for v in visible_vehs:
                self._draw_yolo_box(frame, v)

            self._draw_hud(frame, len(visible_vehs))

            # Build detections list
            detections = [
                {
                    "track_id":   v.track_id,
                    "class_id":   v.cls_id,
                    "label":      v.label,
                    "confidence": v.conf,
                    "bbox":       v.bbox(),
                    "center":     ((v.bbox()[0] + v.bbox()[2]) // 2,
                                   (v.bbox()[1] + v.bbox()[3]) // 2),
                }
                for v in visible_vehs
            ]
            self._last_frame = frame.copy()
            self._last_detections = list(detections)
            self._last_render_at = now
            return frame, detections

    def _process_real_frame(self, frame: np.ndarray):
        """Run real YOLO on a frame from external source."""
        try:
            import cv2  # type: ignore
            if self._yolo_model is None and self._yolo_model_error is None and not self._yolo_model_loading:
                self._yolo_model_loading = True

                def _load_model() -> None:
                    try:
                        from ultralytics import YOLO  # type: ignore

                        self._yolo_model = YOLO("yolov8n.pt")
                    except Exception as exc:
                        self._yolo_model_error = str(exc)
                    finally:
                        self._yolo_model_loading = False

                threading.Thread(target=_load_model, daemon=True).start()

            if self._yolo_model is None:
                return self._process_motion_fallback(frame)

            results = self._yolo_model(frame, conf=0.45, verbose=False)[0]
            detections = []
            VEHICLE_CLS = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = VEHICLE_CLS[cls_id]
                col = BOX_COLORS.get(label, (0, 200, 50))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                detections.append({
                    "track_id": len(detections) + 1,
                    "class_id": cls_id, "label": label, "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                })
            self._draw_hud(frame, len(detections))
            return cv2.resize(frame, (W, H)), detections
        except Exception:
            return self._process_motion_fallback(frame)

    def _process_motion_fallback(self, frame: np.ndarray):
        """Lightweight motion-based fallback when YOLO is unavailable."""
        try:
            import cv2  # type: ignore
            frame = cv2.resize(frame, (W, H))
            if self._bg_subtractor is None:
                self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=400,
                    varThreshold=30,
                    detectShadows=True,
                )

            fg = self._bg_subtractor.apply(frame)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            k = np.ones((3, 3), np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours[:40]:
                area = cv2.contourArea(cnt)
                if area < 550:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 20 or h < 16:
                    continue
                if w > int(W * 0.6) or h > int(H * 0.6):
                    continue

                x1, y1, x2, y2 = x, y, min(W - 1, x + w), min(H - 1, y + h)
                conf = 0.55
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f"vehicle {conf:.2f}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 180, 255),
                    1,
                )
                tid = self._next_id
                self._next_id += 1
                detections.append(
                    {
                        "track_id": tid,
                        "class_id": 2,
                        "label": "car",
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }
                )

            self._draw_hud(frame, len(detections))
            return frame, detections
        except Exception:
            return frame, []
