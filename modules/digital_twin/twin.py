"""
NEXUS-ATMS Digital Twin — Pygame City Renderer
================================================
Real-time 2D visualization of the 4x4 grid city showing:
- Vehicles moving on roads (colored dots)
- Signal states pulsing at each junction
- Emergency corridors glowing blue
- Congestion heat colors on roads
- Click any junction → popup with stats
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PYGAME_OK = False
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    logger.warning("[DigitalTwin] pygame not installed.")


# ------------------------------------------------------------------
# Color Palette
# ------------------------------------------------------------------
BLACK = (20, 20, 30)
DARK_GREY = (50, 50, 60)
ROAD_COLOR = (70, 70, 80)
ROAD_MARKING = (200, 200, 200)
GREEN = (0, 200, 80)
RED = (220, 40, 40)
YELLOW = (255, 200, 0)
AMBER = (255, 165, 0)
BLUE_GLOW = (30, 144, 255)
WHITE = (240, 240, 240)
LIGHT_GREY = (160, 160, 170)

# Congestion heat colors
CONGESTION_COLORS = {
    "free": (0, 180, 80),       # Green
    "moderate": (255, 200, 0),  # Yellow
    "heavy": (255, 120, 0),     # Orange
    "gridlock": (220, 40, 40),  # Red
}

# Vehicle type colors
VEHICLE_COLORS = {
    "car": (255, 230, 100),
    "bus": (100, 180, 255),
    "truck": (255, 160, 80),
    "emergency": (255, 50, 50),
}


@dataclass
class JunctionState:
    """Visual state of a junction."""
    junction_id: str
    x: int
    y: int
    phase: str = "NS_GREEN"     # NS_GREEN, EW_GREEN, YELLOW
    queue_n: int = 0
    queue_s: int = 0
    queue_e: int = 0
    queue_w: int = 0
    wait_time: float = 0.0
    ai_confidence: float = 0.95
    is_corridor: bool = False
    is_overridden: bool = False


@dataclass
class VehicleDot:
    """A vehicle as a moving dot on the digital twin."""
    vehicle_id: str
    x: float
    y: float
    vtype: str = "car"
    speed: float = 10.0
    heading: float = 0.0        # degrees


class DigitalTwin:
    """
    Pygame-based real-time city visualization.

    Call .update() with live data and .render() each frame.
    """

    WINDOW_W = 1200
    WINDOW_H = 800
    GRID_ROWS = 4
    GRID_COLS = 4
    SPACING = 160               # Pixels between junctions
    OFFSET_X = 200              # Left margin
    OFFSET_Y = 100              # Top margin
    JUNCTION_RADIUS = 18
    VEHICLE_RADIUS = 4
    ROAD_WIDTH = 24
    FPS = 30

    def __init__(self, title: str = "NEXUS-ATMS Digital Twin"):
        self.title = title
        self._running = False
        self._screen = None
        self._clock = None
        self._font = None
        self._font_small = None
        self._font_title = None

        self._junctions: Dict[str, JunctionState] = {}
        self._vehicles: List[VehicleDot] = []
        self._corridor_path: List[str] = []
        self._selected_junction: Optional[str] = None

        # Stats overlay
        self._stats = {
            "vehicles": 0,
            "avg_wait": 0.0,
            "co2_saved": 0.0,
            "congestion_pct": 0,
            "ai_confidence": 94,
            "active_incidents": 0,
            "emergency_active": False,
        }

        # Initialize junction positions
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                jid = f"J{r}_{c}"
                self._junctions[jid] = JunctionState(
                    junction_id=jid,
                    x=self.OFFSET_X + c * self.SPACING,
                    y=self.OFFSET_Y + (self.GRID_ROWS - 1 - r) * self.SPACING,
                )

    def init_display(self):
        """Initialize pygame display."""
        if not _PYGAME_OK:
            logger.error("[DigitalTwin] pygame not available")
            return False

        pygame.init()
        self._screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 13)
        self._font_small = pygame.font.SysFont("consolas", 11)
        self._font_title = pygame.font.SysFont("consolas", 20, bold=True)
        self._running = True
        logger.info("[DigitalTwin] Display initialized")
        return True

    # ------------------------------------------------------------------
    # Data Update
    # ------------------------------------------------------------------

    def update_junction(self, junction_id: str, **kwargs):
        """Update a junction's state."""
        if junction_id in self._junctions:
            j = self._junctions[junction_id]
            for k, v in kwargs.items():
                if hasattr(j, k):
                    setattr(j, k, v)

    def update_vehicles(self, vehicles: List[Dict]):
        """Update vehicle positions. Each dict: id, x, y, type, speed, heading."""
        self._vehicles = [
            VehicleDot(
                vehicle_id=v["id"],
                x=self.OFFSET_X + v.get("x", 0) / 200.0 * self.SPACING,
                y=self.OFFSET_Y + (600 - v.get("y", 0)) / 200.0 * self.SPACING,
                vtype=v.get("type", "car"),
                speed=v.get("speed", 10),
                heading=v.get("heading", 0),
            )
            for v in vehicles
        ]

    def update_corridor(self, path: List[str]):
        """Set the active emergency corridor path."""
        self._corridor_path = path

    def update_stats(self, **kwargs):
        """Update overlay stats."""
        self._stats.update(kwargs)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> bool:
        """Render one frame. Returns False if window was closed."""
        if not self._running or not self._screen:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

        self._screen.fill(BLACK)
        self._draw_roads()
        self._draw_corridor()
        self._draw_vehicles()
        self._draw_junctions()
        self._draw_header()
        self._draw_stats_panel()
        self._draw_junction_popup()

        pygame.display.flip()
        self._clock.tick(self.FPS)
        return True

    def _draw_roads(self):
        """Draw road segments between junctions."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                jid = f"J{r}_{c}"
                j = self._junctions[jid]

                # Horizontal road to right
                if c < self.GRID_COLS - 1:
                    j2 = self._junctions[f"J{r}_{c+1}"]
                    pygame.draw.line(self._screen, ROAD_COLOR,
                                     (j.x, j.y), (j2.x, j2.y), self.ROAD_WIDTH)
                    # Center line
                    pygame.draw.line(self._screen, ROAD_MARKING,
                                     (j.x, j.y), (j2.x, j2.y), 1)

                # Vertical road upward
                if r < self.GRID_ROWS - 1:
                    j2 = self._junctions[f"J{r+1}_{c}"]
                    pygame.draw.line(self._screen, ROAD_COLOR,
                                     (j.x, j.y), (j2.x, j2.y), self.ROAD_WIDTH)
                    pygame.draw.line(self._screen, ROAD_MARKING,
                                     (j.x, j.y), (j2.x, j2.y), 1)

    def _draw_corridor(self):
        """Draw emergency corridor glowing blue."""
        if not self._corridor_path or len(self._corridor_path) < 2:
            return

        glow_alpha = int(128 + 127 * math.sin(time.time() * 4))
        for i in range(len(self._corridor_path) - 1):
            j1 = self._junctions.get(self._corridor_path[i])
            j2 = self._junctions.get(self._corridor_path[i + 1])
            if j1 and j2:
                pygame.draw.line(self._screen, BLUE_GLOW,
                                 (j1.x, j1.y), (j2.x, j2.y),
                                 self.ROAD_WIDTH + 6)

    def _draw_vehicles(self):
        """Draw vehicle dots."""
        for v in self._vehicles:
            color = VEHICLE_COLORS.get(v.vtype, VEHICLE_COLORS["car"])
            r = self.VEHICLE_RADIUS + (2 if v.vtype == "emergency" else 0)
            pygame.draw.circle(self._screen, color, (int(v.x), int(v.y)), r)

    def _draw_junctions(self):
        """Draw junction circles with signal colors."""
        for jid, j in self._junctions.items():
            # Outer ring
            border_color = BLUE_GLOW if j.is_corridor else LIGHT_GREY
            pygame.draw.circle(self._screen, border_color,
                               (j.x, j.y), self.JUNCTION_RADIUS + 3, 2)

            # Signal color fill
            if j.phase == "NS_GREEN":
                color = GREEN
            elif j.phase == "EW_GREEN":
                GREEN_ALT = (0, 160, 200)
                color = GREEN_ALT
            elif j.phase == "YELLOW":
                color = YELLOW
            else:
                color = RED

            if j.is_overridden:
                # Pulsing red for manual override
                pulse = int(128 + 127 * math.sin(time.time() * 6))
                color = (pulse, 40, 40)

            pygame.draw.circle(self._screen, color,
                               (j.x, j.y), self.JUNCTION_RADIUS)

            # Junction label
            label = self._font_small.render(jid, True, WHITE)
            self._screen.blit(label, (j.x - label.get_width() // 2,
                                       j.y - label.get_height() // 2))

            # Queue indicators (small bars)
            bar_len = min(j.queue_n + j.queue_s + j.queue_e + j.queue_w, 20) * 2
            if bar_len > 0:
                congestion_color = self._congestion_color(bar_len / 40.0)
                pygame.draw.rect(self._screen, congestion_color,
                                 (j.x - bar_len // 2,
                                  j.y + self.JUNCTION_RADIUS + 5,
                                  bar_len, 3))

    def _draw_header(self):
        """Draw the top header bar."""
        pygame.draw.rect(self._screen, (30, 30, 50), (0, 0, self.WINDOW_W, 50))

        title = self._font_title.render(
            f"NEXUS-ATMS  |  DIGITAL TWIN  |  LIVE",
            True, WHITE,
        )
        self._screen.blit(title, (20, 15))

        ts = self._font.render(
            time.strftime("%H:%M:%S"),
            True, GREEN,
        )
        self._screen.blit(ts, (self.WINDOW_W - 100, 18))

    def _draw_stats_panel(self):
        """Draw the right-side stats panel."""
        panel_x = self.WINDOW_W - 260
        panel_y = 70
        panel_w = 240
        panel_h = 300

        # Background
        s = pygame.Surface((panel_w, panel_h))
        s.set_alpha(200)
        s.fill((30, 30, 50))
        self._screen.blit(s, (panel_x, panel_y))

        # Stats text
        y = panel_y + 10
        lines = [
            ("SYSTEM STATUS", WHITE, True),
            (f"Vehicles Active:  {self._stats.get('vehicles', 0):,}", WHITE, False),
            (f"Avg Wait Time:    {self._stats.get('avg_wait', 0):.0f}s", WHITE, False),
            (f"CO2 Saved Today:  {self._stats.get('co2_saved', 0):.1f} kg", GREEN, False),
            (f"Congestion:       {self._stats.get('congestion_pct', 0)}%", WHITE, False),
            (f"AI Confidence:    {self._stats.get('ai_confidence', 0)}%", GREEN, False),
            (f"Incidents:        {self._stats.get('active_incidents', 0)}", YELLOW, False),
        ]
        if self._stats.get("emergency_active"):
            lines.append(("EMERGENCY ACTIVE", RED, True))

        for text, color, bold in lines:
            font = self._font if not bold else self._font_title if bold and y == panel_y + 10 else self._font
            surf = font.render(text, True, color)
            self._screen.blit(surf, (panel_x + 15, y))
            y += 22 if not bold else 30

    def _draw_junction_popup(self):
        """Draw popup for selected junction."""
        if not self._selected_junction:
            return

        j = self._junctions.get(self._selected_junction)
        if not j:
            return

        popup_w, popup_h = 200, 140
        px = min(j.x + 30, self.WINDOW_W - popup_w - 10)
        py = max(j.y - popup_h // 2, 60)

        s = pygame.Surface((popup_w, popup_h))
        s.set_alpha(230)
        s.fill((40, 40, 70))
        self._screen.blit(s, (px, py))
        pygame.draw.rect(self._screen, BLUE_GLOW, (px, py, popup_w, popup_h), 1)

        lines = [
            (f"  {j.junction_id}", WHITE),
            (f"  Phase: {j.phase}", GREEN if "GREEN" in j.phase else RED),
            (f"  Queue N/S: {j.queue_n}/{j.queue_s}", WHITE),
            (f"  Queue E/W: {j.queue_e}/{j.queue_w}", WHITE),
            (f"  Wait: {j.wait_time:.1f}s", YELLOW if j.wait_time > 30 else WHITE),
            (f"  AI: {j.ai_confidence:.0%}", GREEN),
        ]
        y = py + 8
        for text, color in lines:
            surf = self._font.render(text, True, color)
            self._screen.blit(surf, (px + 5, y))
            y += 20

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click — select nearest junction."""
        mx, my = pos
        min_dist = float("inf")
        closest = None
        for jid, j in self._junctions.items():
            dist = ((mx - j.x)**2 + (my - j.y)**2) ** 0.5
            if dist < self.JUNCTION_RADIUS + 20 and dist < min_dist:
                min_dist = dist
                closest = jid

        self._selected_junction = closest

    def _congestion_color(self, level: float) -> Tuple[int, int, int]:
        """Get color based on congestion level (0-1)."""
        if level < 0.3:
            return CONGESTION_COLORS["free"]
        elif level < 0.6:
            return CONGESTION_COLORS["moderate"]
        elif level < 0.85:
            return CONGESTION_COLORS["heavy"]
        return CONGESTION_COLORS["gridlock"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Clean up pygame."""
        self._running = False
        if _PYGAME_OK:
            pygame.quit()
        logger.info("[DigitalTwin] Closed")

    @property
    def is_running(self) -> bool:
        return self._running
