"""
NEXUS-ATMS — Digital Twin Runner
==================================
Standalone script to launch the Pygame digital twin window.
Connects to the FastAPI backend via WebSocket for live data,
or runs in demo mode with synthetic data.
"""

import json
import math
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.digital_twin.twin import DigitalTwin

# -------------------------------------------------------------------
# Demo data feeder (used when backend WebSocket is not available)
# -------------------------------------------------------------------
def _demo_loop(twin: DigitalTwin):
    """Generate synthetic data to animate the twin."""
    tick = 0
    while twin.is_running:
        tick += 1

        # Update junction states
        for r in range(4):
            for c in range(4):
                jid = f"J{r}_{c}"
                cycle = (tick + r * 10 + c * 7) % 72
                if cycle < 33:
                    phase = "NS_GREEN"
                elif cycle < 36:
                    phase = "YELLOW"
                elif cycle < 69:
                    phase = "EW_GREEN"
                else:
                    phase = "YELLOW"

                twin.update_junction(jid,
                    phase=phase,
                    queue_n=random.randint(0, 8),
                    queue_s=random.randint(0, 8),
                    queue_e=random.randint(0, 6),
                    queue_w=random.randint(0, 6),
                    wait_time=random.uniform(5, 45),
                    ai_confidence=random.uniform(0.85, 0.99),
                )

        # Generate some vehicles
        vehicles = []
        for i in range(random.randint(30, 60)):
            vtype = random.choices(["car", "bus", "truck", "emergency"],
                                   weights=[80, 10, 8, 2])[0]
            vehicles.append({
                "id": f"v_{i}",
                "x": random.uniform(0, 600),
                "y": random.uniform(0, 600),
                "type": vtype,
                "speed": random.uniform(2, 15),
                "heading": random.uniform(0, 360),
            })
        twin.update_vehicles(vehicles)

        # Update stats
        twin.update_stats(
            vehicles=len(vehicles),
            avg_wait=random.uniform(15, 40),
            co2_saved=tick * 0.02,
            congestion_pct=random.randint(20, 65),
            ai_confidence=random.randint(88, 98),
            active_incidents=random.choice([0, 0, 0, 1]),
            emergency_active=(tick % 200 > 150),
        )

        # Demo emergency corridor every 200 ticks
        if tick % 200 > 150:
            twin.update_corridor(["J0_0", "J1_0", "J2_0", "J3_0", "J3_1", "J3_2", "J3_3"])
        else:
            twin.update_corridor([])

        time.sleep(0.5)


# -------------------------------------------------------------------
# WebSocket feeder (connects to backend)
# -------------------------------------------------------------------
def _ws_loop(twin: DigitalTwin):
    """Connect to backend WebSocket and feed data to twin."""
    try:
        import websockets
        import asyncio

        async def _connect():
            uri = "ws://localhost:8000/ws/live"
            try:
                async with websockets.connect(uri) as ws:
                    while twin.is_running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)

                        if "junctions" in data:
                            for j in data["junctions"]:
                                twin.update_junction(j["junction_id"], **j)

                        if "traffic" in data:
                            t = data["traffic"]
                            twin.update_stats(
                                avg_wait=t.get("avg_waiting_time", 0),
                                congestion_pct=int(t.get("total_queue", 0) / 0.6),
                            )
            except Exception:
                # Fall back to demo mode
                _demo_loop(twin)

        asyncio.run(_connect())
    except ImportError:
        _demo_loop(twin)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    twin = DigitalTwin(title="NEXUS-ATMS  |  Digital Twin  |  4x4 Grid City")

    if not twin.init_display():
        print("[ERROR] Could not initialize Pygame display")
        sys.exit(1)

    # Start data feeder in background thread
    feeder = threading.Thread(target=_demo_loop, args=(twin,), daemon=True)
    feeder.start()

    # Main render loop
    print("[NEXUS] Digital Twin running. Close the window to exit.")
    while twin.render():
        pass

    twin.close()
    print("[NEXUS] Digital Twin closed.")


if __name__ == "__main__":
    main()
