"""
Quick test script to verify SUMO + TraCI connection works.
Run: python scripts/test_sumo_connection.py
"""

import os
import sys

# --- Check 1: SUMO_HOME ---
sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    print("[FAIL] SUMO_HOME environment variable is NOT set.")
    print("  Fix: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
    sys.exit(1)
print(f"[OK] SUMO_HOME = {sumo_home}")

# --- Check 2: SUMO binary ---
sumo_bin = os.path.join(sumo_home, "bin", "sumo")
if not os.path.exists(sumo_bin + ".exe") and not os.path.exists(sumo_bin):
    print(f"[FAIL] sumo binary not found at {sumo_bin}")
    sys.exit(1)
print(f"[OK] SUMO binary found")

# --- Check 3: Import traci ---
try:
    import traci
    print(f"[OK] traci imported successfully (version: {traci.constants.TRACI_VERSION})")
except ImportError as e:
    print(f"[FAIL] Cannot import traci: {e}")
    print("  Fix: pip install traci")
    sys.exit(1)

# --- Check 4: Import sumolib ---
try:
    import sumolib
    print(f"[OK] sumolib imported successfully")
except ImportError as e:
    print(f"[FAIL] Cannot import sumolib: {e}")
    sys.exit(1)

# --- Check 5: Network file exists ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
net_file = os.path.join(project_root, "networks", "single_intersection.net.xml")
rou_file = os.path.join(project_root, "networks", "single_intersection.rou.xml")

if not os.path.exists(net_file):
    print(f"[FAIL] Network file not found: {net_file}")
    sys.exit(1)
print(f"[OK] Network file found: {net_file}")

if not os.path.exists(rou_file):
    print(f"[FAIL] Route file not found: {rou_file}")
    sys.exit(1)
print(f"[OK] Route file found: {rou_file}")

# --- Check 6: Actually start SUMO and run a short simulation ---
print("\n--- Starting SUMO simulation (headless, 100 steps) ---")
try:
    traci.start([
        "sumo",  # headless, not sumo-gui
        "-n", net_file,
        "-r", rou_file,
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
        "--time-to-teleport", "-1",
    ])
    print(f"[OK] SUMO started successfully")

    # Run 100 simulation steps
    total_vehicles = 0
    for step in range(100):
        traci.simulationStep()
        num_vehicles = traci.vehicle.getIDCount()
        total_vehicles = max(total_vehicles, num_vehicles)

        if step % 20 == 0:
            # Read traffic light state
            tl_state = traci.trafficlight.getRedYellowGreenState("center")
            # Read queue/waiting info
            queue_north = traci.edge.getLastStepHaltingNumber("north_in")
            wait_north = traci.edge.getWaitingTime("north_in")
            print(
                f"  Step {step:3d}: vehicles={num_vehicles:3d} | "
                f"TL_state={tl_state} | "
                f"queue_N={queue_north} | wait_N={wait_north:.1f}s"
            )

    traci.close()
    print(f"\n[OK] Simulation completed! Peak vehicles in network: {total_vehicles}")
    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED! Your SUMO setup is working correctly.")
    print("  You can now run:  python train.py --agent dqn --demo --gui")
    print("=" * 60)

except Exception as e:
    print(f"[FAIL] SUMO simulation error: {e}")
    try:
        traci.close()
    except Exception:
        pass
    sys.exit(1)
