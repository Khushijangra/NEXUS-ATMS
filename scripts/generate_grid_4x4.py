"""
Generate a 4x4 grid SUMO network for NEXUS-ATMS.
Creates node, edge, traffic light, route, and config files.
"""

import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "networks", "grid_4x4")


def generate_nodes():
    """Generate a 4x4 grid of intersection nodes + boundary nodes."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                 ' xsi:noNamespaceSchemaLocation='
                 '"http://sumo.dlr.de/xsd/nodes_file.xsd">')

    spacing = 200  # metres between intersections

    # 4x4 internal intersection nodes (traffic-light controlled)
    for row in range(4):
        for col in range(4):
            nid = f"J{row}_{col}"
            x = col * spacing
            y = row * spacing
            lines.append(f'  <node id="{nid}" x="{x}" y="{y}" type="traffic_light"/>')

    # Boundary nodes (entry/exit points) — one per edge of the grid
    for col in range(4):
        x = col * spacing
        lines.append(f'  <node id="N_in_{col}" x="{x}" y="{4 * spacing}" type="priority"/>')
        lines.append(f'  <node id="S_in_{col}" x="{x}" y="{-spacing}" type="priority"/>')
    for row in range(4):
        y = row * spacing
        lines.append(f'  <node id="W_in_{row}" x="{-spacing}" y="{y}" type="priority"/>')
        lines.append(f'  <node id="E_in_{row}" x="{4 * spacing}" y="{y}" type="priority"/>')

    lines.append('</nodes>')
    return "\n".join(lines)


def generate_edges():
    """Generate edges (roads) connecting all nodes."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                 ' xsi:noNamespaceSchemaLocation='
                 '"http://sumo.dlr.de/xsd/edges_file.xsd">')

    num_lanes = 2
    speed = "13.89"  # ~50 km/h

    # Internal horizontal edges (E-W)
    for row in range(4):
        for col in range(3):
            src = f"J{row}_{col}"
            dst = f"J{row}_{col+1}"
            lines.append(f'  <edge id="{src}_to_{dst}" from="{src}" to="{dst}"'
                         f' numLanes="{num_lanes}" speed="{speed}"/>')
            lines.append(f'  <edge id="{dst}_to_{src}" from="{dst}" to="{src}"'
                         f' numLanes="{num_lanes}" speed="{speed}"/>')

    # Internal vertical edges (N-S)
    for col in range(4):
        for row in range(3):
            src = f"J{row}_{col}"
            dst = f"J{row+1}_{col}"
            lines.append(f'  <edge id="{src}_to_{dst}" from="{src}" to="{dst}"'
                         f' numLanes="{num_lanes}" speed="{speed}"/>')
            lines.append(f'  <edge id="{dst}_to_{src}" from="{dst}" to="{src}"'
                         f' numLanes="{num_lanes}" speed="{speed}"/>')

    # Boundary edges — North
    for col in range(4):
        lines.append(f'  <edge id="N_in_{col}_to_J3_{col}" from="N_in_{col}" to="J3_{col}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')
        lines.append(f'  <edge id="J3_{col}_to_N_in_{col}" from="J3_{col}" to="N_in_{col}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')

    # Boundary edges — South
    for col in range(4):
        lines.append(f'  <edge id="S_in_{col}_to_J0_{col}" from="S_in_{col}" to="J0_{col}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')
        lines.append(f'  <edge id="J0_{col}_to_S_in_{col}" from="J0_{col}" to="S_in_{col}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')

    # Boundary edges — West
    for row in range(4):
        lines.append(f'  <edge id="W_in_{row}_to_J{row}_0" from="W_in_{row}" to="J{row}_0"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')
        lines.append(f'  <edge id="J{row}_0_to_W_in_{row}" from="J{row}_0" to="W_in_{row}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')

    # Boundary edges — East
    for row in range(4):
        lines.append(f'  <edge id="E_in_{row}_to_J{row}_3" from="E_in_{row}" to="J{row}_3"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')
        lines.append(f'  <edge id="J{row}_3_to_E_in_{row}" from="J{row}_3" to="E_in_{row}"'
                     f' numLanes="{num_lanes}" speed="{speed}"/>')

    lines.append('</edges>')
    return "\n".join(lines)


def generate_routes():
    """Generate vehicle routes for a normal traffic scenario (~800 veh/hr)."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                 ' xsi:noNamespaceSchemaLocation='
                 '"http://sumo.dlr.de/xsd/routes_file.xsd">')

    # Vehicle types
    lines.append('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                 'length="5" maxSpeed="16.67" guiShape="passenger"/>')
    lines.append('  <vType id="bus" accel="1.2" decel="4.0" sigma="0.5" '
                 'length="12" maxSpeed="11.11" guiShape="bus"/>')
    lines.append('  <vType id="truck" accel="1.0" decel="3.5" sigma="0.5" '
                 'length="10" maxSpeed="11.11" guiShape="truck"/>')
    lines.append('  <vType id="emergency" accel="3.0" decel="5.0" sigma="0.2" '
                 'length="6" maxSpeed="22.22" guiShape="emergency" '
                 'vClass="emergency"/>')

    # Define major routes through the grid
    route_defs = [
        # North-South corridors
        ("route_NS_0", "N_in_0_to_J3_0 J3_0_to_J2_0 J2_0_to_J1_0 J1_0_to_J0_0 J0_0_to_S_in_0"),
        ("route_NS_1", "N_in_1_to_J3_1 J3_1_to_J2_1 J2_1_to_J1_1 J1_1_to_J0_1 J0_1_to_S_in_1"),
        ("route_NS_2", "N_in_2_to_J3_2 J3_2_to_J2_2 J2_2_to_J1_2 J1_2_to_J0_2 J0_2_to_S_in_2"),
        ("route_NS_3", "N_in_3_to_J3_3 J3_3_to_J2_3 J2_3_to_J1_3 J1_3_to_J0_3 J0_3_to_S_in_3"),
        # South-North corridors
        ("route_SN_0", "S_in_0_to_J0_0 J0_0_to_J1_0 J1_0_to_J2_0 J2_0_to_J3_0 J3_0_to_N_in_0"),
        ("route_SN_1", "S_in_1_to_J0_1 J0_1_to_J1_1 J1_1_to_J2_1 J2_1_to_J3_1 J3_1_to_N_in_1"),
        ("route_SN_2", "S_in_2_to_J0_2 J0_2_to_J1_2 J1_2_to_J2_2 J2_2_to_J3_2 J3_2_to_N_in_2"),
        ("route_SN_3", "S_in_3_to_J0_3 J0_3_to_J1_3 J1_3_to_J2_3 J2_3_to_J3_3 J3_3_to_N_in_3"),
        # West-East corridors
        ("route_WE_0", "W_in_0_to_J0_0 J0_0_to_J0_1 J0_1_to_J0_2 J0_2_to_J0_3 J0_3_to_E_in_0"),
        ("route_WE_1", "W_in_1_to_J1_0 J1_0_to_J1_1 J1_1_to_J1_2 J1_2_to_J1_3 J1_3_to_E_in_1"),
        ("route_WE_2", "W_in_2_to_J2_0 J2_0_to_J2_1 J2_1_to_J2_2 J2_2_to_J2_3 J2_3_to_E_in_2"),
        ("route_WE_3", "W_in_3_to_J3_0 J3_0_to_J3_1 J3_1_to_J3_2 J3_2_to_J3_3 J3_3_to_E_in_3"),
        # East-West corridors
        ("route_EW_0", "E_in_0_to_J0_3 J0_3_to_J0_2 J0_2_to_J0_1 J0_1_to_J0_0 J0_0_to_W_in_0"),
        ("route_EW_1", "E_in_1_to_J1_3 J1_3_to_J1_2 J1_2_to_J1_1 J1_1_to_J1_0 J1_0_to_W_in_1"),
        ("route_EW_2", "E_in_2_to_J2_3 J2_3_to_J2_2 J2_2_to_J2_1 J2_1_to_J2_0 J2_0_to_W_in_2"),
        ("route_EW_3", "E_in_3_to_J3_3 J3_3_to_J3_2 J3_2_to_J3_1 J3_1_to_J3_0 J3_0_to_W_in_3"),
        # Diagonal / cross routes
        ("route_diag1", "S_in_0_to_J0_0 J0_0_to_J0_1 J0_1_to_J1_1 J1_1_to_J2_1 J2_1_to_J2_2 J2_2_to_J3_2 J3_2_to_N_in_2"),
        ("route_diag2", "W_in_0_to_J0_0 J0_0_to_J1_0 J1_0_to_J1_1 J1_1_to_J1_2 J1_2_to_J2_2 J2_2_to_J3_2 J3_2_to_N_in_2"),
    ]

    for rid, edges in route_defs:
        lines.append(f'  <route id="{rid}" edges="{edges}"/>')

    # Generate vehicle flows
    vid = 0
    import random
    random.seed(42)

    # Flow rates per route (vehicles per hour)
    flow_rates = {
        "route_NS_0": 60, "route_NS_1": 90, "route_NS_2": 80, "route_NS_3": 50,
        "route_SN_0": 50, "route_SN_1": 80, "route_SN_2": 70, "route_SN_3": 40,
        "route_WE_0": 70, "route_WE_1": 100, "route_WE_2": 60, "route_WE_3": 50,
        "route_EW_0": 60, "route_EW_1": 80, "route_EW_2": 50, "route_EW_3": 40,
        "route_diag1": 30, "route_diag2": 20,
    }

    sim_duration = 3600  # 1 hour

    for route_id, rate in flow_rates.items():
        interval = 3600.0 / rate if rate > 0 else 9999
        t = random.uniform(0, interval)
        while t < sim_duration:
            # 80% cars, 10% buses, 10% trucks
            r = random.random()
            vtype = "car" if r < 0.8 else ("bus" if r < 0.9 else "truck")
            lines.append(f'  <vehicle id="v_{vid}" type="{vtype}" '
                         f'route="{route_id}" depart="{t:.1f}" '
                         f'departLane="best" departSpeed="max"/>')
            vid += 1
            t += random.expovariate(1.0 / interval)

    # Add one emergency vehicle at t=600 (10 minutes in)
    lines.append(f'  <vehicle id="ambulance_1" type="emergency" '
                 f'route="route_WE_1" depart="600.0" '
                 f'departLane="best" departSpeed="max"/>')

    lines.append('</routes>')
    return "\n".join(lines)


def generate_sumo_config():
    """Generate .sumocfg file."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
  <input>
    <net-file value="grid_4x4.net.xml"/>
    <route-files value="grid_4x4.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
  <processing>
    <time-to-teleport value="-1"/>
  </processing>
</configuration>"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[NEXUS] Generating 4x4 grid network files...")

    # Write node file
    with open(os.path.join(OUTPUT_DIR, "grid_4x4.nod.xml"), "w") as f:
        f.write(generate_nodes())
    print("  -> grid_4x4.nod.xml")

    # Write edge file
    with open(os.path.join(OUTPUT_DIR, "grid_4x4.edg.xml"), "w") as f:
        f.write(generate_edges())
    print("  -> grid_4x4.edg.xml")

    # Write route file
    with open(os.path.join(OUTPUT_DIR, "grid_4x4.rou.xml"), "w") as f:
        f.write(generate_routes())
    print("  -> grid_4x4.rou.xml")

    # Write SUMO config
    with open(os.path.join(OUTPUT_DIR, "grid_4x4.sumocfg"), "w") as f:
        f.write(generate_sumo_config())
    print("  -> grid_4x4.sumocfg")

    # Build the .net.xml using netconvert
    nod_file = os.path.join(OUTPUT_DIR, "grid_4x4.nod.xml")
    edg_file = os.path.join(OUTPUT_DIR, "grid_4x4.edg.xml")
    net_file = os.path.join(OUTPUT_DIR, "grid_4x4.net.xml")

    sumo_home = os.environ.get("SUMO_HOME", "")
    netconvert = os.path.join(sumo_home, "bin", "netconvert") if sumo_home else "netconvert"

    cmd = f'"{netconvert}" -n "{nod_file}" -e "{edg_file}" -o "{net_file}" --tls.guess true --no-warnings true'
    print(f"\n[NEXUS] Running netconvert:\n  {cmd}")
    ret = os.system(cmd)
    if ret == 0:
        print(f"  -> grid_4x4.net.xml [OK]")
    else:
        print(f"  [WARN] netconvert failed (exit {ret}). You may need to run manually.")

    print("\n[NEXUS] 4x4 grid network generation complete!")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
