"""
Traffic Scenario Generator
Creates SUMO route files for various traffic conditions.
"""

import argparse
import os
import sys
import random
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Scenario definitions
# ------------------------------------------------------------------
SCENARIOS = {
    "rush_hour": {
        "description": "Peak hour — heavy, balanced traffic (1500 veh/hr)",
        "flows": {
            "north_south": 0.20, "south_north": 0.20,
            "east_west": 0.12, "west_east": 0.12,
            "north_east": 0.04, "north_west": 0.04,
            "south_east": 0.04, "south_west": 0.04,
            "east_north": 0.03, "east_south": 0.03,
            "west_north": 0.03, "west_south": 0.03,
        },
        "duration": 3600,
    },
    "normal": {
        "description": "Normal daytime — moderate traffic (800 veh/hr)",
        "flows": {
            "north_south": 0.10, "south_north": 0.10,
            "east_west": 0.08, "west_east": 0.08,
            "north_east": 0.02, "north_west": 0.02,
            "south_east": 0.02, "south_west": 0.02,
            "east_north": 0.02, "east_south": 0.02,
            "west_north": 0.02, "west_south": 0.02,
        },
        "duration": 3600,
    },
    "night": {
        "description": "Night — sparse traffic (200 veh/hr)",
        "flows": {
            "north_south": 0.03, "south_north": 0.03,
            "east_west": 0.02, "west_east": 0.02,
            "north_east": 0.005, "north_west": 0.005,
            "south_east": 0.005, "south_west": 0.005,
            "east_north": 0.005, "east_south": 0.005,
            "west_north": 0.005, "west_south": 0.005,
        },
        "duration": 3600,
    },
    "asymmetric": {
        "description": "Heavy N-S, light E-W",
        "flows": {
            "north_south": 0.25, "south_north": 0.25,
            "east_west": 0.05, "west_east": 0.05,
            "north_east": 0.02, "north_west": 0.02,
            "south_east": 0.02, "south_west": 0.02,
            "east_north": 0.01, "east_south": 0.01,
            "west_north": 0.01, "west_south": 0.01,
        },
        "duration": 3600,
    },
}

EDGE_MAP = {
    "north_south": ("north_in", "south_out"),
    "south_north": ("south_in", "north_out"),
    "east_west":   ("east_in",  "west_out"),
    "west_east":   ("west_in",  "east_out"),
    "north_east":  ("north_in", "east_out"),
    "north_west":  ("north_in", "west_out"),
    "south_east":  ("south_in", "east_out"),
    "south_west":  ("south_in", "west_out"),
    "east_north":  ("east_in",  "north_out"),
    "east_south":  ("east_in",  "south_out"),
    "west_north":  ("west_in",  "north_out"),
    "west_south":  ("west_in",  "south_out"),
}


def generate_route_file(scenario_name: str, output_path: str) -> str:
    """Generate a SUMO .rou.xml from a scenario definition."""
    scenario = SCENARIOS[scenario_name]

    root = ET.Element("routes")
    # Vehicle type
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")
    vtype.set("decel", "4.5")
    vtype.set("sigma", "0.5")
    vtype.set("length", "5.0")
    vtype.set("maxSpeed", "13.89")

    for flow_name, prob in scenario["flows"].items():
        from_edge, to_edge = EDGE_MAP[flow_name]
        flow = ET.SubElement(root, "flow")
        flow.set("id", flow_name)
        flow.set("type", "car")
        flow.set("begin", "0")
        flow.set("end", str(scenario["duration"]))
        flow.set("probability", str(prob))
        flow.set("from", from_edge)
        flow.set("to", to_edge)
        flow.set("departLane", "best")
        flow.set("departSpeed", "max")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate traffic scenario route files")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenario to generate",
    )
    parser.add_argument("--output-dir", type=str, default="networks/scenarios")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    for name in scenarios:
        path = os.path.join(args.output_dir, f"{name}.rou.xml")
        generate_route_file(name, path)
        print(f"[OK] {name:15s} → {path}")

    print(f"\nGenerated {len(scenarios)} scenario(s).")


if __name__ == "__main__":
    main()
