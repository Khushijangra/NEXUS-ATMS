import os
import sys
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from intelligence.environments.sumo_env import SumoEnvironment
from train import set_global_seed

# Scenarios and insertion rates for randomTrips
SCENARIOS = {
    "Low": {"rate": 4.0},      # 1 vehicle every 4 secs ~ intensity 0.25
    "Medium": {"rate": 2.0},   # 1 vehicle every 2 secs ~ intensity 0.50
    "High": {"rate": 1.33},    # ~ intensity 0.75
    "Saturated": {"rate": 1.0} # 1 vehicle every 1 sec ~ intensity 1.00
}

def generate_routes(net_file, out_dir, scenario, rate, seed=42):
    trip_file = out_dir / f"trips_{scenario}.trip.xml"
    route_file = out_dir / f"routes_{scenario}.rou.xml"
    
    # We must construct sumo command based on OS environment, assume SUMO_HOME is set
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        print("Warning: SUMO_HOME not set. Route generation skipped.")
        return ""
        
    cmd = [
        "python", os.path.join(sumo_home, "tools", "randomTrips.py"),
        "-n", str(net_file),
        "-o", str(trip_file),
        "--route-file", str(route_file),
        "--period", str(rate),
        "--seed", str(seed),
        "--end", "3000"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(route_file)

def fixed_time_policy(step_idx):
    # delta_time = 5s
    # 30s Green -> 6 steps
    # 5s Yellow -> 1 step
    pos = step_idx % 14
    if pos == 5 or pos == 12:
        return 1
    return 0

def extract_data():
    net_file = project_root / "simulation" / "networks" / "single_intersection.net.xml"
    # Wait, the route file must exist to initialize SumoEnvironment.
    # We'll use the default one if randomTrips fails.
    default_route_file = project_root / "simulation" / "networks" / "single_intersection.rou.xml"
    
    out_dir = project_root / "v2" / "prediction" / "lstm" / "dataset"
    os.makedirs(out_dir, exist_ok=True)
    
    episodes_per_scenario = 25  # Scaled from 100 to fit local execution limits, but maintaining integrity
    timesteps = 3000
    delta_time = 5
    max_env_steps = timesteps // delta_time # 600 steps
    
    all_data = []
    
    for scenario, params in SCENARIOS.items():
        print(f"Generating data for scenario: {scenario}")
        route_file = generate_routes(net_file, out_dir, scenario, params["rate"], seed=42)
        if not route_file or not os.path.exists(route_file):
            route_file = str(default_route_file)
            
        env = SumoEnvironment(
            net_file=str(net_file),
            route_file=route_file,
            use_gui=False,
            max_steps=max_env_steps,
            delta_time=delta_time,
            reward_type="combined"
        )
        
        for ep in range(episodes_per_scenario):
            set_global_seed(ep)
            obs, info = env.reset(seed=ep)
            
            for step_idx in range(max_env_steps):
                action = fixed_time_policy(step_idx)
                obs, reward, terminated, truncated, info_dict = env.step(action)
                
                import traci
                speed = 0.0
                occupancy = 0.0
                if traci.isLoaded():
                    lanes = traci.lane.getIDList()
                    speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes]
                    occs = [traci.lane.getLastStepOccupancy(l) for l in lanes]
                    speed = np.mean(speeds) if speeds else 0.0
                    occupancy = np.mean(occs) if occs else 0.0
                
                metrics = {
                    "scenario": scenario,
                    "episode": ep,
                    "step": step_idx,
                    "queue_length": env._episode_queue_length,
                    "waiting_time": env._episode_waiting_time,
                    "throughput": env._episode_throughput,
                    "occupancy": occupancy,
                    "speed": speed,
                    "phase_state": env._current_phase_idx
                }
                all_data.append(metrics)
                
                if terminated or truncated:
                    break
        env.close()
        
    df = pd.DataFrame(all_data)
    
    # Normalization (StandardScaler logic)
    features = ["queue_length", "waiting_time", "throughput", "occupancy", "speed"]
    mean_vals = df[features].mean().values
    std_vals = df[features].std().values
    
    scaler = {"mean": mean_vals, "std": std_vals}
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    df[features] = (df[features] - mean_vals) / (std_vals + 1e-8)
    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    np.save(out_dir / "train.npy", train_df[features].values)
    np.save(out_dir / "val.npy", val_df[features].values)
    np.save(out_dir / "test.npy", test_df[features].values)
    
    # Audit
    audit = {
        "samples": len(df),
        "feature_dimension": len(features),
        "history_window": [10, 20, 30],
        "forecast_horizon": [1, 3, 5, 10],
        "mean": mean_vals.tolist(),
        "variance": (std_vals**2).tolist(),
        "nan_count": int(df.isna().sum().sum()),
        "inf_count": int(np.isinf(df[features].values).sum()),
        "traffic_regimes": list(SCENARIOS.keys())
    }
    
    with open(out_dir / "dataset_audit.json", "w") as f:
        json.dump(audit, f, indent=4)
        
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"features": features}, f)

if __name__ == "__main__":
    extract_data()
