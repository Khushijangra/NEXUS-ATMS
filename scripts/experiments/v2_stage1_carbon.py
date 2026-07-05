import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path

# Add project root to sys path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "argus_stream_extracted" / "argus stream A"))

from intelligence.environments.sumo_env import SumoEnvironment
from intelligence.rl_agents.ppo import PPOAgent
from src.models.scorers.mulde import MULDEScorer
from train import load_config, set_global_seed

class NumpyARGUSEngine:
    def __init__(self, features_dir, ckpt_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = MULDEScorer.load_checkpoint(ckpt_path, device=self.device)
        self.scorer.eval()
        
        files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]
        files.sort()
        all_features = []
        for f in files:
            arr = np.load(os.path.join(features_dir, f)).astype(np.float32)
            if len(arr.shape) == 1:
                arr = arr.reshape(1, -1)
            all_features.append(arr)
        
        self.features = np.vstack(all_features)
        self.idx = 0
        self._current_severity = 0.0
        
    def process_frame(self):
        if self.idx >= len(self.features):
            self.idx = 0  
            
        feat = self.features[self.idx:self.idx+1]
        tensor = torch.tensor(feat).to(self.device)
        with torch.no_grad():
            score = self.scorer.score_anomaly(tensor)[0]
            
        self._current_severity = float(score)
        self.idx += 1
        
    def step(self):
        self.process_frame()
        
    def get_current_anomaly(self) -> float:
        return self._current_severity

def run_experiment_b():
    config_path = project_root / "configs" / "default.yaml"
    config = load_config(str(config_path)) if config_path.exists() else {"training": {"seed": 42}}
    
    net_file = str(project_root / "simulation" / "networks" / "single_intersection.net.xml")
    route_file = str(project_root / "simulation" / "networks" / "single_intersection.rou.xml")
    features_dir = project_root / "data" / "features" / "ua_detrac" / "videomae"
    ckpt_path = project_root / "models" / "pretrained" / "stream_a" / "best_clip.pt"
    
    out_dir = project_root / "outputs" / "results_v2"
    os.makedirs(out_dir, exist_ok=True)
    
    seeds = [42, 123, 456] # Reduced for fast evaluation
    results = []

    for reward_type in ["combined", "carbon_combined"]:
        print(f"\\n--- Running Experiment B with reward_type: {reward_type} ---")
        
        for seed in seeds:
            print(f"Seed {seed}")
            set_global_seed(seed, deterministic=True)
            engine = NumpyARGUSEngine(features_dir, ckpt_path)
            
            env = SumoEnvironment(
                net_file=net_file,
                route_file=route_file,
                use_gui=False,
                max_steps=500,
                delta_time=5,
                reward_type=reward_type,
                argus_engine=engine,
            )
            
            log_dir = str(project_root / "outputs" / f"ppo_logs_carbon_{seed}")
            model_dir = str(project_root / "outputs" / f"ppo_models_carbon_{seed}")
            agent = PPOAgent(env=env, config=config, log_dir=log_dir, model_dir=model_dir)
            
            # Train briefly
            agent.train(total_timesteps=1500)
            
            # Evaluate
            obs, info = env.reset()
            done = False
            
            final_metrics = None
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done and "metrics" in info:
                    final_metrics = info["metrics"]

            results.append({
                "reward_type": reward_type,
                "seed": seed,
                "co2_kg": final_metrics.get("co2_kg", 0.0),
                "fuel_l": final_metrics.get("fuel_l", 0.0),
                "waiting_time": final_metrics.get("total_waiting_time", 0.0),
                "throughput": final_metrics.get("throughput", 0)
            })
            
            env.close()

    # Save Results
    with open(out_dir / "experiment_b_carbon.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reward_type", "seed", "co2_kg", "fuel_l", "waiting_time", "throughput"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\\nExperiment B metrics saved to {out_dir / 'experiment_b_carbon.csv'}")

if __name__ == "__main__":
    run_experiment_b()
