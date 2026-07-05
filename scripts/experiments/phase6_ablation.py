import os
import sys
import json
import csv
import torch
import numpy as np
import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "argus_stream_extracted" / "argus stream A"))

from intelligence.environments.sumo_env import SumoEnvironment
from intelligence.rl_agents.ppo import PPOAgent
from src.models.scorers.mulde import MULDEScorer
from train import load_config, set_global_seed

class AblationARGUSEngine:
    def __init__(self, mode, features_dir, ckpt_path):
        self.mode = mode
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
        
    def process_frame(self, frame=None):
        if self.idx >= len(self.features):
            self.idx = 0  
            
        feat = self.features[self.idx:self.idx+1]
        
        if self.mode == "baseline":
            self._current_severity = 0.0
        elif self.mode == "feature":
            # Just mean of VideoMAE feature
            self._current_severity = float(np.mean(feat))
        elif self.mode == "anomaly":
            # VideoMAE + MULDE (No GMM)
            tensor = torch.tensor(feat).to(self.device)
            with torch.no_grad():
                densities = self.scorer.compute_log_densities(tensor)
            self._current_severity = float(np.mean(densities))
        else: # full
            tensor = torch.tensor(feat).to(self.device)
            with torch.no_grad():
                score = self.scorer.score_anomaly(tensor)[0]
            self._current_severity = float(score)
            
        self.idx += 1
        
    def step(self):
        self.process_frame()
        
    def get_current_anomaly(self) -> float:
        return self._current_severity
        
    def get_current_alpha(self) -> float:
        return self._current_severity
        
    def shutdown(self):
        pass

def run_phase6():
    config_path = project_root / "configs" / "default.yaml"
    if not config_path.exists():
        config = {"training": {"seed": 42}, "sumo": {"max_steps": 500, "delta_time": 5, "yellow_time": 3, "min_green": 5, "max_green": 50}, "environment": {"reward": {"type": "combined"}}}
    else:
        config = load_config(str(config_path))
        
    seeds = [42, 123, 456, 789, 999]
    modes = ["baseline", "feature", "anomaly", "full"]
    
    net_file = str(project_root / "simulation" / "networks" / "single_intersection.net.xml")
    route_file = str(project_root / "simulation" / "networks" / "single_intersection.rou.xml")
    features_dir = project_root / "data" / "features" / "ua_detrac" / "videomae"
    ckpt_path = project_root / "models" / "pretrained" / "stream_a" / "best_clip.pt"
    
    out_dir = project_root / "outputs" / "results"
    
    ablation_data = []
    
    for mode in modes:
        for seed in seeds:
            print(f"Running Ablation mode={mode}, seed={seed}")
            set_global_seed(seed, deterministic=True)
            
            engine = AblationARGUSEngine(mode, features_dir, ckpt_path) if mode != "baseline" else None
            
            env = SumoEnvironment(
                net_file=net_file,
                route_file=route_file,
                use_gui=False,
                max_steps=500,
                delta_time=5,
                yellow_time=3,
                min_green=5,
                max_green=50,
                reward_type="combined",
                argus_engine=engine,
            )
            
            log_dir = str(project_root / "outputs" / f"ablation_{mode}_{seed}")
            os.makedirs(log_dir, exist_ok=True)
            
            agent = PPOAgent(env=env, config=config, log_dir=log_dir, model_dir=log_dir)
            agent.train(total_timesteps=1000)
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done and steps < 100:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
                
            ablation_data.append({
                "mode": mode,
                "seed": seed,
                "reward": total_reward
            })
            env.close()

    with open(out_dir / "ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "seed", "reward"])
        writer.writeheader()
        writer.writerows(ablation_data)
            
    print("Saved Phase 6 Ablation metrics.")

if __name__ == "__main__":
    run_phase6()
