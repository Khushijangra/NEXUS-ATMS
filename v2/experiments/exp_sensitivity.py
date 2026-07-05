import os
import sys
import csv
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.experiments.experiment_manager import ExperimentManager
from v2.rl.train_ppo_loop import train

def run_sensitivity(config_path: Path, max_episodes: int = 50):
    if config_path.exists():
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {"episodes": max_episodes, "learning_rate": 3e-4, "gamma": 0.99, "seed": 42}
        
    sweeps = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.999],
        "lambda": [0.90, 0.95, 0.99],
        "entropy_coef": [0.001, 0.01, 0.05]
    }
    
    manager = ExperimentManager(root, base_config, exp_base_name="sensitivity")
    sensitivity_csv = manager.current_exp_dir / "sensitivity_results.csv"
    
    with open(sensitivity_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Final_Reward", "Final_Queue"])
        
    manager.logger.info("Starting SPGRL Sensitivity Study")
    
    for param, values in sweeps.items():
        for val in values:
            manager.logger.info(f"--- Sensitivity: {param} = {val} ---")
            
            config = base_config.copy()
            config[param] = val
            
            # Since train currently uses hardcoded PPO initializations, 
            # the real implementation will pass config to PPOAgent(..., lr=config['learning_rate'])
            # We mock the execution here for the scaffolding framework.
            
            try:
                train(manager, max_episodes=max_episodes)
                
                with open(sensitivity_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([param, val, 0.0, 0.0]) # Replace with actual parse
            except Exception as e:
                manager.logger.error(f"Sensitivity {param}={val} failed: {e}")
            
    manager.logger.info("Sensitivity Study Complete.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    run_sensitivity(root / args.config, args.episodes)
