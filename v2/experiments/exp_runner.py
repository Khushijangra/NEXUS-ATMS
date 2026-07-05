import os
import sys
import yaml
from pathlib import Path
import logging

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.experiments.experiment_manager import ExperimentManager
from v2.rl.train_ppo_loop import train

def run_multiseed(config_path: Path, seeds: list = [42, 7, 99]):
    if config_path.exists():
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {"episodes": 500, "learning_rate": 3e-4, "gamma": 0.99}
        
    print(f"Starting Multi-Seed PPO Validation Run over seeds: {seeds}")
    
    for seed in seeds:
        print(f"\n[{'='*40}]\nInitiating Execution for Seed: {seed}\n[{'='*40}]")
        config = base_config.copy()
        config['seed'] = seed
        
        manager = ExperimentManager(root, config, exp_base_name="ppo_validation")
        
        try:
            train(manager, max_episodes=config.get('episodes', 500))
        except Exception as e:
            manager.logger.error(f"Seed {seed} failed: {e}")
            
    print("\nMulti-Seed Validation Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    args = parser.parse_args()
    
    run_multiseed(root / args.config, args.seeds)
