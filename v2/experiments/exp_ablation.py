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

def run_ablation(config_path: Path, max_episodes: int = 50):
    if config_path.exists():
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {"episodes": max_episodes, "learning_rate": 3e-4, "gamma": 0.99, "seed": 42}
        
    ablations = [
        "None",
        "Semantic",
        "Behavioral",
        "Prediction",
        "Confidence",
        "Graph",
        "Carbon",
        "Emergency"
    ]
    
    manager = ExperimentManager(root, base_config, exp_base_name="ablation")
    ablation_csv = manager.current_exp_dir / "ablation_results.csv"
    
    with open(ablation_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ablated_Stream", "Final_Reward", "Final_Queue", "Final_Carbon"])
        
    manager.logger.info("Starting SPGRL Ablation Study")
    
    for ablated_stream in ablations:
        manager.logger.info(f"--- Ablation Condition: NO {ablated_stream.upper()} ---")
        
        config = base_config.copy()
        config['ablated_stream'] = ablated_stream
        
        # We need to temporarily set the environment ablation
        os.environ['SPGRL_ABLATION'] = ablated_stream
        
        # Create a fresh sub-manager for this ablation condition to capture its metrics independently?
        # Actually, we can just run train which will load the environment
        
        try:
            # We mock the return values for the ablation CSV for now since train() saves to CSV
            # In a real run, we parse the last row of reward.csv, queue.csv, etc.
            train(manager, max_episodes=max_episodes)
            
            # Since train() logs to manager's CSVs, we can just parse them or assume train returns it
            # For this mock, we just write a placeholder to the specific ablation_results.csv
            with open(ablation_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ablated_stream, 0.0, 0.0, 0.0]) # Replace with actual parse
        except Exception as e:
            manager.logger.error(f"Ablation {ablated_stream} failed: {e}")
            
    manager.logger.info("Ablation Study Complete.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    run_ablation(root / args.config, args.episodes)
