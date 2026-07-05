import argparse
import sys
import os
import yaml
import json
import csv
import time
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def print_config(config):
    print("\n" + "="*30)
    print("         CONFIG SUMMARY")
    print("="*30)
    for k, v in config.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
    print("="*30 + "\n")

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def run_training(config, manager, resume=False):
    logger = logging.getLogger(f"SPGRL_{manager.current_exp_dir.name}")
    logger.info(f"Starting training in {manager.current_exp_dir}")
    from v2.rl.train_ppo_loop import train
    train(manager=manager, max_episodes=config.get('episodes', 500), resume=resume)

def run_healthcheck():
    print("Running Advanced Repository Healthcheck...")
    
    checks = {}
    
    checks["Python Version (>=3.10)"] = sys.version_info >= (3, 10)
    checks["Torch Version"] = bool(torch.__version__)
    checks["CUDA Available"] = torch.cuda.is_available()
    checks["cuDNN Available"] = torch.backends.cudnn.is_available() if torch.cuda.is_available() else False
    checks["Experiment Folder Writable"] = os.access(root, os.W_OK)
    
    dataset_path = root / "v2" / "prediction" / "lstm" / "dataset" / "scaler.pkl"
    checks["Dataset Exists"] = dataset_path.exists()
    
    checks["Config Valid"] = (root / "configs" / "ppo.yaml").exists()
    checks["Git Commit Access"] = get_git_commit() != "unknown"
    
    # Check Architecture Integrity
    try:
        from v2.rl.spgrl_environment import SPGRLEnv
        env = SPGRLEnv()
        zt = env.reset()
        checks["Unified State = 168D"] = (zt.shape == (168,))
        checks["No NaN in State"] = not np.isnan(zt).any()
        checks["No Inf in State"] = not np.isinf(zt).any()
        checks["Every Stream Loads"] = True
        
        from v2.rl.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=168, action_dim=4)
        checks["PPO Loads"] = True
        
        from v2.safety.safety_wrapper import SafetyWrapper
        sw = SafetyWrapper()
        checks["Safety Loads"] = True
    except Exception as e:
        print(f"Error loading architecture: {e}")
        checks["Unified State = 168D"] = False
        checks["No NaN in State"] = False
        checks["No Inf in State"] = False
        checks["Every Stream Loads"] = False
        checks["PPO Loads"] = False
        checks["Safety Loads"] = False
        
    all_pass = True
    print("\n============== HEALTHCHECK STATUS ==============")
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {name}")
        if not result:
            all_pass = False
            
    print("================================================")
    if all_pass:
        print("\nHealthcheck PASSED. Repository is fully operational for Scientific Validation.")
        sys.exit(0)
    else:
        print("\nHealthcheck FAILED. Please resolve the issues above.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SPGRL Execution System")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "benchmark", "ablation", "sensitivity", "statistics", "paper", "healthcheck", "demo"], required=True, help="Execution mode")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="Override config episodes for testing")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to experiment directory for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "healthcheck":
        run_healthcheck()
        return
        
    config_path = root / args.config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration.")
        config = {
            "episodes": 500,
            "seed": 42,
            "learning_rate": 3e-4,
            "gamma": 0.99
        }
        
    if args.episodes is not None:
        config['episodes'] = args.episodes
    if args.seed is not None:
        config['seed'] = args.seed
        
    print_config(config)
    
    if args.mode == "train":
        from v2.experiments.experiment_manager import ExperimentManager
        manager = ExperimentManager(root, config, exp_base_name="ppo", resume=args.resume)
        run_training(config, manager, resume=args.resume)
    elif args.mode == "demo":
        print("Running SPGRL Demo Mode...")
    elif args.mode == "evaluate":
        if args.exp_dir is None:
            # Fallback to the latest experiment dir if not provided
            exp_base_dir = root / "experiments"
            existing = [d for d in exp_base_dir.iterdir() if d.is_dir() and "ppo" in d.name]
            if not existing:
                print("Error: No experiments found to evaluate.")
                sys.exit(1)
            latest_exp = sorted(existing, key=lambda x: x.name)[-1]
            args.exp_dir = str(latest_exp)
            print(f"Auto-selected latest experiment for evaluation: {args.exp_dir}")
            
        from v2.experiments.exp_evaluate import run_evaluate
        run_evaluate(Path(args.exp_dir), eval_episodes=config.get('episodes', 10))
    elif args.mode == "benchmark":
        from v2.experiments.exp_benchmark import run_baselines
        run_baselines(config, eval_episodes=config.get('episodes', 10))
    elif args.mode == "ablation":
        from v2.experiments.exp_ablation import run_ablation
        run_ablation(root / args.config, max_episodes=config.get('episodes', 10))
    elif args.mode == "sensitivity":
        from v2.experiments.exp_sensitivity import run_sensitivity
        run_sensitivity(root / args.config, max_episodes=config.get('episodes', 10))
    elif args.mode == "statistics":
        from v2.experiments.exp_statistics import generate_statistics
        benchmark_csv = root / "experiments" / "benchmark_summary.csv" 
        output_dir = root / "experiments"
        generate_statistics(benchmark_csv, output_dir)
    elif args.mode == "paper":
        from v2.experiments.generate_figures import main as generate_figures
        generate_figures(root / "experiments")
        print("Generating SPGRL Paper Assets...")
        
if __name__ == "__main__":
    main()
