"""
Training Script for Smart Traffic Management System
Train DQN or PPO agent on SUMO traffic environment.
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.envs.sumo_env import SumoEnvironment
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for traffic signal control"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["dqn", "ppo"],
        help="Agent type: dqn or ppo (default: ppo)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["rush_hour", "normal", "night"],
        help="Traffic scenario to use",
    )
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Quick demo mode (50K timesteps)",
    )

    args = parser.parse_args()
    log = setup_logger("train")

    # Load config
    config = load_config(args.config)

    # Apply scenario config if specified
    if args.scenario:
        scenario_path = f"configs/scenarios/{args.scenario}.yaml"
        if os.path.exists(scenario_path):
            scenario_cfg = load_config(scenario_path)
            log.info(f"Loaded scenario: {args.scenario}")
        else:
            log.warning(f"Scenario file not found: {scenario_path}")

    # Override config from CLI
    if args.demo:
        config["training"]["total_timesteps"] = 50000
        log.info("Demo mode: 50K timesteps")
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps
    config["training"]["seed"] = args.seed

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.agent}_{timestamp}"
    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Smart Traffic Management System — Training")
    log.info("=" * 60)
    log.info(f"Agent      : {args.agent.upper()}")
    log.info(f"Timesteps  : {config['training']['total_timesteps']:,}")
    log.info(f"Log Dir    : {log_dir}")
    log.info(f"Model Dir  : {model_dir}")
    log.info("=" * 60)

    # Network files
    net_file = config["environment"]["network_file"]
    route_file = config["environment"]["route_file"]

    if not os.path.exists(net_file):
        log.error(f"Network file not found: {net_file}")
        sys.exit(1)
    if not os.path.exists(route_file):
        log.error(f"Route file not found: {route_file}")
        sys.exit(1)

    # Create environment
    log.info("Initialising SUMO environment...")
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
        max_steps=config["sumo"]["max_steps"],
        delta_time=config["sumo"]["delta_time"],
        yellow_time=config["sumo"]["yellow_time"],
        min_green=config["sumo"]["min_green"],
        max_green=config["sumo"]["max_green"],
        reward_type=config["environment"]["reward"]["type"],
    )

    # Create agent
    log.info(f"Creating {args.agent.upper()} agent...")
    AgentClass = DQNAgent if args.agent == "dqn" else PPOAgent
    agent = AgentClass(
        env=env, config=config, log_dir=log_dir, model_dir=model_dir
    )

    # Train
    log.info("Starting training... (Ctrl+C to stop early)")
    try:
        agent.train(
            total_timesteps=config["training"]["total_timesteps"],
            eval_freq=config["training"]["eval_freq"],
            n_eval_episodes=config["training"]["n_eval_episodes"],
            save_freq=config["training"]["save_freq"],
        )
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")
        agent.save(os.path.join(model_dir, f"{args.agent}_interrupted.zip"))
    finally:
        env.close()

    log.info("=" * 60)
    log.info("Training Complete!")
    log.info(f"Models → {model_dir}")
    log.info(f"Logs   → {log_dir}")
    log.info("Next: python evaluate.py --model " + model_dir + "/best/best_model.zip")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
