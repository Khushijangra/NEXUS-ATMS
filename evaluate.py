"""
Evaluation Script for Smart Traffic Management System
Compare trained RL agent against fixed-timing baseline.
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from ai.envs.sumo_env import SumoEnvironment
from ai.rl.dqn import DQNAgent
from ai.rl.d3qn import D3QNAgent
from ai.rl.ppo import PPOAgent
from ai.utils.logger import setup_logger
from ai.utils.metrics import MetricsTracker
from ai.utils.visualization import (
    plot_comparison_bar,
    generate_report_figure,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.get("extends")
    if not base_path:
        return cfg

    parent = Path(config_path).parent / str(base_path)
    with open(parent, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    cfg.pop("extends", None)
    return _deep_update(base_cfg, cfg)


def _deep_update(base: Dict, override: Dict) -> Dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ------------------------------------------------------------------
# Baseline evaluation (fixed-timing signals)
# ------------------------------------------------------------------
def run_baseline(env: SumoEnvironment, n_episodes: int = 10) -> Dict:
    """
    Run evaluation with fixed-timing signals (action = 0 always).

    Returns:
        Aggregated baseline metrics.
    """
    all_rewards: List[float] = []
    all_metrics: List[Dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            total_reward += reward

        all_rewards.append(total_reward)
        if "metrics" in info:
            all_metrics.append(info["metrics"])

    results: Dict = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
    }
    if all_metrics:
        for key in ["avg_waiting_time", "avg_queue_length", "throughput"]:
            vals = [m.get(key, 0) for m in all_metrics]
            results[key] = float(np.mean(vals))
    return results


# ------------------------------------------------------------------
# RL agent evaluation
# ------------------------------------------------------------------
def run_rl_evaluation(
    env: SumoEnvironment,
    model_path: str,
    agent_type: str,
    config: Dict,
    n_episodes: int = 10,
) -> Dict:
    """
    Evaluate a trained RL agent.

    Returns:
        Aggregated RL metrics.
    """
    if agent_type == "dqn":
        AgentClass = DQNAgent
    elif agent_type == "d3qn":
        AgentClass = D3QNAgent
    else:
        AgentClass = PPOAgent
    agent = AgentClass(env=env, config=config)
    agent.load(model_path)
    return agent.evaluate(n_episodes=n_episodes)


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------
def compare_results(baseline: Dict, rl_agent: Dict) -> Dict:
    """Calculate improvement percentages."""
    def metric_value(metrics: Dict, key: str) -> float:
        if key == "throughput":
            return float(metrics.get("throughput", metrics.get("avg_throughput", 0.0)))
        return float(metrics.get(key, 0.0))

    comparison: Dict = {}
    for key in ["avg_waiting_time", "avg_queue_length", "throughput"]:
        bv = metric_value(baseline, key)
        rv = metric_value(rl_agent, key)
        if bv != 0:
            pct = (rv - bv) / abs(bv) * 100
        else:
            pct = 0.0
        comparison[key] = {
            "baseline": bv,
            "rl_agent": rv,
            "change_pct": round(pct, 2),
        }
    return comparison


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agent for traffic signal control"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--agent", type=str, default="ppo", choices=["dqn", "d3qn", "ppo"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", action="store_true", help="Generate visual report")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()
    log = setup_logger("evaluate")

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    net_file = config["environment"]["network_file"]
    route_file = config["environment"]["route_file"]

    log.info("=" * 60)
    log.info("Smart Traffic Management System — Evaluation")
    log.info("=" * 60)

    # Create environment
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

    # --- Baseline ---
    log.info(f"Running baseline evaluation ({args.n_episodes} episodes)...")
    baseline = run_baseline(env, args.n_episodes)
    log.info(f"Baseline — Reward: {baseline['mean_reward']:.2f} "
             f"Wait: {baseline.get('avg_waiting_time', 0):.1f}s "
             f"Queue: {baseline.get('avg_queue_length', 0):.1f}")

    # --- RL Agent ---
    log.info(f"Running {args.agent.upper()} evaluation ({args.n_episodes} episodes)...")
    rl_results = run_rl_evaluation(
        env, args.model, args.agent, config, args.n_episodes
    )
    rl_results["throughput"] = float(
        rl_results.get("throughput", rl_results.get("avg_throughput", 0.0))
    )
    log.info(f"RL Agent — Reward: {rl_results['mean_reward']:.2f} "
             f"Wait: {rl_results.get('avg_waiting_time', 0):.1f}s "
             f"Queue: {rl_results.get('avg_queue_length', 0):.1f}")

    # --- Comparison ---
    comparison = compare_results(baseline, rl_results)

    log.info("-" * 60)
    log.info("COMPARISON RESULTS")
    log.info("-" * 60)
    for metric, vals in comparison.items():
        log.info(
            f"  {metric:25s} | Baseline: {vals['baseline']:8.2f} | "
            f"RL: {vals['rl_agent']:8.2f} | Change: {vals['change_pct']:+.1f}%"
        )

    # Save JSON results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {"baseline": baseline, "rl_agent": rl_results, "comparison": comparison},
            f,
            indent=2,
        )
    log.info(f"Results saved to {results_path}")

    # --- Visual report ---
    if args.report:
        log.info("Generating visual report...")
        chart_path = os.path.join(args.output_dir, "comparison_chart.png")
        plot_comparison_bar(baseline, rl_results, save_path=chart_path)
        log.info(f"Chart saved to {chart_path}")

    env.close()
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
