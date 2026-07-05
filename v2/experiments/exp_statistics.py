import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import csv
from pathlib import Path

def generate_statistics(benchmark_csv: Path, output_dir: Path):
    if not benchmark_csv.exists():
        print(f"Error: {benchmark_csv} not found.")
        return
        
    df = pd.read_csv(benchmark_csv)
    
    # We group by Agent and compute stats over the episodes
    stats_dict = {}
    
    # Metrics to analyze
    metrics = ["Reward", "Queue", "Delay", "Carbon", "SafetyOverrides", "InfTime"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        stats_dict[metric] = {}
        for agent in df['Agent'].unique():
            agent_data = df[df['Agent'] == agent][metric]
            
            mean_val = agent_data.mean()
            std_val = agent_data.std()
            n = len(agent_data)
            
            # 95% Confidence Interval
            se = std_val / np.sqrt(n) if n > 0 else 0
            ci = stats.t.ppf(0.975, n-1) * se if n > 1 else 0
            
            stats_dict[metric][agent] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "ci95": float(ci)
            }
            
    # T-test PPO against Baseline (Random/Actuated)
    if "PPO" in df['Agent'].unique():
        ppo_rewards = df[df['Agent'] == "PPO"]["Reward"]
        for other_agent in df['Agent'].unique():
            if other_agent == "PPO":
                continue
            other_rewards = df[df['Agent'] == other_agent]["Reward"]
            
            # Independent t-test (or paired if we aligned episodes, let's assume independent for evaluation rollouts)
            try:
                t_stat, p_val = stats.ttest_ind(ppo_rewards, other_rewards, equal_var=False)
            except Exception:
                t_stat, p_val = 0.0, 1.0
                
            if "T_Tests" not in stats_dict:
                stats_dict["T_Tests"] = {}
                
            stats_dict["T_Tests"][f"PPO_vs_{other_agent}_Reward"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val)
            }
            
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    # Write to CSV
    with open(output_dir / "statistics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Agent", "Mean", "Std", "CI95"])
        for metric, agents in stats_dict.items():
            if metric == "T_Tests":
                continue
            for agent, s in agents.items():
                writer.writerow([metric, agent, s['mean'], s['std'], s['ci95']])
                
    print(f"Statistical analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    generate_statistics(Path(args.benchmark_csv), Path(args.output_dir))
