import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_phase8():
    out_dir = Path("outputs/results")
    res_dir = Path("results_package")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # We will copy all required files to results_package
    import shutil
    
    csv_files = [
        "dataset_audit.json", "feature_statistics.csv", "feature_distribution.csv", "feature_pca.json",
        "anomaly_scores.csv", "anomaly_distribution.json", "numerical_consistency.json",
        "benchmark_metrics.json", "benchmark_distribution.csv",
        "rewards.csv", "waiting.csv", "queue.csv", "throughput.csv", "ablation.csv", "statistical_analysis.csv"
    ]
    for f in csv_files:
        src = out_dir / f
        if src.exists():
            shutil.copy(src, res_dir / f)

    # Figure 1: Feature PCA
    pca_file = out_dir / "feature_pca.json"
    if pca_file.exists():
        with open(pca_file, "r") as f:
            pca_data = json.load(f)
        plt.figure(figsize=(8,6))
        plt.plot(pca_data["cumulative_variance"], marker='o', linestyle='-')
        plt.title("PCA Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance")
        plt.grid(True)
        plt.savefig(res_dir / "fig_feature_pca.png")
        plt.close()
        
    # Figure 2: Anomaly Histogram
    anom_file = out_dir / "anomaly_distribution.json"
    if anom_file.exists():
        with open(anom_file, "r") as f:
            anom_data = json.load(f)
        plt.figure(figsize=(8,6))
        plt.bar(anom_data["histogram"]["bin_edges"][:-1], anom_data["histogram"]["counts"], width=0.1)
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(res_dir / "fig_anomaly_hist.png")
        plt.close()
        
    # Figure 4: Reward Curve
    rew_file = out_dir / "rewards.csv"
    if rew_file.exists():
        rewards = []
        with open(rew_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row["reward"]))
        plt.figure(figsize=(8,6))
        plt.plot(rewards, marker='s', linestyle='-', color='g')
        plt.title("PPO Reward per Seed")
        plt.xlabel("Seed Index")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(res_dir / "fig_reward_curve.png")
        plt.close()
        
    # Figure 6: Latency Breakdown
    bench_file = out_dir / "benchmark_distribution.csv"
    if bench_file.exists():
        latencies = []
        with open(bench_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                latencies.append(float(row["latency_ms"]))
        plt.figure(figsize=(8,6))
        plt.plot(latencies, alpha=0.5)
        plt.title("Inference Latency Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Latency (ms)")
        plt.grid(True)
        plt.savefig(res_dir / "fig_latency.png")
        plt.close()
        
    # Figure 7: Ablation Comparison
    abl_file = out_dir / "ablation.csv"
    if abl_file.exists():
        data = {"baseline": [], "feature": [], "anomaly": [], "full": []}
        with open(abl_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["mode"]].append(float(row["reward"]))
                
        modes = list(data.keys())
        means = [np.mean(data[m]) if len(data[m])>0 else 0 for m in modes]
        
        plt.figure(figsize=(8,6))
        plt.bar(modes, means, color=['gray', 'blue', 'orange', 'green'])
        plt.title("Ablation Study: Mean Reward")
        plt.ylabel("Reward")
        plt.grid(axis='y')
        plt.savefig(res_dir / "fig_ablation.png")
        plt.close()
        
    print("Saved all figures to results_package/")

if __name__ == "__main__":
    run_phase8()
