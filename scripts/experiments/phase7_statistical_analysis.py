import os
import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats

def compute_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd
    
def compute_cliffs_delta(x, y):
    m, n = len(x), len(y)
    mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if x[i] > y[j]:
                mat[i,j] = 1
            elif x[i] < y[j]:
                mat[i,j] = -1
    return np.sum(mat) / (m * n)

def get_effect_size(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def run_phase7():
    out_dir = Path("outputs/results")
    ablation_file = out_dir / "ablation.csv"
    
    if not ablation_file.exists():
        print("ablation.csv not found, run phase 6 first.")
        return
        
    data = {"baseline": [], "feature": [], "anomaly": [], "full": []}
    
    with open(ablation_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["mode"]].append(float(row["reward"]))
            
    stats_results = []
    
    # 1. Descriptive Stats
    for mode, rewards in data.items():
        arr = np.array(rewards)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1) if len(arr) > 1 else 0
        se = std_val / np.sqrt(len(arr)) if len(arr) > 0 else 0
        ci = stats.t.ppf(0.975, len(arr)-1) * se if len(arr) > 1 else 0
        
        stats_results.append({
            "comparison": mode,
            "metric": "descriptive",
            "mean": mean_val,
            "median": np.median(arr),
            "std": std_val,
            "ci_95": ci,
            "p_value": "",
            "cohens_d": "",
            "cliffs_delta": "",
            "effect_size": ""
        })
        
    # 2. Comparative Stats (vs Baseline)
    baseline = np.array(data["baseline"])
    for mode in ["feature", "anomaly", "full"]:
        target = np.array(data[mode])
        if len(target) > 1 and len(baseline) > 1:
            t_stat, p_val_t = stats.ttest_rel(target, baseline)
            try:
                w_stat, p_val_w = stats.wilcoxon(target, baseline)
            except:
                p_val_w = p_val_t
                
            cohen = compute_cohens_d(target, baseline)
            cliff = compute_cliffs_delta(target, baseline)
            
            stats_results.append({
                "comparison": f"{mode} vs baseline",
                "metric": "comparative",
                "mean": "",
                "median": "",
                "std": "",
                "ci_95": "",
                "p_value": p_val_w,
                "cohens_d": cohen,
                "cliffs_delta": cliff,
                "effect_size": get_effect_size(cohen)
            })

    with open(out_dir / "statistical_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_results[0].keys())
        writer.writeheader()
        writer.writerows(stats_results)
        
    print("Saved statistical_analysis.csv")

if __name__ == "__main__":
    run_phase7()
