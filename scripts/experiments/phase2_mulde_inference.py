import os
import json
import csv
import sys
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis

# Add project root to sys path to import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "argus_stream_extracted" / "argus stream A"))

from src.models.scorers.mulde import MULDEScorer
import torch

def find_peaks(scores, threshold):
    return int(np.sum((scores[:-1] < threshold) & (scores[1:] >= threshold)))

def get_anomaly_duration(scores, threshold):
    return int(np.sum(scores >= threshold))

def run_phase2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dir = Path("data/features/ua_detrac/videomae")
    out_dir = Path("outputs/results")
    
    ckpt_path = Path("models/pretrained/stream_a/best_clip.pt")
    
    scorer = MULDEScorer.load_checkpoint(ckpt_path, device=device)
    scorer.eval()

    files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]
    
    seq_stats = []
    all_scores = []
    
    for file in files:
        arr = np.load(features_dir / file).astype(np.float32)
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
            
        tensor = torch.tensor(arr).to(device)
        
        with torch.no_grad():
            scores = scorer.score_anomaly(tensor)
            
        all_scores.append(scores)
        
        # Calculate sequence level metrics
        mean_anomaly = float(np.mean(scores))
        max_anomaly = float(np.max(scores))
        min_anomaly = float(np.min(scores))
        variance = float(np.var(scores))
        
        # Determine a dynamic threshold for peaks (e.g., mean + 1 std)
        threshold = mean_anomaly + np.sqrt(variance)
        peak_count = find_peaks(scores, threshold)
        duration = get_anomaly_duration(scores, threshold)
        
        seq_stats.append({
            "filename": file,
            "mean_anomaly": mean_anomaly,
            "max_anomaly": max_anomaly,
            "min_anomaly": min_anomaly,
            "variance": variance,
            "peak_count": peak_count,
            "anomaly_duration": duration
        })

    with open(out_dir / "anomaly_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seq_stats[0].keys())
        writer.writeheader()
        writer.writerows(seq_stats)
        
    # Global metrics
    global_scores = np.concatenate(all_scores)
    global_mean = float(np.mean(global_scores))
    global_std = float(np.std(global_scores))
    p95 = float(np.percentile(global_scores, 95))
    p99 = float(np.percentile(global_scores, 99))
    skew_val = float(skew(global_scores))
    kurt_val = float(kurtosis(global_scores))
    
    # Histogram
    hist, bin_edges = np.histogram(global_scores, bins=50)
    
    dist_data = {
        "global_mean": global_mean,
        "global_std": global_std,
        "percentiles": {
            "p25": float(np.percentile(global_scores, 25)),
            "p50": float(np.percentile(global_scores, 50)),
            "p75": float(np.percentile(global_scores, 75)),
            "p95": p95,
            "p99": p99
        },
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
    }
    
    with open(out_dir / "anomaly_distribution.json", "w") as f:
        json.dump(dist_data, f, indent=4)
        
    print("Saved anomaly_scores.csv and anomaly_distribution.json")

if __name__ == "__main__":
    run_phase2()
