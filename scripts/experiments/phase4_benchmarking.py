import os
import json
import csv
import sys
import time
import psutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "argus_stream_extracted" / "argus stream A"))

from src.models.scorers.mulde import MULDEScorer
import torch

def get_gpu_utilization():
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        return float(result.decode("utf-8").strip())
    except:
        return 0.0

def run_phase4():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dir = Path("data/features/ua_detrac/videomae")
    out_dir = Path("outputs/results")
    
    ckpt_path = Path("models/pretrained/stream_a/best_clip.pt")
    
    scorer = MULDEScorer.load_checkpoint(ckpt_path, device=device)
    scorer.eval()

    files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]
    arr = np.load(features_dir / files[0]).astype(np.float32)
    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)
    
    # We take single frames to measure per-frame latency
    tensor = torch.tensor(arr[:1]).to(device)
    
    latencies = []
    
    # Measure Latency
    with torch.no_grad():
        # Cold start
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        _ = scorer.score_anomaly(tensor)
        torch.cuda.synchronize() if device == "cuda" else None
        cold_latency = time.perf_counter() - start
        
        # Warmup
        for _ in range(10):
            _ = scorer.score_anomaly(tensor)
            
        # Benchmark runs
        for _ in range(1000):
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()
            _ = scorer.score_anomaly(tensor)
            torch.cuda.synchronize() if device == "cuda" else None
            latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    warm_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    
    # Throughput (frames per second)
    throughput = 1.0 / warm_latency if warm_latency > 0 else 0
    
    # Resource metrics
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    gpu_util = get_gpu_utilization()
    
    metrics = {
        "cold_start_latency_ms": cold_latency * 1000,
        "warm_latency_ms": warm_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "p95_latency_ms": p95_latency * 1000,
        "p99_latency_ms": p99_latency * 1000,
        "throughput_fps": throughput,
        "ram_utilization_mb": ram_mb,
        "vram_utilization_mb": vram_mb,
        "gpu_utilization_percent": gpu_util
    }
    
    with open(out_dir / "benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    dist_data = [{"iteration": i, "latency_ms": l * 1000} for i, l in enumerate(latencies)]
    
    with open(out_dir / "benchmark_distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "latency_ms"])
        writer.writeheader()
        writer.writerows(dist_data)
        
    print("Saved benchmark metrics")

if __name__ == "__main__":
    run_phase4()
