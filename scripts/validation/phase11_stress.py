import os
import sys
import time
import csv
import psutil
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import GPUtil
    has_gpu = True
except ImportError:
    has_gpu = False

def run_stress_test(num_steps: int, tier_name: str):
    print(f"--- Starting Stress Test {tier_name} ({num_steps} steps) ---")
    
    from intelligence.environments.traffic_env import TrafficEnvironment
    from intelligence.perception.stream_a.engine import ARGUSEngine
    from intelligence.perception.stream_a.provider import SyntheticRenderProvider
    
    # We use SyntheticRenderProvider as a stand-in for real video stream during stress testing
    provider = SyntheticRenderProvider()
    
    ckpt_path = str((PROJECT_ROOT / "argus_stream_extracted" / "argus stream A" / "checkpoints" / "best.pt").resolve())
    print(f"Loading ARGUSEngine with authentic checkpoint: {ckpt_path}")
    
    try:
        engine = ARGUSEngine(frame_provider=provider, mulde_checkpoint=ckpt_path, device="cuda" if has_gpu else "cpu")
    except Exception as e:
        print(f"CRITICAL: Failed to load authentic checkpoints. Are they on disk? Error: {e}")
        return
        
    engine.warmup()
    env = TrafficEnvironment(argus_engine=engine)
    env.reset()
    
    metrics = []
    
    start_time_total = time.time()
    
    for i in range(num_steps):
        step_start = time.time()
        
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        step_latency = (time.time() - step_start) * 1000.0
        
        cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
        gpu_mem = 0.0
        if has_gpu:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_mem = gpus[0].memoryUsed
                
        metrics.append({
            "step": i,
            "latency_ms": step_latency,
            "cpu_mem_gb": cpu_mem,
            "gpu_mem_mb": gpu_mem,
            "anomaly_severity": info.get("anomalies", {}).get("north", 0.0), # sample one
            "buffer_occupancy": len(engine.buffer._buffer)
        })
        
        if (i+1) % 100 == 0:
            print(f"Completed {i+1}/{num_steps} steps... CPU Mem: {cpu_mem:.2f}GB")
            
        if terminated or truncated:
            env.reset()
            
    total_time = time.time() - start_time_total
    print(f"Completed {tier_name} in {total_time:.2f} seconds.")
    
    metrics_file = OUT_DIR / f"stress_test_metrics_{tier_name}.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "latency_ms", "cpu_mem_gb", "gpu_mem_mb", "anomaly_severity", "buffer_occupancy"])
        writer.writeheader()
        writer.writerows(metrics)
        
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    run_stress_test(100, "Tier1")
    # Uncomment to run deeper tiers (can take hours)
    # run_stress_test(1000, "Tier2")
    # run_stress_test(10000, "Tier3")
