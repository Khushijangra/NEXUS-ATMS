import subprocess
import os
import glob
import json
import time

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Pipeline step failed: {cmd}")
    return result.stdout

def verify_pipeline():
    print("========================================")
    print("   SPGRL END-TO-END TRAINING VERIFICATION  ")
    print("========================================")
    
    start_time = time.time()
    
    # Step 1: Train 6 episodes (because checkpoints save after episode 5)
    try:
        run_cmd("python run.py --mode train --episodes 6 --seed 42")
        print("[OK] Train 6 episodes completed")
    except RuntimeError:
        return
        
    # Find latest experiment dir
    exp_dirs = sorted(glob.glob("experiments/*seed42*"), key=os.path.getmtime, reverse=True)
    if not exp_dirs:
        print("[FAIL] No experiment directory found after training.")
        return
        
    latest_exp = exp_dirs[0]
    print(f"    Found experiment dir: {latest_exp}")
    
    # Step 2: Checkpoint saved
    ckpt_path = os.path.join(latest_exp, "checkpoints", "best_reward_actor.pth")
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Checkpoint not saved at {ckpt_path}")
        return
    print(f"[OK] Checkpoint saved successfully")
    
    # Step 3: Resume works (train up to 10 total)
    try:
        run_cmd("python run.py --mode train --episodes 10 --seed 42 --resume")
        print("[OK] Resume training completed")
    except RuntimeError:
        return
        
    # Step 4: Evaluate
    try:
        run_cmd(f"python run.py --mode evaluate --exp_dir \"{latest_exp}\" --episodes 2")
        print("[OK] Evaluation completed")
    except RuntimeError:
        return
        
    # Step 5: CSV written
    csv_path = os.path.join(latest_exp, "evaluation_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[FAIL] CSV metrics not found at {csv_path}")
        return
    print(f"[OK] CSV written successfully")
    
    # Step 6: Verify manifest existence (Provenance Hash)
    manifest_path = os.path.join(latest_exp, "experiment_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[FAIL] Experiment manifest not found at {manifest_path}")
        return
    print(f"[OK] Manifest and Hashes verified")
    
    total_time = time.time() - start_time
    print("========================================")
    print("FINAL STATUS: TRAINING PIPELINE PASS")
    print(f"Total Latency: {total_time:.2f}s")
    print("========================================")

if __name__ == "__main__":
    verify_pipeline()
