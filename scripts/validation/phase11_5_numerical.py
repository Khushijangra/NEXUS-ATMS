import os
import sys
import numpy as np
import torch
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_numerical_consistency():
    print("Starting Phase 11.5: Numerical Consistency Validation...")
    
    from intelligence.perception.stream_a.loader import load_stream_a_models
    try:
        VideoMAEFeatureExtractor, MULDEScorer = load_stream_a_models()
    except Exception as e:
        print(f"Failed to load modules: {e}")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_path = str((PROJECT_ROOT / "argus_stream_extracted" / "argus stream A" / "checkpoints" / "best.pt").resolve())
    
    try:
        videomae = VideoMAEFeatureExtractor(device=device)
        mulde = MULDEScorer.load_checkpoint(ckpt_path, device=device)
        mulde.eval()
    except Exception as e:
        print(f"CRITICAL: Failed to load models with real weights. {e}")
        return
        
    # Generate exactly ONE identical static clip
    static_clip = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    
    runs = 10
    features_list = []
    alphas_list = []
    
    print(f"Running identical clip {runs} times through the offline perception stack...")
    
    for i in range(runs):
        with torch.inference_mode():
            feat = videomae.extract_from_frames([static_clip], batch_size=1)
            feat_tensor = torch.from_numpy(feat).to(device)
            alpha_t = mulde.score_anomaly(feat_tensor)
            
            features_list.append(feat)
            alphas_list.append(float(alpha_t.cpu().numpy()[0]))
            
    # Compute metrics
    alphas = np.array(alphas_list)
    variance = np.var(alphas)
    max_diff = np.max(alphas) - np.min(alphas)
    
    # Feature L2 distances against the first run
    feat_base = features_list[0]
    l2_distances = [np.linalg.norm(f - feat_base) for f in features_list[1:]]
    max_feat_diff = float(np.max(l2_distances)) if l2_distances else 0.0
    mean_feat_diff = float(np.mean(l2_distances)) if l2_distances else 0.0
    
    report = {
        "runs": runs,
        "alpha_variance": float(variance),
        "alpha_max_diff": float(max_diff),
        "alpha_scores": [float(x) for x in alphas],
        "feature_max_l2_diff": max_feat_diff,
        "feature_mean_l2_diff": mean_feat_diff,
        "tolerance_passed": max_feat_diff < 1e-6 and max_diff < 1e-6
    }
    
    report_file = OUT_DIR / "numerical_consistency.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Consistency validated. Passed: {report['tolerance_passed']}")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    run_numerical_consistency()
