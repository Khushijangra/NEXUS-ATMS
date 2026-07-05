import os
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "argus_stream_extracted" / "argus stream A"))

from src.models.scorers.mulde import MULDEScorer
import torch
import torch.nn.functional as F

def run_phase3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dir = Path("data/features/ua_detrac/videomae")
    out_dir = Path("outputs/results")
    
    ckpt_path = Path("models/pretrained/stream_a/best_clip.pt")
    
    scorer = MULDEScorer.load_checkpoint(ckpt_path, device=device)
    scorer.eval()

    # Pick the first file
    file = [f for f in os.listdir(features_dir) if f.endswith(".npy")][0]
    
    arr = np.load(features_dir / file).astype(np.float32)
    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)
        
    tensor = torch.tensor(arr).to(device)
    
    iterations = 1000
    results = []
    
    with torch.no_grad():
        base_scores = scorer.score_anomaly(tensor)
        
        for _ in range(iterations):
            scores = scorer.score_anomaly(tensor)
            results.append(torch.tensor(scores))
            
    # Calculate stability metrics
    stacked = torch.stack(results).to(device)
    base_scores_t = torch.tensor(base_scores).to(device)
    
    mean_scores = stacked.mean(dim=0)
    std_dev = stacked.std(dim=0).mean().item()
    
    # Calculate L1, L2, cosine, max_abs wrt base
    l1_drift = F.l1_loss(stacked, base_scores_t.unsqueeze(0).expand_as(stacked)).item()
    l2_drift = F.mse_loss(stacked, base_scores_t.unsqueeze(0).expand_as(stacked)).item()
    cosine_sim = F.cosine_similarity(stacked.view(iterations, -1), base_scores_t.unsqueeze(0).expand(iterations, -1).view(iterations, -1)).mean().item()
    
    max_abs_error = torch.abs(stacked - base_scores_t.unsqueeze(0)).max().item()
    
    # Coefficient of variation
    cv = (stacked.std(dim=0) / (mean_scores + 1e-8)).mean().item()

    stability_data = {
        "iterations": iterations,
        "l1_drift": l1_drift,
        "l2_drift": l2_drift,
        "cosine_similarity": cosine_sim,
        "max_absolute_error": max_abs_error,
        "std_deviation": std_dev,
        "coefficient_of_variation": cv
    }
    
    with open(out_dir / "numerical_consistency.json", "w") as f:
        json.dump(stability_data, f, indent=4)
        
    print("Saved numerical_consistency.json")

if __name__ == "__main__":
    run_phase3()
