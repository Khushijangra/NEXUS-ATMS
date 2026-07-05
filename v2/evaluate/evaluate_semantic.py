import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

root = Path(__file__).resolve().parents[2]
argus_path = root / "argus_stream_extracted" / "argus stream A"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
if str(argus_path) not in sys.path:
    sys.path.insert(0, str(argus_path))

from src.models.scorers.mulde import MULDEScorer
from src.data.datasets import VideoMAEClipDataset
from src.utils.config import load_config

def main():
    print("Evaluating Semantic Stream (MULDE)...")
    config = load_config(config_dir="argus_stream_extracted/argus stream A/configs", dataset="ua_detrac")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model_path = Path("models/pretrained/stream_a_real/checkpoints/stream_a/best_clip.pt")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    scorer = MULDEScorer.load_checkpoint(model_path, device=device)
    scorer.eval()
    
    # Load validation set
    features_dir = Path("data/features/ua_detrac/videomae")
    metadata_dir = Path("data/metadata")
    
    dataset = VideoMAEClipDataset(
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        split="val",
        mode="eval",
        dataset_name="ua_detrac"
    )
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            scores = scorer.score_anomaly(features)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
            
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_scores)
        auprc = average_precision_score(all_labels, all_scores)
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
    else:
        print("Error: Validation set does not contain both normal and abnormal labels.")
        
if __name__ == "__main__":
    main()
