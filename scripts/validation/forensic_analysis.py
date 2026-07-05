import os
import json
import csv
from pathlib import Path

def inspect_checkpoint(filepath, out_path):
    import torch
    audit = {}
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        audit["model_class"] = "MULDEScorer" if "gmm_components" in data else "Unknown"
        audit["architecture"] = "MULDE (Multi-Level Density Estimator)"
        
        # Check expected input/output shapes from weights if present
        if 'model_state_dict' in data:
            state_dict = data['model_state_dict']
            audit["has_state_dict"] = True
            keys = list(state_dict.keys())
            audit["keys"] = keys[:10]  # Just save a few
            # Try to infer shape
            if keys:
                audit["sample_weight_shape"] = list(state_dict[keys[0]].shape)
        else:
            audit["has_state_dict"] = False
            
        audit["embedding_dimension"] = data.get("feature_dim", "Unknown")
        audit["checkpoint_metadata"] = {k: v for k, v in data.items() if k != "model_state_dict"}
        
        # Check dataset ref
        audit["training_dataset_references"] = "UA-DETRAC" if "ua_detrac" in filepath.lower() else "Unknown"
        audit["config_values"] = {k: data[k] for k in ["hidden_dim", "num_layers", "sigma_low", "sigma_high"] if k in data}
        audit["expected_preprocessing_pipeline"] = "VideoMAE (768-D) -> MULDE"
        
    except Exception as e:
        audit["error"] = str(e)

    with open(out_path, "w") as f:
        json.dump(audit, f, indent=4)
        
def inspect_gmm(filepath, out_path):
    import pickle
    audit = {}
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        audit["sklearn_class"] = str(type(data))
        if hasattr(data, 'n_components'):
            audit["number_of_gaussian_components"] = data.n_components
        if hasattr(data, 'means_'):
            audit["feature_dimensionality"] = data.means_.shape[1]
        if hasattr(data, 'covariance_type'):
            audit["covariance_type"] = data.covariance_type
            
        audit["expected_feature_vector_dimension"] = audit.get("feature_dimensionality", "Unknown")
        
    except Exception as e:
        audit["error"] = str(e)
        
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=4)

def inspect_features(feat_dir, out_path):
    import numpy as np
    features_data = []
    
    count = 0
    for f in os.listdir(feat_dir):
        if f.endswith(".npy"):
            try:
                filepath = os.path.join(feat_dir, f)
                arr = np.load(filepath)
                
                features_data.append({
                    "filename": f,
                    "shape": str(arr.shape),
                    "dtype": str(arr.dtype),
                    "feature_dimension": arr.shape[-1] if len(arr.shape) > 0 else 0,
                    "number_of_frames_represented": arr.shape[0] if len(arr.shape) > 1 else 1,
                    "inferred_content": "VideoMAE embeddings" if arr.shape[-1] == 768 else "Unknown"
                })
                
                count += 1
                if count >= 10:  # Just inspect 10 files
                    break
            except Exception as e:
                pass
                
    with open(out_path, "w", newline="") as f:
        if features_data:
            writer = csv.DictWriter(f, fieldnames=features_data[0].keys())
            writer.writeheader()
            writer.writerows(features_data)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "outputs" / "forensic"
    
    inspect_checkpoint(root / "models/pretrained/stream_a/best_clip.pt", out_dir / "checkpoint_audit.json")
    inspect_gmm(root / "models/pretrained/stream_a/best_clip_gmm.pkl", out_dir / "gmm_audit.json")
    inspect_features(root / "data/features/ua_detrac/videomae", out_dir / "feature_inventory.csv")
    print("Inspection complete")
