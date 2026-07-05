import os
import json
import numpy as np
from pathlib import Path

def run_dataset_audit():
    features_dir = Path("data/features/ua_detrac/videomae")
    out_file = Path("outputs/results/dataset_audit.json")
    
    total_sequences = 0
    total_feature_vectors = 0
    feature_dimension = 0
    seq_lengths = []
    missing_values = 0
    nan_count = 0
    inf_count = 0
    dtypes = {}

    for file in os.listdir(features_dir):
        if file.endswith(".npy"):
            total_sequences += 1
            arr = np.load(features_dir / file)
            shape = arr.shape
            
            if len(shape) == 2:
                seq_len = shape[0]
                feat_dim = shape[1]
            else:
                seq_len = 1
                feat_dim = shape[0]
                
            seq_lengths.append(seq_len)
            total_feature_vectors += seq_len
            feature_dimension = feat_dim
            
            nan_c = np.isnan(arr).sum()
            inf_c = np.isinf(arr).sum()
            nan_count += nan_c
            inf_count += inf_c
            
            dt = str(arr.dtype)
            dtypes[dt] = dtypes.get(dt, 0) + 1

    audit = {
        "total_sequences": total_sequences,
        "total_feature_vectors": total_feature_vectors,
        "feature_dimension": feature_dimension,
        "mean_sequence_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "min_sequence_length": int(np.min(seq_lengths)) if seq_lengths else 0,
        "max_sequence_length": int(np.max(seq_lengths)) if seq_lengths else 0,
        "dtype_distribution": dtypes,
        "missing_values": int(nan_count + inf_count),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count)
    }

    with open(out_file, "w") as f:
        json.dump(audit, f, indent=4)
        
    print(f"Dataset Audit saved to {out_file}")

if __name__ == "__main__":
    run_dataset_audit()
