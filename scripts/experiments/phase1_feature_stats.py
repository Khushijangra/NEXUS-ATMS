import os
import json
import csv
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def compute_entropy(x, bins=50):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) * (x.max() - x.min()) / bins

def run_phase1():
    features_dir = Path("data/features/ua_detrac/videomae")
    out_dir = Path("outputs/results")
    
    files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]
    
    stats_data = []
    all_features = []
    
    for file in files:
        arr = np.load(features_dir / file).astype(np.float32)
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
            
        all_features.append(arr)
        stats_data.append({
            "filename": file,
            "number_of_clips": arr.shape[0],
            "feature_dimension": arr.shape[1],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        })
        
    with open(out_dir / "feature_statistics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_data[0].keys())
        writer.writeheader()
        writer.writerows(stats_data)
        
    print(f"Saved feature_statistics.csv")
    
    # Global feature matrix
    X = np.vstack(all_features)
    
    # Per-dimension variance
    dim_vars = np.var(X, axis=0)
    
    # Covariance spectrum (eigenvalues)
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Entropy & Sparsity
    entropy_val = compute_entropy(X)
    sparsity = float(np.mean(X == 0))
    
    dist_data = [{
        "metric": "feature_entropy", "value": entropy_val
    }, {
        "metric": "feature_sparsity", "value": sparsity
    }]
    
    # Add top 10 eigenvalues for spectrum
    for i in range(min(10, len(eigenvalues))):
        dist_data.append({"metric": f"eigenvalue_{i+1}", "value": float(eigenvalues[i])})
        
    with open(out_dir / "feature_distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(dist_data)
        
    print("Saved feature_distribution.csv")
    
    # PCA
    n_components = min(50, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    pca_data = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist()
    }
    
    with open(out_dir / "feature_pca.json", "w") as f:
        json.dump(pca_data, f, indent=4)
        
    print("Saved feature_pca.json")

if __name__ == "__main__":
    run_phase1()
