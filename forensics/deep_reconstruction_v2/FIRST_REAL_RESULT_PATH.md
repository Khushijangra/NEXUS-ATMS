# FIRST REAL RESULT PATH

**What is the shortest path from the current repository to obtaining ONE genuine publishable metric?**

### Option A: VideoMAE → GMM → AUROC
**Estimated effort: 1-2 days**
*Why?* The heavy lifting is already done. The VideoMAE `.npy` features extracted from `ua_detrac` already exist in `data/features/`. The GMM model `best_clip_gmm.pkl` already exists. 
*Missing:* A single python script (`run_videomae_anomaly.py`) to load the `.npy` files, pass them through the loaded `.pkl` GMM, extract log-likelihoods, and use `sklearn.metrics.roc_auc_score` against synthetic or derived labels.
