# MULDE Retraining Plan

## Objective
Design the `train_mulde_traffic.py` pipeline to fine-tune the ARGUS MULDE backend exclusively on the newly extracted `.npy` traffic features.

## Architecture & Data Splits
MULDE uses a Normalizing Flow to estimate the probability density of "normal" features. 
- **Training**: Uses ONLY *normal* traffic videos (no anomalies).
- **Validation**: Uses mixed normal and anomalous videos to calculate AUC.

## `train_mulde_traffic.py` Design
1. **Data Loading**:
   - Parse `data/features/[dataset]/videomae/train/` (Normal features).
   - Construct PyTorch `DataLoader` (Batch size: 128).
2. **Model Instantiation**:
   - Initialize `MULDEScorer(feature_dim=768, hidden_dim=4096, gmm_components=5)`.
3. **Training Loop**:
   - Optimizer: Adam (lr=1e-4).
   - Loss: Negative Log-Likelihood (NLL).
   - Iterate for 50-100 epochs.
4. **Validation Loop**:
   - Load `data/features/[dataset]/videomae/test/`.
   - Compute anomaly scores.
   - Compare against temporal frame-level annotations.
   - Compute AUROC metric.
5. **Artifact Generation**:
   - Save weights to `checkpoints/mulde_traffic_best.pth`.

## Prerequisites Before Implementation
- Successful completion of the Feature Extraction phase.
- Availability of Frame-level Temporal Annotations mapping frame indices to [0, 1] anomaly labels for AUROC evaluation.
