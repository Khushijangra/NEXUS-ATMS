# Hardware Feasibility Report

## Hardware Profile
- **GPU**: NVIDIA GeForce RTX 2050
- **VRAM**: ~4.0 GB (4096 MB)
- **Environment**: PyTorch on Windows

## Operation VRAM Cost Estimates

| Operation | Model/Component | Est. VRAM Cost |
| :--- | :--- | :--- |
| **Feature Extraction** | VideoMAE-v2 (ViT-B or ViT-L) | ~2.0 GB - 3.5 GB |
| **Anomaly Scoring** | MULDE (GMM + Normalizing Flow) | ~0.5 GB - 1.0 GB |
| **RL Agent** | D3QN (Forward + Backward Pass) | ~0.5 GB - 0.8 GB |
| **Simulation Context** | SUMO (TraCI) + Replay Buffer | System RAM (Negligible VRAM) |

## Simultaneous Execution Feasibility
**Question**: Can the RTX 2050 extract features, retrain MULDE, and run multimodal RL *simultaneously*?
**Answer**: **NO.**

### Analysis
If we attempt real-time video processing during RL training:
`VideoMAE (3GB) + MULDE (1GB) + D3QN (0.8GB) + Windows OS Overhead (0.5GB) = 5.3GB > 4.0GB`
This will immediately cause `CUDA OutOfMemoryError`.

## Mandatory Hardware Mitigations
To proceed with this project on an RTX 2050, the pipeline must be **strictly decoupled into sequential offline stages**:

1. **Phase 1: Feature Extraction (Offline)**
   - Run VideoMAE-v2 on the raw dataset.
   - Save the `[N, 768]` temporal embeddings to `.npy` arrays on disk.
   - *Requires clearing all other models from VRAM.*
2. **Phase 2: MULDE Retraining (Offline)**
   - Load `.npy` arrays.
   - Train MULDE mapping.
   - *Minimal VRAM used.*
3. **Phase 3: RL Training (Online)**
   - Use pre-computed surrogate anomaly scores during D3QN optimization.
   - *Do not load VideoMAE.*

**Conclusion:** End-to-end simultaneous multimodal processing is physically impossible on this hardware.
