# Sprint 1.5 Execution Plan

## Mission
Halt RL execution and formally acquire, process, and map the multimodal traffic datasets required to adapt ARGUS to urban traffic.

## File Changes Required
Before multimodal RL training can be scientifically validated, the following files must be created or heavily modified:
1. `[NEW] data/download_datasets.sh` (or `.py`)
2. `[NEW] scripts/extract_features_traffic.py`
3. `[NEW] scripts/train_mulde_traffic.py`
4. `[MODIFY] ai/envs/env_anomaly.py` (Must map synthetic SUMO incidents to actual extracted traffic feature distributions rather than random Gaussian noise).
5. `[MODIFY] argus_stream_extracted/argus stream A/scripts/inference_server.py` (Must load `checkpoints/mulde_traffic_best.pth`).

## Execution Phases
- **Phase A: Data Acquisition**
  - Secure licensing and download AI City / DoTA / UA-DETRAC.
  - Setup raw directory structures.
- **Phase B: Video Processing**
  - Implement and run `extract_features_traffic.py`.
  - Validate VRAM limits during extraction.
- **Phase C: Backend Retraining**
  - Implement and run `train_mulde_traffic.py`.
  - Achieve >0.75 AUROC on traffic validation sets.
- **Phase D: System Wiring**
  - Point `inference_server.py` to the new traffic checkpoint.
  - Map `env_anomaly.py` to sample from the empirical distribution of traffic anomaly scores.

## Blocking Dependencies
Sprint 1.5 cannot start without user-provided dataset zips or API keys due to academic licensing constraints.
