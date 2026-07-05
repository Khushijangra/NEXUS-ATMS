# Phase 1: Semantic Module - HPC Execution Plan
## Objective
Execute the computational graphs for the Semantic Module to generate genuine empirical telemetry.

## Expected Outputs
The following files MUST be generated and downloaded back to the local repository before proceeding to the next phase:
- auroc.csv
- f1.csv
- roc_curve.png
- pr_curve.png
- confusion_matrix.png
- experiment_config.yaml
- gpu_profile.txt

## Provenance Requirements
- GPU UUID and CUDA Version logged
- PyTorch / Transformers versions logged
- Random seed specified
- SHA256 hashes generated for all CSV/PNG outputs
