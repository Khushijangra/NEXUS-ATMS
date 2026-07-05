# SPGRL Quick Start Guide

Welcome to the **Semantic Predictive Graph Reinforcement Learning (SPGRL)** framework.

This repository is designed for **One-Command Reproducibility**. If you are a Hackathon Judge or a Reviewer, follow these steps to verify the entire system.

## 1. Installation

Clone the repository and install the frozen environment.
```bash
git clone https://github.com/Khushijangra/NEXUS-ATMS.git
cd NEXUS-ATMS
conda env create -f environment.yml
conda activate spgrl
```

Alternatively, using `pip`:
```bash
pip install -r requirements.txt
```

## 2. Healthcheck (Verification)
Before running anything, verify that your CUDA, configs, checkpoints, and models are successfully loaded:
```bash
python run.py --healthcheck
```
**Expected Output**: `Healthcheck PASSED. Repository is fully operational.`

## 3. Demo Mode (Hackathon Evaluation)
To instantly visualize the complete pipeline (YOLO -> VideoMAE -> Graph -> LSTM -> PPO) running locally:
```bash
python run.py --demo
```

## 4. Run Training
To execute the multi-seed scientific validation baseline:
```bash
python run.py --train --config configs/ppo.yaml
```
*Results will automatically generate in `experiments/YYYYMMDD_ppo_seedX/` including `metadata.json` for data provenance.*

## 5. Automated Research Paper Generation
To output the final `.csv` evaluation logs, metrics, LaTeX tables, and 300 DPI Figures:
```bash
python run.py --paper
```
