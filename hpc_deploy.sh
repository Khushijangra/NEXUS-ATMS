#!/bin/bash
# ==============================================================================
# SPGRL HPC Deployment Script (Phase 2.5)
# ==============================================================================
# This script automates the complete scientific execution lifecycle across 
# multiple seeds and baseline evaluations without requiring manual intervention.
# 
# Usage: ./hpc_deploy.sh
# ==============================================================================

set -e

# Base configuration
CONFIG="configs/ppo.yaml"
SEEDS=(42 7 99)
MILESTONES=(500 750 1000)

echo "======================================================"
echo "    SPGRL AUTOMATED SCIENTIFIC COMPUTE PIPELINE       "
echo "======================================================"

# 1. Multi-Seed PPO Execution
for seed in "${SEEDS[@]}"; do
    echo "[*] Initiating Execution for Seed $seed"
    
    # Track resume state
    IS_RESUME=""
    
    for milestone in "${MILESTONES[@]}"; do
        echo "    -> Training to $milestone episodes..."
        python run.py --mode train --episodes $milestone --seed $seed $IS_RESUME
        
        EXP_DIR=$(ls -td experiments/*seed${seed}* | head -1)
        
        echo "    -> Evaluating all checkpoints at $milestone episodes..."
        python run.py --mode evaluate --exp_dir "$EXP_DIR" --episodes 50
        
        IS_RESUME="--resume"
    done
    
    echo "[*] Seed $seed pipeline completed."
    echo "------------------------------------------------------"
done

# 2. Benchmarks, Statistics, and Assets
echo "[*] Running Baseline Benchmarks..."
python run.py --mode benchmark --episodes 100

echo "[*] Running Ablation Studies..."
python run.py --mode ablation --episodes 100

echo "[*] Running Sensitivity Studies..."
python run.py --mode sensitivity --episodes 100

echo "[*] Generating Statistical Significance Tests..."
python run.py --mode statistics

echo "[*] Generating Final Paper Figures..."
python run.py --mode paper

echo "======================================================"
echo "    SPGRL PIPELINE EXECUTION COMPLETE.                "
echo "    All assets saved to experiments/ directory.       "
echo "======================================================"
