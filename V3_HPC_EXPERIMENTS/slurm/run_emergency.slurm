#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu

echo "Starting job on $(hostname)"
python emergency/run_routing.py
