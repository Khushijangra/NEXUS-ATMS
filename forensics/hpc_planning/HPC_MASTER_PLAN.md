# HPC Master Execution Plan
## Pre-flight Checks
1. SSH into `s_01kvfr6ww772zvqw440bbxa89n@ssh.lightning.ai`
2. Verify: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

## SLURM Execution Order
1. Semantic / Behavioral (Parallelizable perception phase)
2. Prediction
3. GNN
4. Carbon / Emergency (Parallelizable constraints)
5. Joint Optimization (Gradient checks)
6. MAPPO (10,000 episode training block)
7. Statistics

## Rule of Progression
Do NOT proceed to step N+1 until all expected outputs and provenance YAMLs for step N are locally secured.
