# MAPPO REALITY
**Did MAPPO ever train?**
NO.

**Exact missing component:**
No `train_mappo.py` exists. The multi-agent environment wrapper and CTDE neural networks (`Actor`, `Critic`) for the specific SPGRL setup are not integrated with a runner. Baseline single-agent PPO logs exist, but MAPPO does not.
