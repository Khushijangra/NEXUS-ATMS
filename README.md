# Semantic Predictive Graph Reinforcement Learning (SPGRL)

SPGRL is a next-generation urban congestion management framework combining Visual Semantic Streams, Predictive Forecasting, Graph Embeddings, and Multi-Agent Reinforcement Learning to optimize urban traffic flow.

## Project Structure
- `v2/`: The core library containing all neural stream wrappers and PPO logic.
- `configs/`: Hyperparameter settings (`ppo.yaml`) enforcing reproducibility.
- `experiments/`: Experiment registry and timestamped output folders.
- `models/`: Central checkpoint registry for best evaluation models.
- `tests/`: 43+ unit tests covering stream dimensionality, reward boundaries, and gradient assertions.

## Reproducibility
Every execution creates a timestamped artifact folder containing a `metadata.json` which tracks `git_commit`, `TorchVersion`, dataset paths, and seed information.

## Getting Started
Please view `QuickStart.md` for Hackathon demonstration and Scientific execution instructions.
