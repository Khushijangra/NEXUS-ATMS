# NEXUS-ATMS

AI-driven urban traffic management prototype with reinforcement learning, prediction, anomaly detection, digital twin visualization, and a FastAPI dashboard.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Khushijangra/NEXUS-ATMS/actions/workflows/ci.yml/badge.svg)](https://github.com/Khushijangra/NEXUS-ATMS/actions/workflows/ci.yml)

## What This Repository Currently Provides

- RL agents (DQN/PPO) and traffic environments (SUMO + standalone synthetic environment)
- Prediction stack: LSTM forecasting and anomaly detection
- Specialty modules: emergency corridor, carbon engine, pedestrian safety, cybersecurity, road maintenance, NL commands, voice broadcast, counterfactual engine
- FastAPI backend with REST + WebSocket endpoints
- Web dashboard and digital twin renderer

## Runtime Modes (Important)

This repository supports two practical operating modes today:

1. Dashboard demo mode
- Works without SUMO
- Streams simulated but realistic traffic snapshots
- Best for presentations and module demonstrations

2. Simulation/training mode
- Uses SUMO where required by training/evaluation scripts
- Produces model checkpoints and benchmark artifacts

Note: Real-world deployment wiring (live camera feeds, physical controllers, production sensor brokers) is not fully integrated end-to-end yet.

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional but recommended) SUMO 1.18+ for simulation scripts
- Windows PowerShell or compatible shell

### Install

```bash
pip install -r requirements.txt
```

### Run Dashboard (Fastest)

```bash
python run_demo.py --dashboard-only
```

Then open:

- http://127.0.0.1:8000 (or the port printed by the backend)

### Train / Evaluate

```bash
python train.py --agent dqn --timesteps 50000 --demo
python evaluate.py --model models/<run>/best/best_model.zip --agent dqn --report
```

## Repository Layout

```text
.
├── dashboard/
│   ├── backend/main.py
│   ├── frontend/index.html
│   └── demo_data.py
├── control/
├── iot/
├── prediction/
├── vision/
├── modules/
├── src/
├── networks/
├── configs/
├── scripts/
├── docs/
├── train.py
├── evaluate.py
├── run_demo.py
└── run_digital_twin.py
```

## Documentation

- Architecture overview: [docs/architecture.md](docs/architecture.md)
- Benchmarks and model metrics: [docs/benchmarks.md](docs/benchmarks.md)
- Implementation audit checklist: [docs/implementation_checklist.md](docs/implementation_checklist.md)

## Governance and Community

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Pull request template: [.github/pull_request_template.md](.github/pull_request_template.md)

## Roadmap (Near-Term)

- Wire real camera/SUMO frame ingestion to vision pipeline in live backend loop
- Wire MQTT broker-backed sensor ingestion for IoT fusion
- Move historical analytics to persistent storage
- Add authenticated production control surface for authority endpoints

## License

Released under the MIT License.
See [LICENSE](LICENSE).
