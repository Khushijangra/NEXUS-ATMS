# 🚦 AI-Based Smart Traffic Management System

> **Reducing urban congestion through Deep Reinforcement Learning and SUMO simulation**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![SUMO](https://img.shields.io/badge/SUMO-1.18+-green)](https://eclipse.dev/sumo/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.1+-orange)](https://stable-baselines3.readthedocs.io/)

---

## 📋 Overview

An AI-powered traffic signal control system that uses **Deep Reinforcement Learning** (DQN / PPO) to optimise traffic light timings at urban intersections. The agent learns to minimise vehicle waiting times and queue lengths by interacting with a realistic **SUMO** microscopic traffic simulator.

### Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **RL Agents** | DQN and PPO implementations via Stable-Baselines3 |
| 🚗 **SUMO Integration** | Full Gymnasium environment with TraCI |
| 📊 **Live Dashboard** | Real-time WebSocket dashboard with Chart.js |
| 📈 **Visualisation** | Learning curves, comparison charts, HTML reports |
| 🎯 **Multiple Scenarios** | Rush hour, normal, night, asymmetric traffic |
| 🛣️ **Multi-Intersection** | 2×2 grid network for scalability demos |
| 🖥️ **Demo Mode** | Works without SUMO for presentations |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SUMO Simulator                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Network  │  │  Routes  │  │  Traffic Lights  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │ TraCI API
                     ▼
┌─────────────────────────────────────────────────────┐
│            Gymnasium Environment (sumo_env.py)       │
│  State: [queues, waits, phase, time] → 13 dims      │
│  Action: {0: keep, 1: switch}                        │
│  Reward: f(waiting_time, queue, throughput)           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              RL Agent (DQN / PPO)                    │
│  • MLP Policy (256 → 256)                            │
│  • Stable-Baselines3                                 │
│  • TensorBoard logging                              │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     [Training]  [Evaluation]  [Dashboard]
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python** 3.9+
- **SUMO** 1.18+ ([install guide](https://sumo.dlr.de/docs/Installing/index.html))
- Set `SUMO_HOME` environment variable

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Demo

```bash
# Full pipeline: train → evaluate → report → dashboard
python run_demo.py

# Dashboard only (no SUMO needed)
python run_demo.py --dashboard-only

# Custom training
python train.py --agent ppo --timesteps 100000
```

---

## 📁 Project Structure

```
Urban congestion/
├── configs/
│   ├── default.yaml                 # Main configuration
│   └── scenarios/                   # Traffic scenario configs
│       ├── rush_hour.yaml
│       ├── normal_traffic.yaml
│       └── night_traffic.yaml
├── dashboard/
│   ├── backend/main.py              # FastAPI + WebSocket server
│   ├── frontend/index.html          # Real-time dashboard UI
│   └── demo_data.py                 # Synthetic data generator
├── docs/
│   ├── architecture.md              # System architecture
│   └── benchmarks.md                # Performance benchmarks
├── networks/
│   ├── single_intersection.net.xml  # 4-way intersection
│   ├── single_intersection.rou.xml  # Traffic routes
│   ├── grid_2x2.net.xml             # 2×2 grid network
│   └── grid_2x2.rou.xml
├── scripts/
│   ├── generate_scenarios.py        # Scenario route generator
│   ├── generate_report.py           # HTML/PNG report builder
│   └── quick_train.py               # Quick demo training
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py             # DQN implementation
│   │   └── ppo_agent.py             # PPO implementation
│   ├── envs/
│   │   └── sumo_env.py              # Gymnasium SUMO environment
│   └── utils/
│       ├── logger.py                # Structured logging
│       ├── metrics.py               # Metrics tracker
│       └── visualization.py         # Charts & figures
├── evaluate.py                      # Baseline vs RL evaluation
├── train.py                         # Main training script
├── run_demo.py                      # One-click demo
├── requirements.txt
└── README.md
```

---

## 🎮 Usage

### Training

```bash
# Train PPO agent (default)
python train.py --agent ppo --timesteps 500000

# Train DQN agent
python train.py --agent dqn --timesteps 200000

# Quick demo (50K steps)
python train.py --agent ppo --demo

# With SUMO GUI
python train.py --agent ppo --gui
```

### Evaluation

```bash
# Evaluate and compare with baseline
python evaluate.py --model models/<run>/best/best_model.zip --agent ppo --report
```

### Dashboard

```bash
# With SUMO data
python dashboard/backend/main.py

# Demo mode (no SUMO)
set DEMO_MODE=true
python dashboard/backend/main.py
# Open http://localhost:8000
```

### Generate Scenarios

```bash
python scripts/generate_scenarios.py --scenario all
```

---

## 📊 Expected Results

| Metric | Fixed-Timing | RL Agent (PPO) | Improvement |
|--------|-------------|----------------|-------------|
| Avg Waiting Time | ~45s | ~32s | **-29%** |
| Avg Queue Length | ~12 veh | ~8 veh | **-33%** |
| Throughput | ~850 veh/hr | ~1020 veh/hr | **+20%** |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Simulation | SUMO + TraCI |
| RL Training | PyTorch + Stable-Baselines3 |
| Environment | Gymnasium |
| Dashboard Backend | FastAPI + WebSocket |
| Dashboard Frontend | HTML5 + Chart.js |
| Visualisation | Matplotlib + Seaborn |
| Logging | Python logging + Rich |

---

## 📚 References

1. Wei et al., *"IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Signal Control"*, KDD 2018
2. Zheng et al., *"Learning Phase Competition for Traffic Signal Control"*, CIKM 2019
3. Alegre, L.N., *"SUMO-RL"*, GitHub 2019
4. Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv 2017

---

## 📄 License

This project is released under the MIT License.
See [LICENSE](LICENSE) for details.

---

## ☁️ Publish To GitHub

If Git is not installed yet on Windows, install it first:

- https://git-scm.com/download/win

Then run:

```bash
git init
git add .
git commit -m "Initial commit: NEXUS-ATMS"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
