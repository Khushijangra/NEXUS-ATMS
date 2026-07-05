# SPGRL Proposed Framework Scientific Consistency Audit

## 1. Complete SPGRL Pipeline
- Raw Video -> VideoMAE -> 768-D -> MULDE+GMM -> $A_s$ (Semantic Anomaly)
- Raw Video -> YOLO+DeepSORT -> Behavioral tracking -> $A_b$ (Behavioral Anomaly)
- Historical Traffic -> LSTM -> $F_t, C_f$ (Prediction & Confidence)
- Neighbor Graph -> GNN -> $G_t$ (Topological Embeddings)
- Real-time Traffic -> Carbon Engine -> $C_t$ (Emission Bounds)
- Traffic Telemetry -> Emergency Routing -> $E_t$ (Priority Override)
- All modalities -> Unified State $Z_t$ -> MAPPO CTDE -> Safety Shield -> Signal Control

## 2. Complete Dependency Graph
- MULDE strictly depends on VideoMAE.
- YOLO strictly provides inputs to DeepSORT.
- $Z_t$ strictly depends on $A_s, A_b, F_t, C_f, G_t, C_t, E_t$.
- MAPPO Actor strictly depends on local $z_i$. MAPPO Critic strictly depends on global $Z_t$.
- Safety Shield maintains terminal override dependency over MAPPO actions.

## 3. Complete Module Inventory
1. Sensing (Camera, Induction loops)
2. Semantic Perception (VideoMAE, MULDE, GMM)
3. Behavioral Perception (YOLO, DeepSORT)
4. Prediction (LSTM)
5. Graph (GNN, GAT)
6. Sustainability (Carbon Engine)
7. Priority (Emergency router)
8. State Constructor ($Z_t$ concatenator)
9. Control (MAPPO)
10. Safety (Deterministic Shield)

## 4. Input-Output Dimensions
- VideoMAE Input: 16-frame 3D temporal volume.
- VideoMAE Output: 768-D embedding.
- MULDE/GMM Output: 1D Scalar $[0, 1]$.
- LSTM Output: Forecast horizon matrix + scalar confidence.
- GNN Output: Reduced dimension topological representation array.
- $Z_t$ Output: 1D Dense flat vector per intersection.

## 5. State Dependencies
- MAPPO Actor policy $\pi(a|z_i)$ cannot compute without $Z_t$.
- $Z_t$ is resilient; if VideoMAE/YOLO fails, the fallback is the isolated numerical subset, though performance degrades gracefully.

## 6. Optimization Dependencies
- MAPPO Critic requires multi-objective reward integrating congestion, $C_t$, $A_t$ ($A_s$ and $A_b$), and $E_t$.
- Joint Optimization enforces gradient similarity between LSTM, GNN, and PPO backbones.

## 7. Execution Dependencies
- Offline Feature Extraction (VideoMAE, YOLO).
- Online Inference (GMM mapping, LSTM forward pass, GNN message passing).

## 8. Safety Dependencies
- Safety Shield is strictly independent of Neural computations. It relies solely on topological collision matrices and minimum statutory green times.

## 9. Training Dependencies
- CTDE mechanism requires a shared centralized buffer aggregating all $Z_t$ and joint rewards.
- Offline pre-training required for VideoMAE and MULDE.

## 10. Inference Dependencies
- Decentralized actors execute locally on isolated edge nodes (e.g., NVIDIA Jetson) using pre-trained Actor weights and real-time sensor ingestion.

## Audit Verdict
**STATUS:** PASS. The 10-layer architecture, dependency structures, and phase executions perfectly align with the intended Phase III parameters. Generation of Section II without equations is fully authorized.
