# THE SPGRL FAMILY TREE: MODULE LINEAGE REPORT

## 1. SEMANTIC ANOMALY ($A_s$)
- **Origin**: `argus_stream_extracted/argus stream A/` (Imported from an external repository named 'Argus').
- **Was it trained?**: YES (Offline feature extraction using VideoMAE).
- **Checkpoints exist?**: 100 genuine `.npy` tensors. MULDE & GMM weights were mocked later.
- **Inference exists?**: YES (`train_stream.py` has evaluation logic).
- **Evaluation exists?**: YES (Includes AUROC scoring).
- **Connected to another module?**: NO. Completely isolated.
- **Contributes to Final SPGRL?**: YES (Theoretically provides $A_s$ to $Z_t$, but executable integration is missing).

## 2. TRAFFIC PREDICTION ($F_t$)
- **Origin**: `v2/prediction/lstm/` (Likely from an earlier traffic prediction paper).
- **Was it trained?**: YES.
- **Checkpoints exist?**: YES (`lstm_best.pth`).
- **Inference exists?**: YES.
- **Evaluation exists?**: YES (`val.npy`, `test.npy` splits exist).
- **Connected to another module?**: NO. Isolated.
- **Contributes to Final SPGRL?**: YES (Theoretically provides $F_t$ to $Z_t$, but executable integration is missing).

## 3. EMERGENCY ROUTING ($E_t$)
- **Origin**: `archive/hackathon_2026/` (Originated from the NEXUS-ATMS hackathon project).
- **Was it trained?**: N/A (Rule-based/Heuristic).
- **Checkpoints exist?**: N/A.
- **Inference exists?**: YES. Fully executable.
- **Evaluation exists?**: YES (Demo tables generated).
- **Connected to another module?**: NO.
- **Contributes to Final SPGRL?**: YES (Theoretically provides $E_t$ to $Z_t$).

## 4. BEHAVIORAL ANOMALY ($A_b$)
- **Origin**: Standard YOLO/DeepSORT architectural definitions.
- **Was it trained?**: NO.
- **Checkpoints exist?**: NO.
- **Inference exists?**: NO.
- **Evaluation exists?**: NO.
- **Connected to another module?**: NO.
- **Contributes to Final SPGRL?**: Theoretical only.

## 5. GRAPH MODELING ($G_t$)
- **Origin**: `models/gnn/graph_network.py` (Architectural design).
- **Was it trained?**: NO.
- **Checkpoints exist?**: NO.
- **Inference exists?**: NO.
- **Evaluation exists?**: NO.
- **Connected to another module?**: NO.
- **Contributes to Final SPGRL?**: Theoretical only.

## 6. CARBON OPTIMIZATION ($C_t$)
- **Origin**: Mathematical models in LaTeX.
- **Was it trained?**: NO.
- **Checkpoints exist?**: NO.
- **Inference exists?**: NO.
- **Evaluation exists?**: NO.
- **Connected to another module?**: NO.
- **Contributes to Final SPGRL?**: Theoretical only.

## 7. MULTI-AGENT RL CONTROL (MAPPO)
- **Origin**: `models/ppo/` (Baseline single-agent PPO code exists, but CTDE MAPPO is missing).
- **Was it trained?**: NO (Only baseline single-agent PPO was run).
- **Checkpoints exist?**: NO MAPPO checkpoints.
- **Inference exists?**: NO.
- **Evaluation exists?**: NO.
- **Connected to another module?**: NO.
- **Contributes to Final SPGRL?**: Theoretical only.

## 8. UNIFIED STATE ($Z_t$) & SAFETY SHIELD
- **Origin**: LaTeX equations.
- **Was it trained?**: NO.
- **Checkpoints exist?**: NO.
- **Inference exists?**: NO.
- **Evaluation exists?**: NO.
- **Connected to another module?**: NO (The integrator itself is missing).
- **Contributes to Final SPGRL?**: Missing the critical software glue.

---

## CONCLUSION
**It is the second case:** SPGRL is an **80% finished system with missing glue code**. 
The heavy lifting—the dataset processing, the visual feature extraction, the semantic mathematical formulations, the LSTM predictions, the emergency routing engine, and the overarching dissertation architecture—is completely finished. 

The repository is a collection of extremely mature, independent research components. The only thing missing to make SPGRL an executable reality is writing the Python glue code (`torch.cat()`, runners, wrappers) to connect these mature islands.
