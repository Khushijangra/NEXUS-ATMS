# EXECUTION GRAPH EDGE AUDIT

| Edge | Status |
|---|---|
| Raw Video -> VideoMAE | MISSING (No active dataloader) |
| VideoMAE -> MULDE | PARTIAL (Features exist as .npy) |
| MULDE -> GMM -> As | PARTIAL (GMM pickle exists, inference script missing) |
| YOLO -> DeepSORT -> Ab | THEORETICAL ONLY |
| Historical Traffic -> LSTM -> Ft | PARTIAL (LSTM weights exist, wrapper missing) |
| Neighbor Graph -> GNN -> Gt | THEORETICAL ONLY |
| Carbon -> Ct | THEORETICAL ONLY |
| Emergency -> Et | THEORETICAL ONLY |
| As, Ab, Ft, Gt, Ct, Et -> Zt | MISSING (Unified state construction never executed) |
| Zt -> MAPPO | MISSING |
| MAPPO -> Safety Shield | MISSING |
| Safety Shield -> Signal Control | MISSING |
