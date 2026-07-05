# EXECUTION GRAPH RECONSTRUCTION
| Edge | Classification | Note |
|---|---|---|
| RawVideo -> VideoMAE | MISSING | No direct ingestion |
| VideoMAE -> MULDE | EXECUTABLE | via train_stream.py |
| MULDE -> GMM | EXECUTABLE | via train_stream.py |
| GMM -> As | EXECUTABLE | via stream_a.py evaluate |
| YOLO -> DeepSORT | THEORETICAL_ONLY | |
| DeepSORT -> Ab | THEORETICAL_ONLY | |
| TrafficHistory -> LSTM | TRAINED_ONLY | Missing wrapper |
| LSTM -> Ft | MISSING | |
| RoadGraph -> GNN | THEORETICAL_ONLY | |
| GNN -> Gt | THEORETICAL_ONLY | |
| Carbon -> Ct | THEORETICAL_ONLY | |
| Emergency -> Et | EXECUTABLE | Isolated script |
| As+Ab+Ft+Gt+Ct+Et -> Zt | MISSING | No fusion code exists |
| Zt -> MAPPO | MISSING | |
| MAPPO -> SafetyShield | MISSING | |
| SafetyShield -> SignalControl| MISSING | |
