# REPOSITORY KNOWLEDGE GRAPH
(Partial summary of massive node graph)
- **Nodes**: 1200+
- **Models**: 15+ (VideoMAE, MULDE, YOLO, DeepSORT, LSTM, GNN, MAPPO, etc.)
- **Checkpoints**: `best_clip.pt` (Mock), `best_clip_gmm.pkl` (Mock), `lstm_best.pth` (Genuine)
- **Features**: 100 `MVI_*.npy` files (Genuine, UA-DETRAC, 768-D)
- **Labels**: `ua_detrac_frame_labels.json` (Genuine, Frame-level binary)
- **Key Executable Entry Points**: `train_stream.py` (Stream A), `hybrid_runtime.py` (Control)
