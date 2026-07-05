# SPGRL REPOSITORY GRAPH
## Stream 1: Semantic Anomaly
```text
RAW VIDEO
    ↓ (Missing dataloader)
VideoMAE (models/semantic/videomae.py exists)
    ↓ (data/features/ua_detrac/videomae/MVI_*.npy exists)
feature.npy
    ↓ (models/semantic/mulde.py exists)
MULDE
    ↓ (models/semantic/gmm.py exists, models/pretrained/stream_a/best_clip_gmm.pkl exists)
GMM
    ↓ (Missing integration)
As
```

## Stream 2: Behavioral Anomaly
```text
RAW VIDEO
    ↓ (Missing dataloader)
YOLO (models/behavioral/yolo.py exists)
    ↓ (Missing checkpoints)
DeepSORT (models/behavioral/deepsort.py exists)
    ↓ (Missing runners)
Ab
```

## Stream 3: Prediction
```text
Historical Traffic
    ↓ (v2/prediction/lstm/dataset/train.npy exists)
LSTM (models/prediction/lstm.py exists, v2/prediction/lstm/lstm_best.pth exists)
    ↓ (Missing integration)
Ft
```

## Stream 4: Graph Topology
```text
Neighbor Graph
    ↓ (Missing adjacency data)
GNN (models/gnn/graph_network.py exists)
    ↓ (Missing checkpoints)
Gt
```
