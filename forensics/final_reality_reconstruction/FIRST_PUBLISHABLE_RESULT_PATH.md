# SHORTEST PATH TO FIRST PUBLISHABLE RESULT
The absolute shortest path to obtain ONE genuine IEEE result is to execute the Semantic Pipeline (Stream A) training.

- **Exact script**: `argus_stream_extracted/argus stream A/src/training/train_stream.py`
- **Exact files**: `data/features/ua_detrac/videomae/*.npy`, `data/processed/ua_detrac_metadata/ua_detrac_frame_labels.json`
- **Exact models**: MULDE, GaussianMixture
- **Exact effort estimate**: 1-2 hours (Script runs on CPU/GPU locally to train MULDE score matching and fit GMM, directly outputting AUROC).
