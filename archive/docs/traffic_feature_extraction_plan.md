# Traffic Feature Extraction Plan

## Objective
Design the workflow to parse raw traffic videos (AI City, DoTA, etc.), run them through VideoMAE-v2, and dump the features to disk without triggering VRAM limits.

## Directory Structure Design
```text
data/
├── raw/
│   ├── aicity/
│   │   ├── videos/
│   │   └── annotations/
│   └── dota/
└── features/
    ├── aicity/
    │   └── videomae/
    │       ├── train/
    │       │   ├── video_001.npy
    │       │   └── video_002.npy
    │       └── test/
    └── dota/
```

## Extraction Workflow
1. **Pre-processing (`scripts/preprocess_videos.py`)**:
   - Resize videos to `224x224`.
   - Sample at 16 frames per clip.
2. **Extraction (`scripts/extract_features_traffic.py`)**:
   - Load `VideoMAEFeatureExtractor`.
   - Batch size strictly limited to `1` or `2` to respect the 4GB VRAM limit.
   - Pass 16-frame chunks into VideoMAE.
   - Extract the pooled output `[768]` dimensional vector.
   - Save sequentially to `data/features/[dataset]/videomae/[split]/[video_name].npy`.

## Pre-requisites Before Implementation
- Download datasets.
- Ensure `timm` and `decord` or `cv2` are installed for fast video loading.
- Verify VideoMAE checkpoint downloads correctly.
