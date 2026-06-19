# Repository Execution Blueprint

To guarantee zero codebase modifications to the core ARGUS files, the workspace must rigidly conform to the following directory and configuration state prior to the first training invocation.

## Required Pre-Execution Architecture

```text
c:\Users\Asus\OneDrive\Desktop\projects\urban congestion\
├── data/
│   ├── raw/
│   │   └── ua_detrac/
│   │       ├── MVI_20011/
│   │       ├── MVI_20012/
│   │       └── ... (All raw video frames or MP4 files)
│   ├── features/
│   │   └── ua_detrac/
│   │       └── videomae/
│   │           ├── MVI_20011.npy
│   │           ├── MVI_20012.npy
│   │           └── ... (FP16 Extracted Numpy features)
│
├── argus_stream_extracted/
│   └── argus stream A/
│       ├── configs/
│       │   └── ua_detrac.yaml               <-- REQUIRED: Must be created to point to features/metadata
│       ├── data/
│       │   └── metadata/
│       │       ├── ua_detrac_splits.json         <-- REQUIRED: Output of metadata generator
│       │       ├── ua_detrac_frame_labels.json   <-- REQUIRED: Output of metadata generator
│       │       └── ua_detrac_scenes.json         <-- REQUIRED: Output of metadata generator
│       ├── outputs/
│       │   ├── checkpoints/
│       │   │   └── ua_detrac/               <-- Auto-created during train.py
│       │   └── reports/
│       │       └── ua_detrac/               <-- Auto-created during eval_frame_level.py
│       │
│       ├── scripts/
│       │   ├── extract_features.py
│       │   ├── train.py
│       │   └── eval_frame_level.py
│       └── src/                             <-- ALL CORE FILES UNTOUCHED
```

## Immutable Assets
Under no circumstances should any scripts inside `src/models/`, `src/training/`, or `src/data/` be modified. If paths do not resolve, the error lies in the `configs/ua_detrac.yaml` mapping or the `metadata/` generation, not the ARGUS pipeline itself.
