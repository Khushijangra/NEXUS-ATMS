# ARGUS Traffic Migration Blueprint

## File-by-File Migration Strategy

### 1. `src/models/backbones/videomae.py`
* **Status:** UNTOUCHED.
* **Justification:** The `VideoMAEFeatureExtractor` class accurately reads PyTorch `.half()` FP16 tensors and processes them via `AutoModel.from_config`. It strictly expects inputs of shape `[B, C, T, H, W]` which will remain standard across any dataset.

### 2. `src/models/scorers/mulde.py`
* **Status:** UNTOUCHED.
* **Justification:** `MULDEScorer` applies Denoising Score Matching natively on `[N, feature_dim]` generic embeddings. It mathematically scales with feature outputs and does not assume any sequence logic or spatial domains natively.

### 3. `src/training/train_stream.py`
* **Status:** UNTOUCHED.
* **Justification:** This file contains the generic loop pulling from the dataset. It does not hardcode `Avenue` or `UBnormal`; it dynamically passes evaluation states downstream.

### 4. `src/training/losses.py`
* **Status:** UNTOUCHED.
* **Justification:** Implements pure `mulde_loss` math targets.

### 5. `src/data/datasets.py`
* **Status:** UNTOUCHED.
* **Justification:** The function `load_metadata(metadata_dir, dataset_name="ubnormal")` hardcodes the expectation for `_splits.json`, `_frame_labels.json`, and `_scenes.json`. **We do not need to rewrite this code.** We simply create metadata that strictly adheres to these naming conventions, and pass `--dataset ua_detrac`.

### 6. `src/evaluation/stream_eval.py`
* **Status:** UNTOUCHED.
* **Justification:** Frame-level AUROC and ROC curve generation are agnostic to the video content, provided the JSON frame-level annotations exist in `data/metadata/`.

## Traffic Adaptation Requirements

**The exact migration path requires ZERO code rewrites within ARGUS.** 

Instead, it requires strict **Metadata Spoofing**:
We must build an external script that parses the raw UA-DETRAC XML/JSON bounding boxes, converts them to binary 1s (anomaly) and 0s (normal) for every frame of every video, and outputs exact `ua_detrac_splits.json`, `ua_detrac_frame_labels.json`, and `ua_detrac_scenes.json` files into `data/metadata/`.

By doing this, ARGUS natively supports UA-DETRAC without realizing the video context has changed from UBnormal to traffic.
