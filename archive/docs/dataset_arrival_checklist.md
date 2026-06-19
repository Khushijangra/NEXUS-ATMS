# Dataset Arrival Checklist

Follow this checklist line-by-line the moment the UA-DETRAC zip files hit the physical hard drive.

- [ ] **1. Ingest Raw Data**
  - [ ] Extract UA-DETRAC archives into `data/raw/ua_detrac/`.
  - [ ] Verify `MVI_XXXX` sequence folders exist immediately under `ua_detrac/`.
  - [ ] Run an OpenCV sanity check to ensure video frames/MP4s are not corrupted.

- [ ] **2. Configuration Scaffolding**
  - [ ] Create `argus_stream_extracted/argus stream A/configs/ua_detrac.yaml`.
  - [ ] Point `features_dir` to `../../data/features/ua_detrac/videomae`.
  - [ ] Point `metadata_dir` to `data/metadata`.

- [ ] **3. Metadata Spoofing**
  - [ ] Execute `generate_ua_detrac_metadata.py` (Script to be built).
  - [ ] Physically verify `ua_detrac_splits.json` exists.
  - [ ] Physically verify `ua_detrac_frame_labels.json` exists.
  - [ ] Physically verify `ua_detrac_scenes.json` exists.
  - [ ] Inspect JSONs to verify zero empty abnormal frame label arrays.

- [ ] **4. Execution**
  - [ ] Run `scripts/extract_features.py`.
  - [ ] Monitor GPU VRAM (`nvidia-smi`) to ensure VideoMAE batch size 1 stays < 4GB.
  - [ ] Verify `.npy` files populate `data/features/ua_detrac/videomae/`.
  - [ ] Run `scripts/train.py`.
  - [ ] Verify `mulde_best.pth` and `mulde_best_gmm.pkl` generate in checkpoints.
  - [ ] Run `scripts/eval_frame_level.py`.
  - [ ] Verify final ROC curve and AUROC metrics generate in `outputs/reports/`.
