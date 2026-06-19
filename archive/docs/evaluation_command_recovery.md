# Evaluation Command Recovery

Based on the ARGUS codebase, evaluation natively loads the checkpoint produced by `scripts/train.py` and processes it via `scripts/eval_frame_level.py`.

### Required Execution Context
* **Working Directory:** `argus_stream_extracted/argus stream A/`
* **Python Path:** `.` (Must include project root)

### Exact Command

```bash
python scripts/eval_frame_level.py \
    --config ../../configs/ua_detrac.yaml \
    --dataset ua_detrac \
    --checkpoint_path outputs/checkpoints/ua_detrac/mulde_best.pth
```

### Mechanism of Action
1. It calls `MULDEScorer.load_checkpoint(path)` from `src/models/scorers/mulde.py`.
2. It loads `data/metadata/ua_detrac_splits.json`.
3. It loads the `abnormal` videos defined under `test` in the split.
4. It calls `compute_multiscale_signal()` and extracts the negative log likelihood.
5. It cross-references the per-clip scores against `ua_detrac_frame_labels.json` using `src/evaluation/stream_eval.py`.
6. It computes and saves the final AUROC, producing the scientific metric.
