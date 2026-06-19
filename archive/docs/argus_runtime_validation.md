# ARGUS Runtime Validation

## Execution Test
The primary ARGUS evaluation pipeline was tested using the frozen benchmark checkpoint to verify the structural integrity of the execution path.

### Command Executed
```powershell
python "argus_stream_extracted\argus stream A\scripts\eval_frame_level.py" --config-dir "argus_stream_extracted\argus stream A\configs" --checkpoint "argus_stream_extracted\argus stream A\outputs\checkpoints\stream_a_locked_videomae_beta1_score_norm_sigma0.pt"
```

### Execution Trace
- **Config Loaded**: Successfully loaded `configs/default.yaml` and resolved absolute paths relative to the ARGUS root.
- **Checkpoint Loaded**: Successfully loaded `stream_a_locked_videomae_beta1_score_norm_sigma0.pt` into the MULDEScorer.
- **First Stage Reached**: Yes. Instantiated `VideoMAEClipDataset(train/train)` with 19,192 clips from 186 videos.
- **Failure Point**: None.

### Conclusion
The script successfully extracted features, performed GMM anomaly scoring, computed the ROC AUC, and exited cleanly. The pipeline is fully functional.
