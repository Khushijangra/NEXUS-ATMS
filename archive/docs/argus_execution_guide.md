# ARGUS Execution Guide

## Correct Baseline Execution Command
To execute the ARGUS evaluation pipeline without throwing the `AttributeError`, the script must be explicitly directed to the internal config directory.

### Command
```powershell
python "argus_stream_extracted\argus stream A\scripts\eval_frame_level.py" --config-dir "argus_stream_extracted\argus stream A\configs" --checkpoint "argus_stream_extracted\argus stream A\outputs\checkpoints\stream_a_locked_videomae_beta1_score_norm_sigma0.pt"
```

*(Note: The `--checkpoint` path assumes the presence of the default frozen checkpoint. If testing a newly trained baseline, replace the checkpoint path appropriately).*

### Why this works:
By overriding `--config-dir`, `src.utils.config.load_config()` will correctly locate the `default.yaml` containing the `evaluation` section, and the recursive path resolution will properly align with the dataset assumptions.
