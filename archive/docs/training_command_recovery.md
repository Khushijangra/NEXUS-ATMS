# Training Command Recovery

Based on the source code structure of `scripts/train.py`, `src/utils/config.py`, and `train_stream.py`, the native ARGUS training entry point must be executed via command line arguments mapping to a YAML configuration file.

### Required Execution Context
* **Working Directory:** `argus_stream_extracted/argus stream A/`
* **Python Path:** `.` (Must include project root)

### Exact Command

```bash
python scripts/train.py \
    --config-dir ../../configs \
    --dataset ua_detrac \
    --output-dir outputs/checkpoints/ua_detrac
```

### Config File Requirements (`ua_detrac.yaml`)
To prevent fallback to Avenue/UBnormal, we must generate a new config containing:
```yaml
dataset:
  name: "ua_detrac"
  features_dir: "../../data/features/ua_detrac/videomae"
  metadata_dir: "../../data/metadata"

model:
  name: "mulde"
  feature_dim: 768
  gmm_components: 10
  beta: 0.1

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.0001
```

By executing `scripts/train.py` directly, `train_stream.py` natively ingests the UA-DETRAC `.npy` features and mathematically processes Denoising Score Matching without mock loops or placeholder checkpoints.
