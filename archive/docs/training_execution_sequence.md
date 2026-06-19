# Training Execution Sequence

The exact sequential pipeline to execute from dataset arrival to final AUROC quantification, utilizing only the native ARGUS entry points. 

*All commands assume the working directory is `argus_stream_extracted/argus stream A/`.*

### Step 1: Metadata Generation
*(This script remains a blocking dependency to be written)*
```bash
python ../../scripts/generate_ua_detrac_metadata.py \
    --input_dir ../../data/raw/ua_detrac/ \
    --output_dir data/metadata/
```

### Step 2: FP16 Feature Extraction
*Extracts VideoMAE-Base hidden states from raw frames into `.npy` files. Forced to batch size 1 to survive the RTX 2050 4GB constraint.*
```bash
python scripts/extract_features.py \
    --config-dir configs \
    --dataset ua_detrac \
    --batch-size 1 \
    --fp16 \
    --device cuda
```

### Step 3: Native MULDE Training
*Trains the Denoising Score Matching Normalizing Flow natively using `train_stream.py` inside `train.py`.*
```bash
python scripts/train.py \
    --config-dir configs \
    --dataset ua_detrac \
    --output-dir outputs/checkpoints/ua_detrac \
    --device cuda
```

### Step 4: Frame-Level AUROC Evaluation
*Loads the pickled GMM and PyTorch model state from the checkpoint and evaluates the negative log-likelihood against `ua_detrac_frame_labels.json`.*
```bash
python scripts/eval_frame_level.py \
    --config-dir configs \
    --dataset ua_detrac \
    --checkpoint-path outputs/checkpoints/ua_detrac/mulde_best.pth \
    --device cuda
```
