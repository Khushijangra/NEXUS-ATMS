# Final Pre-Execution Gate

## Remaining Blocking Dependencies

Before Step 1 of the Execution Sequence can physically run, the following strictly required assets must be acquired or built:

### 1. Data Assets
*   **UA-DETRAC Raw Videos:** The physical `.zip` or `.mp4` / `.jpg` sequence directories must be downloaded and extracted into `data/raw/ua_detrac/`.

### 2. Code Assets (To Be Written)
*   **`generate_ua_detrac_metadata.py`:** A standalone Python script must be explicitly written to parse UA-DETRAC sequences, synthetically inject anomalies (e.g., sudden halts or pseudo-events at the end of clips), and serialize the exact JSON schemas discovered during the metadata audit.
*   **`ua_detrac.yaml` Config:** The Hydra/argparse YAML file must be generated inside `argus_stream_extracted/argus stream A/configs/` to map dataset paths, model dimensions (`768`), and hyperparameters.

### 3. Hardware / Storage Requirements
*   **Storage Disk Space:** UA-DETRAC is large. There must be enough available space for the raw videos **plus** the `.npy` feature files. Given 768-dim FP16 embeddings per 16-frame clip, expect multiple GBs of additional feature overhead.
*   **GPU VRAM Constraints:** The RTX 2050 Mobile contains only 4GB of VRAM. VideoMAEv2-Base parameter loading + forward pass requires strict FP16 precision (`torch.autocast`) and an absolute `batch_size=1` forced onto the `DataLoader` within `extract_features.py` to avoid immediate CUDA Out of Memory (OOM) failures.

**Gate Status:** **NO-GO** (Blocked strictly by the physical acquisition of the UA-DETRAC dataset and the pending creation of the metadata generator script).
