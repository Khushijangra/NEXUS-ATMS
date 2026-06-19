# VideoMAE Integrity Report

## Paths & Logic
- **VideoMAE Backbone File**: `src/models/backbones/videomae.py`
- **Model Factory**: `VideoMAEFeatureExtractor.__init__` uses the Hugging Face `transformers` pipeline (`AutoConfig.from_pretrained`, `AutoModel.from_config`).
- **Checkpoint Loading Logic**: It dynamically downloads `model.safetensors` using `huggingface_hub.hf_hub_download` and loads it via `safetensors.torch.load_file` into CPU memory first (bypassing accelerate meta-tensors) before moving to GPU.

## Integrity Verification
1. **Expected Model Names**: `OpenGVLab/VideoMAEv2-Base` (hardcoded default).
2. **Expected Checkpoint Paths**: No local checkpoint file is required in the repository. It relies entirely on the Hugging Face hub cache (`~/.cache/huggingface/hub/`).
3. **Do Checkpoints Exist**: Yes, the logic correctly targets the public Hugging Face repository, so the checkpoint exists in the cloud and will be cached upon first execution.

## Status: Intact
