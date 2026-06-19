# Mandatory Dependency Report

## Audited Dependencies
- `timm`
- `transformers`
- `gradio`

## Analysis
1. **timm**: 
   - **Status**: MANDATORY
   - **Usage Path**: Required internally by the `transformers` library when instantiating the OpenGVLab/VideoMAEv2-Base model.
   - **Execution Paths**: Any script that initializes the `VideoMAEFeatureExtractor` (e.g., `extract_features.py`, inference pipelines).

2. **transformers**:
   - **Status**: MANDATORY
   - **Usage Path**: Directly imported in `src/models/backbones/videomae.py` (lines 179-181) to load `AutoConfig`, `AutoModel`, and `VideoMAEImageProcessor`.
   - **Execution Paths**: Feature extraction, training, and real-time inference.

3. **gradio**:
   - **Status**: OPTIONAL
   - **Usage Path**: Only used in `demo.py` for the UI. It is wrapped in a try-catch block (`try: import gradio as gr except: ...`) and does not break the core CLI pipelines.
   - **Execution Paths**: Only required when launching the visual `demo.py` tool.
