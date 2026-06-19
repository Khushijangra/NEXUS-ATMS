# Dataset Migration Plan

The following files contain hardcoded or default references to `avenue` and `ubnormal` that must be migrated to `AI City`, `UA-DETRAC`, `DoTA`, and `inD` during Sprint One.

## 1. Documentation
- `argus_stream_extracted\argus stream A\README.md`
- `argus_stream_extracted\argus stream A\CODEX.md`
- `argus_stream_extracted\argus stream A\viva_qa.md`
- `argus_stream_extracted\argus stream A\docs\avenue_dataset_audit.md`
- `argus_stream_extracted\argus stream A\RUN ME FIRST.txt`

## 2. Metadata / Data Loading
- `argus_stream_extracted\argus stream A\src\data\datasets.py`
- `argus_stream_extracted\argus stream A\scripts\scaffold_avenue_metadata.py`
- `argus_stream_extracted\argus stream A\scripts\import_avenue_labels.py`

## 3. Evaluation Scripts & Demos
- `argus_stream_extracted\argus stream A\src\evaluation\stream_a.py`
- `argus_stream_extracted\argus stream A\src\evaluation\metrics.py`
- `argus_stream_extracted\argus stream A\scripts\eval_frame_level.py`
- `argus_stream_extracted\argus stream A\demo.py`
- `argus_stream_extracted\argus stream A\deployment\vercel_app\app\page.tsx`

## 4. Configuration & Shell Scripts
- `argus_stream_extracted\argus stream A\configs\default.yaml`
- `argus_stream_extracted\argus stream A\configs\stream_a_locked.yaml`
- `argus_stream_extracted\argus stream A\run_train_avenue.bat`
- `argus_stream_extracted\argus stream A\run_sweep_avenue_eval.bat`
- `argus_stream_extracted\argus stream A\run_eval_avenue.bat`
- `argus_stream_extracted\argus stream A\run_eval_frozen.bat`
