# Dataset Readiness Audit

## Overview
This report identifies the dataset assumptions currently hardcoded in the repository (specifically within the extracted ARGUS Stream A component) and outlines the files requiring migration to match the finalized Architecture Freeze strategy.

## Current Dataset Assumptions
The codebase currently assumes the use of pedestrian anomaly datasets.
- **Avenue Dataset:** Hardcoded assumptions exist across multiple scripts (e.g., `import_avenue_labels.py`, `scaffold_avenue_metadata.py`, `docs/avenue_dataset_audit.md`).
- **UBnormal Dataset:** The default configurations specify `dataset: ubnormal`.

## Configuration Files Identified for Migration
The following configuration files define the dataset structure and must be updated to replace pedestrian dataset paths with traffic dataset paths:

1. `argus_stream_extracted\argus stream A\configs\default.yaml`
   - Currently sets `data.dataset: ubnormal`. Needs to be configurable for traffic datasets.
2. `argus_stream_extracted\argus stream A\configs\stream_a_locked.yaml`
   - Currently sets `data.dataset: ubnormal` and hardcodes `num_scenes: 29`.
3. `argus_stream_extracted\argus stream A\src\data\datasets.py` (assumed based on standard PyTorch dataset loading).

## Required Dataset Migration Target
To meet the frozen architecture, the datasets must be migrated to:
- **AI City Challenge Track 4** (Normal training & Anomaly validation)
- **UA-DETRAC** (Normal training representation)
- **DoTA** (External anomaly testing subset)
- **inD** (SUMO behavior calibration)

## Status
**NOT READY.** The `data/raw` and `data/processed` folders exist in the root repository, but the ARGUS pipeline scripts have not been pointed to them, and the actual AI City/UA-DETRAC data files are not present or configured.
