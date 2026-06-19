# UA-DETRAC Mapping Specification

To map UA-DETRAC into the ARGUS framework, we must adapt its structure to match the expected UBnormal schema.

## Component Mapping

1.  **Video ID:** The UA-DETRAC sequence folder name (e.g., `MVI_20011`, `MVI_39031`) maps perfectly to the `video_name` string expected by ARGUS.
2.  **Scene ID:** UA-DETRAC sequences are shot at different intersections. The intersection location identifier (e.g., Beijing intersection ID) can map to the `scene_id` integer. Alternatively, it can all be defaulted to `1`.
3.  **Train Split:** UA-DETRAC's native "Train" sequences map to the `train -> normal` array in ARGUS. MULDE trains strictly on normal data.
4.  **Validation Split:** A subset of "Train" or "Test" sequences should map to `val -> normal` and `val -> abnormal`.
5.  **Test Split:** UA-DETRAC's native "Test" sequences map to `test -> normal` and `test -> abnormal`.

## Critical Missing Information: Frame Labels

**The Discrepancy:**
UA-DETRAC is a Multi-Object Tracking and Detection dataset. Its annotations are XML/JSON bounding boxes defining cars, buses, and vans. **It natively lacks anomaly labels.** 

**The ARGUS Expectation:**
ARGUS rigorously expects a binary array of `0`s and `1`s for every single frame in an abnormal video. `datasets.py` line 203 raises an immediate `ValueError` if an abnormal video is missing from `frame_labels.json`.

**Required Resolution:**
To evaluate MULDE's anomaly detection on UA-DETRAC, we must algorithmically or manually assign "anomaly periods" to the UA-DETRAC test set. For example:
- Defining sudden braking or congestion as an anomaly.
- Generating a synthetic anomaly window (e.g., frames 500-600 are marked as `1`).
Without generating this mapping, ARGUS will fail to evaluate.
