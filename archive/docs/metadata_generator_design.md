# Metadata Generator Design

## Core Objective
Design a standalone Python script that reads the UA-DETRAC raw format and translates it into the three rigidly required JSON files for ARGUS, injecting anomaly labels via heuristic rules.

## IO Specification
*   **Input Directory:** `data/raw/ua_detrac/` (Containing sequence folders, video frames, and XML/JSON annotations)
*   **Output Directory:** `argus_stream_extracted/argus stream A/data/metadata/`
*   **Output Files:**
    *   `ua_detrac_splits.json`
    *   `ua_detrac_frame_labels.json`
    *   `ua_detrac_scenes.json`

## Processing Logic
1.  **Inventory Discovery:** Scan the input directory to find all valid sequence folders (e.g., `MVI_20011`).
2.  **Video Profiling:** For each sequence, read the total frame count from the directory or video header.
3.  **Split Assignment:** 
    *   Assign sequences predefined in UA-DETRAC's Train set to `train -> normal`.
    *   Assign a subset of Test sequences to `test -> abnormal`.
4.  **Heuristic Anomaly Injection:** For sequences placed into the `abnormal` splits, define an anomaly window (e.g., the last 30% of frames). Generate an array of `0`s, padding the anomaly window with `1`s.
5.  **Serialization:** Dump the dictionaries using standard Python `json.dump()`.

## Validation & Checksum Rules
*   **Missing Frame Label Detection:** Iterate over every video listed in `test -> abnormal` and `val -> abnormal`. Verify that the key exists in the `frame_labels` dictionary.
*   **Length Verification:** Verify that the length of the `frame_labels[video_name]` array exactly matches the total frame count of the video.
*   **Duplicate Detection:** Ensure `set(train_normal) & set(test_abnormal)` evaluates to an empty set to prevent data leakage.
*   **Empty Array Check:** Raise a fatal error if any `abnormal` array contains only `0`s (making it statistically impossible to calculate an AUROC).

## Failure Conditions
*   Raw UA-DETRAC files cannot be located.
*   Frame counts cannot be resolved for a specific sequence.
*   Generated `frame_labels.json` lacks keys for abnormal sequences.
