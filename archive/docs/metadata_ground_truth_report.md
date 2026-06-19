# Metadata Ground Truth Report

Based on repository evidence found in `argus_stream_extracted/argus stream A/src/data/datasets.py` and existing UBnormal/Avenue JSON files, the metadata format relies on three JSON files with rigid structures.

## 1. splits.json

### Schema Definition
*   **Top-level Keys:** `train`, `val`, `test` (Strings)
*   **Nested Keys:** Inside each split key, two required keys must exist: `normal` and `abnormal`.
*   **Expected Data Types:** Lists of strings. Each string is the exact base filename of a video sequence (without the `.npy` or `.mp4` extension).
*   **Required Fields:** The structural dictionaries for `train`, `val`, and `test` containing `normal` and `abnormal` keys.
*   **Filename Convention:** `[dataset_name]_splits.json`

## 2. frame_labels.json

### Schema Definition
*   **Top-level Keys:** Exact base filenames of the videos (Strings).
*   **Nested Keys:** None.
*   **Expected Data Types:** Arrays of integers (`[0, 1]`). `0` represents a normal frame, `1` represents an anomalous frame.
*   **Required Fields:** `datasets.py` explicitly strictly requires frame labels for any video placed into the `abnormal` list of any split. (Lines 202-207 in `datasets.py`).
*   **Optional Fields:** Normal videos do not require frame-level labels (the code generates `np.zeros`).
*   **Filename Convention:** `[dataset_name]_frame_labels.json`

## 3. scenes.json

### Schema Definition
*   **Top-level Keys:** Exact base filenames of the videos (Strings).
*   **Nested Keys:** None.
*   **Expected Data Types:** Integers representing the Scene ID (e.g., `1`, `2`, `14`).
*   **Required Fields:** None. `datasets.py` contains fallback logic (Lines 60-68) that defaults all videos to Scene `1` if `scenes.json` is missing or incomplete.
*   **Filename Convention:** `[dataset_name]_scenes.json`

## Example Reconstruction

**Example splits.json:**
```json
{
  "train": {
    "normal": ["MVI_20011", "MVI_20012"],
    "abnormal": []
  },
  "val": {
    "normal": ["MVI_20031"],
    "abnormal": ["MVI_20032"]
  },
  "test": {
    "normal": [],
    "abnormal": ["MVI_20051", "MVI_20052"]
  }
}
```

**Example frame_labels.json:**
```json
{
  "MVI_20032": [0, 0, 0, 1, 1, 1, 0, 0],
  "MVI_20051": [0, 0, 0, 0, 0, 0, 1, 1],
  "MVI_20052": [1, 1, 1, 1, 0, 0, 0, 0]
}
```

**Example scenes.json:**
```json
{
  "MVI_20011": 1,
  "MVI_20012": 1,
  "MVI_20031": 1,
  "MVI_20032": 1,
  "MVI_20051": 2,
  "MVI_20052": 2
}
```
