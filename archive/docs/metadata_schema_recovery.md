# Metadata Schema Recovery

Based on the actual JSON files existing in `argus stream A/data/metadata/` (`avenue_splits.json`, `avenue_frame_labels.json`, `avenue_scenes.json`), ARGUS expects the following rigid schemas.

## 1. `[dataset]_splits.json`
Defines the training, validation, and testing logic. MULDE trains strictly on `train -> normal`.

```json
{
  "train": {
    "normal": ["MVI_20011", "MVI_20012"],
    "abnormal": []
  },
  "val": {
    "normal": ["MVI_20031"],
    "abnormal": []
  },
  "test": {
    "normal": [],
    "abnormal": ["MVI_20051", "MVI_20052"]
  }
}
```

## 2. `[dataset]_frame_labels.json`
Maps every single video filename to an array matching its total frame count. `0` = Normal, `1` = Anomaly. For evaluation purposes, UA-DETRAC videos must be padded accurately.

```json
{
  "MVI_20051": [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0
  ],
  "MVI_20052": [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
}
```

## 3. `[dataset]_scenes.json`
Used natively for UBnormal (which has 29 distinct scenes). For UA-DETRAC, we can default all intersections to scene `1` or separate them if desired.

```json
{
  "MVI_20011": 1,
  "MVI_20012": 1,
  "MVI_20031": 1,
  "MVI_20051": 1,
  "MVI_20052": 1
}
```

**Implementation Gate:** Feature extraction cannot begin until these three JSON files are physically constructed and verified for UA-DETRAC.
