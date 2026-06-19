# Dataset Priority Matrix

## Phase D — Dataset Prioritization

Ranked from highest priority to lowest priority for adapting ARGUS to the traffic domain.

### 1. UA-DETRAC
* **Ease of acquisition**: Moderate (Direct download portal available for researchers).
* **Annotation quality**: Extremely high (Bounding boxes for vehicles, occlusion tags, weather tags).
* **Anomaly relevance**: High. Contains natural urban congestion, varied lighting, and dense vehicle tracking useful for establishing a "normal" background flow baseline.
* **Suitability for MULDE**: Excellent. The fixed-camera perspectives closely align with standard intersection surveillance, making it a perfect baseline for Normalizing Flow density estimation.

### 2. AI City Challenge (Track 3/4)
* **Ease of acquisition**: Hard (Requires formal email request, data sharing agreement, and approval delay).
* **Annotation quality**: High (Temporal bounds for vehicular anomalies like crashes and stalled vehicles).
* **Anomaly relevance**: Very High. Specifically designed for vehicular anomaly detection.
* **Suitability for MULDE**: Excellent. Contains exactly the positive/negative labels required to evaluate AUROC during validation.

### 3. DoTA (Dashcam Obstacle/Anomaly)
* **Ease of acquisition**: Easy (Google Drive/Baidu links often publicly available on GitHub).
* **Annotation quality**: High (Temporal bounds, anomaly categories).
* **Anomaly relevance**: High.
* **Suitability for MULDE**: Moderate. The moving dashcam perspective inherently breaks the fixed-background assumption of typical anomaly detection. Normalizing Flow models struggle heavily with moving camera perspectives, making this a secondary choice.

### 4. inD (Intersection Drone Dataset)
* **Ease of acquisition**: Hard (Strict institutional licensing required).
* **Annotation quality**: Very High (Trajectory mapping).
* **Anomaly relevance**: Low to Moderate. It is excellent for micro-simulation calibration, but lacks a high volume of explicit edge-case "anomalies" required for ARGUS stress-testing.
* **Suitability for MULDE**: Low. The top-down drone perspective does not match the angled intersection camera perspectives we are simulating.
