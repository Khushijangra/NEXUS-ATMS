# Dataset Mapping Report

## Traffic Dataset Strategy Mapping
If and when the physical datasets are acquired, they will be strictly mapped to the following architectural purposes to ensure scientific rigor and eliminate data leakage:

| Dataset | Purpose | Usage Constraints |
| :--- | :--- | :--- |
| **UA-DETRAC** | MULDE Backend Training | Exclusively used to fit the MULDE Normalizing Flow to "normal" baseline traffic behavior at intersections. No anomalies included during training. |
| **AI City Challenge** | Backend Validation | Used strictly as a validation set to tune the anomaly threshold and compute AUROC against true vehicular anomalies (crashes, stalled vehicles). |
| **DoTA** | External Stress Testing | Held out for out-of-distribution (OOD) testing to verify robustness against dashcam/moving-perspective anomalies. |
| **inD** | SUMO Environment Calibration | Used purely to calibrate synthetic traffic flow generation in NEXUS (NEXUS simulation ground-truth), not used for visual training. |

**STATUS**: Theoretical Mapping established. Awaiting physical files to execute.
