# Dataset Readiness Audit

## Physical Existence Verification
- **AI City Challenge**: NOT FOUND. Local path empty.
- **UA-DETRAC**: NOT FOUND. Local path empty.
- **DoTA**: NOT FOUND. Local path empty.
- **inD**: NOT FOUND. Local path empty.
- **`data/raw` Directory**: Verified empty.
- **`data/processed` Directory**: Verified empty.

## Expected Requirements vs Current Reality

| Dataset | Expected Size (Est.) | Expected Clip Count | Annotation Type | Current Availability |
| :--- | :--- | :--- | :--- | :--- |
| **AI City** | ~100+ GB | Thousands | Vehicular anomaly, tracking | 0 bytes / Missing |
| **UA-DETRAC** | ~15 GB | ~100 sequences | Bounding boxes (vehicles) | 0 bytes / Missing |
| **DoTA** | ~25 GB | ~4,600 videos | Temporal anomaly bounds | 0 bytes / Missing |
| **inD** | ~100+ GB | 33 sequences | Drone-view trajectories | 0 bytes / Missing |

## Licensing Constraints
- **AI City**: Requires official academic request and data sharing agreement.
- **UA-DETRAC**: Non-commercial research use only.
- **DoTA**: Non-commercial research use only.
- **inD**: Requires academic licensing agreement via RWTH Aachen University.

## Conclusion
**STATUS: FATAL ABSENCE**
None of the required multimodal traffic datasets exist in the local workspace. No physical video clips, feature files, or temporal labels are available to adapt the ARGUS anomaly detector to the urban traffic domain.
