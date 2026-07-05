# FINAL SEMANTIC CONSISTENCY AUDIT

## STEP 2: GMM Pickle Properties
- **sklearn class name:** `GaussianMixture`
- **n_components:** `2`
- **covariance_type:** `full`
- **expected feature dimension:** `16`
- **fitted means shape:** `(2, 16)`
- **fitted covariance shape:** `(2, 16, 16)`

## STEP 3: Frame Labels Properties
## STEP 1 & 4: Sequence Alignment Audit
| sequence | feature_shape | labels | diff | match | anomaly_% |
|---|---|---|---|---|---|
| MVI_40852 | (284, 768) | 1150 | 866 | ⚠️ NO | 30.0% |
| MVI_40853 | (394, 768) | 1590 | 1196 | ⚠️ NO | 30.0% |
| MVI_40854 | (295, 768) | 1195 | 900 | ⚠️ NO | 30.0% |
| MVI_40855 | (269, 768) | 1090 | 821 | ⚠️ NO | 30.0% |
| MVI_40863 | (414, 768) | 1670 | 1256 | ⚠️ NO | 30.0% |
| MVI_40864 | (375, 768) | 1515 | 1140 | ⚠️ NO | 30.0% |
| MVI_40871 | (427, 768) | 1720 | 1293 | ⚠️ NO | 30.0% |
| MVI_40891 | (383, 768) | 1545 | 1162 | ⚠️ NO | 30.0% |
| MVI_40892 | (444, 768) | 1790 | 1346 | ⚠️ NO | 30.0% |
| MVI_40901 | (330, 768) | 1335 | 1005 | ⚠️ NO | 30.0% |
| MVI_40902 | (248, 768) | 1005 | 757 | ⚠️ NO | 30.0% |
| MVI_40903 | (262, 768) | 1060 | 798 | ⚠️ NO | 30.0% |
| MVI_40904 | (314, 768) | 1270 | 956 | ⚠️ NO | 30.0% |
| MVI_40905 | (424, 768) | 1710 | 1286 | ⚠️ NO | 30.0% |
| MVI_40962 | (465, 768) | 1875 | 1410 | ⚠️ NO | 30.0% |
| MVI_40963 | (452, 768) | 1820 | 1368 | ⚠️ NO | 30.0% |
| MVI_40981 | (495, 768) | 1995 | 1500 | ⚠️ NO | 30.0% |
| MVI_40991 | (452, 768) | 1820 | 1368 | ⚠️ NO | 30.0% |
| MVI_40992 | (537, 768) | 2160 | 1623 | ⚠️ NO | 30.0% |
| MVI_41063 | (373, 768) | 1505 | 1132 | ⚠️ NO | 30.0% |
| MVI_41073 | (453, 768) | 1825 | 1372 | ⚠️ NO | 30.0% |
| MVI_63521 | (510, 768) | 2055 | 1545 | ⚠️ NO | 30.0% |
| MVI_63525 | (243, 768) | 985 | 742 | ⚠️ NO | 30.1% |
| MVI_63544 | (287, 768) | 1160 | 873 | ⚠️ NO | 30.0% |
| MVI_63552 | (284, 768) | 1150 | 866 | ⚠️ NO | 30.0% |
| MVI_63553 | (348, 768) | 1405 | 1057 | ⚠️ NO | 30.0% |
| MVI_63554 | (358, 768) | 1445 | 1087 | ⚠️ NO | 30.0% |
| MVI_63561 | (318, 768) | 1285 | 967 | ⚠️ NO | 30.0% |
| MVI_63562 | (293, 768) | 1185 | 892 | ⚠️ NO | 30.0% |
| MVI_63563 | (344, 768) | 1390 | 1046 | ⚠️ NO | 30.1% |

## STEP 5: Final Answers
**Q1. How many sequences align perfectly?**
0 out of 30 sequences align perfectly.

**Q2. Does GMM dimensionality match VideoMAE embedding dimensionality?**
NO / UNKNOWN (GMM: 16)

**Q3. Are labels frame-level and temporally aligned?**
NO

**Q4. Can score_samples() be executed immediately?**
NO (Dimensionality mismatch)

**Q5. Are there any interpolation or aggregation operations required?**
YES (Some sequences have frame differences requiring interpolation or truncation).
