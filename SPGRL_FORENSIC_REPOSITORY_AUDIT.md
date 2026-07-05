# SPGRL FORENSIC REPOSITORY AUDIT

## PHASE 3: Training Provenance Audit

| Module | Training Done | Checkpoint | Logs | Metrics | Status |
|---|---|---|---|---|---|
| VideoMAE | No | No | No | Yes | ❌ Missing |
| MULDE | Yes | No | No | No | ⚠️ No Checkpoint |
| GMM | No | No | No | Yes | ❌ Missing |
| YOLO | No | No | No | Yes | ❌ Missing |
| DeepSORT | No | No | No | Yes | ❌ Missing |
| LSTM | Yes | Yes | No | Yes | ✅ Trained |
| GNN | Yes | No | No | Yes | ⚠️ No Checkpoint |
| MAPPO | Yes | No | No | Yes | ⚠️ No Checkpoint |
| Carbon | No | No | No | Yes | ❌ Missing |
| Emergency | No | No | No | Yes | ❌ Missing |

## Zt (Unified State) Verification

**Result:** Zt construction logic FOUND in codebase. (Partial structural implementation exists).

## PHASE 4: Reality Check

**Case B:** Some modules trained. Some theoretical. (2-3 week problem)

## Master Inventory

### Checkpoints & Weights (.pt, .pth, etc.)
- `dummy.pt`
- `models\pretrained\stream_a\best_clip.pt`
- `v2\prediction\lstm\lstm_best.pth`

### Training & Experiment Scripts
- `v2\rl\gnn.py`
- `v2_phase_d_mappo.py`
- `v2\rl\joint_optimization_real.py`
- `v2\prediction\lstm\lstm_predictor_wrapper.py`
- `v2\rl\mappo.py`
- `v2_phase_c_gnn.py`
- `intelligence\prediction\lstm_predictor.py`
- `v2\rl\mappo_train_real.py`
- `archive\deprecated_training\scripts\train_lstm.py`
- `v2\graph\gnn\gnn_train_real.py`
- `intelligence\orchestration\rl_controller.py`
- `argus_stream_extracted\argus stream A\src\models\scorers\mulde.py`

### Logs (Tensorboard/Wandb/.log)
- `data\runtime_logs\runtime.log`
- `frontend\.next\dev\logs\next-development.log`
- `logs\train_20260702_220256.log`
- `logs\train_20260702_220359.log`
- `logs\train_20260702_220453.log`
- `logs\train_20260702_220537.log`
- `logs\train_20260702_221132.log`
- `logs\train_20260702_221823.log`
- `logs\train_20260702_222726.log`
- `logs\train_20260702_230104.log`

### CSV / PNG Outputs
- `asset_inventory.json`
- `asset_validation_report.json`
- `claim_verification.csv`
- `package-lock.json`
- `package.json`
- `railway.json`
- `table_traceability_report.csv`
- `temp_embeddings.npy`
- `V2_COMPLETENESS_MATRIX.csv`
- `.vscode\settings.json`
- `.vscode\tasks.json`
- `archive\hackathon_2026\presentation_assets\linkedin\01_authority_overview.png`
- `archive\hackathon_2026\presentation_assets\linkedin\02_digital_twin_canvas.png`
- `archive\hackathon_2026\presentation_assets\linkedin\03_citizen_portal.png`
- `archive\hackathon_2026\presentation_assets\linkedin\04_ai_analytics.png`
- `archive\hackathon_2026\presentation_assets\linkedin\05_system_architecture.png`
- `archive\hackathon_2026\presentation_diagrams\complete_ml_pipeline_1769712852554.png`
- `archive\hackathon_2026\presentation_diagrams\deep_rl_pipeline_1769712635336.png`
- `archive\hackathon_2026\presentation_diagrams\dueling_dqn_architecture_1769712676166.png`
- `archive\hackathon_2026\presentation_diagrams\execution_flow_pipeline_1769712894893.png`
- `archive\hackathon_2026\presentation_diagrams\ml_models_overview_1769712920704.png`
- `archive\hackathon_2026\presentation_diagrams\traffic_ml_pipeline_1769712422332.png`
- `archive\hackathon_2026\results\benchmark_d3qn.json`
- `archive\hackathon_2026\results\benchmark_queue_vs_timesteps.png`
- `archive\hackathon_2026\results\benchmark_reward_vs_timesteps.png`
- `archive\hackathon_2026\results\benchmark_summary.csv`
- `archive\hackathon_2026\results\benchmark_waiting_vs_timesteps.png`
- `archive\hackathon_2026\results\comparison_chart.png`
- `archive\hackathon_2026\results\d3qn_gate_report.json`
- `archive\hackathon_2026\results\d3qn_gate_report_relaxed_15_25.json`
- `archive\hackathon_2026\results\d3qn_gate_report_release.json`
- `archive\hackathon_2026\results\d3qn_multiseed_summary.csv`
- `archive\hackathon_2026\results\d3qn_multiseed_summary.json`
- `archive\hackathon_2026\results\evaluation_results.json`
- `archive\hackathon_2026\results\evaluation_results_final.csv`
- `archive\hackathon_2026\results\graph_ab_report.json`
- `archive\hackathon_2026\results\graph_release_candidate.json`
- `archive\hackathon_2026\results\release_candidate.json`
- `archive\hackathon_2026\results\yolo_validation.json`
- `archive\hackathon_2026\results\anomaly\anomaly_detection_plots.png`
- `archive\hackathon_2026\results\anomaly\anomaly_detection_results.json`
- `archive\hackathon_2026\results\d3qn_seed_123\evaluation_results.json`
- `archive\hackathon_2026\results\d3qn_seed_42\evaluation_results.json`
- `archive\hackathon_2026\results\d3qn_seed_999\evaluation_results.json`
- `archive\hackathon_2026\results\demo_submission_20260412_014106\manifest.json`
- `archive\hackathon_2026\results\demo_submission_20260412_014106\assets\evaluation\comparison_chart.png`
- `archive\hackathon_2026\results\demo_submission_20260412_014106\assets\evaluation\evaluation_results.json`
- `archive\hackathon_2026\results\lstm\lstm_scatter.png`
- `archive\hackathon_2026\results\lstm\lstm_training_plots.png`
- `archive\hackathon_2026\results\lstm\lstm_training_results.json`
- *...and 1909 more*