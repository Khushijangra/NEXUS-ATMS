# ARGUS Configuration Recovery

## 1. Audit of Configuration Files
The correct configuration files are located at:
- `argus_stream_extracted\argus stream A\configs\default.yaml`
- `argus_stream_extracted\argus stream A\configs\stream_a_locked.yaml`

These differ drastically from the root NEXUS `configs/default.yaml` which was mistakenly loaded during the crash.

## 2. Required Configuration Sections
Every evaluation/training script expects the config namespace to contain:
- `project`: (name, seed)
- `data`: (dataset, data_dir, train_split, val_split, test_split, num_scenes, fps)
- `backbone`: (scene)
- `stream_a` / `scorer`: (hidden_dim, sigma_low, sigma_high, eval_L, beta, gmm_components, layernorm, batch_size)
- `training`: (batch_size, learning_rate, epochs, optimizer, etc.)
- `evaluation`: (metric, signal_kind, sigma_strategy, gmm_components, single_sigma_index, smoothing_sigma)

## 3. Config Loading Trace (`load_config()`)
1. **Target Directory**: The path is read from `--config-dir` (default: `"configs"`). If executed from the root, this hits the NEXUS configs.
2. **Base Load**: It parses `{config_dir}/default.yaml`.
3. **Override Merge**: It parses `{config_dir}/{dataset}.yaml` (e.g., `stream_a_locked.yaml`) and does a recursive `_deep_merge`.
4. **Path Resolution**: `_resolve_relative_paths` rewrites `data.data_dir` to be absolute relative to `config_path.parent`.
5. **Namespace Construction**: `_dict_to_namespace` recursively converts dictionaries into `SimpleNamespace` objects.

Because the previous run hit the NEXUS config, the `evaluation` dict was missing, meaning `config.evaluation` resulted in the `AttributeError`.
