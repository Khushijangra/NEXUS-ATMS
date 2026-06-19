# MULDE Path Verification Report

## Paths & Implementation
- **MULDE Implementation**: `src/models/scorers/mulde.py`
- **Scorer Initialization**: `MULDEScorer.__init__` handles the initialization of the `MULDENetwork` (a simple PyTorch MLP) and the feature standardization buffers (`feat_mean`, `feat_std`).
- **Evaluation Pipeline**: `src/evaluation/stream_a.py` orchestrates the extraction of features via VideoMAE, computing the log-density signals using `MULDEScorer`, and evaluating AUC.

## Integrity Verification
1. **Can MULDE initialize independently?**: 
   - YES. The `MULDEScorer` relies purely on `torch`, `numpy`, and `sklearn.mixture` (for the GMM). It does not have structural dependencies on VideoMAE or `transformers` for its own initialization.
2. **Dataset assumptions**: 
   - Inside `mulde.py`, none. The scorer expects normalized `[batch, 768]` tensors. 
   - Inside `src/evaluation/stream_a.py`, it assumes `config.data.dataset` is `ubnormal` or `avenue`, and parses paths accordingly.
3. **Required config fields**:
   - For `MULDEScorer`: `feature_dim`, `hidden_dim`, `sigma_low`, `sigma_high`, `eval_L`, `beta`, `gmm_components`, `num_layers`, `use_layernorm`.
   - For Evaluation: `config.evaluation.signal_kind`, `sigma_strategy`.

## Status: Intact
