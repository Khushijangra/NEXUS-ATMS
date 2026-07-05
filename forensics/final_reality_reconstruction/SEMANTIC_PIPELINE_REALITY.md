# SEMANTIC PIPELINE ARCHAEOLOGY
**Q1. Which script generated VideoMAE .npy?**
Unknown upstream script (done offline before freeze).
**Q2. Which script generated best_clip_gmm.pkl?**
`archive/deprecated_training/scripts/regenerate_stream_a_models.py` (Mock data).
**Q3. What operation converts 768 → 16?**
MULDE (Multiscale Log-Density Estimator) evaluates log-likelihoods across L=16 noise scales.
**Q4. What operation converts 1150 → 284?**
`VideoMAEClipDataset` in `datasets.py` (Stride=4, ClipLength=16).
**Q5. Was MULDE used?** YES.
**Q6. Was PCA used?** NO.
**Q7. Was AutoEncoder used?** NO (Denoising score matching network).
**Q8. Was temporal pooling used?** NO. Stride overlapping and center-frame labeling.
**Q9. Can semantic AUROC be reproduced TODAY?**
YES, by executing `train_stream.py` to genuinely train MULDE/GMM on the existing features.
