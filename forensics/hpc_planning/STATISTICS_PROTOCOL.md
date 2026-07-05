# Statistical Validation Protocol
## Objective
Ensure all reported metrics meet the $lpha < 0.05$ significance threshold.

## Required Tests
1. **Shapiro-Wilk Test:** Determine normality of metric distributions across the 5 random seeds.
2. **Welch's t-test:** Compare SPGRL mean metrics against baselines (DQN, PPO) assuming unequal variances.
3. **Mann-Whitney U Test:** Non-parametric alternative if Shapiro-Wilk rejects normality.
4. **Bootstrap Confidence Intervals:** 95% CIs derived from 10,000 resamples.
5. **Cohen's d & Cliff's delta:** Quantify the magnitude of the effect size.

## Expected Outputs
- `pvalues.csv`
- `effect_sizes.csv`
- `confidence_intervals.csv`
