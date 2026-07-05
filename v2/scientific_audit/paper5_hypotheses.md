H0: Joint optimization causes catastrophic interference (cosine sim < 0).
H1: Joint optimization maintains strictly positive cosine similarity across branches.
Expected Effect Size: cosine > 0.1 average
Test: Shapiro-Wilk + t-test
Min Sample: 10k epochs
Confidence: 95%