# Verified findings: First5Last4

## Scope

These findings passed the authority, hash, completeness, aggregate-reproduction,
and statistical-output checks in `analysis_integrity.json`. They apply only to
the internally reconstructed `First5Last4` task and are not official
CareerBuilder competition results.

## Full-catalog design-weighted results

- BM25 (full text): HitRate@20=0.136136, NDCG@20=0.039537, MAP@20=0.022839.
- TF-IDF (combined): HitRate@20=0.090170, NDCG@20=0.021715, MAP@20=0.010932.
- BM25 (title): HitRate@20=0.026873, NDCG@20=0.007070, MAP@20=0.003731.
- Popularity: HitRate@20=0.026320, NDCG@20=0.005685, MAP@20=0.002846.
- MiniLM (title): HitRate@20=0.018585, NDCG@20=0.005573, MAP@20=0.003261.
- Random: HitRate@20=0.000000, NDCG@20=0.000000, MAP@20=0.000000.

The exact eligible catalog averaged 129571.39
jobs per query (median 123600.00; range
81,416--
218,954).

## Sampling distortion and ranking fidelity

The largest mean signed HitRate@20 bias among the evaluated cells was 0.873586 for TF-IDF (combined) under Uniform negatives at nominal 100.

- Popularity-weighted negatives at nominal 100: mean tau-b=0.866667, mean pairwise agreement=0.933333, exact winner-set rate=1.000000.
- Uniform negatives at nominal 100: mean tau-b=0.733333, mean pairwise agreement=0.866667, exact winner-set rate=0.000000.

Interpret the paired contrast intervals in
`publication_table_4_contrast_bootstrap.csv`; do not infer support from mean
values alone. Activity and target-popularity results are descriptive
heterogeneity summaries, not causal or fairness claims.

Fine subgroup-distortion inference is disabled for this temporal sensitivity because its 250-user-per-window cohort supports only protocol-level robustness.
