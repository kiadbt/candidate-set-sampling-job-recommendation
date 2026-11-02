# Verified findings: First6Last3

## Scope

These findings passed the authority, hash, completeness, aggregate-reproduction,
and statistical-output checks in `analysis_integrity.json`. They apply only to
the internally reconstructed `First6Last3` task and are not official
CareerBuilder competition results.

## Full-catalog design-weighted results

- BM25 (full text): HitRate@20=0.125092, NDCG@20=0.039107, MAP@20=0.023465.
- TF-IDF (combined): HitRate@20=0.099367, NDCG@20=0.022822, MAP@20=0.011051.
- MiniLM (title): HitRate@20=0.025478, NDCG@20=0.005995, MAP@20=0.003040.
- Popularity: HitRate@20=0.021796, NDCG@20=0.003724, MAP@20=0.001294.
- BM25 (title): HitRate@20=0.019630, NDCG@20=0.004119, MAP@20=0.001894.
- Random: HitRate@20=0.000000, NDCG@20=0.000000, MAP@20=0.000000.

The exact eligible catalog averaged 127567.66
jobs per query (median 124336.00; range
81,306--
207,215).

## Sampling distortion and ranking fidelity

The largest mean signed HitRate@20 bias among the evaluated cells was 0.856100 for TF-IDF (combined) under Uniform negatives at nominal 100.

- Popularity-weighted negatives at nominal 100: mean tau-b=0.733333, mean pairwise agreement=0.866667, exact winner-set rate=1.000000.
- Uniform negatives at nominal 100: mean tau-b=0.653333, mean pairwise agreement=0.826667, exact winner-set rate=0.400000.

Interpret the paired contrast intervals in
`publication_table_4_contrast_bootstrap.csv`; do not infer support from mean
values alone. Activity and target-popularity results are descriptive
heterogeneity summaries, not causal or fairness claims.

Fine subgroup-distortion inference is disabled for this temporal sensitivity because its 250-user-per-window cohort supports only protocol-level robustness.
