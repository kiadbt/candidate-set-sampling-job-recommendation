# Verified findings: First7Last2

## Scope

These findings passed the authority, hash, completeness, aggregate-reproduction,
and statistical-output checks in `analysis_integrity.json`. They apply only to
the internally reconstructed `First7Last2` task and are not official
CareerBuilder competition results.

## Full-catalog design-weighted results

- BM25 (full text): HitRate@20=0.133615, NDCG@20=0.037980, MAP@20=0.021582.
- TF-IDF (combined): HitRate@20=0.092940, NDCG@20=0.022459, MAP@20=0.011262.
- Popularity: HitRate@20=0.027049, NDCG@20=0.005273, MAP@20=0.002425.
- BM25 (title): HitRate@20=0.025636, NDCG@20=0.007016, MAP@20=0.004022.
- MiniLM (title): HitRate@20=0.021897, NDCG@20=0.007112, MAP@20=0.004601.
- Random: HitRate@20=0.000134, NDCG@20=0.000007, MAP@20=0.000001.

The exact eligible catalog averaged 125599.57
jobs per query (median 124167.00; range
80,424--
194,823).

## Sampling distortion and ranking fidelity

The largest mean signed HitRate@20 bias among the evaluated cells was 0.863647 for TF-IDF (combined) under Uniform negatives at nominal 100.

- Popularity-weighted negatives at nominal 100: mean tau-b=0.733333, mean pairwise agreement=0.866667, exact winner-set rate=1.000000.
- Uniform negatives at nominal 100: mean tau-b=0.973333, mean pairwise agreement=0.986667, exact winner-set rate=0.800000.

Interpret the paired contrast intervals in
`publication_table_4_contrast_bootstrap.csv`; do not infer support from mean
values alone. Activity and target-popularity results are descriptive
heterogeneity summaries, not causal or fairness claims.

Paired sampled-minus-full HitRate@20 and MAP@20 subgroup distortion is reported in `publication_table_9_subgroup_evaluation_bias.csv`; sampled values are per-user means over all five seeds before subtraction.
