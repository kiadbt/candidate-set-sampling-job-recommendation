# Verified statistical findings: strengthened JTH branch

## Integrity gate

The regenerated original condition reproduced all 642,257
retained full-catalog and 100-candidate K=20 rows. Across precision, recall,
HitRate, NDCG, MAP, and MRR, the maximum absolute discrepancy was
0; no
value exceeded the prespecified 1e-12 tolerance. Structural metadata
also matched exactly. This gate passed before the findings below were written.

## Dependence-aware primary contrasts

- BM25 minus MiniLM, full catalog: 0.143028 (two-way 95% interval 0.125648 to 0.159798; bootstrap Pr[difference > 0] = 1.000).
- BM25 minus MiniLM, uniform 100, five-seed query mean: -0.021436 (two-way 95% interval -0.036911 to -0.006497; bootstrap Pr[difference > 0] = 0.002).
- BM25 minus MiniLM, popularity-weighted 100, five-seed query mean: -0.022228 (two-way 95% interval -0.037597 to -0.006939; bootstrap Pr[difference > 0] = 0.002).
- BM25 minus TF-IDF, full catalog: 0.028174 (two-way 95% interval 0.019999 to 0.036844; bootstrap Pr[difference > 0] = 1.000).
- BM25 minus TF-IDF, uniform 100, five-seed query mean: -0.001535 (two-way 95% interval -0.006275 to 0.003006; bootstrap Pr[difference > 0] = 0.264).
- BM25 minus TF-IDF, popularity-weighted 100, five-seed query mean: -0.002949 (two-way 95% interval -0.007883 to 0.001863; bootstrap Pr[difference > 0] = 0.126).

The complete table reports date-cluster, candidate-cluster, and two-way
candidate/date sensitivity intervals for the full catalog, every sampled seed,
and the per-query five-seed sampled mean. These intervals address repeated dates
and/or repeated candidates; the earlier date-only interval should not be treated
as the sole inferential result.

## Known-item and repeat-target sensitivity

The audit found 11 unique
training/test candidate-job overlaps, representing
18 target events in
18 queries. All affected queries were repeat-only,
so the novel-item estimand contains 8,323 of
8,341 original queries and excludes
18 queries. The full-catalog
HitRate@20 winner set is
`BM25` in the original
estimand and `BM25` in the
novel-item estimand. On the matched query cohort, the explicit winner-set comparison is
`bm25` versus `bm25`;
winner-set overlap is 1 and pairwise-order
agreement is 1.000.
Across all 78 model-ranking decisions (six metrics over
the full and sampled protocols), the winner set changed in
0 cells. Pairwise ordering changed in
2 cells, and the maximum absolute HitRate@20 change over
all model/protocol cells was 0.000841. Thus the
known-item restriction is a robustness check, not the source of the much larger
full-versus-sampled winner reversal above.

## Rank fidelity with explicit ties

Across six metrics and five seeds:

- uniform: at size 100, mean tau-b 0.622, pairwise agreement 0.811, and winner-set overlap 0.500; at size 5000, the values are 1.000, 1.000, and 1.000.
- popularity-weighted: at size 100, mean tau-b 0.667, pairwise agreement 0.833, and winner-set overlap 0.500; at size 5000, the values are 1.000, 1.000, and 1.000.

Both best- and worst-case winner regret are in
`rank_fidelity_summary_k20.csv`.

## Descriptive subgroup uncertainty

- training-interaction activity: 0 (cold): 6,611 queries, 1: 841 queries, 2--3: 584 queries, 4+: 305 queries. Across groups, full-catalog HitRate@20 ranges were BM25 0.150820--0.263954, MiniLM 0.045902--0.115262, TF-IDF 0.134426--0.236727.
- target popularity: 0 (unseen): 7,059 queries, >0--2: 429 queries, >2: 853 queries. Across groups, full-catalog HitRate@20 ranges were BM25 0.226107--0.255985, MiniLM 0.062937--0.135991, TF-IDF 0.203986--0.226803.

Every subgroup row reports query, candidate, and date counts together with date-,
candidate-, and two-way cluster intervals. These patterns are descriptive only;
they are not fairness findings and do not identify causal effects. The label
`training-interaction activity` is the released training interaction count, not
CV or work-history depth. The target-popularity analysis includes `unseen_0`
explicitly.

## Corrected time-group label and scope boundary

T1--T4 are dense timestamp-rank quartiles over
389 unique timestamps—not calendar quarters and
not equal-query groups. Descriptive full-catalog HitRate@20 estimates with two-way
candidate/date 95% intervals are:

- T1: BM25 0.338283 [0.306223, 0.371615]; MiniLM 0.148475 [0.123201, 0.176073]; TF-IDF 0.310594 [0.279323, 0.342918].
- T2: BM25 0.221027 [0.187842, 0.260058]; MiniLM 0.081663 [0.061193, 0.104191]; TF-IDF 0.200978 [0.169308, 0.237500].
- T3: BM25 0.277898 [0.234132, 0.321919]; MiniLM 0.118071 [0.088747, 0.148624]; TF-IDF 0.242621 [0.201304, 0.284842].
- T4: BM25 0.175155 [0.146521, 0.204575]; MiniLM 0.086957 [0.067685, 0.109717]; TF-IDF 0.143685 [0.117852, 0.170393].

The static profile text is not verified as an as-of-query
snapshot. Candidate sampling here is post-scoring, so no result supports a claim
of reduced model-scoring or end-to-end computational cost.

## Evaluation-bias estimand across sizes

The exact eligible catalog has mean 5812.92,
median 5808, and range
5,579--6,011
jobs. Mean realized candidate fractions by nominal size are
100: 0.0172; 500: 0.0861; 1,000: 0.1721; 2,500: 0.4303; 5,000: 0.8606.

Each sampled query is averaged over all five seeds before subtracting its exact
full-catalog value. Two-way cluster point estimates at the smallest and largest
candidate sizes are:

- uniform Hitrate@20: BM25 0.476394 at 100 and 0.014842 at 5,000; MiniLM 0.640858 at 100 and 0.007961 at 5,000; TF-IDF 0.506102 at 100 and 0.014818 at 5,000.
- uniform Map@20: BM25 0.351556 at 100 and 0.006439 at 5,000; MiniLM 0.277543 at 100 and 0.002614 at 5,000; TF-IDF 0.345779 at 100 and 0.005997 at 5,000.
- popularity-weighted Hitrate@20: BM25 0.472989 at 100 and 0.013020 at 5,000; MiniLM 0.638245 at 100 and 0.007721 at 5,000; TF-IDF 0.504112 at 100 and 0.013595 at 5,000.
- popularity-weighted Map@20: BM25 0.331191 at 100 and 0.006845 at 5,000; MiniLM 0.261389 at 100 and 0.002862 at 5,000; TF-IDF 0.328183 at 100 and 0.006384 at 5,000.

At uniform size 100, the ranges of HitRate@20 bias across the descriptive groups
are:

- timestamp-rank quartile: BM25 0.378468--0.559069; MiniLM 0.541863--0.721589; TF-IDF 0.401656--0.587159.
- training-interaction activity: BM25 0.422266--0.704262; MiniLM 0.619331--0.745574; TF-IDF 0.452277--0.733115.
- target popularity: BM25 0.467715--0.590676; MiniLM 0.608910--0.751049; TF-IDF 0.496869--0.611189.

Complete date-, candidate-, and two-way cluster intervals are in
`evaluation_bias_cluster_bootstrap.csv` and
`subgroup_evaluation_bias_cluster_bootstrap.csv`. All subgroup cells exceed 30
candidate and 30 date clusters, but the patterns remain descriptive. Target
popularity is outcome-derived and cannot be read as an ex-ante or causal stratum.
