# Figure source map

## `figure_1_cluster_sensitivity_hitrate`

- Source: `primary_contrasts_cluster_bootstrap.csv`
- Caption: BM25-minus-comparator HitRate@20 estimates and 95% percentile intervals under date-cluster, candidate-cluster, and two-way candidate/date pigeonhole-style bootstraps. Sampled estimates average five seeds per query before inference.

## `figure_2_known_item_hitrate`

- Source: `known_item_summary_k20.csv`
- Caption: Signed change in HitRate@20 from the matched-original to novel-item estimand for full-catalog, uniform-100, and popularity-weighted-100 evaluation. Sampled points use per-query five-seed means; the matched cohort isolates eligibility filtering from repeat-only query exclusion.

## `figure_3_rank_fidelity_explicit_ties`

- Source: `rank_fidelity_summary_k20.csv`
- Caption: Full-versus-sampled aggregate rank fidelity at K=20, averaged across six metrics and five seeds. Tau-b uses SciPy's native tie handling; pairwise and winner-set measures use the prespecified numerical tolerance.

## `figure_4_timestamp_rank_quartiles`

- Source: `subgroup_full_catalog_cluster_bootstrap.csv`
- Caption: Full-catalog HitRate@20 across dense timestamp-rank quartiles T1--T4, with two-way candidate/date cluster intervals. These labels are not calendar quarters or equal-query bins.

## `figure_5_subgroup_hitrate_two_way`

- Source: `subgroup_full_catalog_cluster_bootstrap.csv`
- Caption: Descriptive full-catalog HitRate@20 by training-interaction activity and target popularity, with two-way candidate/date cluster intervals. The plot is not a fairness or causal analysis.

## `figure_6_evaluation_bias_across_sizes`

- Source: `evaluation_bias_cluster_bootstrap.csv`
- Caption: Signed sampled-minus-full HitRate@20 and MAP@20 bias across every retained candidate size. Sampled values are averaged over all five seeds per query before paired inference; ribbons are two-way candidate/date cluster intervals.

## `figure_7_subgroup_evaluation_bias`

- Source: `subgroup_evaluation_bias_cluster_bootstrap.csv`
- Caption: Descriptive subgroup variation in uniform-100 HitRate@20 sampled-minus-full bias for BM25, MiniLM, and TF-IDF, with paired two-way candidate/date cluster intervals. Target popularity is outcome-derived; no causal or fairness claim is made.

All PNG files are rendered at 300 dpi. PDF and SVG versions preserve vector text and lines.
