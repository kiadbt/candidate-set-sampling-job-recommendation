# CareerBuilder figure source map

- **Figure 1 — sampled-catalog metric bias.** Signed sampled-minus-full
  differences (rather than unstable ratios) from `sampling_bias_seed_summary.csv`.
- **Figure 2 — ranking fidelity.** Tolerance-explicit Kendall tau-b and
  three-state pairwise order agreement from `rank_fidelity_summary.csv`.
- **Figure 3 — seven temporal windows.** Full-catalog HitRate@20 from
  `per_window_results.csv`; equal-window dispersion is descriptive only.
- **Figure 4 — BM25 contrast distortion.** Five-seed-mean, paired,
  within-window-stratified user-bootstrap estimates from
  `prespecified_contrast_bootstrap.csv`.

- **Figure 5 — subgroup effectiveness.** Descriptive full-catalog HitRate@20
  with within-window-stratified user-bootstrap intervals from
  `subgroup_bootstrap.csv`; query-mean target popularity is outcome-derived.

- **Figure 6 — subgroup evaluation bias.** Paired sampled-minus-full
  HitRate@20 after per-user averaging over seeds 1--5, with users resampled
  within each represented window, from
  `subgroup_evaluation_bias_cluster_bootstrap.csv`. This figure is produced
  only for the primary 1,000-user-per-window audit; target-popularity groups are
  outcome-derived and all subgroup patterns are descriptive.

Every emitted figure is a 300-dpi PNG plus vector PDF and SVG.
