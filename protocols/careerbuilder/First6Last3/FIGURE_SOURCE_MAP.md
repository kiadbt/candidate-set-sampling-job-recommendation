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

- **Subgroup-effectiveness figure intentionally omitted.** Fine subgroup
  inference is disabled for the 250-user-per-window temporal sensitivities.

- **Subgroup-distortion figure intentionally omitted.** The 250-user-per-window
  temporal-sensitivity cohort supports protocol-level robustness but not fine
  subgroup claims; see `SUBGROUP_EVALUATION_BIAS_STATUS.json`.

Every emitted figure is a 300-dpi PNG plus vector PDF and SVG.
