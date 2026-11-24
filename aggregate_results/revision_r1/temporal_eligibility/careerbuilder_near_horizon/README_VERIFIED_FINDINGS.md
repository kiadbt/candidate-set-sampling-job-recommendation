# CareerBuilder Near-Horizon Temporal-Eligibility Sensitivity

## Design

This post-decision sensitivity adds only behavior-consistent same-window jobs satisfying
`cutoff < StartDate <= evaluation_end`. Queries, observed targets, profiles, pre-cutoff
behavior, reference-fitted model parameters, scoring rules, and deterministic tie-breaking
remain fixed. It complements the broader future-opening stress test without adding jobs that
open after the evaluation horizon.

## Verified findings

- Across seven windows, 24,895 near-horizon jobs were retained after the behavior-consistency gate; the design-weighted mean addition was 3,695.2649 jobs per query.
- bm25_full_static: design-weighted HitRate@20 changed from 0.1336 to 0.1313.
- tfidf_combined_static: design-weighted HitRate@20 changed from 0.0929 to 0.0909.
- bm25_title_static: design-weighted HitRate@20 changed from 0.0256 to 0.0246.
- minilm_title_static: design-weighted HitRate@20 changed from 0.0219 to 0.0219.
- The leading configuration was preserved for every metric and cutoff; Kendall's tau-b ranged from 0.8667 to 1.0000.
- Adding alternatives never improved any fixed-score per-query metric, and all 168,000 reference rows reproduced the locked primary evaluation exactly.

## Independent QA

All 190 declared checks passed. The public package excludes row-level user results and absolute local paths.

## Public-safe files

- `careerbuilder_near_horizon_summary.csv`
- `careerbuilder_near_horizon_model_rankings.csv`
- `careerbuilder_near_horizon_window_counts.csv`
- `careerbuilder_near_horizon_k20_design_weighted.csv`
- `NEAR_HORIZON_INDEPENDENT_QA.json`
- `NEAR_HORIZON_INDEPENDENT_QA.md`
