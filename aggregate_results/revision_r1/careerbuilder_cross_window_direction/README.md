# CareerBuilder Cross-Window Directional Consistency

## Purpose

This descriptive robustness check verifies whether the size-100 sampled-minus-exact HitRate@20 difference remains positive in every CareerBuilder provider window and deterministic seed. It uses the locked `First7Last2` per-window output and the design-weighted population estimand.

## Fixed Scope

- Seven provider-defined windows.
- Five deterministic seeds.
- Uniform and popularity-weighted sampling.
- Nominal candidate size 100.
- HitRate@20.
- Full-text BM25, combined-profile/full-text TF-IDF, and combined-profile/title MiniLM.
- The exact eligible catalog for the same model and window is the reference.

The 35 window-by-seed cells in each model-sampler combination are audit cells, not independent inferential units. No p-value is attached to this descriptive check.

## Verified Findings

For each of the six model-sampler combinations, size-100 HitRate@20 inflation was strictly positive in all 35 window-by-seed cells and in all seven window means. Across the complete grid, 210 of 210 window-by-seed cells and 42 of 42 model-sampler-window means were positive.

Cellwise sampled-minus-exact ranges were:

| Sampler | Model | Minimum | Maximum | Positive cells |
|---|---|---:|---:|---:|
| Uniform | Full-text BM25 | 0.7970 | 0.8460 | 35/35 |
| Uniform | Combined-profile/full-text TF-IDF | 0.8410 | 0.8920 | 35/35 |
| Uniform | Combined-profile/title MiniLM | 0.5780 | 0.6630 | 35/35 |
| Popularity weighted | Full-text BM25 | 0.7980 | 0.8460 | 35/35 |
| Popularity weighted | Combined-profile/full-text TF-IDF | 0.8340 | 0.8920 | 35/35 |
| Popularity weighted | Combined-profile/title MiniLM | 0.5780 | 0.6640 | 35/35 |

This strengthens the descriptive robustness claim: the direction of severe small-set HitRate inflation is not driven by a single CareerBuilder window or random seed. It does not establish independence among the seven contiguous periods and does not replace the window-stratified paired-user bootstrap or leave-one-window-out analysis.

## Recommended Manuscript Wording

> As a descriptive cross-window check, size-100 HitRate@20 inflation was positive in every one of the 35 window-by-seed cells for each of full-text BM25, TF-IDF, and MiniLM under both samplers (210/210 cells overall), and in every corresponding window mean. The cells were not treated as independent inferential units because the seven periods are contiguous.

## Files

- `verify_cross_window_directional_consistency.py`: primary deterministic derivation.
- `careerbuilder_size100_hitrate_directional_consistency_detail.csv`: 210 verified window-by-seed-model-sampler cells.
- `careerbuilder_size100_hitrate_directional_consistency_summary.csv`: six model-sampler summaries.
- `careerbuilder_size100_hitrate_directional_consistency_audit.json`: input hash, grid-integrity checks, and primary pass/fail record.
- `independent_qa.py`: independent arithmetic and grouping verification.
- `INDEPENDENT_QA.json`: independent pass/fail record.

Both the primary audit and independent QA pass.
