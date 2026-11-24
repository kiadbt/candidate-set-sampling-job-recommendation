# Revision analysis protocol and public evidence boundary

## Scope

This package supports the article **“Evaluating Candidate-Set Sampling in
Temporal Job Recommendation: Cross-Dataset Evidence on Model-Selection
Agreement.”** It adds reviewer-requested and post-decision robustness and sensitivity analyses to the
two-dataset candidate-set evaluation without redistributing provider records or
individual-level outputs.

## Fixed evaluation principles

- Exact evaluation ranks every job admitted by the dataset-specific temporal
  eligibility rule after declared exclusions.
- Sampled evaluation retains all observed relevant jobs and adds eligible
  logged-non-relevant jobs under uniform or popularity-weighted sampling.
- Model inputs, parameters, relevance judgments, metric definitions, tie
  breaking, and random seeds remain fixed within each comparison.
- CareerBuilder aggregate estimates use the declared design weights where the
  corresponding output column is labelled `weighted_*`; JTH results are
  unweighted query means.
- The reviewer-requested temporal-eligibility sensitivities are post-decision
  analyses. They add future-opening jobs while holding queries, observed
  targets, model definitions, and shared-job scores fixed. The additional
  CareerBuilder near-horizon analysis restricts added jobs to
  `cutoff < StartDate <= evaluation_end` and excludes jobs with any pre-cutoff
  interaction.
- Reciprocal-rank fusion is a fixed post-decision sensitivity with constant
  60. Component ranks and fused scores are recomputed inside each declared
  candidate set.

## Released aggregate evidence

`aggregate_results/revision_r1/` contains only non-row-level outputs:

- `careerbuilder_overlap/`: counts and pairwise overlap summaries for the seven
  provider windows and their half-open periods;
- `careerbuilder_leave_one_window_out/`: aggregate exact-catalog stability,
  principal contrasts, and model-selection-agreement summaries after omitting
  one window at a time;
- `careerbuilder_query_transition/`: aggregate title-BM25 versus title-MiniLM
  hit-state and first-relevant-rank summaries;
- `careerbuilder_local_reference_scale/`: aggregate comparisons between smaller
  candidate sets and the fixed 5,000-candidate local reference;
- `careerbuilder_cross_window_direction/`: descriptive window-by-seed checks
  that size-100 HitRate@20 inflation remains positive for full-text BM25,
  TF-IDF, and MiniLM under both samplers; these cells are not treated as
  independent inferential units;
- `careerbuilder_shared_catalog_feasibility/`: aggregate target-union counts
  showing why a target-complete shared 5,000-job catalog cannot retain the full
  eligible population and would be target-dominated even for the audit cohort;
- `temporal_eligibility/`: CareerBuilder and JTH aggregate effectiveness and
  ranking summaries under reference and future-inclusive catalogs, including
  the CareerBuilder near-horizon sensitivity that excludes jobs opening after
  the evaluation horizon; and
- `rrf/`: CareerBuilder and JTH aggregate metrics for the fixed RRF sensitivity.

The associated implementations are the `revision_r1_*` scripts, the RRF
extensions in the two dataset evaluators, and the targeted
`test_revision_r1_*` verification scripts in `Code/`. In particular,
`revision_r1_cross_window_directional_consistency.py`,
`revision_r1_shared_catalog_feasibility.ps1`, and
`revision_r1_careerbuilder_near_horizon_temporal.py` produce the three
strengthening analyses above. Their public QA records verify the declared
aggregate grids without releasing individual-level inputs.

## Public-data boundary

The package intentionally excludes raw or processed source data, user and job
identifiers, per-user or per-query metrics, paired-record details, candidate
lists, checkpoints, embeddings, local paths, and access credentials. End-to-end
reproduction therefore requires independently authorized access to the source
datasets. The released aggregates permit claim checking without exposing
provider-controlled records.

## Integrity

`RELEASE_MANIFEST.json` records the byte count and SHA-256 digest of every
released file except the manifest itself and Git metadata. The manifest should
be regenerated after any public-file change and verified before release.
