# CareerBuilder internal-temporal statistical analysis protocol

## Scope and authority

This analysis concerns the internally reconstructed `First7Last2`
task only. It is not the CareerBuilder competition test and is not official
competition MAP. Metric files were read only after `RUN_REPORT.json` explicitly
declared `authoritative=true`, `production=true`, `smoke=false`, and
`status=completed`, and after every checkpoint hash and coverage interval passed.

## Estimands

- `audit_cohort`: the unweighted query mean in the fixed
  1,000-query-per-window audit sample
  (7,000 queries).
- `design_weighted_population`: inverse-inclusion-weighted query mean using
  the evaluator's window-specific `N_w / n_w` weights. This is the primary
  finite-population descriptive estimand.
- Equal-window means and standard deviations are retained only as sensitivity
  descriptions; they are not substituted for query-level means.

## Bias and ranking fidelity

Bias is always reported as the signed difference `sampled - full` and its
absolute magnitude; ratios are not used. Candidate fraction is computed from
the realized per-query count divided by that query's exact eligible catalog.
The nominal-100 / k=150 cell is absent by design.

Kendall tau-b uses SciPy 1.18.1 with tolerance-defined tie groups
(`rtol=1e-10`, `atol=1e-12`) before `kendalltau(variant='b')`.
Winner sets contain every model tied with the maximum under the same tolerance.
Pairwise fidelity is exact agreement over the three states {loss, tie, win}.
Regret is the full-catalog maximum minus the best/worst full-catalog score among
the sampled winner set.

## Paired bootstrap

The five prespecified contrasts are title BM25 minus title MiniLM, full-text
BM25 minus title MiniLM, full-text BM25 minus combined TF-IDF, full-text BM25
minus popularity, and full-text BM25 minus random. The pairing unit is the user
query. In each of 2,000 replicates, users are sampled with replacement
independently within each of the seven windows, while the same multiplicities
are applied to both models and—when estimating distortion—to sampled and full
conditions. The fixed primary seed is 20261020. Percentile 95% intervals,
bootstrap standard errors, and smoothed empirical probabilities above/below
zero are reported. Subgroups use the same algorithm within each represented
window subgroup. No interval is described as causal or as a fairness audit.

## Subgroups and computational boundary

Activity is the released pre-cutoff training-interaction count (`cold_0`,
`one_1`, `low_2_4`, `high_5plus`). Target popularity is the query mean of its
targets' pre-cutoff application counts; it is explicitly outcome-derived and
therefore descriptive. For the primary 1,000-user-per-window audit only,
HitRate@20 and MAP@20 sampled-minus-full subgroup distortion is estimated for
all six models, both samplers, and every candidate size. Each sampled value is
averaged over seeds 1--5 within user before subtracting that user's full-catalog
value; the paired differences use the same within-window stratified bootstrap.
Temporal sensitivities use 250 users per window and explicitly disable fine
subgroup-effectiveness and subgroup-distortion inference. The evaluator scores the full catalog before indexing
sampled candidates, so these results do not establish model-scoring or
end-to-end computational savings. The analyzer RSS ceiling is
12.00 GiB.
