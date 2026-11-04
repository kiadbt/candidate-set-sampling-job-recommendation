# Post-review calibration and uncertainty analysis protocol

## Status

These analyses were added after independent manuscript review. They were not
preregistered and are reported as sensitivity or worked-calibration analyses.
They do not replace the prespecified primary contrasts.

## Evidence boundary

The script reads only the completed CareerBuilder First7Last2 authoritative-v2
checkpoints, their admitted analysis package, the completed CareerBuilder
direct-scoring benchmark, the three hash-pinned JTH per-query result files, and
the JTH strengthened-v2 analysis package. It performs no model training, text
encoding, candidate scoring, or resampling of new evaluation candidates.

## Analyses

1. Top-model sets are recomputed under the primary numerical tie rule and
   illustrative absolute equivalence margins of 0.001, 0.005, and 0.010.
2. Uniform-minus-popularity-weighted paired metric differences are calculated
   for the three principal content models after averaging five seeds within
   query. CareerBuilder uses 2000
   user-bootstrap replicates stratified by provider window and retains design
   weights. JTH reports date-cluster, candidate-cluster, and two-way
   candidate/date bootstrap sensitivities with 2000 replicates each. Because
   both sampled protocols share the same full-catalog reference, their sampled
   metric difference is algebraically identical to the difference between
   their two sampled-minus-full biases.
3. Seed-level top-model decisions are reported descriptively. Five seeds do not
   support a precise binomial probability claim.
4. Illustrative decision calibration reports each criterion separately and a
   transparent combined rule. The worked rule requires top-model agreement in
   all five seeds, mean pairwise agreement at least 0.90, maximum reference
   regret at most 0.01, and, for the metric-plus-decision variant, maximum
   absolute bias across the three principal content configurations at most
   0.02. A threshold grid is released; no threshold is claimed as universal.
5. Timing summaries use the 630 primary warm-cache records. Sampled/full ratios
   are paired within model, provider window, and timed repetition.
6. Relevance-set sizes are deduplicated to one record per query.
7. CareerBuilder size 10,000 and JTH size 500 are compared descriptively because
   their mean realized catalog fractions are both approximately 8.5%. This
   single near-match does not isolate candidate fraction from task differences.

## Interpretation

The outputs quantify robustness of the reported decisions and make the
calibration recommendation operational. They must not be described as
preregistered criteria, universal operating points, or a causal decomposition
of cross-dataset differences.
