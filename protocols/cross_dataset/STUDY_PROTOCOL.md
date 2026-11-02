# Cross-dataset synthesis protocol

## Evidence gate

The synthesis reads only a completed JTH v2 package, a passed authoritative
CareerBuilder primary analysis package, and a completed authoritative CareerBuilder
direct-scoring benchmark. It verifies the source packages and hashes before
creating an output directory. Per-query and per-user files are never read.

## Comparability rules

Raw effectiveness values are not pooled, averaged, or used to rank datasets.
Metric bias is compared directionally within shared sampler, candidate-size,
metric, cutoff, and named model-family cells. Signed bias, absolute bias, and
percentage points stay within each dataset. Relative bias is reported only when
the absolute full-catalog value is at least 0.01; otherwise it is explicitly
suppressed as denominator-unstable. Rank and winner fidelity are
metric matched but remain descriptive native-model-set properties because the
evaluated model sets differ.
Candidate fractions are reported beside nominal sizes. BM25 contrasts retain
their exact configuration names; family-level parallels use interval-supported
direction only and are bounded by field and task differences. Subgroup distortion
is compared only for prespecified threshold-aligned groups when both datasets
supply paired sampled-minus-full intervals; event semantics remain dataset
native. Higher activity bins are not aligned. JTH's
chronological quartiles and CareerBuilder's provider-assigned windows remain
separate temporal descriptions, not replications. Measured direct-scoring time is
a CareerBuilder-only boundary; JTH contributes analytical candidate fractions,
not a timing result.

## Classification vocabulary

- `bounded_replication_*`: the datasets share a direction or loss-presence
  finding, subject to the recorded task, model, and configuration boundary.
- `dataset_specific_*`: the direction or decision-loss finding differs.
- `bounded_or_inconclusive_*`: an effect is near zero or its
  dependence-aware interval includes zero.
- `not_comparable_*`: a construct, interval, matched definition, or measured
  boundary is not available in both datasets.

The `classification_category` column collapses detailed labels to `replicated`,
`bounded_replication`, `dataset_specific`, or `not_comparable`. Because datasets,
tasks, model sets, and text fields differ, exact `replicated` labels are not
issued automatically; supported parallels are bounded replications.

These labels describe this study's datasets, protocols, models, and metrics. They
are not universal candidate-size recommendations.
