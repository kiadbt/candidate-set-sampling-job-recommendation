# Strengthened JTH analysis protocol

## Scope

This analysis applies only to the retained JTH experiment. It does not use or
rehabilitate any CareerBuilder result.

## Reproduction gate

Before any finding is written, all 642,257 original-condition K=20 rows for the
full catalog and 100-candidate protocols are joined one-to-one to the retained
production rows on query, model, sampler, seed, and candidate size. All six
metrics must agree within an absolute tolerance of 1e-12, and group, relevance
count, and eligible-catalog size must agree exactly.

## Dependence-aware bootstrap

The preselected contrasts are BM25 minus MiniLM and BM25 minus TF-IDF for
HitRate@20, NDCG@20, and MAP@20. They are evaluated for the full catalog and for
each of five seeds under uniform and popularity-weighted 100-candidate sampling.
The five-seed, per-query mean is also reported as a summary estimand.

Each interval uses 2,000 nonparametric replicates and preserves the query-weighted
mean:

- **Date cluster:** resample the 389 distinct query dates with replacement and
  retain every query on each selected date with its resampling multiplicity.
- **Candidate cluster:** resample the 6,398 candidates with replacement and
  retain every query for each selected candidate with its multiplicity.
- **Two-way candidate/date sensitivity:** independently resample candidate and
  date clusters; each query receives the product of its candidate and date
  multiplicities (a two-way pigeonhole cluster bootstrap). This is a dependence
  sensitivity analysis, not asserted to be a universally optimal small-sample
  correction. The product-weight construction follows the pigeonhole-style
  principle in Owen and Eckles (2012), *Bootstrapping data arrays of arbitrary
  order*, *Annals of Applied Statistics*, 6(3), 895--927,
  https://doi.org/10.1214/12-AOAS547.

Percentile 95% intervals, bootstrap standard errors, and the proportion of
replicates above zero are reported. No bootstrap probability is labelled a
p-value.

## Known-item sensitivity

A job seen for a candidate in the released training split is removed from that
candidate's test-time eligible catalog. Repeat targets are removed from the
relevance set; queries with no novel target are excluded. Scores are unchanged.
For sampled protocols, the original seeded negative order is filtered and then
continued until the requested pool is refilled, avoiding a new random draw.

Three aggregates separate effects: all original queries, original scores on the
novel-query cohort, and the novel-item estimand. The matched original cohort
separates catalog filtering from the effect of excluding repeat-only queries.

## Descriptive subgroup uncertainty

Full-catalog HitRate@20, NDCG@20, and MAP@20 are reported for BM25, MiniLM, and
TF-IDF with date-, candidate-, and two-way cluster intervals across the temporal
groups and both audited subgroup dimensions. The first non-temporal dimension
is named **training-interaction activity** and uses counts from the released
training interactions: 0 (cold), 1, 2--3, and 4+. It must not be described as a
static CV or work-history-depth measure. The second dimension is target popularity,
defined by the mean released-training count of a query's relevant jobs: exactly
0 (unseen), greater than 0 to 2, and greater than 2. Query, candidate, and date
counts accompany every group. These are descriptive heterogeneity results, not
fairness evidence and not causal effects.

## Rank and tie rules

Kendall's coefficient is explicitly SciPy 1.18.1 tau-b (`variant="b"`,
`method="auto"`); SciPy's native exact-value tie handling is retained for that
coefficient. For the separately reported winner sets and pairwise order states,
two values are tied under `numpy.isclose` with rtol=1e-10 and atol=1e-12. A winner
is therefore a set, not an alphabetically selected model. Exact winner agreement
requires identical sets; overlap requires a nonempty intersection. Pairwise order
agreement treats less-than, tie, and greater-than as three distinct states over
all 21 model pairs. Best-case and worst-case regret evaluate every sampled winner
under the full-catalog aggregate score.

## Time groups

T1--T4 are dense timestamp-rank quartiles: each of the 389 unique timestamps is
ranked chronologically and its rank fraction is binned at .25, .50, and .75.
They are not calendar quarters and are not equal-query quantiles.

## Evaluation-bias estimand across sizes

For HitRate@20 and MAP@20, each sampled query/model/sampler/size value is first
averaged over seeds 1--5. Its exact per-query full-catalog value is then
subtracted. The paired difference is summarized over all queries and within
timestamp-rank, training-interaction-activity, and target-popularity groups.
Date-cluster, candidate-cluster, and two-way candidate/date intervals use 2,000
replicates with seeds 20261220--20261222 (offset reproducibly by subgroup).
Subgroup results are descriptive; target popularity is outcome-derived.

## Interpretation limits

The released JTH entity text is static and has not been verified as an as-of-query
snapshot. Candidate sampling was applied after full model scores were available.
Consequently, computational statements concern only post-scoring candidate
ranking and metric calculation; they do not establish model-scoring or end-to-end
latency savings.
