# Candidate-Set Sampling in Temporal Job Recommendation

This repository contains publication-safe reproducibility materials for the study **“Candidate-Set Sampling in Temporal Job Recommendation: Cross-Dataset Evidence on Model-Selection Agreement.”** The study evaluates whether ordinary sampled-candidate protocols reproduce conclusions from sampling-free, protocol-defined logged-catalog evaluation in two temporal job-recommendation tasks.

## Repository scope

The release includes:

- CareerBuilder and JTH preparation, evaluation, sensitivity, and analysis code;
- the cross-dataset synthesis code;
- the post-review decision-stability, calibration, and uncertainty analysis code;
- study protocols and verified-finding records;
- machine-readable aggregate results used in the article and supplement;
- integrity summaries that do not expose local paths or provider records; and
- editable SVG sources for the final figures.

It intentionally excludes:

- raw or processed CareerBuilder and JTH row-level data;
- individual-level, per-user, and per-query outputs;
- cached embeddings, model caches, and checkpoints;
- disclosure-sensitive timestamp definitions;
- manuscripts, submission documents, local-path logs, and temporary files.

The exclusions are deliberate. CareerBuilder source access remains provider controlled, and JTH is distributed separately under CC BY-NC 4.0. See [DATA.md](DATA.md) and [RIGHTS.md](RIGHTS.md).

## Structure

```text
Code/               flat script directory preserving sibling imports
protocols/          dataset-specific and cross-dataset study records
aggregate_results/  non-row-level machine-readable results
figures/            editable SVG figure sources
```

The post-review extension is documented in
[`protocols/post_review/STUDY_PROTOCOL.md`](protocols/post_review/STUDY_PROTOCOL.md).
Its publication-safe outputs are in `aggregate_results/post_review/` and cover
top-set equivalence sensitivity, seed-level decision stability, paired sampler
contrasts, illustrative calibration thresholds, relevance-set distributions,
near-matched candidate-fraction comparisons, and warm-cache timing summaries.
The released seed-detail file contains one row per aggregate experimental
condition and seed; it contains no user, query, candidate, or vacancy
identifier.

`RELEASE_MANIFEST.json` records each released file, byte count, and SHA-256 digest without exposing local source paths.

## Reproducibility boundary

This is not a one-click data reconstruction package. End-to-end reruns require independently authorized access to the source datasets and substantial CPU time and memory. The core CareerBuilder evaluator and direct-scoring benchmark match their authoritative run records. The released CareerBuilder analysis, JTH analysis, and cross-dataset synthesis scripts were rerun in isolated output directories against retained, integrity-checked analysis inputs. File-level comparisons reproduced all 93 retained aggregate CSVs byte for byte, with a maximum absolute numerical difference of 0 and passed integrity checks.

The exact historical analyzer files for the original CareerBuilder, JTH, and
cross-dataset stages were not retained, and those current script hashes differ
from the hashes recorded in the original run records. Their verification
therefore establishes reconstruction of the released aggregate evidence from
retained analysis inputs; it does not establish that the current files are the
missing historical snapshots or constitute an end-to-end rebuild from provider
data. See [REPRODUCIBILITY_VERIFICATION.md](REPRODUCIBILITY_VERIFICATION.md)
for the script hashes, comparison scope, and outcomes, and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the environment and workflow
boundary.

The post-review extension reused retained, hash-checked analysis inputs and
performed no model training, text encoding, candidate scoring, or new candidate
sampling. Its script verified 155 retained source hashes before producing 11
aggregate CSVs and the protocol record. Those withheld source inputs include
individual- or query-level evidence and are not part of this repository; the
public, path-free verification record is
[`aggregate_results/post_review/POST_REVIEW_INTEGRITY_SUMMARY.json`](aggregate_results/post_review/POST_REVIEW_INTEGRITY_SUMMARY.json).

## Data sources

- CareerBuilder Job Recommendation Engine Challenge data page: <https://www.kaggle.com/competitions/job-recommendation/data>
- JTH v1.1 Zenodo record: <https://zenodo.org/records/21390581>
- Official JTH repository: <https://github.com/Aunsiels/JTH>
- Sentence Transformers model used for the MiniLM baseline: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

No provider dataset is redistributed here.

## Citation

Please cite the associated article when its bibliographic record becomes available. Until then, use the metadata in [CITATION.cff](CITATION.cff) and state the repository URL and accessed commit.

## Contact

Kamaluddeen Ibrahim Yarima, School of Digital Science, Universiti Brunei Darussalam.

