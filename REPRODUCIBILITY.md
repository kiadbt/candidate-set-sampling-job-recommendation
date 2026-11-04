# Reproducibility Notes

## Environment recorded for the experiments

The principal CareerBuilder evaluation recorded:

- Python 3.12
- NumPy 2.5.4
- pandas 3.0.1
- SciPy 1.18.1
- scikit-learn 1.9.1
- PyTorch 2.7.1 (CPU)
- Transformers 4.57.6
- Sentence Transformers 5.0.0

The strengthened JTH analysis recorded Python 3.12.14, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, and Matplotlib 3.11.2. The two pandas versions reflect separate recorded analysis environments. The requirements files preserve these dataset-specific records rather than implying that one environment was used for every stage.

The frozen semantic baseline used `sentence-transformers/all-MiniLM-L6-v2` at model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` on CPU.

## Workflow map

1. **Acquire data independently.** Obtain CareerBuilder and JTH from their official sources and comply with their terms.
2. **Prepare the temporal task.** Use the CareerBuilder preprocessing and internal-temporal preparation scripts, or the released JTH chronological task, as described in the corresponding `protocols/` directory.
3. **Evaluate fixed models.** Run the dataset-specific evaluator with the declared candidate sizes, sampling strategies, seeds, temporal eligibility, and tie-breaking rule.
4. **Run sensitivities and uncertainty analyses.** Apply the known-item, temporal-boundary, subgroup, and paired-contrast procedures recorded in the protocols.
5. **Build the cross-dataset synthesis.** Compare the direction, convergence, rank agreement, and decision effects of candidate sampling without pooling raw effectiveness across incomparable tasks.
6. **Run the post-review extension where authorized inputs are available.** Apply the declared top-set equivalence, seed-stability, paired-sampler, worked-calibration, relevance-distribution, matched-fraction, and timing procedures in `Code/analyze_peer_review_revision.py`.
7. **Verify outputs.** Compare generated aggregate files against `RELEASE_MANIFEST.json` and the path-free integrity summaries supplied in `aggregate_results/`.

Scripts expose command-line help through `python <script> --help`. Paths are intentionally not hard-coded in the public release.

## Verification of released analysis code

The released core CareerBuilder evaluator and direct-scoring benchmark match their authoritative configuration hashes. The current CareerBuilder analysis, JTH analysis, and cross-dataset synthesis scripts were rerun in isolated output directories against retained, integrity-checked analysis inputs. File-level comparisons reproduced all 93 retained aggregate CSVs byte for byte: 25 JTH files, 61 CareerBuilder files across the primary protocol, two temporal-boundary sensitivities, and their combined comparison, and 7 cross-dataset synthesis files. Every numerical comparison had a maximum absolute difference of 0, and every regenerated integrity sentinel reported `passed`.

The exact historical analyzer files were not retained, and the current script hashes differ from those recorded in the original run records. This verification therefore establishes reconstruction of the released aggregate evidence from retained analysis inputs. It does not establish that the current files are the missing historical snapshots, and it is not a clean raw-to-result rebuild from provider data. [REPRODUCIBILITY_VERIFICATION.md](REPRODUCIBILITY_VERIFICATION.md) records the current script hashes, comparison procedure, per-stage file counts, integrity outcomes, and the non-numerical wording differences observed in regenerated CareerBuilder findings notes.

## Post-review sensitivity and calibration layer

The post-review extension is explicitly non-preregistered and does not replace
the primary analyses. It uses 2,000 bootstrap replicates for the paired sampler
contrasts, with CareerBuilder users resampled within provider window and JTH
dependence examined through date-, candidate-, and two-way candidate/date
cluster bootstraps. It also reports top-model-set sensitivity under the
numerical tie rule and absolute margins of 0.001, 0.005, and 0.010; five-seed
decision stability; an illustrative threshold grid; relevance-set
distributions; a descriptive near-matched candidate-fraction comparison; and
paired warm-cache timing summaries.

The script performs no model training, encoding, scoring, or new evaluation
candidate sampling. It requires retained authoritative checkpoint and
per-query analysis inputs that are excluded from this public repository. An
authorized rerun can be invoked as follows:

```text
python Code/analyze_peer_review_revision.py --project-root <authorized-project-root> --output-dir <new-output-directory> --bootstrap-reps 2000
```

The release provides the resulting 11 aggregate CSVs in
`aggregate_results/post_review/`, the protocol in
`protocols/post_review/STUDY_PROTOCOL.md`, and a sanitized integrity summary
that omits the internal source-file inventory. The original internal manifest
is intentionally excluded because it names withheld checkpoint and per-query
inputs.

## Computational boundary

The direct-scoring benchmark measures scoring and ranking work under frozen candidates and models. It is not an end-to-end production latency benchmark and excludes data acquisition, initial preprocessing, one-time model acquisition, and unrelated system overhead.

