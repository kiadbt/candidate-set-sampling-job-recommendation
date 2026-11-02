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
6. **Verify outputs.** Compare generated aggregate files against `RELEASE_MANIFEST.json` and the path-free integrity summaries supplied in `aggregate_results/`.

Scripts expose command-line help through `python <script> --help`. Paths are intentionally not hard-coded in the public release.

## Code-version qualification

The released core CareerBuilder evaluator and direct-scoring benchmark match their authoritative configuration hashes. The released CareerBuilder analysis, JTH analysis, and cross-dataset synthesis scripts are maintained post-analysis versions because byte-identical copies of the historical analysis scripts were not retained. Consequently:

- the repository supports inspection and prospective rerunning of the disclosed methods;
- the aggregate outputs are the authoritative evidence package used for reporting; and
- this release does **not** claim byte-for-byte reconstruction of every archived aggregate from the maintained analysis scripts alone.

This qualification is preferable to presenting a stronger reproducibility claim than the retained evidence supports.

## Computational boundary

The direct-scoring benchmark measures scoring and ranking work under frozen candidates and models. It is not an end-to-end production latency benchmark and excludes data acquisition, initial preprocessing, one-time model acquisition, and unrelated system overhead.

