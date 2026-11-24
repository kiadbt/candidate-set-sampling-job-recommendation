# Shared 5,000-Job Catalog Feasibility

## Question

Could a single target-complete 5,000-job CareerBuilder catalog be shared by all evaluated users in a window without changing the cohort or dropping observed relevant jobs?

## Method

For each of the seven provider-defined windows, the analysis counts the union of unique observed future target JobIDs for (1) the 1,000-user audit cohort and (2) the full eligible population. A shared target-complete catalog must contain every job in that union. The calculation uses only the locked prepared cohort and target files; it does not alter sampling, models, scores, or metrics.

## Verified Findings

- In the 1,000-user audit cohorts, the target union ranges from 2,262 to 3,360 jobs, occupying 45.24% to 67.20% of a nominal 5,000-job shared catalog. Only 1,640 to 2,738 positions would remain for non-target jobs.
- In the full eligible populations, the target union ranges from 11,565 to 25,843 jobs. It exceeds 5,000 in every window, so a target-complete shared 5,000-job catalog is impossible without dropping users or observed targets.
- A shared audit-cohort catalog would be technically possible but target-dominated and would allocate a large, window-specific fraction of its capacity to the cohort target union. This changes the negative pool and makes the design less comparable across windows.
- The retained per-user, seed-matched nested design preserves every evaluated user and observed target while changing only local candidate scale. It is therefore the cleaner controlled scale sensitivity for the present estimand.

## Interpretation Boundary

This calculation supports the design choice; it does not establish that 5,000 candidates are universally sufficient or that per-user sampling reconstructs latent exposure or preference.

## Outputs

- shared_catalog_feasibility_by_window.csv: window-level counts and feasibility quantities.
- shared_catalog_feasibility_audit.json: design, integrity checks, source hashes, output hash, and row records.
- analyze_shared_catalog_feasibility.ps1: reproducible analysis script.
