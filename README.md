# Exact Monotonic Decision Trees (EMDT)

The authoritative manuscript is `main.tex`, with bibliography `ref.bib`, figures in `figures/`, and complete statistical tables in `full_statistical_appendix.tex`. The current title is **Monotonicity Certificates Shape Tree Expressivity and Optimization**. `working/editorial_changes.json` records the located prose comparisons and figure/table cross-reference audit; it is a working editorial document separate from the manuscript.

## Overview
EMDT minimizes training error in a fixed-depth tree with training-generated candidates. `monotonic_encoding="subtree"` is the default conservative structural certificate. `monotonic_encoding="cells"` orders adjacent candidate-threshold cells and represents the complete partially monotone candidate-tree class. Cell enumeration is exponential in feature count and is limited to 4,096 cells by default. Decreasing directions must be transformed by the caller. Predicting without a feasible incumbent raises an error.

`experiment/cascade_dp.py` implements exact dynamic programming when every feature is governed and outputs are binary. It does not solve general partially governed trees. `compress_rows=True` groups candidate-outcome vectors, retaining both label counts; candidates are computed before grouping. `constant_hint=True` supplies a complete fit-only majority-tree hint.

## Structure
- `experiment/`: Contains all Python scripts to run the EMDT solver and baseline methods across synthetic and real-world datasets (COMPAS, German Credit, Adult Income, Bank Marketing).
- `figures/`: Contains visualizations of decision boundaries and scalability error bars.

## Dependencies
- `ortools`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pandas`
- `numpy`
- `scipy`

## Current evidence and protocol

The earlier fixed 500-record comparisons are historical artifacts and are no longer manuscript inputs. Full cohorts are COMPAS 6,172, German 1,000, Adult 32,561 (adult.data), and Bank 45,211 (bank-full). Numeric feature counts are 5/5/4/4; all selected features are governed and categorical predictors are excluded. Full-row evaluation does not mean unrestricted full-feature benchmark performance.

Ten split seeds 42–51 control stratified outer 80/20 splits. An inner 80/20 split of the outer training data uses seed + 10,000, yielding approximately 64/16/20 fit/validation/test partitions. All methods use identical fit rows. Min/max scaling is fitted only there; validation/test inputs are clipped and decreasing directions inverted. EMDT-Iso reuses EMDT and learns its two-level nondecreasing calibration map from validation labels only. Its hard decisions may change, so the optimization objective pertains to the original hard tree.

`experiment/full_cohort/all_runs.csv` contains 480 method evaluations (4 datasets × 10 seeds × 12 methods), including 40 calibration maps and 40 independent DP fits. CP-SAT uses depth three, a 30-second search cap, one worker, and the split seed. Fit time includes construction. Four jobs use the CPU concurrently; the first German seed was a serialized smoke run. DP also uses four jobs. Hardware is Intel Core Ultra 9 275HX, 24 logical processors, approximately 32 GB RAM; GPU unused. Timings are conditional on this policy.

Two audits each use 2,000 pairs: uniform profiles ordered by componentwise min/max (seed + 1,000), and test-row anchors with one coordinate increased (seed + 2,000). Both can leave the empirical manifold. Construction guarantees, rather than sampled zeros, establish monotonicity for the constrained models.

CP-SAT certifies all exact-model fits on COMPAS/Adult/Bank. German has 8 OPTIMAL and 2 FEASIBLE for EMDT, and 10 FEASIBLE for both ODT and Cell. Independent DP verifies the same minimum structural objective in all 40 splits, including the two German incumbents; original CP-SAT statuses remain unchanged. Mean DP total fit seconds are 0.005159/0.034743/0.033596/0.044527 for COMPAS/German/Adult/Bank. These are optima over the fully governed structural class and fixed candidates, not all monotone functions or thresholds.

Boolean coverage at one through five variables is 3/3, 6/6, 19/20, 96/168, and 669/7,581. This is a function count rather than real-data prevalence. The cascade family is established prior work, cited in the paper. The certified complete encoding saves mean training errors 20.8 on COMPAS, 69.5 on Adult, and zero on Bank; German cell incumbents do not identify the complete-class optimum.

Fresh synthetic comparisons use 30 seeds 42–71 per condition, N=200, three features, two logistic Bernoulli generators, and extra flips 0.10/0.25. All 120 DP fits attain exact structural optima. Size/depth/feature studies and a memoization ablation use five seeds 42–46.

`summary.csv` reports 624 mean/sample-SD/CI blocks (48 model/dataset combinations × 13 metrics). Intervals use 2,000 seed-level resamples with seed 123. `paired_tests.csv` gives all 88 exploratory accuracy/cost tests with paired intervals, raw p, and joint Holm p. Encoding and synthetic contrasts have separate four-test families. Overlapping partitions limit population-level inference; nonsignificance does not establish equivalence.

`artifact_audit.json` verifies source and execution-code hashes, all 40 partitions, 160 serialized trees, 3,360 held-out metric values, 624 summary blocks, and objective/order/calibration consistency. Per-job artifacts preserve partition indices, source-row mappings, fit extrema, test scores, tree rules, and validation maps. Displayed raw-unit seed-42 rules are rounded; normalized exported thresholds are authoritative.

## Reproduction

Data paths follow `D:/data/CATALOG.md`; edit `SOURCES` in `run_full_cohort.py` for a different machine and verify source hashes. The current `.venv` uses system-site packages from the verified CPU stack; it is not a tested clean-install environment. Relevant actual versions are pinned in `experiment/full_cohort/requirements.txt`; the complete package freeze, hardware, and environment snapshots are alongside it.

```powershell
& .venv/Scripts/python.exe experiment/analyze_boolean_class.py
& .venv/Scripts/python.exe experiment/analyze_model_class.py
& .venv/Scripts/python.exe experiment/verify_compression.py
& .venv/Scripts/python.exe experiment/verify_cascade_dp.py
& .venv/Scripts/python.exe experiment/run_full_cohort.py --workers 4
& .venv/Scripts/python.exe experiment/run_cascade_comparison.py
& .venv/Scripts/python.exe experiment/run_exact_stress.py
& .venv/Scripts/python.exe experiment/ablate_dp_cache.py
& .venv/Scripts/python.exe experiment/report_full_cohort.py
& .venv/Scripts/python.exe experiment/plot_full_cohort.py
& .venv/Scripts/python.exe experiment/verify_full_artifacts.py
& .venv/Scripts/python.exe compile.py
```

The main runner reuses checkpoints with matching protocol/input fingerprints. The fingerprint excludes code: preserve the execution manifest and use a new output directory when changing experimental code. The audit additionally checks the recorded execution-code hashes. Reporting/plotting read saved results; layout changes do not require new fits. XeLaTeX → BibTeX → XeLaTeX twice builds the paper. Generated inputs are `benchmark_tables.tex`, `encoding_table.tex`, `solver_table.tex`, `synthetic_table.tex`, `case_rules.tex`, and `full_statistical_appendix.tex`.

`working/full_cohort_backup/` preserves the first revision and `working/pr_review_backup/` the earlier submitted manuscript. `revision.md` records both stages. No commit, push, archival publication, or journal submission has been performed. The public repository currently contains the earlier reference implementation; current extensions and evidence remain local.


## Computers and Operations Research submission

The target journal is **Computers & Operations Research**. The current manuscript uses Harvard author–year references and discloses the actual assistance of OpenAI Codex. `Cover Letter.docx`, `paperdata.docx`, and `Highlights.docx` match the current manuscript. The five highlights contain 73/79/71/78/74 characters. Existing `title-page.docx` author metadata is also refreshed.

`submission_COR/` collects the upload files, active LaTeX source, and a reproducibility archive with portable data retrieval and verification. `COR_submission_notes.md` explains file roles and the author-guide access limitation. The package has been prepared locally; it has not been submitted or published. Existing public GitHub code does not yet contain this full local evidence.
