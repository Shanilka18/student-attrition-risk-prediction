> Phase 17 delivers a reproducible, artifact-backed viva documentation pack.
> The active run is generated sample data, not a real survey.
> The generated results include explicit evidence gaps and conflicting artifacts.
> The 90 viva answers and 15-slide, 12-minute talk were checked for coverage.
> The documentation command and focused checks pass; final QA was not run.

# Phase 17 Report — Viva Documentation Pack

## (1) What was done

Created the documentation pack for the viva and presentation, including a project index, end-to-end walkthrough, beginner concepts, proposal text, ethics and limitations, reproducibility guide, glossary, 90 viva questions, and a 15-slide presentation guide. The results summary, figure catalog, and 12-number cheat sheet are regenerated from the active run by `python -m mei.cli docs`. The checker verifies artifact equality, links, figures, 300-dpi PNG metadata, paired SVG exports, required document structure, placeholders, and prohibited website descriptions.

Exported 81 PNG figures and 81 paired SVG files to `docs/figures/final/`. Existing vector sources were preserved; SVG wrappers around raster images are identified in the catalog. No website files or content were changed.

Verified viva answers 1–10 against the [project brief](../../PROJECT_BRIEF.md), questionnaire design, and generated [results summary](../19_RESULTS_SUMMARY.md):

| Answer | Evidence check |
|---|---|
| 1 | Consistent with the stated research problem: repeated engagement reports and self-reported intention, not confirmed withdrawal. |
| 2 | Correctly presents an integrated workflow and explicitly avoids claiming a new classifier. |
| 3 | Correctly says generated data can test the pipeline but cannot establish real-student validity. |
| 4 | Correctly identifies the outcome as a self-reported intention item or score, not an administrative dropout record. |
| 5 | The four-time-point design matches the stated survey sequence. |
| 6 | Uses conditional language about possible supportive use; there is no evidence of real-world student benefit. |
| 7 | Correctly distinguishes stated intention from later withdrawal. |
| 8 | Matches the project's question about repeated engagement patterns and incremental value from states. |
| 9 | Correctly disclaims causal, confirmed-dropout, universal-fairness, and operational-readiness claims. |
| 10 | The active classification leaderboard ranks KNN first; the answer also states that the generated-data result is not real-student evidence. |

Answers 3, 6, and 10 were deliberately softened because real-survey validity and student-support benefit are not established. The test-use answers later in the bank were also qualified: the lock marker exists, but saved non-champion test metrics mean strict one-time access cannot be independently confirmed. The weak regression and SHAP evidence is described without upgrading it into a strong claim.

## (2) Files created or modified

- Created the requested documentation files: [00_README_DOCS.md](../00_README_DOCS.md), [18_END_TO_END_WALKTHROUGH.md](../18_END_TO_END_WALKTHROUGH.md), [19_RESULTS_SUMMARY.md](../19_RESULTS_SUMMARY.md), [20_FIGURE_CATALOG.md](../20_FIGURE_CATALOG.md), [21_CONCEPTS_FOR_BEGINNERS.md](../21_CONCEPTS_FOR_BEGINNERS.md), [22_VIVA_QA.md](../22_VIVA_QA.md), [23_PRESENTATION_GUIDE.md](../23_PRESENTATION_GUIDE.md), [24_LIMITATIONS_ETHICS.md](../24_LIMITATIONS_ETHICS.md), [25_PROPOSAL_UPDATE_NOTES.md](../25_PROPOSAL_UPDATE_NOTES.md), [26_REPRODUCIBILITY_AND_RUN_GUIDE.md](../26_REPRODUCIBILITY_AND_RUN_GUIDE.md), and [27_GLOSSARY.md](../27_GLOSSARY.md).
- Generated [RESULTS_SUMMARY.md](../RESULTS_SUMMARY.md) and [_numbers.json](../_numbers.json).
- Created [build_docs_numbers.py](../../scripts/build_docs_numbers.py) and [check_docs.py](../../scripts/check_docs.py); connected the `docs` command in [cli.py](../../src/mei/cli.py).
- Modified [DECISIONS.md](../DECISIONS.md) with decisions D-069–D-072.
- Exported paired images in [figures/final/](../figures/final/).
- Created this [phase report](./phase_17.md).

## (3) Key numbers read from output files

Values below are read from [_numbers.json](../_numbers.json) and rendered in the [generated results summary](../19_RESULTS_SUMMARY.md).

- Source: **sample (generated)**; split cohort **2,184**: **1,528** train, **328** validation, **328** test.
- The champion feature manifest contains **100** features; A1–A3 and named outcome fields are absent.
- Classification champion: k-Nearest Neighbours; test F1 **0.7182** (bootstrap 95% CI **0.6385–0.7914**) and ROC AUC **0.8773** (95% CI **0.8314–0.9198**).
- Regression champion: stacking regressor; test R-squared **0.1147**, with 95% interval **−0.0188–0.2269**. The interval crosses zero; the evidence does not support a strong regression claim.
- K-means selected **3** clusters (silhouette **0.3744**); the HMM selected **4** states and converged.
- State-feature AUC ablations were not statistically significant (logistic regression p = **0.1205**; LightGBM p = **0.8904**). The LightGBM MCC difference was statistically significant (p = **0.0480**) and negative.
- SHAP consistency across **15** folds was weak: mean Spearman **0.0809**, Kendall W **0.1421**, and mean top-ten Jaccard **0.0852**.
- The active run has **36** classification and **41** regression leaderboard rows. The figure catalog covers **81** source figures.
- A training-cleaning report and test calibration metric are not recorded. A separate scoring audit covers **200** rows, flags **5**, retains **200**, and records **0.00%** feature missingness; it is not training-cleaning evidence.

## (4) Checks executed with explicit PASS/FAIL status

- **PASS** — `.venv\Scripts\python.exe -m mei.cli docs`: generated the active-run summaries and figures; the embedded documentation checker passed.
- **PASS** — `.venv\Scripts\python.exe scripts\check_docs.py`: 12 documentation files, 240 local links, 81 figure entries, and artifact-backed results passed.
- **PASS** — `py_compile` for both documentation scripts and `src\mei\cli.py`.
- **PASS** — Ruff and Black checks for `scripts\build_docs_numbers.py` and `scripts\check_docs.py`.
- **PASS** — Presentation coverage: 15 slides, 720 seconds total, 15 speaker notes of 60–71 words; the Q&A bank has 90 questions in the requested group sizes, 15 most-likely marks, and 10 tough questions.
- **FAIL (pre-existing findings outside this change)** — Ruff over `src\mei\cli.py` reports import-order findings in the existing retraining commands and an undefined `logger` at line 3983. Those lines are not part of this phase's CLI change and were not altered.
- **NOT RUN** — Full project/final QA and held-out test evaluation, as explicitly deferred. No test-set evaluation was run during this phase.

## (5) Risks or decisions logged

- D-069 records artifact-backed regeneration; D-070 records the threshold and test-use evidence conflicts; D-071 documents raster-backed SVG wrappers; D-072 records the mismatch between the saved risk-band definitions and the scoring-results chart.
- The classification threshold is **0.29** in the leaderboard and **0.26** in the model card. The test-use lock marker does not resolve the non-champion test metrics in saved leaderboards. Neither discrepancy was silently reconciled or investigated by reopening the test set.
- The active run uses generated data. Its results do not establish real-survey performance. Training cleaning is unavailable, test calibration is absent, regression generalization is weak, and SHAP ranks are unstable.
- The website's scoring-results chart uses different band cutoffs from the active model artifact. This documentation phase records the mismatch and does not modify the website.
- Full project/final QA is intentionally deferred at the user's request.

## Common mistakes

- Treating generated-data metrics as real-student performance.
- Saying that the held-out test was definitely accessed only once despite the conflicting saved artifacts.
- Presenting weak SHAP rankings or regression fit as stable or generalizable findings.
- Reading the website's band chart as though its cutoffs match the active model artifact.

## If the examiner asks

State what the artifacts support, name what they cannot establish, and use the generated results summary for numerical answers. Explain that real survey data, matching data-quality records, reconciled thresholds, and independent validation are needed before any real-world use.
