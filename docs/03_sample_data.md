# EF-04 Synthetic Sample Data

## Purpose and boundaries

These are synthetic, cross-sectional questionnaire-like data for validating the
Environmental & Financial Factors pipeline before a real survey is available.
They are not observations about students and cannot establish real-world
prevalence, causal effects, predictive validity, fairness, or profile recovery.
The hidden factor file is synthetic ground truth reserved for EF-7 profile
recovery checks; do not merge it into model features or upload it as survey data.

The generator is `src/data_gen/sample_survey.py`. Numeric generation, response
quality, calibration, output, and plotting settings are under `sample_data` in
`config/project.yaml`. The source questionnaire and response labels are
`config/schema_envfin.yaml`; Google Forms headers are in
`config/mapping_envfin.yaml`. No real survey data is read. Configuration values
of `seed: 42`, 1,200 full records, 300 small records, and 200 upload examples
make the generation reproducible.

## Latent population design

| Student type | Configured share | F financial strain | A access burden | D digital gap | H home pressure |
|---|---:|---:|---:|---:|---:|
| stable_supported | 0.34 | 0.20 | 0.20 | 0.20 | 0.20 |
| financially_strained | 0.18 | 0.78 | 0.32 | 0.32 | 0.48 |
| commute_burdened | 0.14 | 0.38 | 0.82 | 0.36 | 0.38 |
| digitally_limited | 0.12 | 0.40 | 0.30 | 0.84 | 0.62 |
| working_student | 0.12 | 0.66 | 0.36 | 0.32 | 0.42 |
| multi_pressure | 0.10 | 0.84 | 0.78 | 0.78 | 0.82 |

F/A/D/H are clipped to [0, 1] after adding correlated Gaussian noise with
standard deviation 0.24. The configured correlation matrix (F, A, D, H order)
is `[[1.00, 0.28, 0.22, 0.26], [0.28, 1.00, 0.18, 0.24],
[0.22, 0.18, 1.00, 0.30], [0.26, 0.24, 0.30, 1.00]]`.
The shared factors create plausible overlap rather than mutually exclusive
student categories.

Each questionnaire response uses its own factor, pressure rank/order,
configured cut-points or binary threshold, and Gaussian item noise. This
preserves item-specific scales and avoids making every item a direct copy of
one factor.

| Item | Main factor | Additional rule | Cut-points / threshold | Noise SD | Pressure order |
|---|---|---|---|---:|---|
| income_band | F | — | -1.05, -0.35, 0.35, 1.05 | 0.60 | 4, 3, 2, 1, 0 |
| aid_status | F | — | -0.20, 0.65 | 0.56 | 0, 1, 2 |
| family_support_duty | F | binary threshold 0.62 | 0.62 | 0.60 | 0, 1 |
| cost_difficulty | F | — | -0.58, 0.12, 0.80 | 0.57 | 0, 1, 2, 3 |
| commute_time | A | — | -0.55, 0.15, 0.85 | 0.58 | 0, 1, 2, 3 |
| commute_spend | A | — | -0.55, 0.15, 0.85 | 0.62 | 0, 1, 2, 3 |
| transport_mode | A | — | -0.75, -0.05, 0.60, 1.20 | 0.72 | 0, 2, 1, 3, 4 |
| living_situation | H | — | -0.55, 0.15, 0.85 | 0.70 | 0, 1, 2, 3 |
| quiet_study_place | H | — | -0.10, 0.50 | 0.55 | 0, 1, 2 |
| home_internet | D | — | -0.55, 0.15, 0.85 | 0.58 | 0, 1, 2, 3 |
| main_device | D | — | -0.55, 0.15, 0.85 | 0.66 | 0, 3, 2, 1 |
| mobile_data_plan | D | — | -0.40, 0.42 | 0.58 | 0, 1, 2 |
| part_time_job | F | Type-specific status probabilities | -0.35, 0.45 | 0.66 | 0, 1, 2 |
| job_hours | F | Only employed students have hours | -0.90, -0.20, 0.45, 1.10 | 0.62 | 0, 1, 2, 3, 4 |
| financial_shock | F | — | 0.02, 0.90 | 0.88 | 0, 1, 2 |
| money_worry | F | — | -0.95, -0.35, 0.35, 0.95 | 0.62 | 0, 1, 2, 3, 4 |
| missed_due_to_barriers | H | 0.48 x A also contributes | -0.60, 0.12, 0.85 | 0.72 | 0, 1, 2, 3 |

`online_rate` is 0.10. Fully online respondents receive structural N/A on all
three commute items consistently. Job hours are unavailable for non-working
students; work-status probabilities are configured separately for each student
type in `config/project.yaml`. The largest level of each non-binary item is
limited to 50% by response generation; binary distributions are allowed to
reach 70/30.

## Outcomes and calibration

The synthetic continuous outcome is generated from
`sigmoid(intercept + signal_strength * weighted_factors + noise)`. The
intercept is -2.25; F/A/D/H weights are 0.34/0.24/0.22/0.20; the F x A
interaction weight is 0.70; latent-risk noise SD is 0.80; score noise SD is
8.0. The score is clipped and rounded to 0-100. `at_risk` is positive at score
60 or above. Outcomes are generated after the 17 predictors and are never
included in canonical model features.

Calibration performs five-fold, shuffled stratified cross-validation
(random_state 42) of fold-fitted LightGBM and Logistic Regression pipelines.
The signal starts at 3.00 and advances by 0.10 for at most 8 iterations. The
acceptance bands are at-risk prevalence 24%-32%, LightGBM ROC-AUC 0.78-0.88,
and Logistic ROC-AUC no more than 0.04 above LightGBM. LightGBM uses 110
estimators, learning rate 0.045, depth 3, 7 leaves, minimum child samples 24,
L2 regularization 2.0, one job, and quiet logging. Logistic Regression uses
C=1.0 and maximum 500 iterations.

| Iteration | Signal | At-risk prevalence | LightGBM ROC-AUC | Logistic ROC-AUC | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 3.00 | 25.08% | 0.7702 | 0.8067 | Rejected: LightGBM below 0.78; logistic gap passes |
| 2 | 3.10 | 27.00% | 0.7977 | 0.8121 | Accepted: prevalence, both AUC conditions pass |

The selected synthetic sample therefore meets the configured validation
targets without increasing signal beyond the bounded calibration loop. These
metrics are method-validation results on generated data, not evidence of
expected performance on a real student population.

## Response quality and export mess

Response-quality settings are fixed in the configuration: cost-difficulty
non-disclosure 5%; acquiescence, midpoint, and extreme response styles 4% each;
acquiescence codes 3 and 4; midpoint fraction 0.50; straight-lining 2%;
attention-check failure 2%; random item missingness 2% of item cells; declined
consent 1%; duplicate rows 0.5%; and typos/formatting variants 0.3% of item
cells. A fixed seed controls their placement. Their realized integer counts
vary with sample size and rounding.

The messy upload independently uses 3% blank cells, invalid values in 1% of
rows, an offset of one accepted header variant, and the additional
`legacy_extra_field` column. Configured invalid values are `999 minutes` for
commute time and `90 hours` for job hours.

The generator also writes timestamps from `2026-01-05T08:00:00+05:30`, seven
minutes apart, and IDs with prefix `P-` and five digits. They are synthetic
export metadata, not real collection dates or identifiers.

## Output inventory

| Output | Purpose and contents |
|---|---|
| `data/raw/survey_raw_sample.csv` | Full Google-Forms-style raw export, including shared fields, outcomes, configured missingness, typos, and duplicates |
| `data/raw/survey_raw_sample.xlsx` | Same full raw export in a deterministic XLSX workbook |
| `data/raw/survey_raw_sample_300.csv` | Small seeded raw export |
| `data/raw/ground_truth_hidden.csv` | Type, F, A, D, H, and latent risk; synthetic EF-7-only evaluation truth |
| `data/samples/sample_upload_clean.xlsx` | 200-row canonical feature-only workbook; no shared outcomes |
| `data/samples/sample_upload_messy.csv` | 200-row feature-only upload with header variants, blanks, invalid values, and an unknown field |
| `docs/figures/01_item_distributions.png` | Item response distributions |
| `docs/figures/02_item_correlations.png` | Correlations among pressure-oriented item ranks |
| `docs/figures/03_missingness_map.png` | Missingness in the raw export |
| `docs/figures/04_at_risk_by_type.png` | Synthetic at-risk share by hidden student type |
| `docs/figures/05_at_risk_by_income_band.png` | Synthetic at-risk share by income band |
| `docs/figures/06_configured_vs_achieved.png` | Configured response rates versus achieved sample rates |

All six figures use the configured brand palette and 300 DPI. Regenerate
everything with `python -m efi.cli make-sample-data`; the CLI prints the chosen
signal and calibration metrics. Run settings belong in `config/project.yaml`,
including `project.data_source` (`sample` now, `survey` after approved real
survey import). Never add `ground_truth_hidden.csv` to upload or model feature
lists.

## Limitations

Synthetic associations are chosen for a controlled pipeline test and do not
prove that the item/factor relationships hold in the target population.
Response styles, missingness, duplicates, and type-specific response patterns
are simplified mechanisms. Income bands are locally contextual and sensitive;
the generator does not validate the acceptability or safety of asking them.
Outcomes are generated from the same latent conditions represented by the
predictors, so calibration AUC is not an independent external validation.
There is no time dimension, observed withdrawal outcome, evidence of fairness,
or basis for individual decisions. Real survey collection requires its own
consent, ethics, data-quality, and validation process.
