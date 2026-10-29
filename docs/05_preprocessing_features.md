# Preprocessing, Features, and Evaluation Splits

## Purpose and boundaries

This pipeline prepares questionnaire responses for the Environmental &
Financial Factors models. It builds pressure-oriented item features, four
condition indices, optional profile-membership features, and a single-factor
income baseline. It does not use outcomes as predictors and does not make
causal claims.

`attrition_intent_score` and `at_risk` are labels only. Student identifiers,
timestamps, consent, and attention-check fields are also excluded. The label
is self-reported intention rather than verified withdrawal.

## Split first, then fit

```mermaid
flowchart TD
    A[Clean one row per student] --> B[Stratified outer split: 70% train, 15% validation, 15% test]
    B --> C[Persist shared repeated CV folds on train plus validation]
    C --> D[For each fold, fit imputer, encoder, indices, profiles, selector, and scaler on fold train]
    D --> E[Transform fold validation and evaluate each model]
    B --> F[Fit feature tables on outer train; transform validation]
    F --> G[Diagnostics and leaderboard development]
    B --> H[Keep test indices guarded and unopened]
    H --> I[One-time final evaluation through evaluate_final_on_test()]
```

Splitting precedes every learned transformation so validation and test
observations cannot influence imputation statistics, one-hot fitting, profile
centres, near-constant filtering, or scaling. The outer split is stratified by
`at_risk` and participant. The repeated stratified folds use only the outer
train-plus-validation population: 5 folds by 3 repeats when there are at least
500 such students, otherwise 5 folds by 5 repeats with a warning. Fold
positions are persisted and shared by each compared model.

The held-out test indices are written once in each run's `splits/indices.json`.
The development workflow does not load those indices. Only
`evaluate_final_on_test()` is permitted to read that artifact, and it records
single use before returning test rows.

## Fold-fitted transformations

1. **Imputation:** ordinal items use the training-fold median; nominal items
   use the training-fold mode. A `missing__<item>` indicator is created when
   more than 3% of the training-fold values are missing. Structural commute
   N/A remains a valid category, not a missing value.
2. **Item encoding:** ordinal item responses are rescaled to `[0, 1]` using
   each schema item's pressure direction, so larger values mean more pressure.
   Nominal responses receive one-hot columns and a separate numeric pressure
   value from the schema pressure map. Online, prefer-not-to-say, missing, and
   consistency flags are passed through.
3. **Indices and derived features:** schema item pressures are aggregated into
   the four indices below. Ratio and interaction features use thresholds and
   monetary midpoint assumptions from `config/project.yaml`.
4. **Profiles:** `agg_profiles` currently fits four-cluster KMeans membership
   on the four indices within each training fold. Validation rows are assigned
   to those training-fold clusters. This fold-local implementation is a
   provisional profile-feature path; it does not consume an externally fitted
   EF-7 profile artifact and does not introduce outcome labels into clustering.
5. **Near-constant columns:** columns with training-fold population variance
   at or below `0.005` are removed. The decision is fit on the training fold
   and then applied unchanged to validation.
6. **Scaling:** standard scaling is included only for model families marked
   as scale-sensitive in `config/model_zoo.yaml`. Tree models are not scaled.
7. **Class imbalance:** registered classifiers use class weights by default.
   SMOTE is not enabled in this feature workflow; if compared later, it must be
   fitted inside each training fold.

## Feature formulas

Each item pressure value is the schema-directed rescaling of its response to
`[0, 1]`. For an ordinary increasing-pressure ordinal item with codes from
`min` to `max`, pressure is `(code - min) / (max - min)`. For a reversed item,
it is `(max - code) / (max - min)`. Nominal pressure values follow that item's
configured pressure map. A structural commute response has zero commute
pressure.

The four condition indices are the row-wise mean of their available rescaled
items:

- **Financial Stress:** income-band pressure, aid-status pressure, family
  support duty, difficulty paying study costs, financial shock, money worry,
  and job-hours pressure.
- **Commute Burden:** commute-time, commute-spend, and transport-mode
  pressure. It is explicitly `0` for fully online students.
- **Digital Access Gap:** home-internet pressure, main-device pressure, and
  mobile-data-plan pressure.
- **Study Environment Pressure:** quiet-study-place pressure and
  living-situation pressure.

Additional engineered features:

- **Overall pressure:** mean of the four indices.
- **High-pressure index count:** number of indices strictly greater than
  `0.66`.
- **Travel-spend-to-income ratio:** estimated monthly travel spend divided by
  estimated monthly household income, clipped to `[0, 1]`. Estimates use
  configured income-band midpoints (LKR 12,500; 37,500; 75,000; 150,000;
  250,000) and commute-spend midpoints (LKR 1,250; 3,750; 7,500; 12,500).
  Fractional imputed band codes are linearly interpolated. Fully online
  students receive `0`. These are analysis assumptions, not reported
  household amounts.
- **Low income and no aid:** `1` when the income-band code is at or below the
  configured low-income cutoff and aid-status code represents the highest
  no-aid category; otherwise `0`.
- **Working over 20 hours and long commute:** `1` when the student reports
  part-time work, the job-hours code meets the configured over-20-hours
  cutoff, and commute-time code meets the configured long-commute cutoff.
- **Weak internet and smartphone only:** `1` when the internet code meets the
  configured weak-connection cutoff and the device code is the configured
  smartphone-only category.

### Illustrative student

The following values are an arithmetic example, not a participant record or
observed result. Suppose a student's four indices are Financial Stress `0.60`,
Commute Burden `0.30`, Digital Access Gap `0.70`, and Study Environment
Pressure `0.50`. Overall pressure is `(0.60 + 0.30 + 0.70 + 0.50) / 4 =
0.525`; one index is above `0.66`. If the configured interpolated monthly
travel midpoint is LKR 7,500 and the household-income midpoint is LKR 37,500,
the travel-spend-to-income ratio is `7,500 / 37,500 = 0.20`.

## Feature sets

All sets are defined in `config/model_zoo.yaml`. Near-constant columns may
reduce the final count and are recorded in the run summary.

| Set | Included features | Intended comparison |
|---|---|---|
| `raw` | Rescaled ordinal items; nominal one-hot and pressure-map features; retained missingness, online, prefer-not-to-say, and consistency flags | Establish performance from item-level signals |
| `agg` | `raw` plus four indices, overall pressure, high-pressure index count, travel ratio, and three configured interactions | Test whether compact derived summaries add value |
| `agg_profiles` | `agg` plus four fold-fitted profile-membership indicators | Test whether unsupervised condition patterns add value |
| `single_signal` | Income-band pressure only | Reproduce the one-factor income baseline |

Identifiers, labels, consent, attention checks, and timestamps are absent from
every set. The profile-membership indicators are features only in
`agg_profiles`; discovered profiles remain descriptive outputs elsewhere.

## Diagnostics and reporting

Diagnostics use the aggregate feature table and outer train-plus-validation
rows, not the held-out test partition:

- A feature-by-label effect-size table reports Cohen's *d* for continuous
  features and Cramer's V for binary features.
- A Spearman correlation heat map summarizes feature-to-feature relationships.
- Mutual information ranks features against the training-plus-validation
  labels as an exploratory diagnostic.
- The leakage audit checks that outcome fields are absent from the model
  matrix and flags any absolute feature/label correlation above `0.95`.
- `linear_model_vif.csv` reports variance inflation factors for numeric
  aggregate inputs. It is informational only: no feature is removed based on
  VIF, and tree models do not use VIF to select features.
- Repeated-fold ROC-AUC comparisons are generated for Logistic Regression and
  LightGBM on `raw` and `agg`, with fold-wise 95% quantile intervals. Each
  fold fits a fresh feature pipeline. If a native LightGBM installation error
  occurs, its error is saved and printed explicitly; the affected model's
  metrics are not represented as successful results.

Diagnostics are not a substitute for the configured leaderboard metrics,
calibration checks, group-wise fairness reporting, or final one-time test.
Generated-sample results validate mechanics only and must not be presented as
real survey findings.

## Run and inspect

```powershell
python -m efi.cli features
python -m efi.cli features --input data/raw/survey_raw_sample.csv --run-id feature_check
```

The command prints feature-table shapes, the split artifact path, fold-wise
model comparisons, leakage-audit status, and any model availability errors.
Run outputs are written under `artifacts/runs/<run_id>/features/`; split
indices are stored separately under `artifacts/runs/<run_id>/splits/`.
