# Risk Profiles: Discovery, Features, and Evaluation

## What a profile is

A risk profile is a descriptive grouping of students whose measured study
conditions have similar patterns across Financial Stress, Commute Burden,
Digital Access Gap, and Study Environment Pressure. Profile fitting uses those
four indices plus their overall mean pressure. It does not use attrition
outcomes, student type, or hidden generated-data factors.

Profiles are summaries of a single self-reported response. They are not
diagnoses, individual risk determinations, or causal explanations. Profile
names describe the relative index pattern of the cluster centroid.

## Candidate models and selection rule

The five input features are standardized using the current training partition
only. In each fit, the candidate set is:

- KMeans
- Gaussian Mixture with full covariance
- Gaussian Mixture with diagonal covariance
- Ward agglomerative clustering

Each method is evaluated for `k = 2, 3, 4, 5, 6`. Candidate diagnostics include
silhouette, Calinski-Harabasz, Davies-Bouldin, and GMM BIC where applicable.
Stability is the mean adjusted Rand index over 30 bootstrap resamples. A
candidate is eligible only if its silhouette is at least `0.15`; the candidate
with the greatest mean stability is selected. An exact tie prefers smaller
`k`, then the algorithm order in `config/profile_selection.yaml`.

The selected algorithm, `k`, all candidate metrics, and fitted cluster model
are recorded for each run. Profile naming uses the centroid values converted
back to the original `[0, 1]` index scale. The configured pressure
thresholds determine whether a group is named **Few pressures**, **Several
pressures at once**, or after its dominant pressure (for example **Digital
access gaps** or **Long commute**). Names and profile order are deterministic
for a given centroid set; they are not manually assigned from outcomes.

## Fold-local profile features

`ProfileFeatureTransformer` fits its scaler, candidate comparisons, selected
clusterer, and centroids only on the rows passed to `fit`. Its `transform`
method assigns new rows without refitting. Each model-validation fold gets a
fresh transformer fitted on that fold's training rows.

The transformer emits:

- one-hot profile-id columns;
- soft membership probabilities when the selected method is a Gaussian
  Mixture;
- Euclidean distance to the assigned profile centre in standardized input
  space; and
- the count of index values above the configured high-pressure threshold.

The ablation compares three fixed feature sets on the same EF-06 repeated
stratified folds: `agg` (four indices plus the other aggregate features),
`agg_profiles` (aggregate features plus profile features), and
`profiles_only`. The fixed classifiers are Logistic Regression and LightGBM.
F1, ROC-AUC, and MCC use a fixed probability threshold of `0.5`; no tuning was
performed to make profiles appear more useful.

## Generated-data example and ablation

The recorded run used generated sample data only. It had 941 students in the
outer training-plus-validation population; 774 rows fitted the descriptive
profile model, while validation rows were only transformed. The selected
candidate was **KMeans with k=2**, silhouette `0.261`, and mean bootstrap
stability ARI `0.859`. It passed the configured silhouette minimum; it was
selected by stability, not because `k=2` was specified in advance.

The original-scale selected centroids were named **Digital access gaps** and
**Several pressures at once**. On the combined outer training-plus-validation
rows, their generated-data sizes and at-risk shares were:

| Descriptive profile | Students | At-risk share (95% Wilson CI) |
|---|---:|---:|
| Digital access gaps | 482 | 13.5% (10.7%–16.8%) |
| Several pressures at once | 459 | 44.2% (39.7%–48.8%) |

These are sample-generator outputs, not findings about real students.

| Model / feature set | F1, mean ± SD | ROC-AUC, mean ± SD | MCC, mean ± SD |
|---|---:|---:|---:|
| Logistic Regression / `agg` | 0.5982 ± 0.0416 | 0.7860 ± 0.0346 | 0.4125 ± 0.0667 |
| Logistic Regression / `agg_profiles` | 0.5994 ± 0.0344 | 0.7829 ± 0.0353 | 0.4148 ± 0.0563 |
| Logistic Regression / `profiles_only` | 0.5560 ± 0.0647 | 0.7410 ± 0.0605 | 0.3425 ± 0.1057 |
| LightGBM / all three sets | Not available | Not available | Not available |

The paired two-sided Wilcoxon comparisons of `agg_profiles` with `agg` gave
`p=0.778` for F1, `p=0.0034` for ROC-AUC, and `p=0.683` for MCC. The ROC-AUC
difference was small (about `-0.0032`), while F1 and MCC moved only about
`+0.0012` and `+0.0023`. This is a mixed and practically negligible change:
the evidence does not support a meaningful predictive gain from adding these
profiles in this run.

Compared with `agg`, `profiles_only` had lower mean F1 by about `0.042`,
lower ROC-AUC by about `0.045`, and lower MCC by about `0.070`. The
corresponding paired p-values were `0.0067`, `0.0054`, and `0.0054`. Profile
features alone therefore did not replace the richer aggregate feature set.

LightGBM did not produce metrics: the local Windows native library raised an
access violation on its first fit. The workflow recorded this explicitly and
reported zero completed folds for LightGBM; no substitute model is presented
as LightGBM. The Logistic Regression results are complete for all 15 folds.

The repeated folds reuse students across repeats, so the paired Wilcoxon
p-values are descriptive rather than independent-replicate evidence. The
profile result should be treated as a generated-data pipeline check, not a
real-data estimate. If the design is extended, a pre-specified comparison of
soft mixture memberships, alternative index-only groupings, or an external
real-survey validation could be informative. Do not tune the profile
procedure against the same ablation until it appears beneficial.

## Generated-data type recovery

The hidden student-type recovery check joined generated-data identifiers to
the hidden type file and compared those types with profile assignments. For
the recorded sample run, 941 generated records matched: ARI was `0.100` and
NMI was `0.094`. The confusion matrix is included in
`artifacts/runs/<run_id>/profiles/summary.json`.

**This recovery check is valid only for generated data.** The generator's
student types and latent factors are not observed labels in a real survey, and
this score does not establish that the discovered groups generalize.

## Interpretation limits

- **Self-report:** income, costs, commute, access, and study conditions are
  respondent-reported and may contain recall or reporting error.
- **One time point:** the profiles are cross-sectional snapshots. They do not
  represent changes over time or a student's progression.
- **Generated data:** the example metrics and recovered types come from
  synthetic data and cannot establish real-world validity.
- **Descriptions, not diagnoses:** a profile is not a permanent student
  category and must not be used to make an individual decision.
- **Selection sensitivity:** standardization, the candidate family, the
  stability resamples, and a modest sample can affect selected clusters.
- **Profile labels are summaries:** a dominant centroid index does not mean
  every member has that condition or that the condition caused an outcome.
- **Statistical uncertainty:** CV folds overlap across repeats; confidence
  summaries and paired p-values do not replace independent external
  validation.

## Viva questions and answers

**Q: Why use profiles when there are already four indices?**

A: Profiles summarize recurring combinations of the indices. The ablation
tests whether that summary adds predictive information beyond the indices and
other aggregate features.

**Q: Why was `k=2` selected?**

A: We evaluated `k=2` through `k=6` for four configured algorithms. The
selection rule first requires silhouette of at least `0.15`, then chooses the
highest bootstrap stability; ties prefer smaller `k`. For this generated run,
KMeans with two clusters had the winning eligible stability.

**Q: Were the groups chosen to separate at-risk students?**

A: No. Cluster fitting and selection use only the five condition-pressure
inputs. Outcomes are used afterward for descriptive rates and model
evaluation, not to form the groups.

**Q: How do you prevent validation leakage?**

A: The scaler, candidate clusterers, selected model, and centroids are fitted
on each training fold only. Validation rows are transformed using those
already-fitted values.

**Q: Does the low generated-data ARI mean the method failed?**

A: It means these profiles do not closely recover the generator's six hidden
student types in this sample. The profiles are unsupervised condition
groupings, not intended to reproduce the generator's labels.

**Q: Do the profiles improve the model?**

A: In this generated-data run, adding profiles to aggregate features made
only tiny, mixed changes for Logistic Regression. Profiles alone performed
worse. LightGBM could not be evaluated because its local native library
failed, so there is no evidence here of a gain across both model types.

**Q: Are the paired p-values definitive?**

A: No. Repeated folds overlap, which weakens the independence assumption.
The p-values are descriptive and should be confirmed on independent data.

**Q: Can a profile diagnose a student or trigger an intervention?**

A: No. The labels describe patterns in self-reported, cross-sectional
conditions. They are not diagnoses and are not for individual decisions.

## Explain it like I am five

Imagine sorting cards by five scores about money, travel, internet, and
studying at home. We learn the sorting rule using only the training cards,
then use the same rule on new cards. We check whether those groups help a
separate prediction model. In this example, adding the groups barely changed
the model, and using the groups alone made it worse. The groups describe
patterns; they do not tell us who a person is or what will happen to them.
