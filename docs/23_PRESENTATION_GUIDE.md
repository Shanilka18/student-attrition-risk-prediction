> Summary: This is a 15-slide, 12-minute presentation plan.
> Each slide has one message, one exact figure, speaker notes, a likely question, and timing.
> The live walkthrough uses only the built-in illustrative student profile.
> The 12-number sheet is generated from active run artifacts.
> Generated data and limitations must remain explicit throughout the talk.

# Presentation Guide

## 12-minute storyline

### Slide 1 — A support question, not an automated decision
- **Single message:** The research asks whether engagement patterns can support timely, human-led help.
- **Show:** [ui_overview_1440.png](figures/final/ui_overview_1440.png)
- **Speaker notes:** Open with the student-support problem, not the software. Explain that the project investigates self-reported intention to leave and repeated engagement patterns. The purpose is to help a person consider a supportive conversation. It is not to label, discipline, or make a decision about a student. Say early that the current model run uses generated data so the audience does not confuse pipeline evidence with real-world evidence.
- **Expected question:** Is the system intended to identify confirmed dropouts?
- **Time:** 40 seconds

### Slide 2 — What the questionnaire observes
- **Single message:** Repeated self-reports provide a short view of engagement, not a complete record of student life.
- **Show:** [eda_01_marginal_distributions.png](figures/final/eda_01_marginal_distributions.png)
- **Speaker notes:** Describe the survey as a set of engagement questions repeated over time. The items cover activities such as study, review, participation, and practice. These are reports by respondents, not direct measurements of every learning activity. The pattern plot helps explain the design, but it does not prove that the sample represents enrolled students. Keep the focus on what is measured and what remains unknown.
- **Expected question:** Why use self-report instead of only administrative data?
- **Time:** 40 seconds

### Slide 3 — Data source is a key limitation
- **Single message:** The current evidence comes from generated sample records, not real survey participants.
- **Show:** [eda_06_response_anomalies.png](figures/final/eda_06_response_anomalies.png)
- **Speaker notes:** State the source label directly. Generated data are useful for checking the pipeline and demonstrating methods, but they cannot establish predictive accuracy for real students. Do not call the generated cohort participants. Explain that recruitment, consent, nonresponse, and real missingness still need to be documented. This limitation affects every model result and every interpretation that follows. The next phase is an approved real survey with a matching quality and recruitment record.
- **Expected question:** What can you conclude from generated data?
- **Time:** 45 seconds

### Slide 4 — Quality checks make data limits visible
- **Single message:** Cleaning is a documented audit, and unavailable training counts must stay unavailable.
- **Show:** [eda_05_missingness_map.png](figures/final/eda_05_missingness_map.png)
- **Speaker notes:** Explain that the pipeline checks schema, allowed values, duplicate records, response patterns, and missingness. The active training run does not include its training cleaning report, so its raw and excluded counts cannot be verified. There is a separate scoring audit, but it is not the training cohort. Showing the boundary is more honest than filling a gap with a number from another run.
- **Expected question:** How many training records were removed?
- **Time:** 45 seconds

### Slide 5 — Keep predictors separate from outcomes
- **Single message:** The model uses engagement features while outcome items A1–A3 remain excluded.
- **Show:** [eda_02_indicator_correlations.png](figures/final/eda_02_indicator_correlations.png)
- **Speaker notes:** Describe features as summaries of engagement levels and change. The target items A1–A3 define the outcome and must not enter the predictor matrix. Students are split before model fitting, and learned preprocessing belongs inside each training fold. This prevents the model from borrowing outcome information or learning transformation values from validation and test students. I also check the saved feature manifest and require the leakage audit to pass.
- **Expected question:** How did you prevent target leakage?
- **Time:** 45 seconds

### Slide 6 — States are summaries, not identities
- **Single message:** Clusters describe patterns in the data; they are not validated student types.
- **Show:** [state_06_cluster_silhouette.png](figures/final/state_06_cluster_silhouette.png)
- **Speaker notes:** Explain that clustering groups similar engagement summaries without the target labels. The active run selected a three-cluster K-means solution. The silhouette and stability measures are diagnostics that help compare groupings. They do not prove that the groups have stable meaning outside this generated cohort. Use descriptive names cautiously and avoid describing a student as a permanent type. Replication with another cohort and review of the survey context would be needed.
- **Expected question:** How do you know the clusters are meaningful?
- **Time:** 45 seconds

### Slide 7 — The HMM represents transitions
- **Single message:** A hidden Markov model represents changing latent states across the short survey sequence.
- **Show:** [state_02_transition_matrix.png](figures/final/state_02_transition_matrix.png)
- **Speaker notes:** Point to the transition matrix and explain that each entry summarizes a fitted probability of moving from one hidden state to another. The active artifact selected four states and recorded convergence. Convergence is a computational property, not proof that the states are meaningful. Independent replication and substantive review are needed before giving these labels a strong interpretation. The matrix cannot explain why a transition happened, and the time observations are sparse.
- **Expected question:** What evidence supports the number of hidden states?
- **Time:** 50 seconds

### Slide 8 — Compare models using validation evidence
- **Single message:** The current classification leaderboard ranks a simple neighbour model first.
- **Show:** [class_01_leaderboard_bars.png](figures/final/class_01_leaderboard_bars.png)
- **Speaker notes:** Explain that candidates are ranked using cross-validation evidence rather than the held-out test. The active classification champion is k-nearest neighbours. A simple model can lead when data volume, signal, and structure do not justify more complexity. The ranking is specific to generated data and can change when real responses arrive. Use the results summary for the current metrics and intervals.
- **Expected question:** Why did the deep model not win?
- **Time:** 50 seconds

### Slide 9 — Discrimination is not calibration
- **Single message:** A useful ranking score does not guarantee reliable probabilities.
- **Show:** [class_06_calibration_curves.png](figures/final/class_06_calibration_curves.png)
- **Speaker notes:** Explain that discrimination asks whether higher scores rank positive cases above negative cases. Calibration asks whether predicted probabilities match observed frequencies. The packaged model card records an isotonic calibrator, but a test calibration metric is not present in the active result artifact. The validation ECE should not be relabeled as test evidence. Thresholds also differ between the leaderboard and model card.
- **Expected question:** Which threshold is the deployed bundle using?
- **Time:** 50 seconds

### Slide 10 — Regression generalization is weak
- **Single message:** The regression leader's held-out fit is much weaker than its validation fit.
- **Show:** [reg_01_leaderboard_bars.png](figures/final/reg_01_leaderboard_bars.png)
- **Speaker notes:** The regression leaderboard ranks a stacking regressor first, but the held-out R-squared is low and its bootstrap interval includes values at or below zero. This is a warning against making a strong claim from the validation score. Report errors in the target's units and explain that the result needs confirmation on real survey data. This is a central limitation, not a result to hide behind the leaderboard rank.
- **Expected question:** Does the regression result generalize?
- **Time:** 50 seconds

### Slide 11 — State features did not improve AUC significantly
- **Single message:** The active ablation does not show a statistically significant AUC benefit from adding states.
- **Show:** [state_04_pattern_distribution.png](figures/final/state_04_pattern_distribution.png)
- **Speaker notes:** Explain the ablation comparison between aggregate features and aggregate features with state information. The two tested AUC comparisons are not statistically significant. A LightGBM MCC comparison shows a statistically significant decline, which argues against claiming a general improvement. The plot provides context about pattern counts; the statistical test values are in the generated result summary. These results do not justify adding state features by default later.
- **Expected question:** What did the ablation establish?
- **Time:** 50 seconds

### Slide 12 — Explanations are exploratory
- **Single message:** SHAP can summarize model behavior, but the active feature rankings are unstable.
- **Show:** [fig_shap_05_fold_consistency.png](figures/final/fig_shap_05_fold_consistency.png)
- **Speaker notes:** Explain that SHAP assigns contributions to features for a model prediction. The active run reports weak agreement in feature rankings across folds. Therefore, the top features are exploratory descriptions of this fitted model, not universal drivers or causes. They should not be used to justify an intervention without a separate evaluation and domain review. Separate what the model uses from what might help a student.
- **Expected question:** Can SHAP tell us what causes attrition?
- **Time:** 50 seconds

### Slide 13 — Responsible use requires people and safeguards
- **Single message:** A result may prompt a supportive conversation, never a decision about a student.
- **Show:** [ui_shell_1440.png](figures/final/ui_shell_1440.png)
- **Speaker notes:** Summarize the privacy and fairness limits. The described predictors exclude demographic variables, but nonresponse and access differences can still create bias. Any real use needs consent, purpose limitation, access controls, subgroup evaluation, a correction route, and human oversight. A generated-data result is not ready to inform a real student-support decision. Any later trial should monitor unintended harms and provide a clear appeal route.
- **Expected question:** Does excluding demographic variables guarantee fairness?
- **Time:** 50 seconds

### Slide 14 — The website demonstrates a workflow
- **Single message:** The interface illustrates the workflow; it is not additional validation evidence.
- **Show:** [ui_score_results.png](figures/final/ui_score_results.png)
- **Speaker notes:** Introduce the live walkthrough as a software demonstration using the built-in sample profile. Show the navigation, a declining illustrative pattern, and how the output explanation is displayed. Say clearly that this is not a real student record and not evidence that the model works on real survey data. If a service is unavailable, switch to the prepared screenshots without improvising a real-data upload.
- **Expected question:** What happens if the live demo fails?
- **Time:** 50 seconds

### Slide 15 — Next step: real data and independent validation
- **Single message:** The next scientific step is a consented real survey and a fresh, leakage-safe evaluation.
- **Show:** [ui_overview_1440.png](figures/final/ui_overview_1440.png)
- **Speaker notes:** Close with the boundary between a working research pipeline and a validated real-data result. Before operational use, collect approved survey responses, document the cohort and cleaning, preserve student-level separation, fit transformations inside folds, and use the held-out test once. Reconcile the threshold records and measure calibration and subgroup performance. The responsible conclusion today is that the workflow is ready for evidence gathering, not decision-making.
- **Expected question:** What must happen before the model is used with students?
- **Time:** 60 seconds

**Total speaking time: 12 minutes.**

## Five-minute live-demo script

Use the running local site with its sample backend. Do not upload real student data during the talk. If the single-score route is unavailable, move immediately to the screenshot plan below.

| Time | Exact click | What to say |
|---|---|---|
| 0:00–0:35 | Click **Overview** in the left navigation. | “This is the project overview. The active research results are generated-data results, not estimates validated on real students.” |
| 0:35–1:10 | Click **Model Leaderboard**. Point to the classification and regression tracks and the leading rows. | “The leaderboard is a validation comparison. The classification and regression tasks have different outcomes and different metrics.” |
| 1:10–1:45 | Click **Engagement Patterns**. Point to a trajectory or transition visualization. | “These are fitted summaries of engagement patterns. They are descriptive and are not student diagnoses.” |
| 1:45–2:20 | Click **What Drives Risk**. Point to the global feature view. | “These are model attributions. Fold stability is weak, so I treat them as exploratory rather than causal.” |
| 2:20–2:45 | Click **Score Students**. Click **Score an Individual Student**. | “I will use the interface's sample profile, not a real student record.” |
| 2:45–3:10 | In **Quick Profiles**, click **Declining Pace**. Leave the sample identifier unchanged. | “This preset fills an illustrative declining pattern across the survey periods.” |
| 3:10–3:40 | Click **Calculate Engagement Index**. | “This output is an interface demonstration. It is not a validated estimate for an actual student.” |
| 3:40–4:20 | Point only to the individual result and **Waterfall Driver Explanation**; do not interpret the cohort band chart. | “The explanation shows how this output is assembled, not what causes an outcome. The current cohort chart uses different band cutoffs from the saved model artifact, so I will not treat that chart as validation evidence.” |
| 4:20–5:00 | Close the modal. Return to **Overview**. | “The responsible next step is real-data validation, consent, calibration, fairness review, and human oversight.” |

## Backup plan if the live demo fails

1. Open [ui_overview_1440.png](figures/final/ui_overview_1440.png) and explain the sample-data limitation.
2. Show [ui_leaderboard_1440.png](figures/final/ui_leaderboard_1440.png) for the comparison page.
3. Show [ui_patterns_1440.png](figures/final/ui_patterns_1440.png) for the pattern view.
4. Show [ui_drivers_1440.png](figures/final/ui_drivers_1440.png) for feature attribution.
5. Show [ui_score_single_student.png](figures/final/ui_score_single_student.png) and [ui_score_results.png](figures/final/ui_score_results.png) for the sample scoring flow.
6. State that these are static screenshots, then answer questions from [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).

## One-page cheat sheet: 12 numbers to remember

Data source for this sheet: generated from [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md). The values refresh when `python -m mei.cli docs` runs.

<!-- GENERATED CHEAT SHEET START -->
| Remember | Current artifact value |
|---|---|
| Split-cohort total | 2,184 |
| Training cohort | 1,528 |
| Validation cohort | 328 |
| Test cohort | 328 |
| Test at-risk percentage | 25.30 |
| Champion CV leaderboard score | 0.7949 |
| Classification test F1 | 0.7182 |
| Classification test F1 95% CI | [0.6385, 0.7914] |
| Classification test ROC AUC | 0.8773 |
| Regression test R-squared | 0.1147 |
| K-means clusters | 3 |
| SHAP mean Spearman | 0.0809 |
<!-- GENERATED CHEAT SHEET END -->

## What changes when real data arrive

Every result number in the cheat sheet must be regenerated from the new active run. Replace the sample source label only when the model card identifies the real survey. Report real recruitment, consent, exclusions, and cleaning counts; preserve the distinction between self-reported intentions and confirmed dropout. The sequence has four time points, and the held-out test must still be used once. Reassess calibration, uncertainty, state meaning, SHAP stability, subgroup error, and the threshold discrepancy before presenting any operational claim.

## Common mistakes

- Speaking beyond the time assigned to a slide.
- Calling generated observations real survey participants.
- Claiming that a model, state, or explanation causes an outcome.
- Using a live upload with real student data.
- Forgetting the threshold mismatch or weak regression and SHAP evidence.

## If the examiner asks

Answer from the generated [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md), explain the limitation, and distinguish validation from held-out test evidence. If a demo result conflicts with the artifact summary, stop and use the static evidence rather than improvising.
