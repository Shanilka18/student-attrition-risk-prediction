> Summary: This bank contains 90 questions in the requested topic groups.
> The 15 marked questions are the suggested first-practice set.
> Answers describe the active generated-data run honestly.
> Held-out metrics and exact intervals are in the generated results summary.
> Ten tough questions are marked and answered without overstating evidence.

# Viva Questions and Model Answers

Read [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md) beside this document. It is generated from the active run. Any numerical answer in a viva should be read from that summary immediately before the presentation.

## Research problem and novelty (10)

1. **[MOST LIKELY] What problem does the project address?**  
   It tests whether repeated self-reported engagement patterns can support early identification of students who report an intention to leave a course. It does not predict confirmed dropout.

2. **[MOST LIKELY] What is the research contribution?**  
   The contribution is the integrated, leakage-aware workflow for repeated survey data, state summaries, model comparison, uncertainty, and human-support boundaries. I do not claim a new classifier.

3. **[TOUGH] Why should we believe results on generated data?**  
   Generated data let me test the pipeline and check whether code and evaluation steps run. They cannot establish real-world validity. I will not claim effectiveness until the real survey is collected and evaluated.

4. What is the target outcome?  
   The target is a self-reported intention item or score defined by the questionnaire. It is not an administrative record of withdrawal.

5. What is novel about the repeated design?  
   It lets the analysis compare current engagement summaries with changes and sequence states over four time points. That design choice is useful, but it does not by itself prove a better prediction.

6. Who could benefit from the work?  
   Researchers and student-support staff could use validated evidence to consider voluntary, supportive outreach. A score must not decide a student's status or access to education.

7. **[TOUGH] Is attrition intention the same as dropout?**  
   No. Intention is self-reported and may not become action. A student can leave without reporting intention, or report intention and later remain.

8. What is the main research question?  
   Whether engagement patterns collected repeatedly can help distinguish self-reported intention levels, and whether state-derived features add validated value beyond aggregate features.

9. What is the project not claiming?  
   It is not claiming causal drivers, confirmed dropout prediction, fair performance for every group, or readiness for real-world decisions.

10. **[MOST LIKELY] What is the strongest result and the biggest caveat?**  
    The current classification run ranks a k-nearest-neighbours model first. The biggest caveat is that the active run uses generated data, so its score is not evidence of performance on real students.

## Data and questionnaire (12)

11. **[MOST LIKELY] What data does the project use now?**  
    The active run uses generated sample records, not completed real survey responses. The source label and cohort are in [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).

12. What are the planned data inputs?  
    Repeated questionnaire responses about engagement, plus the outcome items defined by the survey.

13. Why use a questionnaire?  
    It captures experiences that may not be present in administrative records. Self-report is also subjective and must be described as such.

14. How many time points are in the survey sequence?  
    Four time points are represented in the current design. The sequence is still short and does not capture every change between surveys.

15. What is the unit of analysis?  
    The student is the unit of analysis for splitting. Repeated observations from one student must not appear in separate partitions.

16. What do A1–A3 represent?  
    They are outcome items. They are excluded from predictors to prevent the model from receiving the answer it is supposed to predict.

17. Does the current sample represent enrolled students?  
    It represents generated records only. Representativeness cannot be inferred until recruitment and real response patterns are documented.

18. What will be recorded about participants?  
    The proposal should report eligibility, recruitment, consent, response dates, exclusions, and the final analyzed cohort from the real survey records.

19. How will identities be protected?  
    Use a study identifier in the analysis set, store any linkage key separately, restrict access, and minimize retention.

20. What is the intention label's limitation?  
    It captures what a respondent reports at one point. It may change and is not the same as observed withdrawal.

21. **[MOST LIKELY] What happens when real data arrive?**  
    Re-run data checks, record participant flow, create a new student-level split, refit within CV, evaluate a held-out test once, and regenerate all result documents.

22. **[TOUGH] What if the least engaged students do not answer the survey?**  
    That creates nonresponse bias that modeling cannot remove automatically. I would report participation and missingness, compare available response patterns where ethically permitted, and limit claims to the observed sample.

## Cleaning and preprocessing (8)

23. What does data cleaning check?  
    It checks schema and headers, valid ordinal values, duplicates, careless responding, and missingness under declared rules.

24. What do the current cleaning counts say?  
    The active run has no training-cleaning report, so its raw and dropped-row counts are unavailable. A separate scoring audit is shown separately in the results summary.

25. How are missing values handled?  
    The policy is explicit and learned imputation must be fitted on training data only. The active training-cleaning artifact does not provide missingness counts.

26. **[TOUGH] Can the separate 200-row audit stand in for training cleaning?**  
    No. It is a later scoring-mode audit with a different run identifier. Its flagged and retained rows must not be reported as training exclusions.

27. What is a careless-response check?  
    It flags patterns such as identical answers across many items for review. A flag is not an automatic claim that a person responded dishonestly.

28. How do you handle duplicates?  
    The documented rule resolves duplicate submissions using the project's declared policy. For real data, the rule and count should be reported from the matching training quality artifact.

29. When are scaling and imputation fitted?  
    After the student-level split, inside each training fold. Validation and test records do not fit preprocessing parameters.

30. **[MOST LIKELY] How do you prevent outcome leakage?**  
    I exclude A1–A3 from the feature matrix and keep each student in one partition. I also fit learned preprocessing within each CV fold and check the feature manifest.

## States, clustering, and HMM (12)

31. What is clustering used for?  
    It groups similar engagement summaries without using the target label. The resulting groups are descriptive, not diagnoses.

32. **[MOST LIKELY] How many clusters were selected in the active run?**  
    The active state artifact selects three K-means clusters. See the generated summary for the selection metrics and names.

33. What is a silhouette score?  
    It summarizes how close an observation is to its own cluster compared with other clusters. It is one diagnostic, not proof that clusters are real kinds of students.

34. What does bootstrap stability measure?  
    It checks whether clustering assignments remain similar across resampled data. Stability supports repeatability of a grouping, not its educational meaning.

35. What does an HMM add?  
    It models a sequence as transitions among hidden states that emit observed engagement patterns.

36. **[MOST LIKELY] How many HMM states did the active run select?**  
    The active HMM artifact selected four states and recorded convergence. Their labels summarize fitted patterns; they are not validated student categories.

37. Why use both clustering and an HMM?  
    Clustering describes groups of summaries; an HMM represents state transitions over time. They answer related but different descriptive questions.

38. **[TOUGH] How do you know the HMM states mean anything?**  
    I do not know from fit statistics alone. They need replication, substantive review, sensitivity checks, and validation against an independent outcome before receiving a strong interpretation.

39. How are state names assigned?  
    Names summarize fitted response patterns for communication. They are not ground-truth labels or claims about a student's motivation.

40. Do state features improve the model?  
    The active ablation does not show a statistically significant AUC gain for either tested classifier. One LightGBM MCC comparison shows a statistically significant decline, so a blanket benefit claim would be wrong.

41. What does a non-significant ablation mean?  
    The comparison did not provide sufficient evidence of a difference under its test and assumptions. It does not prove the two feature sets are identical.

42. What is the main limitation of the sequence analysis?  
    There are only four scheduled observation points, and the active evidence is generated. The estimated states may not transfer to real student histories.

## Models and leaderboard (14)

43. **[MOST LIKELY] Which model leads the active classification track?**  
    The active leaderboard and model card identify k-nearest neighbours as the classification champion. Its exact validation and held-out estimates are in [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md).

44. Which model leads the active regression track?  
    The active regression leaderboard ranks a stacking regressor first. Its held-out fit is much weaker than its validation fit, so I would not present the validation result alone.

45. Why include a logistic-regression baseline?  
    It offers a simple, interpretable comparison and helps show whether added complexity is useful.

46. Why compare tree ensembles?  
    They can represent nonlinear relationships and interactions that a linear model may miss.

47. **[TOUGH] Why did a simple model beat the deep one?**  
    The active sample is generated and may not contain enough reliable sequence signal for a deep model. The observed ranking is run-specific; it does not prove that simple models always win.

48. Does a higher leaderboard score prove practical usefulness?  
    No. It is a validation comparison under the chosen metric. Real-data validation, calibration, uncertainty, fairness, and human factors still matter.

49. What does the leaderboard score represent?  
    It is the stored selection score used to rank candidates in this run. The results summary reports it alongside component metrics so it is not mistaken for a probability or a test metric.

50. **[TOUGH] Why not select the model with the highest test score?**  
    That would use the held-out test for model selection and make the final estimate optimistic. Choices should be made from training and validation evidence before a single test evaluation.

51. Why are there two prediction tracks?  
    Classification predicts a defined binary intention label. Regression predicts a continuous outcome score. Their metrics answer different questions.

52. Why compare models from different categories?  
    A common split and selection process allows a transparent comparison of simple, ensemble, and temporal candidates.

53. What does the generalization gap show?  
    It compares fitted performance with validation performance. A large gap is a warning that training results may not carry to new cases.

54. Why is regression performance a concern?  
    The current regression test R-squared is low and its bootstrap interval includes values at or below zero. This weakens claims that the continuous score generalizes.

55. How should the winner be chosen for a real deployment?  
    There is no deployment recommendation yet. A future choice needs real-data validation, predefined metrics, calibration, governance, and review of harms and benefits.

56. What is the current threshold discrepancy?  
    The leaderboard records 0.29 and the frozen model card records 0.26. The bundle follows the model card, but the disagreement must be reconciled before a later release.

## Overfitting and validation (12)

57. What is overfitting?  
    It is learning training-specific detail that does not carry to unseen students.

58. How is overfitting monitored?  
    Compare training and validation performance, inspect fold variability, and keep a genuinely held-out test for one final check.

59. **[TOUGH] Was the test set used more than once?**  
    The active run has a single-use marker, but the saved leaderboards contain test metrics for non-champion models. The artifact set cannot independently verify strict one-time access for one frozen model. I would not reopen or tune against this test.

60. Why split by student?  
    It prevents repeated responses from the same person leaking across training and evaluation sets.

61. Why use stratified folds?  
    They aim to keep class proportions reasonably similar across folds for classification comparisons.

62. Where does preprocessing happen?  
    In the training portion of each fold. It must not learn imputation or scaling values from the validation or test portion.

63. **[MOST LIKELY] What should “test used once” mean?**  
    For a clean evaluation, model and threshold choices are made without test scores, then one frozen model is evaluated and the test is not reused. The current artifacts do not fully confirm that protocol because non-champion test metrics are also saved.

64. What is a validation set for?  
    It provides evidence for comparing models and choices while the test remains untouched.

65. What does a confidence interval add?  
    It communicates uncertainty around an estimate. It does not guarantee that the next cohort's result will fall inside the interval.

66. Is a p-value the probability the model is correct?  
    No. It describes how compatible the observed comparison is with a specified null model and assumptions.

67. When do you call a comparison non-significant?  
    When the recorded test does not meet the prespecified significance rule. The current state-feature AUC comparisons are reported as not statistically significant.

68. What would you do if folds disagree?  
    Report the variability, investigate instability without peeking at the test, simplify the claim, and collect more representative data if possible.

## Explainability (8)

69. **[MOST LIKELY] What does SHAP explain?**  
    It attributes a model output among the input features under the method's assumptions. It does not explain a causal process in the student.

70. What are the top active-run SHAP drivers?  
    The leading entries include change from peak to last engagement, missing review-time information for one period, and minimum participation. Their exact ranks are in the generated result table.

71. **[TOUGH] SHAP consistency is low. Can you still use the ranking?**  
    Only as an exploratory description of this fitted run. Low across-fold stability means I should not present the order as a reliable universal driver list.

72. Does a high SHAP value mean changing that feature would change the outcome?  
    No. It describes the model's use of the feature, not the effect of an intervention.

73. What is global feature importance?  
    A summary across observations of how strongly features contribute to model predictions, under the selected explanation method.

74. What is a local explanation?  
    An explanation for one prediction. It can help a reviewer inspect the model output but should not label the student.

75. How can explanations be validated?  
    Check stability across folds and model versions, compare with domain review, test plausible perturbations, and communicate uncertainty.

76. What is the explanation's main limitation here?  
    The rankings vary across folds and come from generated data, so they cannot support causal or population-wide claims.

## Ethics, privacy, and limitations (8)

77. **[MOST LIKELY] What is the main ethical boundary?**  
    Use a result to start a supportive conversation, not to make a decision about a student.

78. How will consent be handled?  
    Real collection must be voluntary, informed, and approved under the institution's required review process before recruitment.

79. **[MOST LIKELY] Are demographic features used?**  
    The described feature design excludes demographic predictors. That does not prove fairness because survey response and missingness can still be biased.

80. **[TOUGH] Could the tool unfairly stigmatize students?**  
    Yes. A risk label can stigmatize or invite inappropriate intervention. The system should not issue a decision; any support must be voluntary, contextual, and reviewed by a person.

81. How will privacy be protected?  
    Minimize data, pseudonymize the analysis file, restrict access to identifiers, set retention limits, and keep any identity key separately.

82. **[MOST LIKELY] What is the largest limitation?**  
    The active evidence is generated and the target is self-reported intention. Real-data validity and the relation to observed course withdrawal remain untested.

83. What bias could remain without demographic inputs?  
    Differences in participation, access, language, course context, missingness, and trust can still produce unequal error rates.

84. What is required before real-world use?  
    Real-data validation, consent and privacy review, leakage and calibration checks, subgroup assessment, governance approval, human oversight, and a process to challenge or correct data.

## System and website (6)

85. What does the system provide?  
    It provides an analysis workflow for survey data, model evaluation, and governed results. It is not an automatic educational decision-maker.

86. **[MOST LIKELY] What will you demonstrate?**  
    I will show the project overview, the approved sample workflow, and the results and limitations screen using the exact steps in [23_PRESENTATION_GUIDE.md](./23_PRESENTATION_GUIDE.md).

87. What happens if an uploaded file is invalid?  
    The workflow should report the validation problem and stop scoring rather than silently treating bad inputs as valid.

88. How are model versions governed?  
    A run should preserve its model card and artifacts; activation and rollback should be auditable and restricted to an authorized operator.

89. Where is the viva documentation stored?  
    It is in the repository's `docs/` folder and is not website content.

90. What is the backup if the live system fails?  
    Use the prepared screenshots in `docs/figures/final/`, explain they are static, and continue with the same limitations and result evidence.

## What changes when real data arrive

The source marker, cohort flow, cleaning counts, model rankings, intervals, calibration, states, explanation stability, and all metric answers must be refreshed from the new artifacts. Until then, answer every results question as evidence from generated data only. Intention remains distinct from confirmed dropout, the sequence remains four time points, and the held-out test must still be used once.

## Common mistakes

- Treating generated results as real-cohort evidence.
- Calling intention confirmed attrition.
- Tuning on the held-out test.
- Claiming states, SHAP values, or a predictive score are causal.

## If the examiner asks

Answer directly, state the evidence source, and give the limitation. When an exact value is requested, read it from [19_RESULTS_SUMMARY.md](./19_RESULTS_SUMMARY.md) rather than from memory.
