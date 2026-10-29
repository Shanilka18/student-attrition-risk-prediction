> Summary: This project is research support, not an automated student decision.
> The active run uses generated data and self-reported intention labels.
> Intention is not confirmed withdrawal.
> Survey respondents may not represent the wider student population.
> Real-world use requires consent, governance, validation, and human oversight.

# Limitations and Ethics

## Evidence limits

- **Generated data:** The active run is sample-generated. Its metrics show that the pipeline runs; they do not establish performance on real students.
- **Self-report:** The label measures stated intention. A student may later stay, and a student with no stated intention may still leave.
- **Short sequence:** The survey design has four time points. It cannot show every change between surveys or establish long-term trajectories.
- **Test-use provenance:** The intended protocol reserves a held-out test for one final evaluation. A lock marker exists, but saved leaderboard rows also contain non-champion test metrics, so the artifacts do not independently confirm strict one-time use. Do not reopen or tune against this test.
- **Limited generalization:** Results may depend on the simulated data-generating process, institution, course, response rate, survey wording, and time period.
- **Regression uncertainty:** Consult the active results summary. Weak test fit or intervals crossing a null benchmark do not support strong continuous-score claims.
- **State meaning:** Clusters and HMM states are data-driven summaries. Their labels need independent validation and do not describe a student's identity.
- **Explanation stability:** The active SHAP consistency metrics are weak. Rank order is not reliable enough to present as a stable universal driver list.
- **Missing documentation:** The active run lacks a training-cleaning report and a test calibration metric. Do not estimate either.
- **Threshold conflict:** The model card and leaderboard record different thresholds. Reconcile this before any later release.
- **Website band mismatch:** The scoring-results chart displays different band cutoffs from the active model's saved band definitions. Do not present that chart as the active run's band verification; reconcile the interface before a later release.

## Privacy and consent

Collect only the minimum survey fields needed for the research question. Obtain informed, voluntary consent and explain the purpose, retention period, access controls, and withdrawal route. Do not include names, email addresses, or direct identifiers in analysis exports or screenshots. Store the identity key separately with restricted access. Use role-based access, encryption at rest and in transit, retention limits, and documented deletion procedures. Do not use student responses for discipline, grading, admissions, or eligibility decisions.

## Fairness

The model feature design does not use demographic features. This does not guarantee fairness. Participation and missingness may differ by language, disability, schedule, course, access to technology, or trust in the institution. A self-selected survey may underrepresent students who are already disengaged. Before a real-data study, predefine an ethical subgroup evaluation plan, suppress small groups that could identify people, and consult appropriate student and institutional representatives.

## Responsible use

**Use a result to start a supportive conversation, not to make a decision about a student.** A student must be able to correct the record, decline support, or ask how a result was produced. A staff member must consider context and must not treat a score or state as a diagnosis.

## Before any real-world use

1. Collect real survey data under approved consent and privacy procedures.
2. Validate the intention items and confirm that the target answers the research question.
3. Document cohort recruitment, exclusions, missingness, and the final data flow.
4. Split by student before fitting; fit preprocessing only inside CV folds.
5. Keep A1–A3 out of all predictors and audit the saved feature manifest.
6. Pre-register model selection and threshold decisions; evaluate a held-out test once.
7. Assess calibration, uncertainty, subgroup performance, and shift across courses and time.
8. Reconcile the current threshold discrepancy and verify the exact deployed bundle.
9. Conduct a privacy, security, fairness, and human-factors review.
10. Obtain institutional approval, define an appeal route, monitor harm, and stop use if safeguards fail.

## Common mistakes

- Calling intention a confirmed dropout outcome.
- Saying “no demographic inputs” proves fairness.
- Treating a generated-data score as real-world accuracy.
- Using a risk band to restrict opportunity or label a student.

## If the examiner asks

Say that the system is not ready for operational decisions. Explain the evidence limits, the human-support boundary, and the specific validation and governance work still required.
