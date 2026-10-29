# EF-02 Source Reading Notes

## Sources reviewed

- `reference/TAF_KND_CDAP_IT_05.pdf`, read with `pdfplumber`.
- `reference/IT23445154.pdf`, read with `pdfplumber`. This is the proposal PDF; the separate proposal `.docx` named in the task was not present in `reference/`.
- `PROJECT_BRIEF.md`, `docs/DECISIONS.md`, and `docs/questionnaire_envfin_v2.md`.

The source files in `reference/` were not changed. The proposal PDF contains an unnumbered Abstract, followed by numbered proposal sections. The replacement files follow the extracted proposal structure; where a DOCX-only layout detail could not be checked, no assumption is made about it.

## TAF structure and style

The TAF is version 2.2. Section 10, **Objectives and Novelty**, contains the member table with **Member Name / Registration No**, **Sub Objective**, **Tasks**, and **Novelty** columns. The Environmental & Financial row spans PDF pages 6–7. Other member rows use formal prose: task cells state a sequence of actions, name the data and methods, compare models, describe explainability, and finish with an index; novelty cells briefly distinguish the component from simpler or existing approaches.

Visible TAF word limits are:

| TAF location | Limit |
|---|---:|
| Section 1, Topic | 12 words |
| Section 5, Research problem | 300 words, references excluded |
| Section 6, Existing research and systems | 300 words |
| Section 8, Solution nature and conceptual diagram | 500 words |

Section 10 does not state a separate word limit for each member's Tasks or Novelty cell. The replacements are therefore kept concise and comparable in length and style to the other member rows. The current Environmental & Financial row includes institutional records and causal inference; those elements do not match the specified closed-ended, cross-sectional questionnaire design.

## Proposal structure and replacement locations

The proposal PDF has no stated overall or per-section word limits in the sections reviewed. The requested paste locations are:

| Requested replacement | Proposal location |
|---|---|
| Abstract | Unnumbered Abstract |
| Specific objectives | Section 2.2 |
| Data, participants, inputs | Section 3.2; participant procedures are in 3.2.1 |
| Baselines | Section 3.3, Baseline and Benchmark Models |
| Evaluation metrics | Section 3.4, Table 2 |
| Validation strategy | Section 3.5 |
| Architecture | Section 4.1, Overview of Proposed Solution |
| Website requirements | Sections 5.1 and 5.2 |
| Ethics | Section 3.2.1, Data Collection and Participant Procedures |

The proposal currently describes optional institutional records, an attrition outcome, causal analyses, and individual risk explanations. These must be reconciled with the questionnaire-only design, the shared self-reported intention outcome, fold-contained profiles, the EFI, and the requirement not to use the website for individual decisions. No proposal word limit was found for the replacement sections.

## Source-of-truth points used

- The Environmental & Financial instrument is the 17 closed-ended items Q23–Q39, with conditional job-hours routing and structural not-applicable travel responses.
- Shared outcome fields are `attrition_intent_score` (0–100) and `at_risk` (1 when the score is at least 60). The score is stated intention, not observed withdrawal. Consent, attention check, `student_id`, and `timestamp` are not model features.
- The EFI is a 0–100 higher-risk score. The project brief defines it from calibrated probability when calibration is adequate.
- Four derived dimensions are Financial Stress, Commute Burden, Digital Access Gap, and Study Environment Pressure. Latent profiles are for descriptive reporting and are not inputs to the primary classifier.
- The TAF's existing causal-inference and Causal Forest wording is not carried into the corrected row.
- No empirical findings, subgroup results, approval status, supervisor details, or study dates were supplied as verified results for these replacement texts.
