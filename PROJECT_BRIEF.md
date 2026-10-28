# PROJECT BRIEF — Multimodal Student Engagement Intelligence

## 1. What we are building
A complete, final-quality research system for ONE component of a four-component student attrition-risk project. The component produces a **Multimodal Engagement Index (MEI), 0–100**, for each student, where **higher = higher engagement-related attrition risk**. It also explains every score in plain language.
Pipeline: questionnaire data → cleaning → preprocessing/feature engineering → engagement states (clustering + Hidden Markov Model) → many candidate models (linear, shallow/classical, boosting, deep, ensembles, hybrid) → leaderboard with overfitting checks → best model chosen → SHAP explanations → MEI → website that shows the leaderboard, patterns, drivers, and scores uploaded student files.

## 2. Data truth (this overrides the old proposal text)
- We do NOT use campus data (no Moodle/LMS/Zoom/mouse logs). All data are collected by ONE anonymous, closed-ended questionnaire (Google Forms style) filled in by students. The same questionnaire also feeds the other three components; our component uses only the engagement part.
- Real survey data do not exist yet. We first build and test everything on GENERATED sample data that imitates a Google Forms export (text answers, long headers, timestamps, missing answers, careless responders, duplicates). Later the real file will replace it WITHOUT code changes (config/mapping only). Therefore the code must be data-source agnostic and must record `data_source: sample | survey` in every run.

## 3. Questionnaire design (engagement part)
- The semester is split into **4 periods** (P1 early, P2 early-middle, P3 late-middle, P4 recent; about 4 weeks each).
- **7 engagement indicators**, each asked for all 4 periods (28 numeric inputs). Each indicator is one grid question: rows = P1..P4, columns = 5 ordered options:
  1. resource_access — days per week the student opened online learning materials (0 / 1–2 / 3–4 / 5–6 / every day)
  2. study_time — hours per week studying (<2 / 2–5 / 6–10 / 11–15 / >15)
  3. review_time — hours per week reviewing past material (none / <1 / 1–2 / 3–4 / >4)
  4. quiz_practice — attempted quizzes/practice questions (never / rarely / sometimes / often / always)
  5. assignment_engagement — started and submitted assignments on time (never … always)
  6. study_consistency — "I studied on a regular schedule" (strongly disagree … strongly agree)
  7. participation — took part in lectures, discussions, group work (never … always)
  Coded 1–5 (ordinal).
- 3 **outcome items** (5-point agreement, coded 1–5): A1 "I have seriously thought about leaving my degree", A2 "I have considered deferring or taking a break", A3 "I do not plan to continue next semester". These define the label; they are NEVER model inputs.
- 1 attention-check row (instruction to select "Sometimes"), consent item, anonymous student_id, timestamp.
- Known limitation to document: answers are retrospective self-reports (recall bias); the label is self-reported intention (a proxy for real attrition).

## 4. Targets and the index
- Regression target: `attrition_intent_score` = ((mean(A1,A2,A3) − 1) / 4) × 100.
- Classification target: `at_risk` = 1 if score ≥ 60 (configurable in config/labels.yaml).
- MEI = round(100 × calibrated probability of at_risk) from the champion classification model. Bands (configurable): Low 0–34, Moderate 35–64, High 65–100. Decision threshold for the yes/no flag is tuned on validation data (not fixed at 0.5).

## 5. Engagement states and features
- Per student: composite engagement E_t (0–100) per period; derived features per indicator (mean, std, min, max, OLS slope, last−first, recent drop, volatility/reversals), composite drift features.
- States: (a) clustering (KMeans, GMM, Ward) on aggregated features; (b) Gaussian HMM (hmmlearn) over the 4-period sequences, K chosen from 2–4. Student-level PATTERN from the decoded state path: Consistent, Declining, Sporadic, Improving; plus overall LEVEL (High/Medium/Low).
- All state models are fitted INSIDE each training fold (see G3).

## 6. Models and the two leaderboard tracks
- Track A (primary): classification of at_risk. Track B (supporting): regression of attrition_intent_score.
- Categories: Baseline (single-signal rule/logistic using resource_access only — imitates the Moodle login indicator), Linear, Classical/Shallow, Boosting/Tree ensembles, Deep learning, Ensembles (voting/stacking), Hybrid (proposed: states + aggregated features → boosting).
- Final champion for MEI = best model of Track A under the selection rule in §7. Track B is reported as supporting evidence (R², etc.).

## 7. Metrics and selection rule
- Classification: accuracy, balanced accuracy, precision, recall, specificity, F1 (at-risk), macro-F1, AUC-ROC, PR-AUC, MCC, Cohen's kappa, log-loss, Brier score, ECE (calibration error).
- Regression: R², adjusted R², RMSE, MAE, median AE, explained variance, Pearson, Spearman, % of predictions within ±10 points.
- Also: train vs CV-validation vs test values, generalisation gap, CV standard deviation, seed variation, training time, inference time, model size.
- Leaderboard Score (Track A) = mean(F1_at_risk, AUC_ROC, balanced_accuracy, (MCC+1)/2) measured on cross-validation, minus overfit penalty = 2 × max(0, gap − 0.05), where gap = train F1 − CV-validation F1. Track B score = mean(R², Spearman) minus the same penalty on R² gap.
- Overfit status: gap ≤ 0.05 "Good fit"; 0.05–0.10 "Slight overfit"; > 0.10 "Overfit". A model with status "Overfit" cannot be champion unless it has been remediated and re-evaluated.
- Test set is touched ONCE for the finalists in Phase 9.

## 8. Website rules (final product, not a demo)
- Pages: Overview, Model Leaderboard, Engagement Patterns, What Drivers Risk, Score Students (upload CSV/Excel), Train on New Data.
- NO viva, presentation, thesis, or proposal content in the website.
- Never use the words prototype, demo, beta, mock, placeholder, "AI-powered", "smart", "magic".
- Plain language for non-technical staff; every technical term has a plain label with a tooltip for the formal name.
- Must not look AI-generated: see Phase 12 design rules. No purple/indigo/violet, no gradients, no glow, no glassmorphism, no emojis, no sparkle/robot icons.
- A small label on result pages states the training data: "Sample survey responses (generated)" or "Your survey data". Never present generated data as real findings.

## 9. Global rules (G1–G13)
G1 One phase at a time. Finish only the current phase, run its checks, write its report, then STOP. Never start the next phase; its instructions arrive in the next prompt.
G2 Never invent numbers. Every metric, figure, count or claim in code output, docs or UI must come from files produced by real runs. If something fails or is weak, say so plainly.
G3 No leakage. Split by student first. Every learned transformer (imputer, scaler, encoder, KMeans, GMM, HMM, calibrator, selector) is fitted on the training portion only and inside CV folds (use sklearn Pipelines / custom transformers). Outcome items A1–A3 and anything derived from them are never features (assert this in code and in a test). Test set is used once in Phase 9.
G4 Reproducibility. Global seed 42 (config). Log package versions. Each run is stored in artifacts/runs/<run_id>/ with manifest.json (data_source, n_rows, seed, config hash, git commit, timestamps).
G5 Config over hard-coding: paths, column names, thresholds, bands, model lists, time budgets live in config/*.yaml.
G6 Honesty about generated data (see §2, §8).
G7 Code quality: type hints, docstrings, logging (no print), small functions, tests for every module, ruff + black clean, pytest green.
G8 Documentation duty: every phase writes/updates docs/NN_<name>.md in SIMPLE English with sections: What it is · Why we do it · How it works step by step · Inputs/Outputs · Formulas · Decisions and rejected alternatives · Common mistakes · "How to explain this in a viva" · "Explain like I am 5". Phase reports go to docs/phase_reports/phase_NN.md.
G9 Figures: static figures with matplotlib/seaborn, 300-dpi PNG + SVG, one consistent style (palette in §10), clear titles, labelled axes with units, a caption in docs/figures/INDEX.md.
G10 Portability and hardware: Windows/Linux/macOS, 8 GB RAM, no GPU. Time-box every heavy step through config (default total training budget for the full run ≤ 60 minutes).
G11 Log every decision to docs/DECISIONS.md (date, decision, reason, alternatives rejected).
G12 Ask me a question only if truly blocked; otherwise choose the documented default and record it.
G13 Reply format at the end of every phase: (1) what you did, (2) files created/changed, (3) key numbers read from files, (4) checks run with PASS/FAIL, (5) risks or decisions I should know about. Then STOP.

## 10. Palette used by figures and the website
paper #F5F2EC · surface #FFFFFF · ink #1A1D21 · muted #5B6168 · line #DDD8CE · primary forest #1F4E46 · info blue #2B5C8A · warning amber #C98A00 · risk-low #2E7D4F · risk-moderate #C98A00 · risk-high #B3261E · extra categorical: #8C4A3B, #6E7F3F, #7A6F63
