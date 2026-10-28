# Multimodal Student Engagement Intelligence — Antigravity Prompt Pack

Project: KND_CDAP_IT_05 · Component: Multimodal Student Engagement Intelligence (Vimukthi H.A.V.C.)

This file has 4 parts:

- **Part A** – the corrected "Tasks" and "Novelty" text for your Topic Assessment Form (survey-based).
- **Part B** – how to run these prompts in Antigravity (explained very simply).
- **Part C** – what your overfitting screenshots show, and the graphs we will copy the idea of.
- **Part D** – the 19 prompts (Prompt 0 to Prompt 18). Paste them **one at a time, in order**.

---

# PART A — Corrected TAF text (Section 10, your row)

**Sub Objective:** Multimodal Student Engagement Intelligence

**Tasks**

Collect and preprocess student engagement data through a structured closed-ended questionnaire, capturing multiple engagement indicators such as frequency of accessing online learning resources, time spent studying and reviewing learning materials, quiz/practice activity, assignment engagement, study consistency, and participation in academic learning activities. Use the collected questionnaire responses to construct session/period-level behavioural representations and identify distinct engagement states (e.g., consistent, declining, sporadic) rather than relying on a single engagement measure. Apply clustering or a Hidden Markov Model (HMM) to the engagement-pattern data to discover distinct behavioural states. Use these discovered engagement states, alongside aggregated engagement features, as inputs to a Gradient Boosting (XGBoost/LightGBM) classifier, and compare against Logistic Regression and Random Forest to select the best-performing model. Apply SHAP to identify the engagement factors and behavioural states contributing most to attrition risk, and develop the Multimodal Engagement Index (0–100).

**Novelty**

The proposed component combines multiple student-reported engagement indicators, including learning-resource usage, study time, quiz/practice behaviour, assignment engagement, study consistency, and academic participation, rather than relying on a single engagement measure such as LMS login counts. By transforming these multiple engagement responses into distinct behavioural states and integrating them with aggregated engagement features, the proposed approach provides a more comprehensive assessment of student engagement. The use of explainable AI further identifies the engagement factors and behavioural states contributing most significantly to attrition risk.

> Note: your proposal report (vihu_1__2_.docx) still describes Moodle/Zoom/mouse-activity logs in the Abstract, Objectives, Section 3.2, Section 3.4 and the requirements. Prompt 17 produces replacement text for those parts so the proposal matches the survey-based design.

---

# PART B — How to use this in Antigravity (very simple)

Think of Antigravity as a very fast helper who can type code, run commands and open a browser, but who forgets things between chats. So we give it a **rule book** first (Prompt 0), and then we give it **one small job at a time**.

**Before you start**

1. Install Antigravity and sign in with your Google account.
2. Make a new empty folder on your computer, for example `engagement-intelligence`.
3. Inside it, make a folder called `reference` and put these two files in it: your proposal `vihu_1__2_.docx` and your `TAF_KND_CDAP_IT_05.pdf`. (If you also have the real SLIIT questionnaire template, put it there too.)
4. In Antigravity, open that folder as your workspace (Open Workspace / Open Folder).

**For every prompt**

1. Open the Agent Manager (the agent panel) for this workspace.
2. Set the mode to **Planning** (not Fast). Planning makes the agent write a plan first, which is what we want for big jobs.
3. Pick the strongest model offered to you (you can switch models per task in the settings).
4. Paste **one** prompt (starting with Prompt 0). Press send.
5. The agent shows a plan. Read it quickly. If it looks sensible, approve it. If it wants to do things from a *later* phase, tell it: "Only do this phase. Stop after the checks."
6. Wait. When it finishes, it will show a short report: what it did, the files, the numbers, and the checks. Open `docs/phase_reports/phase_NN.md` and read it.
7. Check the "ACCEPTANCE CHECKS" yourself (each prompt lists them). If one fails, paste this: *"Check X failed. Here is the error: … Fix it, re-run all checks of this phase, update the phase report, then stop."*
8. Only when the phase is clean, save your work (the agent already runs `git commit`) and paste the **next** prompt.

**Golden rules for you**

- Never paste two prompts at once. Never skip a prompt.
- Each prompt ends with "STOP". That is on purpose — the next instructions come in the next prompt.
- If the agent says it cannot install something (for example TensorFlow), let it switch to the fallback written in the prompt, and make sure it writes the change in `docs/DECISIONS.md`.
- If the agent's chat gets very long or confused, open a **new** agent chat in the same workspace and start with: *"Read AGENTS.md and PROJECT_BRIEF.md, then read the latest file in docs/phase_reports/. Then wait for my next prompt."* The rule book (created in Prompt 0) is how the agent remembers.
- Antigravity automatically reads `AGENTS.md` in the workspace root. Prompt 0 creates it.

**Where you do the real-data swap later:** Prompt 16 builds a "Train on New Data" page. When your real survey results arrive, you upload the file there, the system checks it, retrains every model, shows a new leaderboard, and you press "Use this model".

---

# PART C — Your overfitting screenshots (yes, I understand them)

What they show:

- **Split table:** train accuracy ≈ 0.91, validation ≈ 0.82, test ≈ 0.79. A model that scores much higher on training data than on unseen data has memorised some of the training data. That difference is the **generalisation gap**.
- **Loss curve:** the training loss keeps falling, but the validation loss flattens early and stays higher. After the point where validation stops improving, extra training only fits noise.
- **Accuracy curve:** training accuracy climbs towards ~0.95 while validation levels off around ~0.8.
- **Dashed vertical line ("start fine-tuning")** marks a change of training stage; the spike right after it is the learning rate jump.
- **Footer line:** "Best validation loss 0.4661 at epoch 23, val acc 0.8200." This is early stopping: keep the model from the best validation epoch, not the last epoch.

What we will build (research-paper style) for the engagement models:

1. Train / validation / test table for every model, with a "gap" column and a status tag (Good fit / Slight overfit / Overfit).
2. Loss and accuracy curves for every deep model, with a dashed line where training stage 2 starts, a marker for the best epoch, and a shaded "overfitting zone".
3. Boosting curves (error per tree) and learning curves (score versus training size) for the other models.
4. Cross-validation fold box plots (stability) and a "before vs after regularisation" figure for any model that overfit.
5. The same figures are shown in the website (without any viva/presentation material).

---

# PART D — THE PROMPTS

Paste each block (the text inside the box) into Antigravity, one at a time.

---

## PROMPT 0 — Set up the workspace, rule book and project brief

````text
ROLE: You are a senior research software engineer and ML lead. You are starting a new project from an empty workspace. Work carefully; correctness matters more than speed. Other phases will come in later prompts, so do NOT rush ahead.

GOAL OF THIS PHASE: Create the project skeleton, the environment, the permanent rule book (PROJECT_BRIEF.md + AGENTS.md), and the first documentation. No modelling yet.

STEP 1 — Read the reference material
Read everything in ./reference (a .docx proposal and a .pdf Topic Assessment Form; use pandoc / python-docx / pdfplumber). Understand: it is one of four components of a student attrition-risk project (Academic, Engagement, Psychosocial, Environmental & Financial). MY component is "Multimodal Student Engagement Intelligence". IMPORTANT CORRECTION: the proposal text still says the data come from Moodle/LMS/Zoom/mouse logs. That is OUTDATED. The truth is in PROJECT_BRIEF.md below: all data come from a closed-ended student questionnaire. Where the reference and the brief disagree, the brief wins.

STEP 2 — Create this folder structure (pathlib only, no absolute paths)
engagement-intelligence/
  AGENTS.md, PROJECT_BRIEF.md, README.md, pyproject.toml, requirements.txt, Makefile, .gitignore
  config/        project.yaml, schema.yaml (later), labels.yaml (later), model_zoo.yaml (later), mapping.yaml (later)
  data/          raw/ interim/ processed/ samples/ real/ templates/
  src/mei/       __init__.py, cli.py, config.py, io/, cleaning/, features/, states/, models/, evaluation/, explain/, registry/, api/, utils/
  artifacts/     runs/  (one sub-folder per run id)  and  active_run.txt
  frontend/      (empty for now)
  docs/          figures/  phase_reports/  DECISIONS.md
  tests/
  scripts/

STEP 3 — Environment
- Python 3.11 virtual environment in .venv. Install and PIN versions in requirements.txt: numpy, pandas, scipy, scikit-learn, statsmodels, xgboost, lightgbm, catboost, tensorflow (CPU) with keras, hmmlearn, shap, optuna, imbalanced-learn, matplotlib, seaborn, pyyaml, pydantic, joblib, openpyxl, xlsxwriter, rapidfuzz, typer, rich, fastapi, uvicorn[standard], python-multipart, pytest, pytest-cov, ruff, black, pandoc-free docx/pdf readers you used.
- If TensorFlow cannot be installed on this machine, install PyTorch (CPU) instead, and record that in docs/DECISIONS.md. Everything else must still work.
- Assume the target laptop is Windows, 8 GB RAM, NO GPU. Every heavy step must be time-boxed and configurable.
- Build a CLI: `python -m mei.cli <command>` (use typer). Create stub commands now: setup, make-sample-data, clean, features, states, train-classic, train-deep, train-regression, finalize, explain, serve, retrain, docs, all. Each stub prints "not implemented yet" for now. Also write the same commands as Makefile targets, and a `scripts/tasks.ps1` for Windows users without make.
- Configure ruff + black + pytest. Add one smoke test. All must pass.

STEP 4 — Write PROJECT_BRIEF.md exactly as the text between the lines BRIEF START and BRIEF END below (do not shorten it). Then write AGENTS.md (max 25 lines): "Read PROJECT_BRIEF.md before every task. Obey Global Rules G1–G13. Work on ONE phase at a time and stop when it is finished. Write docs/phase_reports/phase_NN.md at the end of each phase."

----- BRIEF START -----
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
- Pages: Overview, Model Leaderboard, Engagement Patterns, What Drives Risk, Score Students (upload CSV/Excel), Train on New Data.
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
----- BRIEF END -----

STEP 5 — Documentation
Write docs/00_environment.md (versions, how to set up, how to run the CLI) and docs/01_project_context.md (the 4-component project, my component, what changed from the proposal to the survey-based design, a one-paragraph plain explanation). Start docs/DECISIONS.md.

STEP 6 — Git
git init, .gitignore (ignore .venv, data/real/*, artifacts/runs/* except manifest files, __pycache__, node_modules), commit "phase 0: skeleton and rules".

ACCEPTANCE CHECKS
- `python -m mei.cli --help` lists all commands. `pytest` passes. `ruff check` and `black --check` pass.
- PROJECT_BRIEF.md contains all 10 sections unchanged. AGENTS.md exists.
- docs/00_environment.md, docs/01_project_context.md, docs/DECISIONS.md, docs/phase_reports/phase_00.md exist.

FINISH: Reply in the G13 format and STOP. Do not start Phase 1.
````

---

## PROMPT 1 — Questionnaire, data dictionary and label rules

````text
ROLE: Senior survey-methodology + data engineer.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md (sections 2–4), docs/DECISIONS.md, docs/phase_reports/phase_00.md.

GOAL: Turn the brief's questionnaire into exact, machine-readable definitions, so that sample data, real data, cleaning, the website upload and the template all use ONE source of truth.

STEPS
1. Create config/schema.yaml listing every canonical column: name, type, allowed codes, text labels for each code (the exact answer wording a student sees), role (id | consent | attention_check | feature | outcome | meta), and period (1–4 or null). Canonical feature names: `<indicator>_p<k>` for the 7 indicators and k=1..4 (28 columns), e.g. `resource_access_p1`. Outcome columns: `intent_leave`, `intent_defer`, `intent_not_continue`. Meta: `student_id`, `timestamp`, `consent`, `attention_check`.
2. Write the exact option wording for every grid (the 5 options per indicator from the brief). Keep the wording simple, non-judgemental, and in English only.
3. Create config/mapping.yaml: for each canonical column, a list of accepted header variants (Google Forms style long headers such as "How many days per week did you open online learning materials? [Period 1: Weeks 1–4]"), so that a real Google Forms export maps automatically. Add a fuzzy-matching fallback design using rapidfuzz (threshold 88) that is used only to SUGGEST mappings, never to silently accept them.
4. Create config/labels.yaml: outcome items, score formula, at_risk threshold (60), MEI bands (0–34, 35–64, 65–100).
5. Implement src/mei/io/schema.py (pydantic models that load and validate the YAML) and src/mei/io/codec.py (text label ⇄ code conversion in both directions, tolerant of case, extra spaces, and en-dash vs hyphen).
6. Write docs/questionnaire_engagement.md: the final questionnaire ready to copy into Google Forms: title, consent text (voluntary, anonymous, closed-ended, used for research, can stop any time), instructions, definition of the 4 periods (with example week ranges for a 14-week semester), the 7 grid questions with row/column labels, the 3 outcome statements, the attention-check row, estimated completion time (target ≤ 6 minutes), and a short paragraph per indicator: what it measures, why it matters for attrition, and which published engagement construct it relates to (state honestly if you are unsure; do NOT invent citations — write "citation to be added by author").
7. Write docs/02_data_design.md (the data dictionary in a table, the label rule with a worked example, and the limitations: self-report, recall bias, label is a proxy of real attrition).
8. Write data/templates/survey_template.csv and survey_template.xlsx (headers only + a second sheet "Data dictionary" in the xlsx).
9. Tests: schema loads; codec round-trips every label; mapping covers every canonical column; outcome columns are never marked role=feature.

ACCEPTANCE CHECKS
- pytest green; `python -m mei.cli setup` validates all three YAML files and prints a summary table (columns per role).
- The template opens in Excel/LibreOffice with correct headers (open it programmatically and re-read it).
- docs/questionnaire_engagement.md has all 7 grids + outcomes + attention check.

FINISH: G13 reply, write phase_01.md, git commit, STOP. Do not start sample-data generation.
````

---

## PROMPT 2 — Generated sample data that behaves like real survey data

````text
ROLE: Senior data scientist who builds realistic synthetic survey data for method validation.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, config/schema.yaml, config/labels.yaml.

GOAL: Write a generator that creates a Google-Forms-style raw export from a hidden latent model, with realistic mess, so every later step (cleaning, states, models) is tested honestly. The data must be HARD ENOUGH that results are believable (not 99 %).

DESIGN (implement exactly; all numbers in config/project.yaml under `sample_data:`)
- N = 2400 students (configurable). Seed 42.
- Archetype per student (probabilities): consistent_high 0.28, consistent_moderate 0.24, consistently_low 0.10, declining 0.17, sporadic 0.13, improving 0.08.
- Latent engagement e_t ∈ [0,1] for t=1..4:
  • consistent_*: level b ~ Beta by archetype (high ≈ 0.78, moderate ≈ 0.55, low ≈ 0.25) + AR(1) noise (σ 0.04).
  • declining: start 0.7–0.9, linear fall of 0.08–0.2 per period, random onset.
  • sporadic: base 0.5–0.7 with random alternating shocks of ±0.15–0.30.
  • improving: mirror of declining.
- Student-level diligence random effect (σ 0.08) shared by all indicators; indicator-specific noise (σ 0.10). Indicator loadings: resource_access 0.90, study_time 0.85, review_time 0.80, quiz_practice 0.75, assignment_engagement 0.80, study_consistency 0.85, participation 0.70. Convert to the 5 ordinal codes with fixed cut-points so each indicator has a sensible marginal distribution (no indicator should have >45 % in one code).
- Response styles: 10 % acquiescent (+0.5 level on agreement-type items), 5 % extreme responders, 5 % midpoint responders, 2 % straight-liners (same answer everywhere), 2 % fail the attention check.
- Missingness: 2 % random missing items; plus 3 % of students miss a whole period, with higher probability for low engagement (missing-at-random).
- Mess: 0.5 % duplicated submissions (same student_id, later timestamp), 0.3 % impossible values or typos in labels (e.g. "Alwys", extra spaces, en-dash/hyphen differences), 1 % without consent = "No" (must be removed in cleaning).
- Outcome: latent risk r = sigmoid(w0 + w1·(1 − mean e) + w2·max(0, e1 − e4) + w3·std(e) + w4·(1 − e4) + ε), ε ~ N(0, σ_ε) per student. Generate A1..A3 as 5-point answers around r with item noise. Compute attrition_intent_score and at_risk with labels.yaml. Choose w0 so that at-risk prevalence ≈ 28 % (accept 24–32 %).
- `signal_strength` parameter scales w1..w4. CALIBRATION LOOP: after generating, fit a quick Logistic Regression and a LightGBM with 5-fold CV on the 28 raw features; adjust signal_strength (max 8 iterations) until LightGBM CV AUC-ROC lies in 0.86–0.92 and LR AUC is at least 0.03 lower than the boosted model OR within 0.03 if the structure turns out linear (record which). Save the final value in config and the calibration log in docs/03_sample_data.md.
- Hidden ground truth file data/raw/ground_truth_hidden.csv (student_id, archetype, e_1..e_4, latent_risk). It is used ONLY to evaluate how well the HMM/clustering recover the true archetypes (in Phase 5) — never as a feature.
- Output the RAW export exactly like Google Forms: data/raw/survey_raw_sample.csv and .xlsx with long question headers (taken from config/mapping.yaml), a "Timestamp" column, text answers (not codes), and the grid answers as separate columns per period. Also write data/samples/sample_upload_clean.xlsx (200 students, clean, canonical headers, NO outcome columns — used for testing the website's scoring page) and data/samples/sample_upload_messy.csv (200 students, renamed headers, some blanks, some out-of-range values, one extra unknown column — used for testing column mapping and validation).

EXPLORATORY CHECKS (write code + figures, 300 dpi, brand palette)
- Marginal distribution of every item, correlation heat map of the 7 indicators, mean engagement per period by archetype, at-risk rate by archetype, missingness map, duplicate/typo counts.
- A "reality check" table: prevalence, missing rate, response-style rates, attention-check failures — compared with the configured values.

DOCS: docs/03_sample_data.md (how the generator works, every parameter, why it is realistic, what it can NOT prove, how real data will differ, calibration log) + a data card docs/data_card_sample.md.

TESTS: same seed → identical file; N and prevalence in range; no outcome column leaks into the sample upload files; ground truth file not in any feature list.

ACCEPTANCE CHECKS
- `python -m mei.cli make-sample-data` regenerates everything in under 2 minutes.
- Calibration result printed: LR AUC, LightGBM AUC, prevalence. All in the target ranges.
- 6 EDA figures exist in artifacts and are indexed in docs/figures/INDEX.md.

FINISH: G13 reply, phase_02.md, git commit, STOP. Do not start cleaning.
````

---

## PROMPT 3 — Cleaning and validation module (works on any survey file)

````text
ROLE: Senior data-quality engineer.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, config/*.yaml, docs/phase_reports/phase_02.md.

GOAL: One reusable cleaning module used by the CLI, the retraining job and the website upload. It must work for the generated file today and a real Google Forms file tomorrow, and it must produce a clear, honest "data quality report".

BUILD src/mei/cleaning/ with these stages, in this order. Each stage returns the data AND a log entry (rule, rows affected, columns affected, action):
1. Read CSV/XLSX/XLS (encoding detection, header row detection, drop fully empty rows/columns, strip whitespace, normalise dashes/quotes).
2. Header mapping using config/mapping.yaml (exact, then normalised match; fuzzy suggestions with confidence returned to the caller for the UI; never auto-accept below the threshold). Report missing required columns and unknown extra columns.
3. Consent filter: drop consent != Yes. Attention-check filter: drop students who failed it (configurable on/off).
4. Convert text labels to codes via the codec; tolerate typos using fuzzy match with a 90 threshold; values that cannot be converted become missing and are logged.
5. Range and type validation: codes must be 1–5; out-of-range → missing.
6. Duplicates: keep the LAST valid submission per student_id; log the removed ones.
7. Careless responding: flag straight-liners (zero variance across all 28 answers, or across ≥ 6 of 7 indicators for all periods), and extreme speed if timestamps allow (not available → skip). Default action: flag + drop, configurable.
8. Missing data policy (configurable): rows with > 25 % missing feature cells are dropped; remaining missing values are NOT imputed here (imputation belongs inside the model pipeline to avoid leakage — see G3). Produce a missingness summary per column and per period.
9. Outcome handling: if outcome columns exist, compute attrition_intent_score and at_risk (labels.yaml); rows with a missing outcome are kept only for scoring, excluded from training.
10. Output a tidy canonical dataframe saved to data/interim/clean_<run_id>.parquet, and a JSON data quality report with: rows in/out, per-rule counts, missingness, class balance (if labelled), distribution of each item, warnings (e.g. "sample size below 300", "at-risk class below 15 %"), and a traffic-light status (green/amber/red) with plain-language messages.

MODES: `training` (outcomes required) and `scoring` (outcomes optional, nothing dropped except unusable rows — rows that cannot be scored are returned with a reason, not silently removed).

DOCS: docs/04_cleaning.md with a flow diagram (Mermaid), every rule with its reason, and a table of what happened to the sample data (rows removed per rule, read from the report). Include "common mistakes" such as cleaning after splitting, imputing before splitting, dropping instead of flagging.

TESTS: one test per rule using tiny hand-made frames; end-to-end test on the raw sample file asserts the removed counts match the injected mess within tolerance (duplicates ≈ 0.5 %, non-consent ≈ 1 %, attention-check fails ≈ 2 %, straight-liners ≈ 2 %); messy upload file produces suggestions and warnings but no crash.

ACCEPTANCE CHECKS
- `python -m mei.cli clean --input data/raw/survey_raw_sample.csv --mode training` writes the parquet and the JSON report; status green.
- `python -m mei.cli clean --input data/samples/sample_upload_messy.csv --mode scoring` returns mapping suggestions, warnings and per-row reasons.
- pytest green; coverage of src/mei/cleaning ≥ 85 %.

FINISH: G13 reply, phase_03.md, git commit, STOP. Do not start feature engineering.
````

---

## PROMPT 4 — Preprocessing, feature engineering and leakage-proof splits

````text
ROLE: Senior ML engineer specialised in tabular and short-sequence data.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, docs/phase_reports/phase_03.md.

GOAL: Build the feature pipeline and the evaluation splits. Everything learned from data lives inside sklearn-compatible transformers so cross-validation is honest.

SPLITS (src/mei/features/splits.py)
- Outer split: stratified by at_risk (and by archetype-proxy if available — NOT in real data), 70 % train / 15 % validation / 15 % test, by student_id (one row per student already). Save index files to artifacts/runs/<run_id>/splits/. The test indices are written once and never loaded except by the Phase 9 function `evaluate_final_on_test()` (enforce with a guard + a test).
- CV scheme for the leaderboard: repeated stratified 5-fold × 3 repeats on train+validation (15 folds). If n < 500, switch automatically to 5 × 5 and log a warning. Provide a fixed list of fold indices saved to disk so all models see identical folds.

TRANSFORMERS (each fit on training folds only)
1. `PeriodReshaper`: 28 columns ⇄ array (n, 4, 7) for sequence models.
2. `Imputer`: median per indicator-period for ordinal codes, plus missing-indicator columns for any column with > 3 % missing; also an option "interpolate within student across periods" evaluated as an alternative (choose by CV on a quick LR; record the result in DECISIONS.md).
3. `EngagementFeatures` — for every indicator: mean, std, min, max, OLS slope across the 4 periods, last − first, last − mean(first two), count of direction reversals. Composite per period E_t = mean of the 7 indicators rescaled to 0–100 and its own features: mean, std, slope, max single-period drop, number of reversals, area under the curve, last-period value, drop from best to last.
4. `Scaler`: StandardScaler for linear/NN/SVM/KNN; none for trees (registry decides).
5. Feature sets (named, saved in config/model_zoo.yaml later): `raw28` (the 28 coded answers), `agg` (raw28 + aggregated features), `agg_states` (agg + state features from Phase 5), `single_signal` (resource_access_p1..p4 only; the Moodle-login imitation baseline), `sequence` (the (4,7) tensor).
6. Constant/near-constant column removal and multicollinearity report (VIF) for linear models only (report, don't drop trees' inputs).
7. Class imbalance: compute class weights; implement optional SMOTE only inside the training fold (imblearn Pipeline). Default = class weights; SMOTE is compared later in Phase 6.

DIAGNOSTICS (figures + tables, brand palette)
- Distribution of every engineered feature by at_risk, effect sizes (Cohen's d / Cramér's V), Spearman correlation heat map, mutual information ranking, and a leakage audit: the audit must assert that no feature column has |corr| > 0.95 with the label and that outcome columns are absent.
- Fit a quick LR/LightGBM on `raw28` vs `agg` using the CV scheme and show whether feature engineering helps (with fold-wise confidence intervals).

DOCS: docs/05_preprocessing_features.md — each feature explained in plain words with a formula and an example student, why splits are done first, what leakage is and exactly how we prevent it (with a diagram), the CV design, the imbalance policy. Include a table "Feature → meaning → why it signals risk".

TESTS: transformer fit/transform shape tests; leakage tests (fit on train, transform on test never refits; shuffle-label sanity test: AUC ≈ 0.5 ± 0.05 when labels are shuffled); the guard that blocks reading test indices.

ACCEPTANCE CHECKS
- `python -m mei.cli features` creates splits + feature tables for the run and prints the shapes per feature set.
- Shuffled-label test passes. Leakage audit passes. All figures indexed.

FINISH: G13 reply, phase_04.md, git commit, STOP. Do not fit HMM/clustering yet.
````

---

## PROMPT 5 — Engagement states: clustering and Hidden Markov Model

````text
ROLE: Senior researcher in sequence modelling and unsupervised learning.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md (sections 5–6), docs/DECISIONS.md, docs/phase_reports/phase_04.md.

GOAL: Discover engagement states and patterns from the 4-period answers, turn them into features, and prove with an ablation whether they add value. Everything is fitted inside training folds (G3).

A. CLUSTERING (src/mei/states/clustering.py)
- Student-level feature matrix: per-period composite E_1..E_4 + slope + volatility + level (mean). Standardise.
- Algorithms: KMeans, Gaussian Mixture (full/diag), Agglomerative (Ward). Optionally tslearn TimeSeriesKMeans (DTW) if installable (skip silently if not).
- k from 2 to 6. Metrics: silhouette, Calinski–Harabasz, Davies–Bouldin, BIC (GMM), and BOOTSTRAP STABILITY (adjusted Rand index between clusterings of 30 bootstrap resamples). Choose with a documented rule: highest stability among candidates whose silhouette ≥ 0.15, ties → smaller k. Name clusters by their profile (level + shape).

B. HIDDEN MARKOV MODEL (src/mei/states/hmm.py)
- hmmlearn GaussianHMM, covariance_type diag, over sequences of length 4 with the 7 standardised indicators per period (missing imputed by the pipeline). Sequence lengths array handled correctly.
- K ∈ {2,3,4}; 10 random restarts each, keep best log-likelihood; select K by BIC on training data AND log-likelihood on validation folds; prefer the simpler K if BIC difference < 10. Check convergence and report it.
- Order states by their mean composite engagement (State 1 = lowest). Name them Low / Medium / High (for K=3). Report: start probabilities, transition matrix, emission means per indicator, expected dwell time.
- Decode the most likely state path per student (Viterbi) and posterior probabilities (forward–backward).
- Honest warning to document: sequences are only 4 time-points long, so the HMM is weakly identified; hence we also compare with clustering and run the ablation below.

C. PATTERN LABELS (src/mei/states/patterns.py) — deterministic rules on the decoded path, written in config:
- Consistent: the state does not change (or changes by at most one level once without returning).
- Declining: net drop of ≥ 1 state level from the first to the last period with no recovery in the last period.
- Improving: net rise of ≥ 1 level.
- Sporadic: ≥ 2 direction changes (up-down or down-up).
- Overall LEVEL = High/Medium/Low from the mean state.
Return both PATTERN and LEVEL.

D. STATE FEATURES for models: decoded state at each period (ordinal), time spent in each state, number of transitions, final state, posterior probabilities of the last period, sequence log-likelihood, cluster id (one-hot) and distance to own cluster centre, pattern (one-hot), level. Implement as `StateFeatureTransformer` (fit on training fold only; refits clustering + HMM per fold; stores fitted parameters).

E. EVALUATION OF STATES
- Because the data are generated, compute how well states recover the hidden archetypes (adjusted Rand index, normalised mutual information, confusion matrix archetype × pattern). Label this clearly as "valid only for generated data".
- Predictive value: ablation with a fixed LightGBM and LR on 3 feature sets (`agg`, `agg_states`, `states_only`) using the Phase 4 CV; report mean ± std for F1, AUC, MCC and a paired Wilcoxon test across folds for agg vs agg_states. Report the result honestly even if states add little; if they add little, say so in the phase report and in docs, and propose what to try (do not tune until it "works").
- Figures: state emission profiles (heat map), transition matrix heat map, mean composite trajectory by pattern, pattern distribution, at-risk rate per pattern with 95 % CI, cluster silhouette plot, K-selection curves (BIC, validation log-likelihood).

DOCS: docs/06_engagement_states.md — what a state is, what an HMM is in plain words with a weather/mood example, how K was chosen, how patterns are defined (rule table), clustering vs HMM comparison, ablation result, limitations (short sequences, self-report), viva-style explanation.

TESTS: HMM on a toy sequence with known transitions recovers them; state ordering is deterministic; pattern rules on hand-made paths; fold-wise fitting does not touch validation rows.

ACCEPTANCE CHECKS
- `python -m mei.cli states` saves fitted state models, the state-feature table, all figures, and artifacts/runs/<run_id>/states/summary.json (chosen K, chosen clustering, ARI, ablation numbers).
- Ablation table is present with confidence intervals and the Wilcoxon p-values.

FINISH: G13 reply, phase_05.md, git commit, STOP. Do not train the leaderboard models yet.
````

---

## PROMPT 6 — Leaderboard engine and all non-deep classification models (Track A, part 1)

````text
ROLE: Senior ML engineer and benchmarking specialist.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md (sections 6–7), docs/DECISIONS.md, phase reports 04 and 05.

GOAL: Build the benchmarking engine (used for every model, including deep ones and later the retraining job) and train every linear, classical/shallow, boosting and ensemble classifier with fair tuning, cross-validation and full metrics. Deep models come in the next prompt.

ENGINE (src/mei/models/engine.py, src/mei/evaluation/metrics.py)
- Model registry in config/model_zoo.yaml: id, display name, plain name, category, task (classification|regression), feature set, scaler need, class-weight support, search space, time budget.
- `evaluate_model(spec)` does, for the identical 15 CV folds: build Pipeline(imputer → [StateFeatureTransformer if feature set uses states] → scaler → estimator) → tune → fit → collect OOF predictions and per-fold metrics for BOTH the training part and the validation part of each fold → store everything.
- Tuning: Optuna (TPE, seed 42), nested properly: the inner objective uses 3-fold CV INSIDE the training part of each outer fold only for the top-tier models; for time reasons use this protocol: (a) tune hyperparameters once on the train split with 5-fold CV (n_trials from config; default 30 for boosting, 20 for others, 10 for slow ones), (b) then run the 15-fold outer evaluation with the frozen hyperparameters. Document the small optimism this introduces and why it is acceptable (the test set remains untouched).
- Time budget per model from config; stop the study when exceeded and log it.
- Metrics (every one in the brief, §7) at (i) threshold 0.5 and (ii) threshold tuned on the training folds' OOF predictions (maximise F1 of at-risk). Also ROC/PR curve points, calibration curve points, confusion matrices (counts + normalised), training time, inference time per 1000 rows, serialized model size, and the fitted pipeline saved with joblib.
- Leaderboard Score, gap, status (Good fit / Slight overfit / Overfit) exactly as in the brief. Save leaderboard rows as JSON + CSV, per-model detail JSON (everything the website needs: curves, fold metrics, hyperparameters, confusion matrix, roc, pr, calibration), and a models index.
- Learning-curve data for each model: score vs training-set size (10 sizes, 5-fold) and a validation curve for ONE key hyperparameter (e.g. max_depth, C, k, n_estimators). For boosting models also store the per-iteration train/validation loss (XGBoost eval_set, LightGBM record_evaluation, CatBoost eval metrics, sklearn staged_predict) so we can plot loss vs trees.

MODELS TO TRAIN (feature sets in brackets; use `agg_states` for the "Proposed/Hybrid" family and `agg` for the others, EXCEPT the stated ones)
- Baseline: single-signal Logistic Regression [single_signal]; majority-class dummy; simple rule "mean resource_access below 2 ⇒ at risk" (imitating the login indicator).
- Linear: Logistic Regression (L2), Logistic Regression (Elastic-Net), Linear SVM (calibrated), Ridge classifier (calibrated), LDA.
- Classical / shallow: k-Nearest Neighbours, Gaussian Naive Bayes, Decision Tree, SVM (RBF), single hidden-layer MLP from sklearn (shallow net), Extra-Trees.
- Boosting / tree ensembles: Random Forest, Gradient Boosting (sklearn), HistGradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost.
- Hybrid (proposed): XGBoost [agg_states], LightGBM [agg_states], CatBoost [agg_states], Random Forest [agg_states], Logistic Regression [agg_states]. Each hybrid has a twin on `agg` so that the value of states is visible in the leaderboard.
- Ensembles: Soft-voting (top 3 by CV from the groups above), Stacking (LR meta-learner on out-of-fold probabilities of the top 5; stacking must use proper OOF to avoid leakage).
- Also run, for the best 2 boosting models only, a SMOTE-in-fold variant versus class weights and report which is better.

OUTPUT per run: artifacts/runs/<run_id>/models/<model_id>/{pipeline.joblib, detail.json, curves.json}, artifacts/runs/<run_id>/leaderboard_classification.json/.csv, `per_category_best.json`.

FIGURES (static, for docs): leaderboard bar chart, train-vs-validation gap chart, metric heat map (models × metrics), ROC/PR overlays of the best model per category, calibration plots of top models, boosting loss-vs-iteration curves for XGBoost/LightGBM/CatBoost, learning curves, validation curves.

DOCS: docs/07_model_zoo.md (each model in plain words: what it is, strengths, weaknesses, when it overfits, key hyperparameters with plain meaning) and docs/08_training_validation.md (tuning protocol, CV design, metrics definitions with formulas, how the leaderboard score is computed, tie-breaking, time budgets). Include a short "Explain like I am 5" for each model family.

TESTS: engine reproduces identical scores when run twice with the same seed; folds identical across models; stacking uses OOF only (test with a leakage canary: a feature equal to the label must be caught by the audit); metrics match sklearn on a toy case.

ACCEPTANCE CHECKS
- `python -m mei.cli train-classic --budget-minutes 40` completes within the budget on the 8 GB laptop assumption (state the real time).
- Leaderboard has ≥ 25 classification rows with every metric present and zero NaN in the primary columns. Best-per-category file exists.
- Report in phase_06.md: top 5 models, their gaps and statuses, and honest comments if any result looks too good (> 0.95 AUC) or too poor (< 0.70).

FINISH: G13 reply, phase_06.md, git commit, STOP. Do not train deep models yet.
````

---

## PROMPT 7 — Deep learning models with proper learning curves (Track A, part 2)

````text
ROLE: Senior deep-learning engineer for small tabular and short-sequence data.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, phase_06 report, src/mei/models/engine.py.

GOAL: Add the deep-learning category to the classification leaderboard using the SAME engine, folds and metrics, and generate research-grade training curves that make over-training visible (like a paper's loss/accuracy figures with a "start of stage 2" marker, best-epoch marker and a train/validation/test table).

MODELS (Keras/TensorFlow CPU; PyTorch fallback only if TF is unavailable)
1. MLP-Wide: Dense(128)-BN-ReLU-Dropout → Dense(64)-… → Dense(32)-… → sigmoid [agg or agg_states].
2. MLP-Residual: 2 residual blocks with LayerNorm [agg_states].
3. GRU: input (4,7) [sequence]; GRU(32) → Dropout → Dense(16) → sigmoid.
4. LSTM: same with LSTM(32).
5. Bi-GRU or Bi-LSTM (choose one, justify).
6. 1D-CNN: Conv1D(32,k=2)-ReLU → Conv1D(32,k=2) → GlobalMaxPool → Dense.
7. Tiny Transformer encoder: 1–2 blocks, d_model 32, 2 heads, positional encoding for 4 steps, pooling → Dense.
8. Hybrid sequence + tabular: sequence branch (GRU) + tabular branch (state features + aggregates) concatenated → Dense. This is the "multimodal" deep model.
All models: L2 1e-4, Gaussian input noise (σ 0.05, training only), dropout, class weights, binary cross-entropy, metrics = loss, accuracy, AUC, precision, recall logged per epoch.

TRAINING PROTOCOL (the same for all)
- Stage 1: Adam lr 1e-3, max 150 epochs, batch 64, EarlyStopping(monitor val_loss, patience 15, restore_best_weights), ReduceLROnPlateau(factor 0.5, patience 6).
- Stage 2 ("refine"): start from the best stage-1 weights, lr 1e-4, max 60 epochs, patience 10. Keep stage 2 ONLY if the best validation loss improves; otherwise record "stage 2 not used". Record the epoch where stage 2 starts so the figure can draw a dashed vertical line, exactly like a "start fine-tuning" marker.
- Hyperparameter search: Optuna, ≤ 15 trials per model (units, dropout, L2, lr, batch size), 3-fold inner CV on the training split, each trial capped at 40 epochs; total budget per model from config.
- Final evaluation: the identical 15 outer folds with the frozen hyperparameters. For each fold keep the history. For the best 3 deep models also train 5 seeds and report seed standard deviation.
- Curves to store: per-epoch train/validation loss and accuracy and AUC (mean and ±1 std across folds, plus one representative fold), best epoch, stage-2 start epoch, epochs run, "overfitting zone" = epochs after the best epoch where validation loss exceeds its minimum by > 3 % while training loss keeps falling.
- Overfit status uses the same gap rule as every other model.

FIGURES (per deep model, saved as PNG+SVG in the brand palette)
- Twin panel "Loss" and "Accuracy" with train (solid, forest green), validation (amber), dashed vertical line at stage-2 start, dotted vertical line at best epoch, shaded overfitting zone, and a caption line: "Best validation loss X at epoch N (stage), val acc Y". Style them like the reference screenshots but cleaner.
- Train / validation / test-ready table (the test column stays blank until Phase 9) with accuracy, balanced accuracy, F1, AUC, loss and the gap.
- Mean ± std learning curves across folds; weight-norm vs epoch (optional); calibration plot.
- A comparison figure of all deep models' validation loss curves in one chart.

OVERFIT DEMONSTRATION (important for the report): for the strongest deep model, ALSO train a deliberately over-parameterised version (no dropout, no L2, no early stopping, 300 epochs, 512-256-128 units) and plot its curves next to the regularised version ("before vs after regularisation"). This single figure explains the whole overfitting story in a viva. Add it to the leaderboard detail data but NOT as a leaderboard row.

DOCS: docs/09_deep_learning_and_overfitting.md — what each architecture is in plain words, why short sequences favour small models, what overfitting is, how to read the curves, how early stopping/dropout/L2/noise work, what stage 2 does, the before/after figure with explanation, "Explain like I am 5".

TESTS: models build and overfit a tiny toy set; early stopping restores best weights; history files have the expected keys; same seed gives same first-epoch loss within tolerance.

ACCEPTANCE CHECKS
- `python -m mei.cli train-deep --budget-minutes 25` finishes in budget; leaderboard now includes ≥ 8 deep rows; curve JSON exists for each.
- All figures exist and are listed in docs/figures/INDEX.md. phase_07.md states honestly which deep models overfit, which did not, and how they rank against boosting models.

FINISH: G13 reply, phase_07.md, git commit, STOP. Do not start regression.
````

---

## PROMPT 8 — Regression leaderboard with R² and the full regression metric set (Track B)

````text
ROLE: Senior applied statistician and ML engineer.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, phase reports 06 and 07.

GOAL: Build Track B: predict `attrition_intent_score` (0–100) with the same folds, features, engine and honesty rules, and report R² and the other regression metrics. This supports Track A and gives staff a continuous score.

MODELS (same categories; use `agg` for general models and `agg_states` for hybrids)
- Baseline: mean predictor; single-signal Linear Regression [single_signal].
- Linear: Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge, Huber regressor.
- Classical/shallow: KNN regressor, SVR (RBF), Decision Tree, shallow sklearn MLPRegressor, Extra-Trees.
- Boosting/ensembles: Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost.
- Deep (reuse Phase 7 architectures with linear output and Huber/MSE loss, same two-stage protocol and curve outputs): MLP, Residual MLP, GRU, LSTM, 1D-CNN, Transformer, Hybrid sequence+tabular.
- Hybrid (proposed): XGBoost/LightGBM/CatBoost [agg_states] and their `agg` twins.
- Ensembles: Voting (top 3), Stacking (Ridge meta-learner, OOF only).

METRICS (every one, per fold, train and validation parts): R², adjusted R², RMSE, MAE, median AE, explained variance, max error, Pearson r, Spearman ρ, Kendall τ (top models only), share of predictions within ±5 and ±10 points, residual mean (bias), residual std. Also report the Phase 6 classification metrics obtained by thresholding the predicted score at 60 (so Track B can be compared with Track A on F1/AUC). Clip predictions to 0–100 and say whether clipping was needed.
- Residual diagnostics for the top models: residual vs fitted, QQ plot, residuals by pattern and by level, heteroscedasticity check (Breusch–Pagan), calibration of predicted vs actual (binned), learning curves (R² vs training size), validation curve for one hyperparameter, per-iteration boosting loss, deep-model epoch curves (same figure style as Phase 7).
- Leaderboard Score (Track B) = mean(R², Spearman) − 2 × max(0, R²_train − R²_cv − 0.05). Status labels as in the brief using the R² gap.
- Statistical comparison: for the top 5 models, paired Wilcoxon tests on fold-wise R² with Holm correction; report which differences are NOT significant, so the leaderboard is not over-interpreted.

CONSISTENCY WITH TRACK A: compute Spearman between the Track A champion's probability and the Track B champion's predicted score on the validation folds; report it. Do NOT merge the two tracks into one index; the MEI comes from Track A (brief §4).

OUTPUT: leaderboard_regression.json/.csv, per-model detail JSON in the same shape as Track A (plus residual data), per_category_best_regression.json.

DOCS: docs/10_regression_track.md — what R² means in plain words (with an example: R² = 0.62 ⇒ the model explains 62 % of the ups and downs in intent scores), why adjusted R² exists, RMSE vs MAE, why we also look at Spearman, why Track B is supporting evidence, interpretation of the residual plots, "Explain like I am 5".

TESTS: metrics match sklearn/scipy on toy data; adjusted R² formula test; no NaN in primary columns; thresholding at 60 reproduces labels exactly on a toy vector.

ACCEPTANCE CHECKS
- `python -m mei.cli train-regression --budget-minutes 30` completes in budget; ≥ 28 rows; all metrics present.
- phase_08.md lists the top 5 with R², RMSE, MAE, gap and status, and states clearly whether differences between the top models are statistically significant.

FINISH: G13 reply, phase_08.md, git commit, STOP. Do not do the final test evaluation yet.
````

---

## PROMPT 9 — Overfitting audit, remediation, one-time test evaluation, champion, calibration and the MEI

````text
ROLE: Senior ML engineer and evaluation auditor. Be strict; this phase decides the final model.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md (sections 4, 6, 7), docs/DECISIONS.md, phase reports 06–08.

GOAL: Audit every leaderboard row, fix overfitting where possible, evaluate the finalists on the untouched test set exactly once, choose the champion by the brief's rule, calibrate it, define the MEI and freeze everything for the website.

STEP 1 — Overfitting audit (both tracks)
For every model compute: gap (train − CV validation) on F1 (Track A) or R² (Track B), CV std, seed std (where available), epochs-after-best for deep models, and status. Produce `overfit_audit.json` and a figure "train vs validation vs (later) test per model" sorted by gap.

STEP 2 — Remediation ladder (automatic, logged, bounded)
For each model with status "Overfit" or "Slight overfit" among the TOP 12 of its track, try in order, stopping when status becomes "Good fit" or the ladder ends: (1) stronger regularisation (higher L2/min_child_weight/min_samples_leaf, lower max_depth, higher learning-rate decay), (2) reduce capacity (fewer units/trees/leaves), (3) stronger dropout/noise or subsampling (row and column), (4) feature selection by in-fold SHAP/permutation importance keeping the top features covering 90 % importance, (5) early stopping with a smaller patience. Every attempt is re-evaluated on the same 15 folds and stored as a new leaderboard row marked "remediated (step n)"; the original row stays visible. Save the "before vs after" figure for each remediated model.

STEP 3 — Finalists and the one-time test evaluation
Finalists = best model of each category in Track A + overall top 5 (union, deduplicated) and the same for Track B. Implement `evaluate_final_on_test()` (the ONLY code path allowed to read the test indices): refit each finalist on train+validation with frozen hyperparameters, predict the test set once, compute all metrics, and store them in detail.json under `test`. After this call, write a lock file `artifacts/runs/<run_id>/TEST_USED.lock` containing the time and finalist ids, and make the function refuse to run again for the same run id (a new run id is needed to evaluate again). Never tune on test results; never reorder the leaderboard because of test results EXCEPT through the pre-declared selection rule below.
- Bootstrap 95 % confidence intervals (2000 resamples) on test F1, AUC, MCC (Track A) and R², RMSE (Track B) for the champion and runner-up.
- Test-vs-CV agreement check: flag any finalist whose test metric differs from its CV estimate by more than 2 standard errors and explain.

STEP 4 — Champion selection (pre-declared rule, write it in DECISIONS.md BEFORE looking at test numbers)
1. Candidates: Track A rows with status "Good fit" or remediated to "Good fit"/"Slight overfit".
2. Highest CV Leaderboard Score wins; if the top two differ by less than one pooled CV standard error (or the Wilcoxon test is not significant), prefer the simpler/faster/more explainable model in this order: boosting-family hybrid > linear > other classical > deep > stacked ensembles.
3. Record the champion, the runner-up, and a plain-language justification paragraph. Also record the best model of every category (this feeds the website's "best in each category" cards) for both tracks.

STEP 5 — Calibration, threshold and the MEI
- Calibrate the champion's probabilities with isotonic (n ≥ 1000) or Platt/sigmoid (smaller n) using out-of-fold predictions on train+validation only; report ECE and Brier score before and after, and the reliability diagram.
- Tune the decision threshold on OOF predictions (maximise F1 of at-risk; also report the threshold that gives recall ≥ 0.80 as an alternative "sensitive" mode).
- MEI = round(100 × calibrated probability). Bands from labels.yaml. Verify that the observed at-risk rate rises across the bands on the test set (monotonic) and report the table (band, n students, observed at-risk %, mean MEI).
- Save the final scoring bundle: `artifacts/runs/<run_id>/champion/` containing the fitted pipeline (including state models and calibrator), feature list, schema version, threshold, bands, SHAP background sample (200 rows, anonymised), model card JSON (name, category, data source, n, date, metrics with CI, limitations). Provide `src/mei/registry/score.py::score_dataframe(df)` that takes CLEAN canonical data and returns MEI, band, flag, pattern, level, state path, and top-3 reasons placeholder (filled in Phase 10).
- Set artifacts/active_run.txt to this run id.

DOCS: docs/11_final_evaluation_and_champion.md — the audit table, the remediation story with figures, why the test set is used once and how we enforced it, the selection rule and result, calibration and what the MEI means with a worked example (a student with a 0.71 calibrated probability gets MEI 71, band High), confidence intervals in plain words, honest limitations. Add docs/12_model_card.md.

TESTS: calling the test evaluation twice raises; test indices cannot be read elsewhere (static check via a test that greps the code); calibration monotonicity; score_dataframe on the sample upload returns values in 0–100 with the expected columns.

ACCEPTANCE CHECKS
- `python -m mei.cli finalize` creates the champion bundle, the lock file, the model card and the full leaderboards with test columns filled for finalists only (other rows show "not evaluated on test", which is intentional).
- `python -m mei.cli score --input data/samples/sample_upload_clean.xlsx` prints a table with MEI and bands.
- phase_09.md states the champion, runner-up, the test metrics with CIs, the band table, and every honest caveat.

FINISH: G13 reply, phase_09.md, git commit, STOP. Do not start SHAP/explanations.
````

---

## PROMPT 10 — Explainability: SHAP, stability, and plain-language reasons

````text
ROLE: Senior ML engineer specialised in explainable ML for non-technical audiences.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, phase_09 report, artifacts/runs/<active>/champion/.

GOAL: Explain the champion globally and per student, quantify how stable the explanations are (the proposal's "SHAP consistency" metric), and convert them to short plain-English reasons that the website will show.

STEPS
1. Choose the explainer by model type: TreeExplainer for tree/boosting models (use probability output or log-odds consistently and say which), LinearExplainer for linear, KernelExplainer/DeepExplainer with the saved background sample for others (time-box; if too slow use a 100-row background and 300 explained rows). Because the pipeline contains transformers, explain at the level of the MODEL INPUT FEATURES and also build a mapping from engineered features back to the 7 indicators, 4 periods, and state features.
2. Global explanations on the validation data of each fold and on the test set (report both): mean |SHAP| ranking; beeswarm; dependence plots for the top 8 features with the best interaction feature; interaction values for top 5 (tree models only); grouped importance by (a) indicator (7 groups), (b) period (P1–P4), (c) feature family (raw answers, aggregates/drift, state features). Report the share of total importance carried by state features versus aggregates.
3. SHAP consistency across folds (proposal metric): for each of the 15 fold-models compute the top-10 ranking; report mean Spearman correlation of mean |SHAP| rankings between folds and the Kendall's W coefficient, and the Jaccard overlap of top-10 sets. Show a plot. Interpretation text must be careful: stable ≠ causal.
4. Cross-method check: compare SHAP ranking with permutation importance (on validation folds) and with the Logistic Regression standardised coefficients; report rank correlations and discuss disagreements honestly. Optionally add LIME for 5 example students and compare with SHAP (qualitative).
5. By pattern: mean SHAP per feature within Consistent / Declining / Sporadic / Improving students, so we can say "what drives risk in declining students versus sporadic ones". Report group sizes and warn when small.
6. Local explanations: for any student return top-3 risk-increasing and top-2 risk-decreasing contributions. Create `src/mei/explain/reasons.py` that turns feature names into simple sentences using a template table in config/reason_templates.yaml. Examples (the agent must write the full table for every engineered feature; wording must be plain, non-judgemental, and never blame the student): "Study time dropped from period 1 to period 4", "Takes part in class activities less often than most students", "Practice and quiz activity is low in recent weeks", "Engagement goes up and down from period to period", "Spends most of the time in the low-engagement state". Contribution size is shown in points of MEI (approximate conversion of the log-odds contribution scaled so that contributions sum to MEI − baseline; explain the conversion and test that the parts add up within tolerance).
7. Implement `explain_dataframe(df)` returning for each row: reasons (list of {text, direction, points}), a compact SHAP vector for the waterfall chart, and state path/pattern. Add it to the scoring function from Phase 9 (reasons are now filled).
8. Save global SHAP artifacts as JSON for the website (importance lists, grouped importance, dependence samples ≤ 500 points per feature, beeswarm points ≤ 1500 points, pattern-wise importance, consistency numbers) plus PNG/SVG figures.

DOCS: docs/13_explainability.md — what SHAP is in plain words (a fair way to share credit among features, with a team-project analogy), how to read bar/beeswarm/dependence/waterfall figures, how reasons are generated, consistency results, comparison with permutation importance, limitations (correlated features, short sequences, explanation ≠ causation, sample-data caveat). Include "Explain like I am 5" and viva-style Q&A for SHAP.

TESTS: SHAP values sum to model output − expected value within tolerance; reason templates exist for every model input feature (test fails if one is missing); points add up; explanations are deterministic.

ACCEPTANCE CHECKS
- `python -m mei.cli explain` generates all artifacts in < 15 minutes.
- `python -m mei.cli score --input data/samples/sample_upload_clean.xlsx --explain` prints MEI, band, pattern, and 3 readable reasons for 5 example students.
- phase_10.md lists the top 10 drivers, the consistency numbers, and any disagreements between methods.

FINISH: G13 reply, phase_10.md, git commit, STOP. The modelling work is now complete; the website starts in the next prompt. Do not start the backend.
````

---

## PROMPT 11 — Backend API (reads only real artifacts)

````text
ROLE: Senior backend engineer (Python, FastAPI).
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, phase reports 09 and 10, the artifacts/runs/<active>/ folder.

GOAL: Build the API that the website will use. It must serve REAL artifacts only (G2): no hard-coded numbers, no fake endpoints.

BUILD src/mei/api/ (FastAPI, pydantic response models, CORS for localhost, structured errors with plain messages, request logging, a /health endpoint). Everything is read through a `RunStore` class that loads the active run from artifacts/active_run.txt and caches JSON files; switching the active run clears the cache.

ENDPOINTS (all JSON unless stated)
- GET /api/status → active run id, data_source (sample|survey), n_students, trained_at, champion summary, model card essentials.
- GET /api/leaderboard?track=classification|regression → rows with all metrics, gap, status, category, plain names, rank, test columns (null when not evaluated), flags (champion, category_best, remediated).
- GET /api/categories/best?track=… → best model per category.
- GET /api/models/{model_id}?track=… → full detail: metrics by split (train, CV validation, test), fold metrics, hyperparameters, confusion matrix, ROC/PR/calibration points, threshold table, timings, status explanation text.
- GET /api/models/{model_id}/curves?track=… → epoch curves (deep), per-iteration curves (boosting), learning curves, validation curves, overfit-zone info, stage-2 marker, best-epoch marker; also "before vs after regularisation" if present.
- GET /api/compare?ids=a,b,c&track=… → side-by-side data for up to 4 models.
- GET /api/patterns → state profiles, transition matrix, pattern distribution, mean trajectory by pattern, risk rate by pattern with CI, clustering comparison table, ablation (with vs without states).
- GET /api/drivers → global importance, grouped importance (indicator, period, family), dependence samples, beeswarm points, by-pattern importance, consistency numbers.
- GET /api/template?format=csv|xlsx → the blank template (file download). GET /api/dictionary → data dictionary JSON.
- POST /api/score/validate (multipart file: csv/xlsx/xls, max 20 MB) → runs cleaning in scoring mode: detected columns, auto-mapping with confidences, unmapped required columns, unknown columns, quality summary and traffic light, row-level problems. Returns a `upload_id`.
- POST /api/score/run {upload_id, mapping_overrides} → scores all rows: per row student_id, MEI, band, flag (using the tuned threshold), pattern, level, state path, top reasons, and `not_scored_reason` where needed. If the file also contains the outcome questions, add an optional `check` block: actual vs predicted counts, F1, AUC, confusion matrix (clearly labelled "on your file").
- GET /api/score/{upload_id}/results?page=&sort=&filter= and GET /api/score/{upload_id}/download?format=csv|xlsx (all columns incl. reasons joined with "; ").
- POST /api/score/single {answers…} → one student, with full explanation vector for a waterfall chart.
- GET /api/runs → all runs with data source, date, champion, metrics; POST /api/runs/{id}/activate; (retraining endpoints are added in Phase 16).
- Background tasks: use a small job manager (threading/multiprocessing) with job states queued/running/finished/failed/cancelled, progress 0–1, a message, and persisted job JSON so a server restart does not lose history.

SECURITY AND ROBUSTNESS: file type/size checks, safe temp files deleted after 24 h, no pickle loading from uploads, never log student rows, uploads stay on the machine, rate-limit scoring to protect memory (chunked scoring ≥ 5000 rows), friendly error messages (no stack traces to the client).

CONTRACT: generate an OpenAPI file at docs/api/openapi.json and a TypeScript types file the frontend will import (use openapi-typescript later, or hand-write a typed client in frontend/src/api/types.ts now and test it against the OpenAPI schema).

DOCS: docs/14_backend_api.md — endpoint table with example requests/responses (taken from real responses), data flow diagram (Mermaid), how the active run works.

TESTS: pytest with httpx TestClient for every endpoint using the real active run; upload tests with the clean and messy sample files; large-file test (20 000 rows) finishes in < 60 s; an "honesty test" asserts that /api/leaderboard values equal the values in artifacts files.

ACCEPTANCE CHECKS
- `python -m mei.cli serve` starts on port 8000; /docs (Swagger) lists all endpoints; every endpoint returns 200 with the sample run; uploading the messy sample file returns mapping suggestions and warnings; scoring the clean sample returns 200 rows with reasons.

FINISH: G13 reply, phase_11.md, git commit, STOP. Do not start the frontend.
````

---

## PROMPT 12 — Website foundation and design system (make it look like a real product, not like AI)

````text
ROLE: Senior product designer and front-end engineer who designs research and analytics tools for universities. You have strong taste and you hate generic template look.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md (§8, §10), docs/DECISIONS.md, docs/api/openapi.json, phase_11 report.

GOAL: Create the front-end project, the design system, the application shell and the shared components. No page content yet except an empty-state placeholder per route (use proper empty states, not lorem ipsum).

STACK: React 18 + Vite + TypeScript, React Router, TanStack Query, TanStack Table, Apache ECharts (echarts-for-react) for charts, Tailwind CSS configured with our own tokens (NOT default colours), Radix UI primitives for accessible dialogs/tooltips/tabs/popovers, react-dropzone, xlsx only if needed client-side (prefer server parsing). Fonts are bundled locally (no external CDN, so the site works offline): "Source Serif 4" (headings, big numbers) and "Public Sans" (UI). Use tabular numerals for all numbers.

DESIGN RULES — follow literally
1. Palette = PROJECT_BRIEF §10 only. Page background paper #F5F2EC, cards #FFFFFF with 1 px #DDD8CE border, 6 px radius, NO drop shadows except on menus/popovers. Primary actions forest #1F4E46. Risk colours only for risk meaning. Contrast ≥ 4.5:1 everywhere; colour never the only signal (add text/icon/pattern).
2. Forbidden: purple, indigo, violet, magenta, neon, gradients, glow, glassmorphism, blobs, 3D, emojis, sparkles/robot/brain icons, "AI" badges, centred marketing hero, oversized rounded pill buttons, stock illustrations, animated backgrounds, dark-mode-by-default. Use simple outline icons (Lucide) sparingly, only where they help recognition.
3. Layout: fixed left sidebar 232 px (product name "Engagement Index" in serif, then nav: Overview · Model Leaderboard · Engagement Patterns · What Drives Risk · Score Students · Train on New Data), top bar with the active model name and data source label, content max-width 1280 px, 8-px spacing grid, generous but not wasteful whitespace. Tables are dense, with sticky headers, right-aligned numerals, subtle row hover, no zebra.
4. Typography scale: page title 28/34 serif 600; section title 18/26 serif; body 15/24 Public Sans; table text 14; captions 12.5 muted. Sentence case everywhere.
5. Charts: flat, thin gridlines (#E7E2D8), direct labels instead of legends where possible, no chart junk, no rounded bars, no gradients, consistent categorical order and colours (primary forest, info blue, amber, clay #8C4A3B, olive #6E7F3F, stone #7A6F63). Units on axes. Always show sample size.
6. Motion: only 150 ms fades/slides for menus and tabs; respect prefers-reduced-motion. Skeleton loaders, not spinners, for page content.
7. Copy: plain English at about 8th-grade level, short sentences, no jargon. Every technical term shows a plain label first and the formal name in a tooltip, e.g. "Catches at-risk students" with tooltip "Recall (sensitivity)". Never use the words prototype, demo, beta, mock, placeholder, smart, magic, powered by AI.
8. States: every data view has loading, empty, error (with a "Try again" button and a plain message) and partial states. 404 page. Keyboard navigation and visible focus rings (2 px forest outline).
9. Responsive: designed for 1440 and 1024 widths first; usable at 390 px (sidebar collapses to a menu; tables scroll horizontally inside their card).
10. A small, quiet data-source label in the top bar: "Training data: Sample survey responses (generated)" or "Training data: Your survey data" (from /api/status). No other disclaimers or tags.

SHARED COMPONENTS (build with Storybook-free simple demo route /_components only in development, excluded from production build): Button, IconButton, Card, Stat (big number + label + note), Tag/Status badge (Good fit / Slight overfit / Overfit; Low/Moderate/High; Consistent/Declining/Sporadic/Improving), Tabs, Tooltip with formal-term help, DataTable (sorting, column visibility, search, CSV export, pagination), ChartCard (title, subtitle, sample size, download PNG/CSV), EmptyState, ErrorState, FileDropzone, Stepper, Drawer, Dialog, Toast, Skeleton, Term (inline glossary term).
Create a glossary file frontend/src/content/glossary.ts with plain labels + formal names + one-sentence explanations for every metric and term used in the UI (F1, AUC, MCC, R², RMSE, MAE, balanced accuracy, calibration, generalisation gap, SHAP, HMM state, pattern, level, MEI, band). Write them for a non-technical lecturer.

API CLIENT: typed fetch wrapper with TanStack Query hooks per endpoint; handles errors consistently; base URL from env with a dev proxy to :8000. In production the FastAPI app serves the built frontend from /frontend/dist (add that to the backend now) so ONE command runs the whole product: `python -m mei.cli serve`.

DOCS: docs/15_frontend_architecture.md (structure, design tokens table, component list, how data flows, how to add a page) and docs/design_tokens.md.

TESTS: Vitest + React Testing Library for Button, DataTable sorting, Term tooltip; ESLint + TypeScript strict + Prettier clean; a script that fails the build if forbidden words/colours appear in src (grep for purple/indigo/violet hex ranges and the forbidden copy list).

ACCEPTANCE CHECKS
- `npm run build` passes; `python -m mei.cli serve` shows the shell with working navigation to six routes with empty states.
- Open the site in the built-in browser and take screenshots at 1440, 1024 and 390 widths; save to docs/figures/ui_shell_*.png and LOOK at them: check alignment, contrast, spacing; fix anything that looks like a generic template.
- Forbidden-words check passes.

FINISH: G13 reply (include the screenshots), phase_12.md, git commit, STOP. Do not build the pages yet.
````

---

## PROMPT 13 — Model Leaderboard page and Model Detail page

````text
ROLE: Senior front-end engineer and data-visualisation designer.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/design_tokens.md, docs/15_frontend_architecture.md, docs/api/openapi.json, phase_12 report.

GOAL: Build the Model Leaderboard (the centrepiece) and the Model Detail page, with real data, in the established design system.

PAGE: /leaderboard
1. Header: title "Model leaderboard"; one plain sentence: "We trained many kinds of models on the same survey data and ranked them fairly. The top model is used to calculate the Engagement Index."; track switch (segmented control): "At-risk or not" (classification) / "Risk score 0–100" (regression). Show training data label and number of students and models.
2. Champion banner (not a hero): model name, category, leaderboard score, 4 key numbers (for classification: Catches at-risk students [recall], Correct when flagged [precision], Overall balance [F1], Ranking quality [AUC]; for regression: R², typical error [MAE], RMSE, Rank agreement [Spearman]) each with a confidence interval from the test set where available, and a one-line reason why it was chosen (from the model card).
3. "Best in each category" strip: one card per category (Baseline, Linear, Classical, Boosting & trees, Deep learning, Ensembles, Proposed hybrid) with the best model's name, score, and a mini bar; clicking filters the table. Mark the champion.
4. Main table (TanStack Table): Rank · Model · Category · Leaderboard score · F1 / R² · AUC / RMSE · Accuracy / MAE · Balanced accuracy · MCC · Train–validation gap (small bar + status tag) · Test score (blank dash with tooltip "Only the finalists are tested") · Training time. Filters: category chips, status filter, text search; column chooser; sort; sticky header; export CSV. Rows flagged remediated show a small "Tuned to reduce overfitting" tag with tooltip. Row click opens the detail page; checkbox lets users select up to 4 for comparison ("Compare" button opens a drawer or /compare view).
5. Charts below the table (ChartCards): (a) Leaderboard score by model (horizontal bars, top 15, coloured by category); (b) Train vs validation vs test (grouped bars or dumbbell) for the top 8 — this is the paper-style generalisation view; (c) Metric heat map for the top 12 models; (d) Training time vs score scatter with labelled points; (e) Category comparison box plot of CV scores. Each chart has a caption explaining how to read it in one sentence, shows n, and can be downloaded as PNG.
6. "With and without engagement states" card: shows the score of each hybrid model next to its twin without states (data from /api/patterns ablation or leaderboard twins), plain wording: "Do engagement patterns help?".

PAGE: /leaderboard/:modelId
- Header: model name, category, status tag, plain description (from API), tiny "what it is" paragraph.
- Tabs: Summary · Training curves · Splits · Errors · Stability · Settings.
  • Summary: key metrics with CI, rank, champion/category-best flags, status explanation in plain text ("Training score is 0.93 and validation score is 0.84. A gap this large means the model learned some details of the training students that do not carry over.").
  • Training curves: for deep models, the twin Loss and Accuracy charts (train vs validation lines, dashed vertical line at the start of stage 2, dotted line at the best epoch, shaded "over-training zone", annotation "Best validation loss X at epoch N"), mean ± std band across folds with a toggle for one fold; for boosting models loss vs number of trees; for others learning curve (score vs training size) and validation curve for the key setting. If a "before vs after regularisation" figure exists show it with a caption. Use the exact look of research figures: restrained, labelled, brand colours.
  • Splits: table with Train / Validation (cross-validated) / Test columns and rows for each metric and the gap (like a research paper's results table); test column shows "Not tested" when not a finalist.
  • Errors: confusion matrix (counts and percentages toggle), ROC, precision–recall, calibration plot, and a threshold slider that updates precision/recall/F1/flag counts live from the API's threshold table. For regression: predicted vs actual, residuals vs predicted, residual histogram, error by pattern.
  • Stability: fold-wise scores box/strip plot, seed variation (if present), confidence intervals.
  • Settings: hyperparameters as a readable table with plain meanings, training time, inference time, model size, feature set used, whether states were used.
- "Compare" view /compare?ids=…: overlay ROC/PR (or predicted-vs-actual), side-by-side metric table with the best value in each row highlighted by weight/underline (not colour only).

QUALITY: virtualise long tables if > 200 rows; remember filters in the URL; every chart has a text alternative (a visually hidden data table); all numbers formatted consistently (3 significant decimals for scores, thousands separators, "—" for missing).

DOCS: add a "Leaderboard and model pages" section to docs/15_frontend_architecture.md (data sources per element).

TESTS: component tests for status tags and gap bar; a test that the table's first row equals the first row of the API; a Playwright (or Cypress) smoke test: open leaderboard, switch track, filter a category, open a model, switch tabs, no console errors.

ACCEPTANCE CHECKS
- Use the built-in browser to open both pages, take screenshots at 1440 and 1024 widths, save to docs/figures/ui_leaderboard_*.png, and LOOK at them. Compare the values on screen with artifacts files for 5 random cells and report the check.
- No console errors; Lighthouse accessibility ≥ 90 for both pages; forbidden-words check passes.

FINISH: G13 reply with screenshots, phase_13.md, git commit, STOP. Do not build the other pages.
````

---

## PROMPT 14 — Overview, Engagement Patterns and What Drives Risk pages

````text
ROLE: Senior front-end engineer and data-storytelling designer for non-technical audiences.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/design_tokens.md, docs/15_frontend_architecture.md, phase_13 report.

GOAL: Build three pages with real data and plain language.

PAGE: /overview
- Title "Engagement Index". One sentence: "A score from 0 to 100 that shows how much a student's study engagement signals a risk of leaving. Higher means higher risk." 
- Row of 4 Stat cards: Students in the data · Models compared · Model in use (name + category) · Share flagged as higher risk.
- "How the score is made" — a 4-step horizontal diagram built with HTML/SVG in the design system (not a stock graphic): 1 Students answer a short questionnaire → 2 Answers are checked and cleaned → 3 Study patterns are found across the semester → 4 The best-performing model gives each student a score with reasons. Include the real counts from the run under each step.
- Score distribution histogram with band shading (Low/Moderate/High), band table with counts and the observed at-risk share per band (from Phase 9).
- Shortcuts: "Score students" and "Train on new data" buttons; "Best model in each category" mini-list linking to the leaderboard.

PAGE: /patterns
- Intro: "Students do not study at a constant pace. We looked for typical patterns across the four parts of the semester."
- Pattern cards: Consistent, Declining, Sporadic, Improving — each with a one-line plain meaning, share of students, mean engagement trajectory (sparkline across P1–P4), and the observed at-risk rate with 95 % CI.
- Trajectory chart: mean composite engagement (0–100) across periods for each pattern with ±1 SD bands (toggle bands).
- "Engagement levels" (the HMM states): heat map of what each state looks like (7 indicators × K states; plain names Low/Medium/High), state sizes, the transition matrix as a labelled heat map with a one-line reading guide ("Read across: from the row state to the column state"). Expected time students stay in each state.
- "How we grouped students" table comparing clustering methods and the HMM (quality scores with tooltips) and the plain reason for the chosen approach (from the run, not invented).
- "Do patterns help predict risk?" chart: with-states vs without-states scores (F1, AUC, MCC with intervals) and the plain verdict generated from the numbers (if the improvement is not significant the page must say so).
- Pattern × level matrix (counts and at-risk rate).

PAGE: /drivers
- Intro: "What in the answers pushes the score up or down."
- Top drivers: horizontal bar chart of the 10 most important factors in plain names (e.g., "Study time in the most recent period"), with a switch for grouping by: Factor / Question / Part of semester / Type (raw answers, change over time, patterns).
- "How each factor works": dependence view — pick a factor from a list; chart shows how the factor's values relate to the risk contribution (jittered points + trend line), with a plain sentence generated from the data ("Students who study fewer than 2 hours per week in the recent period get about +9 points on average").
- Beeswarm in friendly form: use ECharts scatter with the same ordering as SHAP summary plot, a clear legend "Low value ... High value", and a one-sentence reading guide.
- "Different patterns, different drivers": small multiples for the four patterns (top 5 drivers each), with group sizes and a caution when a group is small.
- "How steady are these results?": the SHAP consistency numbers expressed in plain words and a rank-stability chart across folds.
- A short caution card: "These are associations in the survey answers, not proof of cause."

QUALITY RULES: all text generated from data must come from backend fields or deterministic templates in frontend/src/content/sentences.ts (unit-tested; no hard-coded numbers); numbers formatted consistently; every chart has a one-sentence reading guide and n.

TESTS: unit tests for sentence templates; Playwright smoke tests for the three pages; accessibility checks (axe) pass with no serious issues.

ACCEPTANCE CHECKS
- Screenshots at 1440 and 1024 widths of all three pages saved to docs/figures/ui_*.png; look at them and fix anything crowded, misaligned or generic. Verify 5 numbers per page against the artifacts files and report the verification.
- Lighthouse accessibility ≥ 90; no console errors; forbidden-words check passes.

FINISH: G13 reply with screenshots, phase_14.md, git commit, STOP. Do not build the scoring page.
````

---

## PROMPT 15 — Score Students: upload any CSV/Excel and get the Engagement Index

````text
ROLE: Senior full-stack engineer focused on file-upload workflows and trust in results.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/design_tokens.md, docs/14_backend_api.md, phase_14 report, data/samples/*.

GOAL: Build /score. A user uploads a CSV or Excel file with student answers (any column order, possibly different header text) and gets, from the CURRENT champion model, the Engagement Index for each student, with patterns and reasons.

FLOW (Stepper: 1 Upload · 2 Check columns · 3 Results)
1. UPLOAD: drag-and-drop or choose file (.csv, .xlsx, .xls; ≤ 20 MB). Show requirements in plain words with a link to download the template and the data dictionary. Also offer "Score one student" (a form with the 7 questions × 4 periods as compact grids with the same option wording as the questionnaire) that returns the full explanation.
2. CHECK COLUMNS (calls /api/score/validate): show a table of required questions vs the columns found. Each row: required item (plain name), matched column (select box to change), match confidence (High / Check this / Not found), a sample of values. Auto-accept only high-confidence matches; low-confidence ones must be confirmed by the user. Show the data quality summary as a traffic light with plain messages (rows found, rows that can be scored, rows with missing answers, answers outside the allowed range, duplicates), and a short list of fixable problems with an explanation of what the system will do about each. Allow "Continue with rows that can be scored" and "Download list of problem rows".
3. RESULTS (calls /api/score/run):
   • Summary tiles: students scored · average score · High risk count · Moderate · Low.
   • Distribution histogram by band.
   • Results table: Student ID · Engagement Index (bar + number) · Band tag · Flag · Pattern · Level · Main reasons (first reason visible, +2 in a tooltip) · Rows not scored show the reason. Filters by band/pattern/level, search by ID, sort, pagination (server-side for > 2000 rows).
   • Row click opens a side drawer: engagement trajectory over the four periods (composite and each indicator toggle), the waterfall explanation in points ("Starts from the average of X; Study time −… ; Practice activity +…"), the pattern/level with plain meaning, and a "What this does not mean" note ("The score describes study engagement answers. It is not a prediction about a person's character and should be used to start a supportive conversation.").
   • If the file contained the outcome questions: a "How well did the model do on your file?" card (confusion matrix, recall, precision, F1, AUC) clearly labelled as computed on this file only.
   • Download: CSV and Excel with all columns (IDs, MEI, band, flag, pattern, level, reasons, not-scored reason) and a second Excel sheet "Notes" containing the model name, data source, date, and how to read the score. File names include the date.
4. Privacy note under the uploader: files are processed on this computer, are not shared, and are removed automatically after 24 hours (this must be TRUE; implement and test the cleanup).

ROBUSTNESS: test with (a) clean sample, (b) messy sample, (c) a file with extra and missing columns, (d) a file with 20 000 rows, (e) a file with Sinhala/Unicode in extra columns, (f) a Google-Forms-style export with long headers and text answers, (g) an empty file, (h) a wrong file type. Each must give a clear message and never crash. Store these as fixtures in tests/fixtures/.

DOCS: docs/16_scoring_workflow.md — user guide (with screenshots), column-matching rules, error messages table, how scores are produced step by step.

TESTS: unit tests for the mapping UI logic; Playwright end-to-end test uploading the clean sample and the messy sample and asserting the results table, the drawer, and the downloaded file's contents (columns and values equal API output).

ACCEPTANCE CHECKS
- Run the full flow in the built-in browser with data/samples/sample_upload_messy.csv: screenshot every step, save to docs/figures/ui_score_*.png, look at them, and confirm the downloaded Excel opens correctly (re-read with openpyxl and compare 10 rows with the API).
- Lighthouse accessibility ≥ 90; no console errors; forbidden-words check passes.

FINISH: G13 reply with screenshots, phase_15.md, git commit, STOP. Do not build the retraining page.
````

---

## PROMPT 16 — Train on New Data (swap generated data for the real survey)

````text
ROLE: Senior ML-platform engineer.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, phase reports 06–11 and 15, src/mei/cli.py, src/mei/models/engine.py.

GOAL: When the real survey results arrive, the user must be able to retrain EVERYTHING on the new file (cleaning → features → states → all models → leaderboard → final evaluation → explanations) from the website or the command line, review the new leaderboard next to the old one, and then choose to use it. No code changes needed.

BACKEND
1. Orchestrator `src/mei/registry/retrain.py::run_full_pipeline(dataset_path, mode, label_config, seed)` that calls the existing steps in order with a NEW run id and writes `artifacts/runs/<new_id>/`. Modes: `quick` (fewer trials, drop the slowest models, ≈ 10–15 min), `full` (everything, ≤ 60 min budget from config). The orchestrator is the same code the CLI uses: `python -m mei.cli retrain --data data/real/survey.xlsx --mode quick`.
2. SMALL-DATA GUARDS (real surveys are likely 100–600 rows): if n < 150 → refuse training and explain; if 150 ≤ n < 500 → automatically use 5×5 repeated CV, reduce deep-model sizes and trials, disable the Transformer and stacking, warn clearly in the report ("With fewer than 500 students, deep models are unlikely to beat simpler ones; results have wide uncertainty"), and widen reported confidence intervals via bootstrap; if the positive class is < 15 % → enable class weights + report PR-AUC prominently; if the label column cannot be built → explain exactly which outcome questions are missing.
3. LABEL OPTIONS: default = the 3 outcome questions (brief §4). Also allow the user to choose a different existing 0/1 column or a numeric column as the label, with the threshold, but only through config (labels.yaml copy stored in the run) and with a clear note in the report that results are not comparable with the default label.
4. Validation endpoint POST /api/retrain/validate (reuses the cleaning module in training mode): returns rows, usable rows, class balance, missingness, warnings, estimated time, and the traffic light. POST /api/retrain/start {upload_id, mode, options} → job id; GET /api/retrain/{job}/status (progress 0–1, current step, current model, models finished, elapsed, ETA, last log lines); POST /api/retrain/{job}/cancel; POST /api/retrain/{job}/activate (switch active run; keep the old run for rollback); GET /api/runs/compare?a=&b= (leaderboards side by side: rank changes, score changes, champion change, metric differences).
5. Safety: each run isolated; the previous champion is never deleted; activation writes an audit line (who/when/run ids) to artifacts/audit.log; real data are copied to data/real/ and listed in .gitignore; the data_source for such runs is `survey`.
6. Data-card generator: for every run write `data_card.md` and `model_card.json` automatically (rows, class balance, missingness, quality warnings, champion and metrics with CI, limitations text that depends on sample size).

FRONTEND: /train
- A 6-step stepper: 1 Get the template (download template + dictionary; checklist of what the file must contain) · 2 Upload your survey file (same dropzone and column matching as the score page, but with outcome questions required) · 3 Check your data (traffic light, class balance chart, missing-answer chart, plain warnings, sample size advice) · 4 Choose how to train (Quick / Full cards with time estimates; the label rule shown in plain words with a link to the page that explains it; advanced options collapsed) · 5 Training (live progress: current step, each model with a status icon (waiting/running/done/skipped), a running mini-leaderboard, elapsed time, Cancel button; safe to leave the page and return) · 6 Review and decide (old vs new leaderboard side by side with rank changes, champion comparison, key metric differences with intervals, a plain-language summary generated from the numbers, and two buttons: "Use the new model" / "Keep the current model").
- "Previous runs" table below the stepper: date, data source, students, champion, main score, status, "Use this one" action (rollback).
- Wording rules apply (no "prototype/demo"). The page may state plainly: "Models trained on sample data should be replaced once your real survey results are in." in the data-source explanation text.

DOCS: docs/17_retraining_real_data.md — step-by-step guide for the author: how to build the Google Form from docs/questionnaire_engagement.md, how to export results, where to put the file, how to retrain from the website or CLI, what to check before trusting the new model, how to update the documents and viva numbers afterwards (the docs command in Phase 17 regenerates them), what to do if the sample is small, how to interpret changes in the leaderboard.

TESTS: run the whole pipeline end to end on a 300-row subset of the sample data in quick mode in < 15 min as an integration test (mark as slow); guards trigger correctly for n = 100, 300, 800; cancel works; activation switches the API's active run; rollback works; the audit log is written.

ACCEPTANCE CHECKS
- Simulate real data: generate a second sample with a different seed and N = 350 and different header wording, save as data/real/test_survey.xlsx, upload through the website, train in Quick mode, compare, activate, and roll back. Screenshots of each step saved to docs/figures/ui_train_*.png. Delete or clearly mark the simulated file so it is never mistaken for real data.
- The leaderboard page shows the new run's data after activation, and the data-source label updates correctly.

FINISH: G13 reply with screenshots and timings, phase_16.md, git commit, STOP. Do not write the viva/presentation documents yet.
````

---

## PROMPT 17 — Documentation pack: how everything works, viva and presentation (Markdown only, NOT in the website)

````text
ROLE: Senior research mentor and technical writer who prepares students for final-year vivas and presentations.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, ALL files in docs/ and docs/phase_reports/, and the active run's artifacts (leaderboards, model card, SHAP results, states summary, data quality report).

GOAL: Produce a complete, accurate, readable documentation pack in Markdown under docs/, for my viva and presentation. All numbers must be pulled automatically from artifacts by a script (scripts/build_docs_numbers.py → docs/_numbers.json and docs/RESULTS_SUMMARY.md) so the documents can be regenerated after the real data arrive (`python -m mei.cli docs`). Do NOT put any of this into the website.

CREATE THESE FILES (simple English, short sentences, tables and Mermaid diagrams where helpful; each file starts with a 5-line summary and ends with "Common mistakes" and "If the examiner asks")
00_README_DOCS.md — index and reading order (30-minute quick read, 2-hour full read).
18_END_TO_END_WALKTHROUGH.md — the whole project in order: problem → data → cleaning → features → states → models → leaderboard → champion → explanations → website → retraining, each in a paragraph with a pointer to the detailed file and the key figure.
19_RESULTS_SUMMARY.md — auto-generated: data used, sample sizes, cleaning counts, states chosen, top 10 leaderboard rows (both tracks), best per category, champion with CI, calibration, band table, SHAP top drivers, consistency, ablation verdict. Include the line "Data source: sample (generated)" or "survey" automatically.
20_FIGURE_CATALOG.md — every figure with file name, what it shows, how to read it in one sentence, when to use it in the presentation. Export final versions to docs/figures/final/ at 300 dpi (PNG + SVG).
21_CONCEPTS_FOR_BEGINNERS.md — "Explain like I am 5" for: survey data, cleaning, missing values, features, train/validation/test, cross-validation, overfitting, underfitting, regularisation, early stopping, hyper-parameter tuning, logistic regression, random forest, gradient boosting, deep nets, GRU/LSTM, clustering, HMM, SHAP, calibration, R², F1, AUC, MCC, confidence intervals, leaderboard. Each: a 5-year-old explanation, a student-level explanation, one example from OUR project.
22_VIVA_QA.md — at least 80 questions with model answers, grouped: research problem and novelty (10) · data and questionnaire (12) · cleaning and preprocessing (8) · states, clustering and HMM (12) · models and leaderboard (14) · overfitting and validation (12) · explainability (8) · ethics, privacy and limitations (8) · system and website (6). Answers use OUR real results and limitations (including the facts that data are generated until real data arrive, labels are self-reported intentions, sequences have 4 time points, and test set was used once). Mark the 15 "most likely" questions. Include 10 tough questions with honest answers (e.g., "Why should we believe results on generated data?", "Why did a simple model beat the deep one?", "How do you know the HMM states mean anything?", "Is attrition intention the same as dropout?").
23_PRESENTATION_GUIDE.md — a 15–20 slide storyline for a 12-minute talk: for each slide: title, the single message, what to show (exact figure file), speaker notes (≈ 60–90 words), expected question, and time. Add a 5-minute live-demo script for the website (exact clicks and what to say), a backup plan if the live demo fails (use screenshots from docs/figures), and a one-page cheat sheet of the 12 numbers to remember.
24_LIMITATIONS_ETHICS.md — honest limitations and mitigations; privacy and consent; fairness (no demographic features used; possible bias in who answers surveys); responsible use ("start a supportive conversation, not a decision about a student"); what must happen before real-world use.
25_PROPOSAL_UPDATE_NOTES.md — replacement text (ready to paste) for my proposal report so it matches the survey-based design: Abstract (revise data description), Section 2.2 specific objectives, Section 3.2 Data/participants/inputs, Section 3.3 baselines, Section 3.4 evaluation metrics table (add balanced accuracy, MCC, PR-AUC, R², RMSE, MAE, calibration, SHAP consistency, generalisation gap), 3.5 validation strategy (repeated stratified CV + one-time held-out test), 4.x architecture text, functional requirements for the website. Also a short list of things to change in the Topic Assessment Form (Section 10 Tasks and Novelty are already updated). Mark places where the author must insert real numbers after the real survey.
26_REPRODUCIBILITY_AND_RUN_GUIDE.md — how to install, run every phase, regenerate sample data, retrain on real data, run tests, and where each output lives; troubleshooting table.
27_GLOSSARY.md — all terms, A–Z, plain definitions.

RULES: no invented citations (write "citation to be added" where unsure; list only sources you can name accurately, e.g., library documentation); every number comes from _numbers.json; say clearly when a result is not statistically significant; keep a "What changes when real data arrive" section in files 19, 22 and 23.

TESTS: a script `scripts/check_docs.py` that (1) verifies every number in 19_RESULTS_SUMMARY.md matches artifacts, (2) checks all internal links and image paths, (3) fails if the documents contain "TODO", "lorem", or the forbidden words from the website rules in places that describe the website itself, (4) checks that every figure in 20_FIGURE_CATALOG.md exists.

ACCEPTANCE CHECKS
- `python -m mei.cli docs` regenerates numbers and summary; `python scripts/check_docs.py` passes.
- Read VIVA_QA.md answers 1–10 and verify each against the artifacts; list any answer you had to soften because the evidence was weak.

FINISH: G13 reply, phase_17.md, git commit, STOP. Do not run the final QA yet.
````

---

## PROMPT 18 — Final quality gate, hardening and hand-over

````text
ROLE: Principal engineer doing a release review, and a sceptical external examiner at the same time.
READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, every phase report.

GOAL: Prove the whole product works from a clean machine, remove anything that looks unfinished or "AI-made", and hand over a clear run guide. Fix what you find; do not just report it.

A. CLEAN-ROOM TEST
Clone the repository into a new folder, create a fresh virtual environment, run setup, then the full pipeline (`python -m mei.cli all` with budgets from config), start the server, and run the whole test suite (pytest, vitest, Playwright). Record timings. The command `python -m mei.cli all` must create data → clean → features → states → classic → deep → regression → finalize → explain → docs, and `python -m mei.cli serve` must then show everything. Fix every failure.

B. SCIENCE AUDIT (be an examiner)
1. Leakage: re-run the shuffled-label test and the canary-feature test on the final pipeline; confirm outcome items are absent from all feature lists; confirm HMM/clustering/imputer/scaler/calibrator are fitted only on training data in every fold (read the code, then write a test that inspects fitted objects' training row counts).
2. Test-set discipline: confirm the lock file and that no notebook/script reads test indices elsewhere.
3. Numbers: compare 25 random numbers from the website, the docs and the artifacts; they must agree to the displayed precision.
4. Plausibility: check that results are neither suspiciously perfect nor trivial; check class-balance, calibration and band monotonicity; confirm the leaderboard is not dominated by models that have status "Overfit".
5. Seeds: re-run the champion with 3 new seeds and report the spread of the main metrics.
6. Honesty labels: the website shows the correct data-source label; docs state that results come from generated data until real survey data are used.

C. PRODUCT AUDIT
1. Forbidden-words and forbidden-colour scan across frontend/src and the built bundle. Zero hits for prototype/demo/beta/mock/placeholder/"AI-powered"/"smart"/"magic"; no purple/indigo/violet hues.
2. Visual QA in the built-in browser at 1440, 1024, 390 for all six pages in normal, loading, empty and error states. Check alignment, truncation, overflow, contrast, focus order, keyboard use, tooltips, chart readability. Fix every issue; save the final screenshots to docs/figures/final_ui/.
3. Accessibility: axe has no serious/critical issues on any page; Lighthouse accessibility ≥ 90; all charts have text alternatives.
4. Performance: first load < 3 s on localhost production build; leaderboard interaction < 100 ms; scoring a 5 000-row file < 20 s.
5. Offline: disconnect the network and verify the site and fonts still work.
6. Wording pass: read every string in the UI as a non-technical lecturer; simplify anything unclear; make sure each page explains itself in two sentences.
7. Failure drills: kill the server mid-training and restart (job state survives); upload a corrupt file; remove the active run folder and confirm a friendly error with recovery advice.

D. HAND-OVER
- README.md (what it is, 5-minute quick start, the 5 commands that matter, folder map, how to retrain with real data, how to regenerate docs), `start.ps1` and `start.sh` that activate the environment and launch the server, a `Makefile` target `release-check` that runs every check in this phase.
- A one-page `HANDOVER_CHECKLIST.md` for me: (1) what to do the day the real survey results arrive, in 8 numbered steps; (2) what to re-read in the viva documents afterwards; (3) the 12 numbers to update in the slides; (4) known limitations; (5) how to share the project folder (what to exclude: .venv, node_modules, data/real).
- Tag the repository v1.0.0.

FINISH: G13 reply with the release-check results (PASS/FAIL list), the final screenshots, remaining known issues (if any, honestly), and the exact commands to start the product. STOP.
````

---

# After Prompt 18 — what you do yourself

1. Read `docs/19_RESULTS_SUMMARY.md` and `docs/22_VIVA_QA.md` once, with the website open next to them.
2. Show your supervisors the label definition (attrition-intent items) and the 4-period questionnaire design before you collect real data. They must agree to both.
3. When real data arrive, follow `docs/17_retraining_real_data.md`, then regenerate the documents.
4. Update your proposal with `docs/25_PROPOSAL_UPDATE_NOTES.md`.

# prompt 19
ROLE: Senior ML engineer doing a controlled migration of a COMPLETED project (phases 0–18 are all finished and tagged). Read AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md and every file in docs/phase_reports/. Then read reference/Engagement_Questionnaire_Final.md: it is the NEW source of truth for the questionnaire.

WHY: The questionnaire changed. The old design assumed every item has 5 ordinal levels. The new one uses MIXED formats with different numbers of levels: resource_access 5, study_time 5, review_time 2 (No/Yes), quiz_practice 4, assignment_engagement 4, study_consistency 5, participation 3 (Low/Medium/High); one typed-number item study_hours_exact (0–60, messy text allowed); an attention-check row inside the study_consistency grid ("select Neutral" = code 3); outcome items intent_leave (5 levels), intent_defer (3 levels: No/Maybe/Yes), intent_continue (5 levels, reverse: true; renamed from intent_not_continue). All other design decisions stay (4 periods, 7 indicators, HMM/clustering, two tracks, leaderboard rules, G1–G13).

THIS PATCH COVERS: archive, brief/config, data generator, cleaning, features and splits. It does NOT retrain models (Patch B) or touch the website (Patch C). Stop after this patch.

STEP 1 — Archive, do not delete
git tag legacy-5level. Move artifacts/runs/* to artifacts/legacy/, data/raw|interim|processed|samples to data/legacy/, empty artifacts/active_run.txt. Mark old figures as legacy in docs/figures/INDEX.md. Old numbers must never appear in new docs. Create docs/phase_reports/patch_A_inventory.md listing what existed.

STEP 2 — Brief and config
- Replace PROJECT_BRIEF §3 with the new questionnaire exactly (items, wording, n_levels, codes, formats, direction, attention check, typed-number item, outcome items, reverse flag). Q1–Q7: higher code = more engagement.
- §4: attrition_intent_score = 100 × mean of the three outcome items after rescale01 (intent_continue reversed). Keep threshold 60 and the bands.
- §5: every indicator is rescaled with rescale01(code, item) = (code − 1) / (n_levels − 1) before building composite E_t, the sequence tensor and any NN/linear input; tree models may use raw codes.
- Update config/schema.yaml (n_levels, ordered labels, format type, direction, reverse), config/mapping.yaml (new header variants), config/labels.yaml. Log all in docs/DECISIONS.md.

STEP 3 — Code
- Search the whole codebase for hard-coded 5-level assumptions (patterns such as "- 1) / 4", "range(1, 6)", "<= 5", "max=5", "== 5", "5 options", "1–5") and remove them. Add a repository test that fails if one remains.
- io/codec.py: label⇄code for items with different levels; tolerant of case, spaces, en-dash vs hyphen, "hrs/hours"; typed-number parser (leading number; "a-b" → midpoint; words/garbage → missing; outside 0–60 → missing); rescale01 with reverse flag.
- Regenerate the template (csv/xlsx + data dictionary) from the schema.
- Sample generator: convert latent engagement to each item's OWN n_levels with fixed cut-points (no code above 45 %, except the yes/no item may be 70/30); text answers with exact option wording; study_hours_exact drawn inside the student's study_time range with heaping on 0/2/5/10/15/20, 8 % inconsistent by more than one bin, 10 % typed with units, 5 % as ranges, 2 % words/text, 1 % out of range; outcomes A1 (5), A2 (3), A3 (5, answered as likelihood = reverse-coded risk) around the latent risk; acquiescence/extreme/midpoint styles only on agreement and 3-level items; attention-check row failed by 2 %. Re-run the calibration loop (LightGBM CV AUC 0.86–0.92, prevalence 24–32 %). Regenerate data/raw/*, ground_truth_hidden.csv, data/samples/sample_upload_clean.xlsx and sample_upload_messy.csv.
- Cleaning: per-item range checks (1..n_levels); typed-number parsing; attention-check on the new row; CROSS-CHECK rule (if study_hours_exact differs from the mean study_time bin by more than one bin keep the row and set inconsistent_study_hours = True, counted in the report); straight-lining rule = same relative position (lowest/highest option) on ≥ 6 of 7 indicators across all periods.
- Features: use rescale01 everywhere; add study_hours_exact (cleaned, capped at 60) and inconsistent_study_hours to the `agg` family and test in the quick CV whether they help (report honestly); regenerate splits (same seed, new data, new run id). Rebuild reason-template needs: list every engineered feature and flag the ones whose wording must change (review_time is binary; intent_* are never features).
- Update tests: items with 2, 3, 4 and 5 levels exist; rescale01 stays in [0,1] and respects reverse; parser cases ("10", "10 hours", "8-10", "ten", "90"); attention check; cross-check flag; leakage and shuffled-label tests still pass; all earlier tests green.
- Update docs: questionnaire, data design, sample data, cleaning, preprocessing (no sentence may say "all items are 5-point"), regenerate their figures and numbers.

ACCEPTANCE CHECKS: pytest green; ruff/black clean; calibration result in range and printed; grep for legacy patterns returns nothing; the raw sample contains the new wording for all 7 grids, the typed-hours column and the 3 new outcome items; the cleaning report shows the cross-check count.

FINISH: write docs/phase_reports/patch_A.md, git commit, G13 reply, STOP. Do not retrain models.

# 
Patch A iwara una passe patch_A.md eke calibration (AUC, prevalence) eka range eke thiyenawada balanna. Eka out nam Patch B ekata yanna epa.
Patch B eke old vs new comparison table eka balanna. Champion model eka wenas wenna puluwan. Eka normal, but docs wala eka honestly liyanna ona.

# prompt 20
ROLE: Same engineer. READ FIRST: AGENTS.md, PROJECT_BRIEF.md (now the new version), docs/DECISIONS.md, docs/phase_reports/patch_A.md.

GOAL: Re-run phases 5 to 10 on the new data with the ORIGINAL rules (HMM/clustering inside folds, same 15-fold CV, leaderboard score and overfit rules, remediation ladder, one-time test evaluation, pre-declared champion rule, calibration, SHAP). Everything gets a NEW run id.

STEPS
1. Run in order: states → classic classification → deep → regression → finalize → explain. Use the existing CLI commands and time budgets. After each step run its tests and write a short note in docs/phase_reports/patch_B.md (timings, top results). Fix anything that breaks (typical breakages: sequence tensor shape or dtype, NN inputs not rescaled, feature name lists, reason templates for new or renamed features, SHAP background sample).
2. IMPORTANT: write the champion-selection rule into DECISIONS.md again for this run BEFORE looking at the new test results; the TEST_USED.lock of the legacy runs does not apply to the new run id.
3. Do NOT copy numbers from the legacy runs. Compare old vs new only in a table in the patch report (champion, top 5 per track, AUC/F1/R², gaps) to show whether the questionnaire change altered conclusions; explain honestly any changes.
4. Re-check: shuffled-label test, canary leakage test, state-recovery ARI vs the hidden archetypes, ablation (with vs without states, Wilcoxon), calibration, band monotonicity, SHAP consistency, overfit audit (no "Overfit" champion).
5. Set artifacts/active_run.txt to the new run. Regenerate docs numbers for phases 5–10 (docs files 06–13) so they describe the new run only.

ACCEPTANCE CHECKS: new leaderboard files for both tracks; champion bundle; model card; SHAP artifacts; `python -m mei.cli score --input data/samples/sample_upload_clean.xlsx --explain` works with readable reasons; all tests green.

FINISH: docs/phase_reports/patch_B.md, git commit, G13 reply (include the old-vs-new comparison table), STOP. Do not touch the website yet.

# prompt 21
ROLE: Same engineer. READ FIRST: AGENTS.md, PROJECT_BRIEF.md, docs/DECISIONS.md, patch_A and patch_B reports.

GOAL: Make the API, website and documentation pack match the new questionnaire and the new run, then repeat the final quality gate.

STEPS
1. API: regenerate OpenAPI and TypeScript types; update validate/score/single endpoints for items with different level sets, the typed hours field, the attention-check handling and the renamed outcome items; template/dictionary downloads come from the new schema. Update the `check` block (file with outcome questions).
2. Website: (a) Score one student form: 7 grids with their DIFFERENT option sets (use the exact wording from schema) + the typed hours field with plain validation; (b) column-matching screen labels and sample values; (c) glossary and plain labels for review_time (yes/no) and other formats; (d) Overview/Patterns/Drivers texts must come from the new run; (e) data-source label still correct; (f) Train on New Data page: template link, checklist and validation messages mention the new items and the cross-check warning. Test with the new messy sample, a Google-Forms-style export with long headers, and a file with typed hours like "10 hours", "8-10" and "ten".
3. Docs: run `python -m mei.cli docs` and update 17–27 (viva Q&A, presentation guide, figure catalog, proposal update notes, limitations): every number from the new run; add viva questions about mixed answer formats (why ranges instead of exact numbers, how rescale01 makes items comparable, why a binary item is acceptable, what the typed-hours cross-check does); update 25_PROPOSAL_UPDATE_NOTES.md with the mixed-format questionnaire description. Re-export final figures. `scripts/check_docs.py` must pass.
4. Final gate (same as the original Phase 18): clean-room run (fresh clone → `python -m mei.cli all` → serve), leakage and test-discipline audit, 25-number consistency check between website/docs/artifacts, forbidden words/colours scan, visual QA at 1440/1024/390 for all pages, axe + Lighthouse ≥ 90, offline check, failure drills, 3-seed champion spread. Fix everything you find. Take new screenshots into docs/figures/final_ui/.
5. Update README.md, HANDOVER_CHECKLIST.md; tag v1.1.0.

ACCEPTANCE CHECKS: release-check target passes; no legacy numbers anywhere (grep for the legacy champion name and legacy metric values from artifacts/legacy); website values equal artifacts for 25 random numbers.

FINISH: docs/phase_reports/patch_C.md, git commit, G13 reply with the release-check PASS/FAIL list and start commands, STOP.