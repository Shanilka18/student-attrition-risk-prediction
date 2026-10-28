# Phase 17 — Retraining on Real Survey Data

This guide is for the author (a university lecturer) who has collected real survey responses and wants to retrain the MEI model on that data. No code changes are required.

---

## 1. Build and Export the Google Form

Use `docs/questionnaire_engagement.md` as the source of truth for all question text.

1. Open [Google Forms](https://forms.google.com) and create a new form titled **MEI Engagement Survey**.
2. Add each of the 31 items in order:
   - Items Q1–Q7 × Periods P1–P4 (28 Likert items, scale 1–5 labelled "Never" to "Always").
   - Attention check item: "Please select 3 for this question." (Likert 1–5).
   - Outcome items A1 (`intent_leave`), A2 (`intent_defer`), A3 (`intent_not_continue`): Likert 1–5 labelled "Not at all likely" to "Very likely".
3. Add a **Student ID** short-answer question (mark as required).
4. Add a **Consent** checkbox (value 1 = agrees).
5. Enable **Response validation** on all Likert items: integer between 1 and 5.
6. Share the form link with students through the LMS at the end of the semester.

### Exporting responses

1. Open the form → **Responses** tab → click the green Sheets icon to link to Google Sheets.
2. In Google Sheets: **File → Download → Microsoft Excel (.xlsx)** or **Comma-separated values (.csv)**.
3. Save the file somewhere safe on your computer.

> **Column names do not have to match exactly.** The system uses fuzzy header matching to find the right columns. However, keeping names close to the template (download from the `/train` page) reduces the chance of ambiguous matches.

---

## 2. Where to Put the File

For retraining from the website:
- Upload via the `/train` page (Step 2). The file is stored in a secure temporary location and automatically deleted after 24 hours.

For retraining from the command line:
- Copy the file to `data/real/`. Files in this folder are listed in `.gitignore` and are never committed.

```powershell
# Example
Copy-Item "$HOME\Downloads\mei_survey_2027.xlsx" "C:\path\to\MEI_Project\data\real\survey.xlsx"
```

---

## 3. Retrain from the Website (`/train`)

1. Navigate to `/train` in the MEI web interface.
2. **Step 1 — Get template**: Download the template to verify your file matches the expected format.
3. **Step 2 — Upload file**: Drag and drop your `.xlsx` or `.csv` file.
4. **Step 3 — Check data**: Review the traffic light. Green means the data is ready. Amber means warnings but retraining is allowed. Red means the data cannot be used (reason shown).
5. **Step 4 — Choose mode**:
   - **Quick** (~10–15 min): fewer Optuna trials, skips the slowest models (Transformer, Stacking). Good for a first check.
   - **Full** (~45–60 min): all models, all trials. Use before a final decision.
6. **Step 5 — Training**: Watch the live progress bar. Each model shows Waiting / Running / Done / Skipped. You can leave the page and return — the job continues in the background.
7. **Step 6 — Review and decide**: The new leaderboard appears side-by-side with the current active leaderboard. Rank changes, metric differences, and a plain-language summary are shown. Click **Use the new model** to activate it, or **Keep the current model** to leave the active champion unchanged.

---

## 4. Retrain from the Command Line

```powershell
# Quick mode
.\.venv\Scripts\python.exe -m mei.cli retrain `
    --data data/real/survey.xlsx `
    --mode quick

# Full mode with a custom seed
.\.venv\Scripts\python.exe -m mei.cli retrain `
    --data data/real/survey.xlsx `
    --mode full `
    --seed 2027

# Retrain and immediately activate the result
.\.venv\Scripts\python.exe -m mei.cli retrain `
    --data data/real/survey.xlsx `
    --mode quick `
    --activate
```

The new run artifacts are written to `artifacts/runs/<new_run_id>/`. The previous champion is never overwritten.

### Activate a run manually

```powershell
# List available runs
.\.venv\Scripts\python.exe -m mei.cli runs

# Activate a specific run
.\.venv\Scripts\python.exe -m mei.cli activate <run_id> --actor "Author name"
```

Activation writes a line to `artifacts/audit.log`:
```
2026-10-28T12:00:00Z | ACTIVATE | run_20261028_120000_abc123 <- run_20261026_124514_1407b3 | actor=Author name
```

### Rollback

The previous champion is stored in `artifacts/runs/<old_run_id>/`. To roll back:
```powershell
.\.venv\Scripts\python.exe -m mei.cli activate <old_run_id> --actor "Rollback"
```

---

## 5. Small-Data Guards

| Students (N) | Behaviour |
|---|---|
| N < 150 | **Refused.** Training cannot proceed. Explanation shown. |
| 150 ≤ N < 500 | **Allowed with restrictions.** 5×5 repeated cross-validation, reduced Optuna trials, Transformer and Stacking disabled. Wide bootstrap confidence intervals. Warning shown in report. |
| N ≥ 500 | **Full pipeline.** All models enabled, standard 5-fold CV. |
| Positive class < 15 % | Class weights enabled automatically. PR-AUC reported prominently. |

When 150 ≤ N < 500 the phase report includes this note:
> *With fewer than 500 students, deep models are unlikely to beat simpler ones; results have wide uncertainty. The reported confidence intervals are based on 1 000 bootstrap resamples.*

---

## 6. What to Check Before Trusting the New Model

Run through this checklist before activating:

1. **Class balance**: Is the positive class (intent to leave/defer) at a reasonable rate? If under 10%, the minority label may be too sparse to model reliably.
2. **Test F1 vs previous champion**: Is the new F1 at least as good? A drop of more than 0.03 on the test set is a signal to investigate.
3. **Calibration curve**: Does the predicted probability track the observed rate? (Shown in the model detail page.)
4. **Overfit audit**: Is the gap between CV score and holdout score under 0.05? Larger gaps suggest the model is overfitting.
5. **SHAP drivers**: Do the top features make sense for your cohort? A driver that seems implausible (e.g., `study_time_p4` negative when you expect it to be positive) may indicate a data quality issue.
6. **Confusion matrix**: Check recall on the positive class. For attrition risk, recall matters more than precision.
7. **Data card**: Read `artifacts/runs/<new_run_id>/data_card.md` for the row count, missingness, class balance, and any quality warnings.

---

## 7. Interpreting Changes in the Leaderboard

When you click **Review and decide** (Step 6), the side-by-side table shows:

| Column | Meaning |
|---|---|
| Rank change | ↑ / ↓ / = compared to the same model in the old run |
| Score Δ | Leaderboard score difference (positive = improved) |
| Test F1 Δ | Holdout F1 difference |
| Champion change | Whether the champion model changed |

A **plain-language verdict** is generated automatically:
- *"The new run has a stronger champion (Test F1 0.89 vs 0.88). Metrics are within expected variation."*
- *"The new run has a weaker champion (Test F1 0.82 vs 0.88). Review the confusion matrix before activating."*

---

## 8. Label Configuration (Advanced)

By default the label is built from the three outcome questions (A1, A2, A3). If you want to use a different label — for example, an existing binary column from your student records — edit `config/labels.yaml` before retraining:

```yaml
label_source: column          # 'outcomes' (default) or 'column'
label_column: withdrew        # column name in your CSV
label_threshold: 0.5          # only used when label_source='numeric'
```

> **Note**: Results from a custom label are not comparable with the default label. The phase report will contain a warning.

---

## 9. Updating Documents and Viva Numbers After Retraining

After activating a new run, regenerate the metrics tables:

```powershell
# Regenerate the docs command (Phase 17)
.\.venv\Scripts\python.exe -m mei.cli docs
```

This updates:
- `docs/model_card.md` and `artifacts/runs/<active_run_id>/model_card.json`
- `docs/data_card.md` and `artifacts/runs/<active_run_id>/data_card.md`
- Phase report tables that reference leaderboard metrics

For viva presentations, the key numbers to quote are in:
- `artifacts/runs/<active_run_id>/summary.json` — champion model, Test F1, AUC, threshold
- `artifacts/runs/<active_run_id>/data_card.md` — N students, class balance, missingness

---

## 10. Endpoint Reference

| Endpoint | Description |
|---|---|
| `POST /api/retrain/validate` | Validate uploaded file; returns traffic light, class balance, estimated time |
| `POST /api/retrain/start` | Start a retraining job; returns `job_id` |
| `GET /api/retrain/{job_id}/status` | Poll for progress, current step, elapsed time, ETA |
| `POST /api/retrain/{job_id}/cancel` | Cancel a running job |
| `POST /api/retrain/{job_id}/activate` | Activate the completed run as the new champion |
| `POST /api/retrain/activate` | Activate any completed run by `run_id` |
| `GET /api/runs/compare?a=&b=` | Compare two runs side by side |

---

## 11. Error Messages

| Message | Cause | Fix |
|---|---|---|
| `Need at least 150 students to train reliable models` | N < 150 | Collect more responses |
| `Outcome questions missing: intent_leave, intent_defer` | A1–A3 not in file | Add outcome question columns |
| `Column mapping failed for: study_time_p3` | Header not recognisable | Rename the column or use the template |
| `File exceeds 20 MB` | File too large | Split into batches or reduce columns |
| `Unsupported file type` | Not CSV/XLSX/XLS | Convert the file |
| `Positive class rate 8 % — class weights enabled` | Imbalanced label | Not an error; proceed with caution |
