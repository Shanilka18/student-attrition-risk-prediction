# Phase 7 — FastAPI Backend Documentation

## Simple Explanation (ELI5)

Imagine the Psychosocial Intelligence Framework as a clinic reception desk. Students fill in a questionnaire (their "form"), hand it to the receptionist (the API), and the receptionist:

1. **Checks the form** — is everything filled in correctly? Any missing fields? (`/api/upload`)
2. **Runs it through the specialist** — predicts a risk score, flags anything concerning (`/api/predict`)
3. **Gives you a typed-up report** — a nice summary with a risk category, the top factors to watch, and a plain-English explanation (`/api/export`)

The clinic **never confuses one patient's file with another's** (session-based upload IDs), and it always stamps every report with a reminder: *"This is educational research support — not a clinical diagnosis."*

---

## Technical Explanation

The backend is a **FastAPI** application (`src/api/main.py`) that exposes a stateless REST API over HTTP. State is held in an in-memory `UploadStorage` (TTL-evicted sessions). All I/O-heavy operations (file parse, inference, retraining) are separated into dedicated modules so the API layer stays thin.

### Architecture

```
Client → FastAPI (main.py)
              ├── Upload & Validation  (schema.py / storage.py)
              ├── Inference Engine     (inference.py → preprocessor.joblib + champion.joblib)
              ├── Explainability       (explainer.py / local_explain.py / narrative.py)
              ├── Retraining Worker    (retrain.py via BackgroundTasks)
              └── Export               (CSV / Fusion JSON contract)
```

---

## Running the Server

```bash
# From project root (venv activated)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger UI available at: `http://localhost:8000/docs`

> **RESEARCH NOTICE:** The PRI is an academic decision-support indicator for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.

---

## Endpoint Reference with `curl` Examples

### 1. `GET /health`

Health check — confirms the server is alive.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "project": "Psychosocial Intelligence Framework - PRI Component",
  "version": "1.0.0",
  "disclaimer": "RESEARCH NOTICE: ..."
}
```

---

### 2. `GET /api/meta`

Model metadata — version, training date, data hash, `is_synthetic` flag.

```bash
curl http://localhost:8000/api/meta
```

**Response:**
```json
{
  "model_version": "1.0.0",
  "training_date": "2026-10-18",
  "is_synthetic": true,
  "data_hash": "...",
  "champion_regression": "hist_gradient_boosting",
  "champion_classification": "hist_gradient_boosting"
}
```

---

### 3. `GET /api/leaderboard`

Retrieve the model comparison leaderboard with optional filters.

```bash
# All models
curl http://localhost:8000/api/leaderboard

# Filter by track and category
curl "http://localhost:8000/api/leaderboard?track=regression&category=Boosting"
```

**Query Parameters:**

| Parameter    | Type   | Description                                      |
|--------------|--------|--------------------------------------------------|
| `track`      | string | `regression` or `classification`                |
| `category`   | string | `Shallow`, `Ensemble`, `Boosting`, `Deep`, `Meta`|
| `feature_set`| string | `factor_scores` or `raw_items`                  |

---

### 4. `GET /api/models/{name}`

Retrieve metrics, parameters, and plots for a specific model.

```bash
curl http://localhost:8000/api/models/hist_gradient_boosting
```

---

### 5. `GET /api/champion`

Get details on the current champion model (best on validation set).

```bash
curl http://localhost:8000/api/champion
```

---

### 6. `GET /api/reliability`

Psychometric reliability metrics — Cronbach's alpha per factor, alpha-if-item-deleted, inter-factor correlation matrix.

```bash
curl http://localhost:8000/api/reliability
```

---

### 7. `GET /api/shap/global`

Global SHAP feature importance — mean |SHAP| per factor, beeswarm summary data.

```bash
curl http://localhost:8000/api/shap/global
```

---

### 8. `GET /api/eda`

Exploratory Data Analysis summary — descriptive statistics, factor distributions, missing-value rates, class balance.

```bash
curl http://localhost:8000/api/eda
```

---

### 9. `POST /api/upload`

Upload a questionnaire CSV/XLSX/XLS file. Returns a validation report and fuzzy column mapping suggestions. **Does NOT run predictions yet.**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@survey_responses.csv;type=text/csv"
```

**Accepted formats:** `.csv`, `.xlsx`, `.xls`  
**Max file size:** 10 MB (configurable in `config/default.yaml`)

**Response:**
```json
{
  "upload_id": "upl_abc123",
  "filename": "survey_responses.csv",
  "row_count": 120,
  "column_count": 16,
  "detected_columns": ["study_id", "S1", "S2", ...],
  "suggested_mappings": {"Q1_stress": "S1", ...},
  "mapping_confidences": {"Q1_stress": 0.92},
  "validation_report": {
    "status": "PASSED",
    "error_count": 0,
    "warning_count": 2,
    "messages": [...]
  },
  "has_outcomes": false,
  "created_at": "2026-10-18T10:00:00+00:00",
  "expires_at": "2026-10-18T11:00:00+00:00"
}
```

---

### 10. `POST /api/predict`

Run batch inference on a previously uploaded file. Returns per-student PRI scores, risk categories, SHAP explanations, and a cohort summary.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "upl_abc123",
    "confirmed_mapping": {"Q1_stress": "S1"},
    "model_name": null
  }'
```

**Request Body:**

| Field               | Type            | Required | Description                                 |
|---------------------|-----------------|----------|---------------------------------------------|
| `upload_id`         | string          | ✅       | ID from `/api/upload`                       |
| `confirmed_mapping` | dict (str→str)  | ❌       | Override column mapping (defaults to suggested) |
| `model_name`        | string          | ❌       | Override champion model; default: champion  |

**Response (abbreviated):**
```json
{
  "upload_id": "upl_abc123",
  "model_used": "hist_gradient_boosting",
  "students": [
    {
      "study_id": "STU0001",
      "pri_score": 68.4,
      "risk_category": "Moderate",
      "class_probabilities": {"High": 0.22, "Moderate": 0.58, "Low": 0.20},
      "factor_scores": {"stress": 3.7, "motivation": 2.8, ...},
      "top_risk_contributors": [...],
      "top_protective_contributors": [...],
      "narrative_headline": "Psychosocial risk is MODERATE ...",
      "narrative_drivers": "...",
      "narrative_buffers": "..."
    }
  ],
  "cohort_summary": {
    "total_students": 120,
    "high_risk_count": 18,
    "average_pri": 55.3,
    "risk_distribution": {"Low": 60, "Moderate": 42, "High": 18}
  },
  "disclaimer": "RESEARCH NOTICE: ..."
}
```

---

### 11. `POST /api/retrain`

Trigger background retraining on a user-uploaded dataset that contains an outcome column (`attrition_label`, `risk_category`, or `dropout_intention`). Requires `allow_retraining: true` in config.

```bash
curl -X POST http://localhost:8000/api/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "upl_abc123",
    "target_column": "attrition_label"
  }'
```

**Response (202 Accepted):**
```json
{
  "job_id": "job_xyz789",
  "status": "queued",
  "progress_pct": 0.0,
  "message": "Job queued"
}
```

> **Note:** Retraining runs in a background thread and is completely isolated from the primary synthetic benchmark pipeline.

---

### 12. `GET /api/jobs/{job_id}`

Poll the status of a background retraining job.

```bash
curl http://localhost:8000/api/jobs/job_xyz789
```

**Response (while running):**
```json
{
  "job_id": "job_xyz789",
  "status": "running",
  "progress_pct": 65.0,
  "message": "Training benchmark model candidate suite"
}
```

**Response (completed):**
```json
{
  "job_id": "job_xyz789",
  "status": "completed",
  "progress_pct": 100.0,
  "message": "Retraining completed. Champion: hist_gradient_boosting (Macro F1 = 0.74)",
  "results": {
    "champion_model": "hist_gradient_boosting",
    "leaderboard": [...],
    "dataset_type": "user_dataset"
  }
}
```

---

### 13. `GET /api/export/{upload_id}.csv`

Download predictions as a CSV file (requires prior `/api/predict` call).

```bash
curl -o predictions.csv \
  http://localhost:8000/api/export/upl_abc123.csv
```

**CSV Columns:** `study_id`, `pri_score`, `risk_category`, `prob_high`, `prob_moderate`, `prob_low`, `factor_stress`, `factor_motivation`, `factor_belongingness`, `factor_self_efficacy`, `factor_peer_support`

---

### 14. `GET /api/export/fusion/{upload_id}.json`

Export scored records in the standardized Fusion Framework contract format for downstream integration with Academic, Engagement, and Environmental indices.

```bash
curl -o fusion_export.json \
  http://localhost:8000/api/export/fusion/upl_abc123.json
```

**Response:**
```json
{
  "component": "psychosocial",
  "version": "1.0.0",
  "export_timestamp": "2026-10-18T10:05:00+00:00",
  "is_synthetic": true,
  "disclaimer": "RESEARCH NOTICE: ...",
  "records": [
    {
      "study_id": "STU0001",
      "pri": 68.4,
      "category": "Moderate",
      "subscale_scores": {"stress": 3.7, "motivation": 2.8, ...},
      "top_factors": ["stress", "motivation"],
      "confidence": 0.58
    }
  ]
}
```

---

## Configuration

All API behaviour is driven by `config/default.yaml`:

```yaml
api:
  max_upload_size_mb: 10       # Max upload size
  upload_ttl_seconds: 3600     # Session TTL (1 hour)
  allow_retraining: true       # Enable POST /api/retrain
```

---

## Error Codes

| Code | Meaning                                                    |
|------|------------------------------------------------------------|
| 400  | Bad request — invalid file format, missing columns, etc.  |
| 404  | Upload session or job not found (expired or wrong ID)     |
| 413  | File exceeds `max_upload_size_mb` limit                   |
| 403  | Retraining disabled by server configuration               |
| 500  | Internal inference or model error (see server logs)       |

---

## Ethics & Disclaimer

> **RESEARCH NOTICE:** The Psychosocial Risk Index (PRI) is a statistical research and educational decision-support indicator developed for academic research at SLIIT Kandy, Sri Lanka. It is **not** a psychiatric evaluation, clinical psychological assessment, or diagnostic tool. All records are identified solely by pseudonymized Study IDs. PRI outputs must be interpreted by qualified staff in the context of a comprehensive student support framework.

This disclaimer is embedded in **every** `/api/predict` response, every export, and every API metadata endpoint.
