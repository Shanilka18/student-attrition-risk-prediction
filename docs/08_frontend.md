# Phase 8 — Frontend Architecture & Interactive Dashboard

## Simple Explanation (ELI5)

Imagine you are a university academic counselor managing hundreds of students. Rather than reading raw numbers from 15 survey questions or sifting through complex code, you sit down at an interactive **Attrition Intelligence cockpit**:

1. **At a Glance (Overview):** You immediately see how many students are at low, moderate, or high risk, alongside proof that the machine learning models meet rigorous scientific standards.
2. **Leaderboard:** You can compare every algorithm (from simple regressions to complex gradient boosted ensembles) to see which one accurately predicts risk without overfitting.
3. **Score My Data:** You can drop in an Excel or CSV file from a new semester. The system automatically inspects column headers, fixes typos or abbreviations, scores the entire cohort in seconds, and provides a plain-language summary for each student.
4. **Methodology:** Examiners and faculty can review the mathematical foundations and psychometric proofs directly inside the application.

Every screen features a persistent banner: **"Research indicator only, not a clinical diagnosis"**, reminding staff that this system guides proactive academic advising—not medical evaluation.

---

## Technical Architecture

The frontend is implemented in **React 18** with **TypeScript**, powered by **Vite 5** and **Tailwind CSS**. It connects to the FastAPI backend (`http://localhost:8000`) and serves real-time telemetry from serialized model artifacts and cross-validation leaderboards.

### Tech Stack
- **Framework:** React 18 + Vite 5 + TypeScript (strict typing)
- **Styling:** Tailwind CSS (Dark Navy "Attrition Intelligence" theme: `#070b14`, `#0b1120`, `#0f172a`, `#1e293b`)
- **Visualizations:** Recharts (Donuts, Radar Charts, Bar Charts, Scatterplots)
- **Icons:** Lucide React
- **Chart Exports:** `html-to-image` for high-resolution PNG snapshots
- **Markdown Docs:** `react-markdown` + `remark-gfm`
- **Testing:** Vitest 2.1 (`npx vitest run`)

---

## Page-by-Page Feature Matrix

| # | Page | Key Visualizations & Features | Data Source |
|---|------|-------------------------------|-------------|
| 1 | **Overview** | Headline KPI cards (Champions, R², Macro-F1, Sample Size, Alphas), Risk Distribution Donut, 8-phase architecture pipeline stepper | `GET /api/meta`, `GET /api/champion`, `GET /api/eda`, `GET /api/reliability` |
| 2 | **Data & Reliability** | Sample demographics, Nunnally alpha threshold verification table (α ≥ 0.70), item-total correlation diagnostics, inter-factor correlation matrix heatmap | `GET /api/reliability`, `GET /api/eda` |
| 3 | **Leaderboard** | Regression / Classification toggle, category filters (Linear, Ensemble, Boosting, Baseline), sortable multi-metric table with champion crown, accuracy vs. latency scatterplot, top-N bar chart race, and slide-over model inspection drawer | `GET /api/leaderboard`, `GET /api/models/{name}` |
| 4 | **Champion & Evaluation** | Holdout test set verification, overfitting gap audit, confusion matrix and regression diagnostics, proposal baseline comparison table (Logistic Regression vs. Random Forest vs. XGBoost vs. Champion) | `GET /api/champion`, `/artifacts/plots/*` |
| 5 | **Explainability (SHAP)** | Shapley axiom verification audit (additivity, seed stability, ground truth correlation), global factor importance horizontal bar chart, beeswarm and dependence plots | `GET /api/shap/global?track={track}` |
| 6 | **Score My Data** | Drag-and-drop CSV/Excel file upload, schema validation report, fuzzy column mapping with confidence percentages, batch prediction execution, student risk table, export CSV, export Fusion JSON, and background retraining trigger | `POST /api/upload`, `POST /api/predict`, `POST /api/retrain`, `GET /api/jobs/{id}` |
| 7 | **Individual Student View** | Circular PRI score gauge (0–100), 5-factor psychometric radar chart, local additive SHAP waterfall bars, deterministic plain-language academic explanation box | Presets & local SHAP telemetry |
| 8 | **Methodology & Docs** | In-app viva documentation viewer rendering `/docs/*.md` files, multi-modal Fusion Framework architecture diagram illustrating PRI component placement | `GET /api/docs`, `GET /api/docs/{doc_name}` |

---

## Governance & Ethics Enforcement

In strict compliance with **Section 3 & 7 of the project brief (AGENTS.md)**:
1. **Persistent Ethics Banner:** Displayed across the top of all pages:
   > *"RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis."*
2. **Synthetic Data Badges:** Every chart, table, and data card derived from probabilistic copula generation is watermarked with a `SYNTHETIC DATA` pill badge.
3. **Presentation Mode:** Single-click toggle in the sidebar that switches to high-contrast large typography and clean layouts suitable for viva slide presentations or board room displays.

---

## Verification & Execution Guide

### 1. Launching the FastAPI Backend
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 2. Launching the Frontend Dev Server
```bash
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

### 3. Running Frontend Tests
```bash
cd frontend
npx vitest run
```
*Result: 5 passed (100% test pass rate).*

### 4. Building Production Bundle
```bash
cd frontend
npm run build
```
*Result: Clean TypeScript compilation and optimized Vite asset bundling.*

---

## Verification Click-Through Checklist

- [x] **Top Banner:** Amber warning banner persists on all views with the non-clinical diagnostic notice.
- [x] **Overview Page:** 6 KPI cards load live data from API; donut chart displays synthetic risk breakdown.
- [x] **Reliability Page:** Clicking a factor in the reliability table updates the 3-item subscale diagnostics.
- [x] **Leaderboard Track Toggle:** Switching between Regression and Classification updates candidate models and champion crown.
- [x] **Leaderboard Drawer:** Clicking "Inspect" slides out tuned Optuna parameters and diagnostic plot artifacts.
- [x] **Chart Export:** Camera button on chart cards downloads high-resolution PNG image snapshots.
- [x] **Upload & Scoring Flow:** Dropping a questionnaire file validates headers and populates fuzzy mappings.
- [x] **Exports:** Download Scored CSV and Download Fusion JSON buttons trigger valid file downloads.
- [x] **Cohort State Persistence:** Scored survey cohorts are preserved across sidebar tab navigation with sessionStorage synchronization.
- [x] **Cross-Tab Student Inspection:** Dedicated inspection navigation seamlessly transitions from cohort tables into individual student diagnostic gauges, factor radars, and local SHAP waterfalls.
- [x] **Methodology Viewer:** Clicking document titles renders viva-ready markdown with GitHub Flavored Markdown tables.
- [x] **Presentation Mode:** Clicking "Presentation Mode" optimizes typography for projector display.
