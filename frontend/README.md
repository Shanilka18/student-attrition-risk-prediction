# Academic Intelligence Framework: Interactive Research Portal

**Component**: Academic Intelligence Framework (IT23164680)  
**Project**: Multi-Factor Student Attrition Risk Prediction (`KND_CDAP_IT_05`)  
**Technology Stack**: React 18, Vite 6, TypeScript 5, Tailwind CSS, Recharts, Lucide Icons  

---

## 1. Quick Start Guide

### Prerequisites
- Python 3.10+ with active project virtual environment
- Node.js 18+ and npm 9+

### Option A: One-Command Execution via CLI Orchestrator
```bash
# Terminal 1: Launch FastAPI Backend Server (Port 8000)
python run.py serve

# Terminal 2: Launch React Frontend Development Server (Port 5173)
python run.py frontend
```

### Option B: Direct NPM Commands
```bash
cd frontend

# Install dependencies (if not already installed)
npm install

# Start Vite development server
npm run dev

# Open http://localhost:5173 in your browser
```

### Option C: Production Build & Optimized Preview
```bash
cd frontend

# Compile TypeScript and bundle production assets
npm run build

# Launch optimized preview server with API reverse proxy
npm run preview
```

---

## 2. Portal Architecture & Key Features

This frontend serves as the public and viva-defense research results portal for the SLIIT Centre of Excellence for AI (CoEAI).

1. **Zero Hardcoded Numbers**: All telemetry, metrics, hyperparameters, and predictions are dynamically loaded in real-time from the Phase 8 FastAPI backend (`/health`, `/meta`, `/leaderboard`, `/models/{key}`, `/data/summary`, `/explain/global`, `/predict/upload`).
2. **Persistent Synthetic Data Provenance**: An eye-catching amber banner persists across all views whenever `meta.data_mode === "SYNTHETIC"` to enforce academic honesty pending institutional IRB records clearance.
3. **Presentation Mode**: Click the **Presentation Mode** button in the header bar to optimize the interface for auditorium projector screens (enlarged font scale, amplified chart borders, decluttered contrast).
4. **Dark / Light Theme**: Instant toggle between academic midnight slate (`#0b0f19`) and clean light paper themes.
5. **Interactive File Ingestion**:
   - Drag & drop CSV or Excel files into **PREDICT (Upload)**.
   - Built-in schema validation flags out-of-bounds metrics and duplicate records with exact row numbers.
   - Live download of batch results as CSV or JSON.
6. **Single-Student Trajectory Simulator**: Adjust semester GPA, attendance, exam marks, and assignment delays on-the-fly to test real-time risk index changes and TreeSHAP diagnostic narratives.

---

## 3. Screen Structure

1. **Overview**: Project synopsis, research question, headline metrics, 4-stage pipeline diagram.
2. **Data & Cohort**: Synthetic cohort audit (5,000 students), 4 academic signals dictionary, 7 behavioral archetypes, audit plot gallery.
3. **Leaderboard**: Filterable table of 28 models, category champions card row, composite scores, confidence intervals.
4. **Model Detail**: Deep inspection of any model: parameters, Optuna tuning history, training curves, confusion matrix.
5. **Compare**: Pick 2 to 4 models for side-by-side metric tables and overlay bar charts.
6. **Early Warning**: Prognostic lead-time distribution (~12–15 months runway) and risk separation curves.
7. **Explainability**: Global TreeSHAP feature importance (22 features), faithfulness deletion curve, interactive student explorer.
8. **PREDICT**: Batch CSV/XLSX upload, schema error feedback, single-student trajectory builder.
9. **Methodology & Ethics**: Zero-leakage splitting, Platt calibration, metrics glossary, ethical non-punitive triage, Fusion JSON schema.
10. **Export Center**: Download leaderboard CSVs, template observations, and publication PNGs.
