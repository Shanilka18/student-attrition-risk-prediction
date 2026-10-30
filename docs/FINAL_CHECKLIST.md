# Final Project Verification Checklist

**Project**: Multi-Factor Student Attrition Risk Prediction (Academic Intelligence Framework)  
**Project ID**: KND_CDAP_IT_05  
**Candidate**: V. S. H. Gunasekara (IT23164680)  
**Supervisors**: Ms. Chathurika Koswatte, Ms. Anuththara Ekanayake  
**Cluster**: Centre of Excellence for AI (CoEAI), SLIIT  
**Verification Date**: 2026-10-19  

---

## Acceptance Checks Across All Phases

| Phase | Requirement & Acceptance Check | Status | Verification Evidence / Artifact Reference |
|---|---|:---:|---|
| **Phase 1: Setup & Baselines** | Python 3.12+ environment, unified directory layout, reproducible seed control, cross-platform build wrapper (`run.py`, `Makefile`, `make.bat`). | **PASS** | `run.py setup` exits 0; directories verified in `config.py`; seed=42 verified in `backend/tests/test_config.py`. |
| **Phase 2: Synthetic Data Engine** | 5,000-student longitudinal cohort, AR(1) momentum (α=0.65), 7 latent archetypes, non-separable profiles, 15–25% attrition base rate, task prefixing ($t \in [2,6]$). | **PASS** | `data/synthetic/academic_long.csv` (24,260 records, 5,000 students), `task_samples.csv` (21,985 samples, 6.37% sample rate, 23.6% student rate). Audit plots generated in `artifacts/plots/data_audit/`. |
| **Phase 3: Preprocessing & Leakage Safety** | Student-grouped partitioning (70/15/15), zero student ID overlap, training-only fitted scalers & imputers, 22 longitudinal trend features, padded 3D PyTorch tensors. | **PASS** | `data/splits/train_students.json` (3,500), `val_students.json` (750), `test_students.json` (750). Mathematical intersection = $\emptyset$. `scripts/phase11_verify.py` Section 1 passes. `artifacts/models/preprocessor.joblib`. |
| **Phase 4: Model Zoo (28 Models)** | Unified `BaseAcademicModel` interface, `@register` decorator pattern, 5 architectural families (Linear, Bagging, Boosting, Tabular NN, Deep Sequence), masked PyTorch sequence handling. | **PASS** | 28 registered models across Categories A–F (`backend/src/models/`). All implement `fit`, `predict_proba`, `save`, `load`. Unit test suite passes `test_model_zoo.py`. |
| **Phase 5: Training & Tuning Engine** | Optuna Bayesian hyperparameter search isolated strictly to training CV folds, Platt probability calibration, recall-constrained decision thresholding ($\ge 0.80$), multi-seed stability (3 seeds). | **PASS** | Models trained and saved under `artifacts/models/`. Platt calibration yielded 72.06% Brier improvement on validation (0.1907 $\rightarrow$ 0.0533). Multi-seed std $\le 0.004 \le 0.05$ gate. |
| **Phase 6: Evaluation & Leaderboard** | Single test-set evaluation locked with cryptographic SHA-256 fingerprint, student-grouped bootstrap 95% CIs (1,000 resamples), paired hypothesis testing, multi-criteria composite ranking. | **PASS** | `artifacts/metrics/test_evaluation_provenance.json` (SHA-256: `f82970c1...` constant across all evaluations). Leaderboard in `artifacts/leaderboard/leaderboard.json` and `.md`. CatBoost & sequence models evaluated without failure. |
| **Phase 7: Explainability & Risk Index** | Calibrated Academic Risk Index (0–100), 4-tier clinical risk bands anchored to validation cut-points, TreeSHAP attributions, deterministic plain-language narratives, faithfulness deletion test, stability check ($\rho \ge 0.85$), Pydantic Fusion contract. | **PASS** | `artifacts/explanations/faithfulness_deletion_results.json` confirms monotonic risk drop ($13.71 \rightarrow 3.38$, 75.3% reduction). Explanation stability $\rho = 0.9605 \ge 0.85$. `sample_fusion_contract.json` validates against Pydantic schema. |
| **Phase 8: FastAPI Backend** | 12 REST endpoints, OpenAPI 3.1 schema, multipart CSV/Excel upload with row-indexed error reporting, wide-to-long auto-pivoting, zero-PII in-memory streaming, NFR1 latency SLA ($< 2,000$ ms). | **PASS** | `backend/src/api/main.py` verified; test suite (`backend/tests/test_api.py`) runs 25 dedicated endpoint tests; single student inference = 18.2 ms; batch per-student = 9.39 ms. |
| **Phase 9: React Frontend** | Single-page React 18 + Vite 6 + TypeScript + Tailwind CSS portal, 10 functional pages, zero-hardcoding contract consuming live API, high-contrast Presentation Mode, persistent SYNTHETIC data indicator. | **PASS** | `frontend/dist/` builds with 0 errors (`npm run build`). All 10 views rendered; 14 Playwright full-page screenshots preserved in `docs/screenshots/`. |
| **Phase 10: Viva & Presentation Pack** | Complete 10-minute viva script, process walkthrough, 83-question Q&A bank across 14 topic clusters, weaknesses & rated defences, 15-slide presentation deck, 5-minute live demo script with fallback, 30-term 3-level glossary, cheatsheet, Appendix 3 AI disclosure. | **PASS** | All 9 markdown documents in `docs/viva/`, `docs/presentation/`, and `docs/AI_USE_DISCLOSURE.md` verified without placeholders. All numbers cross-cited to artifact files. |
| **Phase 11: Final Hardening** | Full pipeline executed (`python run.py all`), 209 automated tests passing with code coverage report, zero-leakage verified, one-touch test evaluation confirmed, upload security/privacy reviewed, pinned dependencies, handover guide. | **PASS** | `python run.py all` executed cleanly. `pytest` shows **209 passed, 0 failed** (63% total coverage, $\ge 85\%$ on critical inference/model files). `phase11_verify.py` passes all checks. `requirements.txt` pinned to exact installed versions. |

---

## Critical Module Code Coverage Summary

```
Name                                       Stmts   Cover   Status
-----------------------------------------------------------------
backend/src/api/main.py                     126     97%    CRITICAL PASS
backend/src/api/schemas.py                  193     99%    CRITICAL PASS
backend/src/api/service.py                  219     95%    CRITICAL PASS
backend/src/models/sequence_models.py       354     92%    CRITICAL PASS
backend/src/models/tabular_models.py        311     88%    CRITICAL PASS
backend/src/training/calibration.py          57     95%    CRITICAL PASS
backend/src/training/threshold.py            33    100%    CRITICAL PASS
backend/src/explain/risk_index.py            47     98%    CRITICAL PASS
backend/src/explain/fusion_contract.py       58     97%    CRITICAL PASS
-----------------------------------------------------------------
TOTAL TEST SUITE: 209 PASSED, 0 FAILED (100% test success rate)
```

---

## Final Security & Privacy Verification

- **File Upload Extension Check**: Strictly limited to `.csv`, `.xlsx`, `.xls` via allowlist checking.
- **Upload Size Ceiling**: Capped at 10 MB (`MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024`), returns HTTP 413.
- **In-Memory Streaming**: File payloads are ingested via `io.BytesIO`; no raw file bytes are persisted to disk.
- **Zero-PII Logging**: Server logs record only file row counts, batch sizes, and execution times; student IDs and personal features are never written to application logs.
- **Temporary Cache Cleanup**: Batch export sessions expire via a 50-session FIFO eviction policy.

---

## Verdict

**ALL SYSTEM ACCEPTANCE CHECKS: PASS.**  
The Academic Intelligence Framework is fully hardened, documented, reproducible, and ready for viva defense.
