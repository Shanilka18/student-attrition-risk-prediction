# 14 — Ethics, Privacy, Fairness, & Limitations: Research Governance

## 1. Simple Explanation (Age-5 Level)

When a doctor gives you a thermometer reading of 38°C, the thermometer does **not** give you medicine or tell you what illness you have—it just tells your doctor that your body is warm so they can look closer and help you.

Our computer tool is like an educational thermometer. It tells university advisors: *"Hey, this student is carrying a heavy backpack of stress right now."* 

It **never** labels a student as "depressed" or "broken," and it **never** makes decisions by itself (like kicking someone out). 

We also make sure nobody can see a student's real name—every student is given a secret code number so their privacy is always protected.

---

## 2. Non-Negotiable Research Disclaimer

The following legal and operational notice is permanently displayed across the frontend dashboard banner, API metadata responses, prediction payloads, and exported JSON/CSV files:

> **RESEARCH NOTICE:**  
> The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy, Sri Lanka. It is **not** a clinical psychological evaluation, diagnostic assessment, or psychiatric determination. All outputs represent statistical indicators intended solely for academic advising and retention support under qualified human institutional oversight.

---

## 3. Data Governance & Student Privacy Protection

1. **Strict Pseudonymization:** No personally identifiable information (PII) such as student full names, National Identity Card (NIC) numbers, home addresses, or phone numbers are accepted by the pipeline. Records are identified exclusively by pseudonymized Study IDs (e.g., `SLIIT_00514`).
2. **Stateless Session Management:** Uploaded survey files stored via `/api/upload` exist in an in-memory session cache with an automatic Time-To-Live (TTL) eviction policy configured to 3,600 seconds (1 hour).
3. **Sealed Holdout Isolation:** To guarantee empirical truthfulness and eliminate p-hacking, the holdout test set (`data/processed/test.csv`) was generated with an automated write-lock file (`test.lock`) and was evaluated exactly once for the champion model bundle (`artifacts/models/final/`).

---

## 4. Subgroup Fairness & Demographic Disparity Audits

In accordance with Phase 5 requirements, the champion models were audited for disparate impact and predictive equity across demographic subgroups on the sealed test set (`artifacts/models/final/classification_champion_bundle.json` lines 106–160):

### Gender Subgroup Disparity Audit

| Subgroup | Sample Size ($n$) | Primary Macro-F1 | High-Risk Recall | Disparity vs. Majority |
| :--- | :---: | :---: | :---: | :---: |
| **Male** | 241 | 0.9566 | 93.94% | Baseline |
| **Female** | 194 | 0.9602 | 97.67% | +0.0036 (F1) / +3.7% (Recall) |
| **Non-Binary** | 13 | 0.4000 | 50.00% | Under-represented minority sample |

> **Fairness Finding & Recommendation:** Performance is exceptionally balanced between male and female undergraduates ($0.9566$ vs. $0.9602$ F1). However, the small non-binary subgroup ($n=13$, reflecting true low-proportion population demographics) exhibited high variance. System recommendation: small demographic subgroups must receive human-in-the-loop qualitative advising rather than automated thresholding.

### Year of Study Longitudinal Equity

| Year of Study | Sample Size ($n$) | Primary Macro-F1 | High-Risk Recall | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Year 1** | 158 | 0.9551 | 95.74% | High Performance (Freshman transition focus) |
| **Year 2** | 116 | 0.9245 | 89.66% | Consistent Generalization |
| **Year 3** | 101 | 0.9760 | 100.00% | Zero missed high-risk students |
| **Year 4** | 73 | 0.9406 | 92.31% | Consistent Generalization |

---

## 5. Known Threats to Scientific Validity

1. **Self-Report Bias (Survey Inaccuracy):** Likert surveys are susceptible to social desirability bias (students under-reporting stress) or momentary affective spikes (students reporting extreme stress immediately before an exam).
2. **Cross-Sectional vs. Longitudinal Limitations:** Current survey instruments capture a single cross-sectional snapshot in time. Real psychological resilience is dynamic and fluctuates across the 14-week academic semester.
3. **Synthetic Generative Bounds:** The primary benchmark dataset was generated using latent Gaussian copulas. While carefully calibrated to Nunnally psychometric standards ($\alpha \in [0.82, 0.85]$), it cannot simulate the full socio-economic complexity of Sri Lankan undergraduate life during economic crises.
