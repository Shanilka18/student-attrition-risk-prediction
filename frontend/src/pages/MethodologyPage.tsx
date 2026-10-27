import React from 'react';
import {
  BookOpen,
  ShieldCheck,
  AlertTriangle,
  FileCode,
  Layers,
  HelpCircle,
  Lock,
  GitBranch,
} from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  const glossary = [
    {
      term: 'ROC-AUC',
      fullName: 'Area Under Receiver Operating Characteristic',
      desc: 'Measures the probability that a randomly chosen attriting student receives a higher risk score than a retained student across all classification thresholds.',
    },
    {
      term: 'PR-AUC',
      fullName: 'Area Under Precision-Recall Curve',
      desc: 'The gold standard metric for severe class imbalance (~6.4% attrition prevalence). Focuses exclusively on minority class identification without being inflated by true negatives.',
    },
    {
      term: 'Brier Score',
      fullName: 'Mean Squared Probability Error',
      desc: 'Measures the accuracy of probabilistic predictions: (1/N) * sum((p_i - y_i)^2). Lower is better. Calibrated down from 0.1869 to 0.0527 via Platt scaling.',
    },
    {
      term: 'ECE',
      fullName: 'Expected Calibration Error',
      desc: 'The weighted difference between predicted confidence and observed empirical frequency across 10 probability bins. Measures whether an 80% risk score truly corresponds to an 80% attrition rate.',
    },
    {
      term: 'F2-Score',
      fullName: 'Recall-Weighted F-Measure (beta = 2)',
      desc: 'Places twice as much penalty on false negatives (missed at-risk students) than on false alarms, reflecting the ethical imperative in educational retention.',
    },
    {
      term: 'Prognostic Lead Time',
      fullName: 'Intervention Horizon Runway',
      desc: 'The average number of academic semesters between when the early-warning system first crosses its operating threshold and when the student actually withdraws.',
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-indigo-400" />
          Methodology, Scientific Rigor & Ethical Governance
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Detailed documentation of experimental controls, zero-leakage splitting, probability calibration, and ethical triage protocols.
        </p>
      </div>

      {/* 2. Experimental Design Principles */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-1">
            01
          </div>
          <h3 className="font-bold text-sm text-white">Student-Grouped Splitting</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Multi-semester observation rows are partitioned strictly by <code>student_id</code> (70% train, 15% validation, 15% test).
            Row-level random splitting is strictly prohibited to prevent cross-semester memorization leakage.
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-1">
            02
          </div>
          <h3 className="font-bold text-sm text-white">Platt Probability Calibration</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Tree ensembles output uncalibrated leaf margins. Platt sigmoid scaling fitted strictly on validation sets aligns raw outputs
            with true empirical attrition probabilities, reducing Brier error by <strong>71.8%</strong>.
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-1">
            03
          </div>
          <h3 className="font-bold text-sm text-white">Recall-Constrained Thresholding</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Standard 0.5 classification cutoffs fail under 6.4% imbalance. Operating thresholds are selected to guarantee at least <strong>80% Recall</strong> on validation data, ensuring faculty advisors capture the vulnerable cohort.
          </p>
        </div>
      </section>

      {/* 3. Metrics Glossary */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-indigo-400" />
          Evaluation Metrics Glossary & Statistical Interpretations
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {glossary.map(item => (
            <div key={item.term} className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
              <div className="flex items-baseline justify-between">
                <span className="font-bold text-xs text-indigo-300 font-mono">{item.term}</span>
                <span className="text-[10px] text-slate-500">{item.fullName}</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 4. Limitations & Ethical Considerations */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-amber-500/30 bg-amber-950/15 p-5 space-y-3">
          <h3 className="text-sm font-bold text-amber-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Research Limitations & Synthetic Data Stamping
          </h3>
          <ul className="space-y-2 text-xs text-slate-300 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Synthetic Proxy Cohort:</strong> All empirical findings are evaluated on a latent-process synthetic cohort. While modeling AR(1) momentum, course load shocks, and realistic noise, findings must be re-validated once genuine institutional records receive IRB clearance.
            </li>
            <li>
              <strong>Course Curricula Differences:</strong> The framework assumes a generalized semester progression; specialized medical or architecture curricula with clinical rotations may exhibit different drop-off dynamics.
            </li>
          </ul>
        </div>

        <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/15 p-5 space-y-3">
          <h3 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
            <Lock className="w-4 h-4" />
            Ethical Governance & Non-Punitive Triage
          </h3>
          <ul className="space-y-2 text-xs text-slate-300 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Supportive vs Punitive Design:</strong> Risk scores are strictly intended for academic support, counseling outreach, and peer tutoring. Automated de-enrollment or scholarship revocation based on predictions is explicitly prohibited.
            </li>
            <li>
              <strong>Strict Anonymization:</strong> All student identifiers are hashed to <code>Participant-ID</code> tokens. Raw names, emails, and demographic attributes are excluded from model training to prevent algorithmic bias.
            </li>
          </ul>
        </div>
      </section>

      {/* 5. Fusion Framework JSON Contract Schema */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <FileCode className="w-4 h-4 text-indigo-400" />
          Teammates' Shared Fusion Framework Integration Contract (Pydantic v2)
        </h3>
        <p className="text-xs text-slate-400">
          The Academic Intelligence Framework exports standardized risk payloads consumed by teammates' Multi-Modal Fusion Engine:
        </p>
        <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-indigo-300 overflow-x-auto max-h-56">
{`{
  "participant_id": "STU_00002",
  "sample_id": "STU_00002_t4",
  "window_semester": 4,
  "academic_risk_index": 21.6,
  "risk_band": "High",
  "top_factors": [
    {
      "name": "delay_mean",
      "display_name": "Average Assignment Delay",
      "direction": "RISK_INCREASING",
      "contribution": 0.312,
      "feature_value": 6.0,
      "unit": "Days"
    }
  ],
  "model_name": "catboost",
  "model_version": "1.0.0",
  "calibrated": true,
  "explanation_text": "High risk (21.6/100): Key drivers indicate that Average Assignment Delay is 6.00 Days...",
  "timestamp_utc": "2026-10-19T14:02:09Z",
  "provenance_tag": "SYNTHETIC"
}`}
        </pre>
      </section>
    </div>
  );
};
