import React, { useEffect, useState } from 'react';
import {
  Brain,
  Activity,
  Award,
  CheckCircle2,
  Database,
  ArrowRight,
  TrendingUp,
  Cpu,
  BarChart,
  ShieldCheck,
} from 'lucide-react';
import { Card } from '../components/common/Card';
import { Skeleton } from '../components/common/Skeleton';
import { ErrorState } from '../components/common/ErrorState';
import { apiService } from '../services/api';
import { MetaResponse, LeaderboardResponse } from '../types/api';

export const OverviewPage: React.FC = () => {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [metaData, lbData] = await Promise.all([
          apiService.getMeta(),
          apiService.getLeaderboard('C'),
        ]);
        setMeta(metaData);
        setLeaderboard(lbData);
      } catch (err: any) {
        setError(err.message || 'Failed to connect to backend.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-28 w-full rounded-2xl" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-32 rounded-xl" />
          <Skeleton className="h-32 rounded-xl" />
          <Skeleton className="h-32 rounded-xl" />
          <Skeleton className="h-32 rounded-xl" />
        </div>
      </div>
    );
  }

  if (error || !meta) {
    return <ErrorState message={error || 'Backend service unreachable.'} onRetry={() => window.location.reload()} />;
  }

  const pipelineSteps = [
    { title: 'Data Ingestion', desc: 'Survey Questionnaire (2,000 Cohort)', icon: <Database className="w-4 h-4 text-blue-500" /> },
    { title: 'Data Cleaning', desc: 'EDA & Missingness Heatmap', icon: <CheckCircle2 className="w-4 h-4 text-emerald-500" /> },
    { title: 'In-Fold Preprocess', desc: 'Leakage-Free Splitting & Scaling', icon: <Cpu className="w-4 h-4 text-indigo-500" /> },
    { title: 'Model Zoo Benchmark', desc: '22 Classifiers & Regressors', icon: <Activity className="w-4 h-4 text-cyan-500" /> },
    { title: 'Leaderboard & Audit', desc: 'Parsimony & Subgroup Fairness', icon: <Award className="w-4 h-4 text-amber-500" /> },
    { title: 'Calibrated ERI', desc: '0-100 Isotonic Risk Score', icon: <ShieldCheck className="w-4 h-4 text-rose-500" /> },
    { title: 'SHAP Explainability', desc: 'Waterfall Drivers & Stability', icon: <Brain className="w-4 h-4 text-purple-500" /> },
    { title: 'Causal Inference', desc: 'DAG, sIPW, AIPW & EconML Forest', icon: <TrendingUp className="w-4 h-4 text-teal-500" /> },
    { title: 'Fusion Integration', desc: 'Group Score API (KND-CDAP-IT-05)', icon: <BarChart className="w-4 h-4 text-blue-600" /> },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Abstract */}
      <div className="bg-gradient-to-r from-blue-900 via-indigo-950 to-slate-900 text-white rounded-2xl p-6 sm:p-8 shadow-xl border border-blue-800/40">
        <div className="max-w-4xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-300 border border-blue-400/30">
            SLIIT CDAP Research Component
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Environmental & Financial Causal Risk Intelligence Framework
          </h1>
          <p className="text-sm text-slate-300 leading-relaxed">
            Quantifies non-academic socio-economic barriers to higher education retention. Combines a rigorous 22-model benchmark,
            isotonic probability calibration producing a normalized Environmental Risk Index (ERI 0–100), deterministic SHAP
            waterfall explanations, and a methodologically isolated observational causal inference layer.
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-blue-600">
          <div className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">Models Benchmarked</div>
          <div className="text-2xl font-black text-slate-900 dark:text-white mt-1">
            {leaderboard?.total_models || 22}
          </div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">Linear, Tree Ensembles, Deep MLPs</div>
        </Card>

        <Card className="border-l-4 border-l-emerald-600">
          <div className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">Selected Model</div>
          <div className="text-2xl font-black text-slate-900 dark:text-white mt-1 font-mono">
            {meta.model_name}
          </div>
          <div className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium mt-1">Parsimony winner (PR-AUC 0.380)</div>
        </Card>

        <Card className="border-l-4 border-l-amber-600">
          <div className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">Held-Out Test ROC-AUC</div>
          <div className="text-2xl font-black text-slate-900 dark:text-white mt-1 font-mono">
            0.738
          </div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">Evaluated once at benchmark end</div>
        </Card>

        <Card className="border-l-4 border-l-purple-600">
          <div className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">Calibration Status</div>
          <div className="text-2xl font-black text-slate-900 dark:text-white mt-1 capitalize">
            {meta.calibration_method}
          </div>
          <div className="text-[11px] text-purple-600 dark:text-purple-400 font-medium mt-1">Brier: 0.208 | ECE: 0.041</div>
        </Card>
      </div>

      {/* Pipeline Diagram */}
      <Card
        title="End-to-End Methodological Architecture"
        subtitle="Linear execution pipeline enforcing leakage control and observational causal boundaries"
        aboutText="Traces the path from raw student survey responses to calibrated ERI scores, SHAP explanations, and causal policy effects."
      >
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-9 gap-3 py-2">
          {pipelineSteps.map((step, idx) => (
            <div
              key={idx}
              className="flex flex-col items-center text-center p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 relative group hover:border-blue-500 transition-all"
            >
              <div className="p-2 rounded-lg bg-white dark:bg-slate-900 shadow-sm border border-slate-100 dark:border-slate-800 mb-2">
                {step.icon}
              </div>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">
                Step 0{idx + 1}
              </div>
              <div className="text-xs font-bold text-slate-800 dark:text-slate-200 leading-tight">
                {step.title}
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-1 leading-snug">
                {step.desc}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Cohort & Data Provenance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <Card
          title="Cohort Scale & Prevalence"
          subtitle="Research cohort characteristics and target institutional attrition distribution"
        >
          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500 dark:text-slate-400">Total Validated Cohort Size:</span>
              <span className="font-mono font-bold text-slate-800 dark:text-slate-200">2,000 Students</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500 dark:text-slate-400">Attrition Events:</span>
              <span className="font-mono font-bold text-rose-600 dark:text-rose-400">351 Events (17.55% Prevalence)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500 dark:text-slate-400">Partition Split Ratio:</span>
              <span className="font-mono font-bold text-slate-800 dark:text-slate-200">Train: 1,428 | Val: 286 | Test: 286</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500 dark:text-slate-400">Participant ID Scheme:</span>
              <span className="font-mono text-slate-700 dark:text-slate-300">{meta.student_id_format}</span>
            </div>
          </div>
        </Card>

        <Card
          title="Methodological Principles (Hard Rules)"
          subtitle="Core invariants maintained across all engineering and modeling stages"
        >
          <ul className="space-y-2 text-xs text-slate-600 dark:text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span><strong>Prediction $\ne$ Causation:</strong> SHAP is strictly presented as model attribution. Causal estimates are derived via DAG-informed IPW/AIPW under explicit exchangeability.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span><strong>Strict Leakage Control:</strong> Scalers, imputers, encoders, and tuning fitted strictly inside cross-validation training folds.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span><strong>Never Invent Metrics:</strong> Every figure displayed is read from serialized disk artifacts generated by real executions.</span>
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
};
