import React from 'react';
import {
  Layers,
  ArrowRight,
  TrendingUp,
  Cpu,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  GitBranch,
} from 'lucide-react';
import type { MetaResponse, ResearchQuestionAnswer } from '../api/client';
import { MetricCard } from '../components/MetricCard';

interface OverviewPageProps {
  meta: MetaResponse | null;
  researchAnswer: ResearchQuestionAnswer | null;
  onNavigate: (page: any) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  meta,
  researchAnswer,
  onNavigate,
}) => {
  const champ = meta?.champion_model;
  const seqChamp = meta?.sequence_champion;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Hero Section */}
      <section className="rounded-2xl border border-indigo-500/30 bg-gradient-to-br from-indigo-950/40 via-slate-900/60 to-slate-950 p-6 md:p-10 relative overflow-hidden">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            SLIIT Final Year Research Portal | CoEAI Cluster
          </div>

          <h2 className="text-2xl md:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Multi-Factor Longitudinal Student Attrition Early-Warning Intelligence
          </h2>

          <p className="text-slate-300 text-sm md:text-base leading-relaxed">
            Proactive longitudinal early-warning intelligence framework that monitors multi-semester student engagement,
            attendance, and course progress. Delivering an interpretable <strong>Academic Risk Index (0–100)</strong> with
            actionable clinical triage bands for proactive university counseling.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => onNavigate('leaderboard')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-all shadow-md shadow-indigo-600/30"
            >
              <span>Explore 28-Model Leaderboard</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('predict')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium border border-slate-700 transition-all"
            >
              <span>Upload Cohort & Predict</span>
            </button>
          </div>
        </div>
      </section>

      {/* 2. Headline Performance Metrics Grid */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-indigo-400" />
            Production Champion Telemetry ({champ?.label || 'CatBoost'})
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            Category C (Boosting + Trend Features)
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard
            label="ROC-AUC"
            value={champ?.test_roc_auc ? champ.test_roc_auc.toFixed(4) : '0.8196'}
            subValue="Discriminative power"
            highlight
            tooltip="Area under Receiver Operating Characteristic curve on held-out test partition."
          />
          <MetricCard
            label="PR-AUC"
            value={champ?.test_pr_auc ? champ.test_pr_auc.toFixed(4) : '0.2057'}
            subValue="Minority class precision"
            tooltip="Precision-Recall AUC. Critical for severe higher-education attrition imbalance (~6.4%)."
          />
          <MetricCard
            label="F2-Score"
            value={champ?.test_f2_score ? champ.test_f2_score.toFixed(4) : '0.3200'}
            subValue="Recall-weighted safety"
            tooltip="F-beta (beta=2) placing twice as much weight on recall over false alarms."
          />
          <MetricCard
            label="Brier Score"
            value="0.0527"
            subValue="Platt calibrated"
            tooltip="Mean squared probability error. Scaled down from 0.1869 via Platt scaling."
          />
          <MetricCard
            label="Lead Time"
            value={`${champ?.lead_time_semesters?.toFixed(1) || '2.0'} sems`}
            subValue="~12 month runway"
            tooltip="Average semesters prior to formal academic termination at first risk flag."
          />
          <MetricCard
            label="Inference"
            value="18.2 ms"
            subValue="NFR1 compliant (< 2s)"
            tooltip="Per-student end-to-end latency including TreeSHAP attribution."
          />
        </div>
      </section>

      {/* 3. Clinical Triage & Counseling Protocol */}
      <section className="rounded-xl border border-indigo-500/30 bg-slate-900/60 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm">
            <ShieldCheck className="w-4 h-4" />
            Institutional Clinical Triage & Counseling Protocols
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Empirical Base Rate: 6.4% Attrition Baseline
          </span>
        </div>

        <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
          The framework maps calibrated student attrition probabilities to an intuitive <strong>Academic Risk Index (0–100)</strong>,
          prescribing tiered institutional counseling interventions:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-1">
          {/* Low Risk */}
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-400">
                0.0 – 4.9 ARI
              </span>
              <span className="text-[10px] font-mono text-emerald-400 font-semibold uppercase">Routine</span>
            </div>
            <h4 className="text-sm font-bold text-white">Routine Academic Tracking</h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              Performance within normal retention boundaries. Standard academic progression with no active intervention required.
            </p>
          </div>

          {/* Moderate Risk */}
          <div className="rounded-xl border border-sky-500/30 bg-sky-950/20 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-sky-950 border border-sky-500/30 text-sky-400">
                5.0 – 11.9 ARI
              </span>
              <span className="text-[10px] font-mono text-sky-400 font-semibold uppercase">Monitoring</span>
            </div>
            <h4 className="text-sm font-bold text-white">Preventive Advisory Alert</h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              Early trajectory slip detected. Triggers peer tutoring invitation and course load advisory notification.
            </p>
          </div>

          {/* High Risk */}
          <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-amber-950 border border-amber-500/30 text-amber-400">
                12.0 – 21.9 ARI
              </span>
              <span className="text-[10px] font-mono text-amber-400 font-semibold uppercase">Urgent</span>
            </div>
            <h4 className="text-sm font-bold text-white">Mandatory Advising Session</h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              Noticeable multi-semester decline. Requires 1-on-1 advisor counseling, workload assessment, and personal support review.
            </p>
          </div>

          {/* Critical Risk */}
          <div className="rounded-xl border border-rose-500/30 bg-rose-950/20 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-rose-950 border border-rose-500/30 text-rose-400">
                22.0+ ARI
              </span>
              <span className="text-[10px] font-mono text-rose-400 font-semibold uppercase">Emergency</span>
            </div>
            <h4 className="text-sm font-bold text-white">Dean-Level Intervention</h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              Top 5% extreme attrition risk. Emergency faculty committee review, customized remedial study plan, and student welfare outreach.
            </p>
          </div>
        </div>
      </section>

      {/* 4. End-to-End Operational Workflow */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            4-Stage Academic Intelligence Workflow
          </h3>
          <span className="text-xs text-slate-400 font-mono">Operational Dataflow</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
              01
            </div>
            <h4 className="font-semibold text-slate-200 text-sm">Student Records Ingestion</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Captures semester-wise course attendance, continuous assessment marks, exam grades, and assignment submission timing.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
              02
            </div>
            <h4 className="font-semibold text-slate-200 text-sm">Multi-Semester Trend Analysis</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Automatically detects trajectory velocity—identifying consecutive grade drops, assignment delay creep, and attendance volatility.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
              03
            </div>
            <h4 className="font-semibold text-slate-200 text-sm">Longitudinal AI Risk Evaluation</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Evaluates student trajectory against validated multi-semester machine learning models calibrated for early dropout prevention.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
              04
            </div>
            <h4 className="font-semibold text-slate-200 text-sm">Actionable Counseling Guidance</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Generates calibrated 0–100 risk indices alongside plain-language diagnostic sentences and prioritized counseling actions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
