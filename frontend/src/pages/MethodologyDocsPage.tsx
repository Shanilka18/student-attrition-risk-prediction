import React from 'react';
import { Layers, ArrowDown, Network, Database, Cpu, ShieldCheck } from 'lucide-react';

export const MethodologyDocsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-2 border-b border-navy-800">
        <h2 className="text-xl font-bold text-white tracking-tight">System Architecture & Fusion Integration</h2>
        <p className="text-xs text-slate-400 mt-1">
          Technical specification for the multi-factor student attrition prediction framework (CDAP IT05) and downstream Fusion layer contracts.
        </p>
      </div>

      {/* Multi-Factor Architecture Flow */}
      <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            <h3 className="font-semibold text-sm text-slate-100">Multi-Factor Component Architecture (CDAP IT05)</h3>
          </div>
          <span className="text-[10px] font-mono text-indigo-300 px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/30">
            Modular Federated Structure
          </span>
        </div>

        {/* 4 Multi-Factor Domains */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
          {/* Our Component */}
          <div className="p-4 rounded-xl bg-navy-950 border-2 border-indigo-500/60 shadow-lg relative overflow-hidden flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-indigo-400 uppercase tracking-widest font-bold">This Component (Ours)</span>
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <div className="font-bold text-slate-100 text-sm mt-1.5">Psychosocial Risk (PRI)</div>
              <p className="text-[11px] text-slate-300 mt-1 font-sans">
                15-Item Psychometric Questionnaire measuring Stress, Motivation, Belongingness, Self-Efficacy, and Peer Support.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-navy-800 space-y-1 text-[10px] text-slate-400">
              <div className="flex justify-between">
                <span>Model Champion:</span>
                <span className="text-indigo-300 font-bold">Ridge (R²=0.9908)</span>
              </div>
              <div className="flex justify-between">
                <span>Primary Signal:</span>
                <span className="text-emerald-300 font-bold">PRI (0–100 Scale)</span>
              </div>
              <div className="flex justify-between">
                <span>Explainability:</span>
                <span className="text-amber-300">SHAP Factor Attribution</span>
              </div>
            </div>
          </div>

          {/* Component A */}
          <div className="p-4 rounded-xl bg-navy-950/70 border border-navy-800 flex flex-col justify-between">
            <div>
              <span className="text-[10px] text-slate-400 uppercase tracking-widest block font-bold">Teammate Component A</span>
              <div className="font-bold text-slate-200 text-sm mt-1.5">Academic Index</div>
              <p className="text-[11px] text-slate-400 mt-1 font-sans">
                Cumulative GPA trajectories, credit completion ratios, repeated module counts, and prerequisite failures.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-navy-800 space-y-1 text-[10px] text-slate-400">
              <div className="flex justify-between">
                <span>Data Source:</span>
                <span className="text-slate-300">SIS Registry</span>
              </div>
              <div className="flex justify-between">
                <span>Primary Signal:</span>
                <span className="text-slate-300">Academic Score (0–100)</span>
              </div>
              <div className="flex justify-between">
                <span>Temporal Window:</span>
                <span className="text-slate-300">Semester Term Trajectory</span>
              </div>
            </div>
          </div>

          {/* Component B */}
          <div className="p-4 rounded-xl bg-navy-950/70 border border-navy-800 flex flex-col justify-between">
            <div>
              <span className="text-[10px] text-slate-400 uppercase tracking-widest block font-bold">Teammate Component B</span>
              <div className="font-bold text-slate-200 text-sm mt-1.5">Engagement Index</div>
              <p className="text-[11px] text-slate-400 mt-1 font-sans">
                LMS access frequency, lecture attendance logs, forum discussion interactions, and assignment submission recency.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-navy-800 space-y-1 text-[10px] text-slate-400">
              <div className="flex justify-between">
                <span>Data Source:</span>
                <span className="text-slate-300">Moodle LMS Telemetry</span>
              </div>
              <div className="flex justify-between">
                <span>Primary Signal:</span>
                <span className="text-slate-300">Engagement Score (0–100)</span>
              </div>
              <div className="flex justify-between">
                <span>Temporal Window:</span>
                <span className="text-slate-300">Weekly Pulse Logs</span>
              </div>
            </div>
          </div>

          {/* Component C */}
          <div className="p-4 rounded-xl bg-navy-950/70 border border-navy-800 flex flex-col justify-between">
            <div>
              <span className="text-[10px] text-slate-400 uppercase tracking-widest block font-bold">Teammate Component C</span>
              <div className="font-bold text-slate-200 text-sm mt-1.5">Environmental Index</div>
              <p className="text-[11px] text-slate-400 mt-1 font-sans">
                Commute distance/duration, financial scholarship status, living accommodation stability, and part-time employment hours.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-navy-800 space-y-1 text-[10px] text-slate-400">
              <div className="flex justify-between">
                <span>Data Source:</span>
                <span className="text-slate-300">Student Affairs DB</span>
              </div>
              <div className="flex justify-between">
                <span>Primary Signal:</span>
                <span className="text-slate-300">Environmental Score (0–100)</span>
              </div>
              <div className="flex justify-between">
                <span>Temporal Window:</span>
                <span className="text-slate-300">Annual Intake Profiling</span>
              </div>
            </div>
          </div>
        </div>

        {/* Downstream Synthesis Banner */}
        <div className="mt-4 p-3.5 bg-indigo-950/40 border border-indigo-500/30 rounded-xl flex flex-col md:flex-row items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400 shrink-0" />
            <span className="text-slate-200 font-sans">
              <strong>Downstream Fusion Engine:</strong> Multi-modal weighted stacking layer synthesizing the 4 domain indices into an institutional student attrition probability.
            </span>
          </div>
          <span className="font-mono text-[10px] px-2.5 py-1 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shrink-0">
            Output Contract: JSON Schema v1.0
          </span>
        </div>
      </div>
    </div>
  );
};
