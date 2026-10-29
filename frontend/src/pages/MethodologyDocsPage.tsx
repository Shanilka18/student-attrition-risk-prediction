import React from 'react';
import { Layers } from 'lucide-react';

export const MethodologyDocsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-2 border-b border-navy-800">
        <h2 className="text-xl font-bold text-white tracking-tight">System Architecture & Fusion Integration</h2>
        <p className="text-xs text-slate-400 mt-1">
          Architectural specification for the multi-factor student attrition prediction framework and downstream Fusion layer contracts.
        </p>
      </div>

      {/* Fusion Framework Architecture Blueprint */}
      <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="w-4 h-4 text-indigo-400" />
          <h3 className="font-semibold text-sm text-slate-100">Multi-Factor Student Attrition Architecture (Fusion Context)</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-navy-950 border border-indigo-500/40 shadow-sm relative overflow-hidden">
            <span className="text-[10px] text-indigo-400 uppercase tracking-widest block font-bold">This Component (Ours)</span>
            <div className="font-bold text-slate-100 text-sm mt-1">Psychosocial Risk (PRI)</div>
            <p className="text-[11px] text-slate-400 mt-1">15-Item Likert (Stress, Motivation, Belonging, Efficacy, Peer)</p>
            <div className="mt-2 text-indigo-300 font-bold text-[10px]">Outputs PRI (0-100) + SHAP</div>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component A</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Academic Index</div>
            <p className="text-[11px] text-slate-400 mt-1">GPA trajectories, credit completion, repeating courses</p>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component B</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Engagement Index</div>
            <p className="text-[11px] text-slate-400 mt-1">LMS activity, lecture attendance, forum interaction</p>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component C</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Environmental Index</div>
            <p className="text-[11px] text-slate-400 mt-1">Commute distance, financial aid status, accommodation</p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-indigo-950/20 border border-indigo-500/20 rounded-lg flex items-center justify-between text-xs text-indigo-200">
          <span>Downstream Fusion: Combined multi-modal attrition risk score for university administration</span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/30">
            Strict Standard Export Contract
          </span>
        </div>
      </div>
    </div>
  );
};
