import React from 'react';
import {
  Download,
  FileSpreadsheet,
  ExternalLink,
  Code2,
} from 'lucide-react';
import { api } from '../api/client';
import type { DataSummaryResponse } from '../api/client';

interface ExportPageProps {
  dataSummary?: DataSummaryResponse | null;
}

export const ExportPage: React.FC<ExportPageProps> = () => {
  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <Download className="w-6 h-6 text-indigo-400" />
          Data Templates & Export Center
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Download observation upload templates, system evaluation summaries, and API integration specifications.
        </p>
      </div>

      {/* 2. Key Data & Leaderboard Downloads */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
            <FileSpreadsheet className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white">Full Leaderboard (CSV)</h3>
            <p className="text-xs text-slate-400 mt-1">
              Test set evaluation metrics, composite ranks, and inference latency for all registered models.
            </p>
          </div>
          <a
            href="/leaderboard"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            View Leaderboard JSON
          </a>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
            <Code2 className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white">Upload Template (CSV)</h3>
            <p className="text-xs text-slate-400 mt-1">
              Canonical schema-compliant CSV template containing pre-formatted semester observation examples.
            </p>
          </div>
          <a
            href={api.getTemplateUrl()}
            download="academic_observation_template.csv"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download Template CSV
          </a>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center">
            <ExternalLink className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white">OpenAPI Documentation</h3>
            <p className="text-xs text-slate-400 mt-1">
              Interactive Swagger UI and ReDoc endpoints for system integration and contract validation.
            </p>
          </div>
          <div className="flex gap-2">
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              Swagger UI
            </a>
            <a
              href="http://127.0.0.1:8000/redoc"
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              ReDoc
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};
