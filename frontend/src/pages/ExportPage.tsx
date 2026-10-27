import React from 'react';
import {
  Download,
  FileSpreadsheet,
  FileText,
  ImageIcon,
  FolderArchive,
  ExternalLink,
  Code2,
} from 'lucide-react';
import { api } from '../api/client';
import type { DataSummaryResponse } from '../api/client';

interface ExportPageProps {
  dataSummary: DataSummaryResponse | null;
}

export const ExportPage: React.FC<ExportPageProps> = ({ dataSummary }) => {
  const plots = dataSummary?.plots_catalog || [];

  const docs = [
    { name: '01_PROJECT_SCAFFOLD.md', desc: 'Environment initialization and reproducibility standards' },
    { name: '02_DATA_GENERATION.md', desc: 'Latent-variable synthetic cohort generator and Sri Lankan baselines' },
    { name: '03_PREPROCESSING_AND_FEATURES.md', desc: 'Zero-leakage split protocol, dual-paradigm feature engineering' },
    { name: '04_TRAINING_AND_VALIDATION.md', desc: 'Optuna tuning, Platt calibration, recall-safe thresholding' },
    { name: '05_EVALUATION_AND_LEADERBOARD.md', desc: '28-model test evaluation, bootstrap CIs, paired hypothesis tests' },
    { name: '06_EXPLAINABILITY_AND_RISK_INDEX.md', desc: 'TreeSHAP, Integrated Gradients, Academic Risk Index, faithfulness' },
    { name: '07_API.md', desc: 'FastAPI serving architecture, endpoints catalog, and OpenAPI specs' },
    { name: '08_FRONTEND_GUIDE.md', desc: 'Interactive research portal guide and viva demonstration talking points' },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <Download className="w-6 h-6 text-indigo-400" />
          Research Asset Export & Telemetry Repository
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Download evaluation matrices, publication plots, OpenAPI schemas, and complete viva defense documentation.
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
              Test set evaluation metrics, composite ranks, and inference latency for all 28 registered models.
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
              Interactive Swagger UI and ReDoc endpoints for teammate integration and contract validation.
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

      {/* 3. Publication Plots Catalog */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-indigo-400" />
          Publication-Grade Visual Assets (19 High-Resolution Figures)
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {plots.map(p => (
            <div
              key={p.key}
              className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center justify-between text-xs group"
            >
              <div className="truncate mr-2">
                <span className="font-semibold text-slate-200 group-hover:text-indigo-300 transition-colors block truncate">
                  {p.title}
                </span>
                <span className="text-[10px] text-slate-500 font-mono block">
                  {p.category} / {p.filename}
                </span>
              </div>
              <a
                href={p.url_path}
                download={p.filename}
                target="_blank"
                rel="noreferrer"
                className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800"
                title="Download PNG"
              >
                <Download className="w-3.5 h-3.5" />
              </a>
            </div>
          ))}
        </div>
      </section>

      {/* 4. Viva Documentation Directory */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-400" />
          Research Documentation & Viva Defense Guides (/docs)
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {docs.map(d => (
            <div key={d.name} className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs space-y-1">
              <span className="font-mono font-bold text-indigo-300 block">{d.name}</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
