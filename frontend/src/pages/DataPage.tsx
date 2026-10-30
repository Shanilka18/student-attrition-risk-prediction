import React, { useState } from 'react';
import {
  Database,
  Image as ImageIcon,
} from 'lucide-react';
import type { DataSummaryResponse, PlotCatalogItem } from '../api/client';
import { MetricCard } from '../components/MetricCard';

interface DataPageProps {
  dataSummary: DataSummaryResponse | null;
}

export const DataPage: React.FC<DataPageProps> = ({ dataSummary }) => {
  const [selectedPlot, setSelectedPlot] = useState<PlotCatalogItem | null>(null);

  const auditPlots =
    dataSummary?.plots_catalog?.filter(p => p.category === 'data_audit') || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Page Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <Database className="w-6 h-6 text-indigo-400" />
          Academic Cohort & Trajectory Audit
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Longitudinal distributions, retention metrics, and statistical diagnostics across the student cohort.
        </p>
      </div>

      {/* 2. Headline Dataset Numbers */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Cohort Size"
          value={dataSummary?.total_students ? dataSummary.total_students.toLocaleString() : '5,000'}
          subValue="Enrolled student records"
        />
        <MetricCard
          label="Prefix Samples"
          value={dataSummary?.prefix_prediction_samples ? dataSummary.prefix_prediction_samples.toLocaleString() : '21,985'}
          subValue="Multi-horizon sliding windows (t=2..6)"
        />
        <MetricCard
          label="Base Attrition Rate"
          value={`${dataSummary?.base_attrition_rate_pct || '23.6'}%`}
          subValue="Cohort retention benchmark"
        />
        <MetricCard
          label="Sequence Length"
          value={`${dataSummary?.num_semesters || 8} Semesters`}
          subValue="4-Year Honors Degree Pathway"
        />
      </div>

      {/* 3. Cohort Audit Plots Gallery */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-indigo-400" />
            Statistical Cohort Audit Visualizations
          </h3>
          <span className="text-xs text-slate-400">Click any plot to expand</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {auditPlots.map(plot => (
            <div
              key={plot.key}
              onClick={() => setSelectedPlot(plot)}
              className="group cursor-pointer rounded-xl border border-slate-800 bg-slate-900/40 p-3 hover:border-indigo-500/50 transition-all"
            >
              <div className="aspect-[4/3] rounded-lg bg-slate-950 overflow-hidden mb-2 relative">
                <img
                  src={plot.url_path}
                  alt={plot.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  onError={e => {
                    // Fallback placeholder if image load fails
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
                <div className="absolute inset-0 bg-slate-950/20 group-hover:bg-transparent transition-colors" />
              </div>
              <h4 className="text-xs font-semibold text-slate-200 group-hover:text-indigo-300 transition-colors">
                {plot.title}
              </h4>
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                {plot.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* 7. Plot Zoom Modal */}
      {selectedPlot && (
        <div
          className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setSelectedPlot(null)}
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full p-6 space-y-4 max-h-[90vh] overflow-y-auto"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-base font-bold text-white">{selectedPlot.title}</h3>
              <button
                onClick={() => setSelectedPlot(null)}
                className="px-2.5 py-1 rounded bg-slate-800 text-slate-400 hover:text-white text-xs"
              >
                Close
              </button>
            </div>
            <img
              src={selectedPlot.url_path}
              alt={selectedPlot.title}
              className="w-full rounded-lg border border-slate-800"
            />
            <p className="text-xs text-slate-300">{selectedPlot.description}</p>
          </div>
        </div>
      )}
    </div>
  );
};
