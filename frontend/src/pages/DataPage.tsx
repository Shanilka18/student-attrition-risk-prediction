import React, { useState } from 'react';
import {
  Database,
  FileText,
  TrendingDown,
  Layers,
  Image as ImageIcon,
  ExternalLink,
  Info,
  CheckCircle,
  XCircle,
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
          Synthetic Academic Cohort & Trajectory Audit
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Detailed telemetry, longitudinal distributions, and feature dictionary for the 5,000-student simulated SLIIT cohort.
        </p>
      </div>

      {/* 2. Headline Dataset Numbers */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Cohort Size"
          value={dataSummary?.total_students ? dataSummary.total_students.toLocaleString() : '5,000'}
          subValue="Anonymized student identities"
        />
        <MetricCard
          label="Prefix Samples"
          value={dataSummary?.prefix_prediction_samples ? dataSummary.prefix_prediction_samples.toLocaleString() : '21,985'}
          subValue="Multi-horizon sliding windows (t=2..6)"
        />
        <MetricCard
          label="Base Attrition Rate"
          value={`${dataSummary?.base_attrition_rate_pct || '23.6'}%`}
          subValue="Calibrated to Sri Lankan benchmarks"
        />
        <MetricCard
          label="Sequence Length"
          value={`${dataSummary?.num_semesters || 8} Semesters`}
          subValue="4-Year Honors Degree Pathway"
        />
      </div>

      {/* 3. Motivating Finding: Naive GPA Cutoff Baseline */}
      <section className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-5 space-y-3">
        <div className="flex items-center gap-2 text-amber-400 text-sm font-semibold">
          <TrendingDown className="w-4 h-4" />
          Research Motivation: Limitations of Conventional Point-in-Time GPA Thresholds
        </div>
        <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
          Traditional university retention mechanisms rely on a static rule: flag a student once their GPA drops below 2.0 (Academic Probation).
          When evaluated on our multi-horizon cohort, this static baseline achieved:
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center pt-1">
          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
            <span className="text-xs text-slate-400 block">Static F1-Score</span>
            <span className="text-lg font-bold text-amber-400">0.1200</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
            <span className="text-xs text-slate-400 block">Static Precision</span>
            <span className="text-lg font-bold text-rose-400">0.0600</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
            <span className="text-xs text-slate-400 block">Average Lead Time</span>
            <span className="text-lg font-bold text-slate-300">0.5 Semesters</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
            <span className="text-xs text-slate-400 block">False Alarm Ratio</span>
            <span className="text-lg font-bold text-amber-300">15.6 : 1</span>
          </div>
        </div>
        <p className="text-xs text-slate-400 italic">
          Finding: By the time a student's cross-sectional GPA hits 2.0, attrition is already imminent (only 0.5 semesters of warning).
          In contrast, longitudinal trend and sequence modeling achieves an average lead time of <strong>2.0 to 2.5 semesters</strong> (~12 months).
        </p>
      </section>

      {/* 4. Interactive Data Dictionary */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-400" />
          Core Longitudinal Indicators (4 Academic Signals)
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                <th className="py-2.5 px-3">Variable Name</th>
                <th className="py-2.5 px-3">Domain</th>
                <th className="py-2.5 px-3">Allowed Scale</th>
                <th className="py-2.5 px-3">Derived Longitudinal Features</th>
                <th className="py-2.5 px-3">Prognostic Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans text-slate-300">
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-3 font-mono font-medium text-indigo-300">gpa</td>
                <td className="py-3 px-3">Academic Performance</td>
                <td className="py-3 px-3 font-mono">0.00 – 4.00</td>
                <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                  gpa_slope, gpa_rolling_mean, gpa_volatility, consecutive_declines, gpa_max_drop
                </td>
                <td className="py-3 px-3 text-slate-400">
                  Lagging indicator of course mastery. Captures academic shocks and cumulative trajectory.
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-3 font-mono font-medium text-indigo-300">exam_marks</td>
                <td className="py-3 px-3">Examination Mastery</td>
                <td className="py-3 px-3 font-mono">0.0% – 100.0%</td>
                <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                  exam_mark_slope, exam_mark_mean
                </td>
                <td className="py-3 px-3 text-slate-400">
                  Continuous examination score average, un-binned into GPA bands.
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-3 font-mono font-medium text-indigo-300">attendance_pct</td>
                <td className="py-3 px-3">Course Engagement</td>
                <td className="py-3 px-3 font-mono">0.0% – 100.0%</td>
                <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                  attendance_slope, attendance_mean, attendance_variance, sems_below_att_65
                </td>
                <td className="py-3 px-3 text-slate-400">
                  Leading indicator of disengagement. Accounts for 16.7% of global prognostic signal.
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-3 font-mono font-medium text-indigo-300">assignment_delay_days</td>
                <td className="py-3 px-3">Behavioral Punctuality</td>
                <td className="py-3 px-3 font-mono">0.0 – 45.0 Days</td>
                <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                  delay_trend, delay_mean, delay_max, late_count_total
                </td>
                <td className="py-3 px-3 text-slate-400">
                  Early behavioral creep indicator. Precedes formal GPA drops by 1 to 2 semesters.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. Latent Trajectory Archetypes */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          7 Latent Student Behavioral Archetypes
        </h3>
        <p className="text-xs text-slate-400">
          The synthetic generator simulates realistic cognitive and behavioral trajectories grounded in educational psychology:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {[
            { name: 'Stable High', desc: 'Consistently high ability and resilience. Low attrition rate (< 2%).' },
            { name: 'Stable Average', desc: 'Predictable average performance with stable attendance.' },
            { name: 'Slow Decline', desc: 'Gradual loss of momentum across 3+ semesters; high intervention potential.' },
            { name: 'Sudden Drop', desc: 'External life shock or health event causing steep mid-degree plunge.' },
            { name: 'Recovering', desc: 'Initial drop followed by upward slope and academic rebound.' },
            { name: 'Volatile', desc: 'High semester-to-semester variance in attendance and assignment timeliness.' },
            { name: 'Struggling from Start', desc: 'Persistent low entry scores requiring immediate Year 1 foundational support.' },
          ].map(arch => (
            <div key={arch.name} className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs">
              <span className="font-semibold text-slate-200 block mb-1">{arch.name}</span>
              <span className="text-slate-400 text-[11px] leading-tight block">{arch.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* 6. Cohort Audit Plots Gallery */}
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
