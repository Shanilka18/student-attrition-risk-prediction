import React, { useState } from 'react';
import {
  GitCompare,
  Check,
  Plus,
  Trash2,
  TrendingUp,
  Clock,
  Sparkles,
  ArrowRight,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import type { LeaderboardResponse, LeaderboardModel, ResearchQuestionAnswer } from '../api/client';

interface ComparePageProps {
  leaderboard: LeaderboardResponse | null;
  researchAnswer: ResearchQuestionAnswer | null;
}

export const ComparePage: React.FC<ComparePageProps> = ({
  leaderboard,
  researchAnswer,
}) => {
  const models = leaderboard?.models || [];

  // Default comparison: Champion CatBoost, Proposal Sequence GRU, Proposal LSTM, and Naive Static Snapshot
  const [selectedKeys, setSelectedKeys] = useState<string[]>([
    'catboost',
    'gru',
    'lstm',
    'gpa_threshold',
  ]);

  const toggleModel = (key: string) => {
    if (selectedKeys.includes(key)) {
      if (selectedKeys.length > 2) {
        setSelectedKeys(selectedKeys.filter(k => k !== key));
      }
    } else {
      if (selectedKeys.length < 4) {
        setSelectedKeys([...selectedKeys, key]);
      }
    }
  };

  const selectedModels = selectedKeys
    .map(key => models.find(m => m.model_key === key))
    .filter(Boolean) as LeaderboardModel[];

  // Prepare data for bar chart comparison
  const chartData = [
    {
      metric: 'ROC-AUC',
      ...Object.fromEntries(selectedModels.map(m => [m.label, m.test_roc_auc])),
    },
    {
      metric: 'PR-AUC',
      ...Object.fromEntries(selectedModels.map(m => [m.label, m.test_pr_auc])),
    },
    {
      metric: 'F2-Score',
      ...Object.fromEntries(selectedModels.map(m => [m.label, m.test_f2_score])),
    },
    {
      metric: 'Lead Time (x10 sems)',
      ...Object.fromEntries(selectedModels.map(m => [m.label, (m.test_lead_time || 0) / 10])),
    },
  ];

  const colors = ['#818cf8', '#38bdf8', '#34d399', '#f87171'];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <GitCompare className="w-6 h-6 text-indigo-400" />
          Multi-Model Comparative Evaluation
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Direct head-to-head comparison between sequence architectures, boosting baselines, and point-in-time threshold rules.
        </p>
      </div>

      {/* 2. Model Selection Pills */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300 uppercase tracking-wider">
            Select 2 to 4 Models to Compare:
          </span>
          <span className="text-slate-400 font-mono">
            {selectedKeys.length} / 4 Selected
          </span>
        </div>

        <div className="flex flex-wrap gap-2">
          {models.slice(0, 12).map(m => {
            const isSelected = selectedKeys.includes(m.model_key);
            return (
              <button
                key={m.model_key}
                onClick={() => toggleModel(m.model_key)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  isSelected
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-sm shadow-indigo-600/40'
                    : 'bg-slate-950/60 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {isSelected ? <Check className="w-3.5 h-3.5" /> : <Plus className="w-3.5 h-3.5 text-slate-500" />}
                <span>{m.label}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* 3. Side-by-Side Comparison Table */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-lg shadow-black/20">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-3 px-4">Evaluation Metric</th>
                {selectedModels.map((m, idx) => (
                  <th key={m.model_key} className="py-3 px-4 text-right">
                    <span className="font-bold text-slate-200 block" style={{ color: colors[idx % colors.length] }}>
                      {m.label}
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono lowercase block">
                      cat {m.category}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">ROC-AUC (Test)</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right font-bold text-slate-100">
                    {m.test_roc_auc?.toFixed(4)}
                  </td>
                ))}
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">PR-AUC (Minority Class)</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right text-indigo-300">
                    {m.test_pr_auc?.toFixed(4)}
                  </td>
                ))}
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">F2-Score (Safety Weighted)</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right text-slate-200">
                    {m.test_f2_score?.toFixed(4)}
                  </td>
                ))}
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">Brier Score (Calibration)</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right text-slate-400">
                    {m.test_brier_score?.toFixed(4)}
                  </td>
                ))}
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">Early Warning Lead Time</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right text-emerald-400 font-semibold">
                    {m.test_lead_time?.toFixed(1)} Semesters
                  </td>
                ))}
              </tr>
              <tr className="hover:bg-slate-800/20">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">Inference Latency</td>
                {selectedModels.map(m => (
                  <td key={m.model_key} className="py-2.5 px-4 text-right text-slate-400">
                    {m.inference_time_ms ? `${m.inference_time_ms.toFixed(1)} ms` : '< 10 ms'}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 4. Visual Comparison Bar Chart */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-indigo-400" />
          Comparative Metric Overview
        </h3>

        <div className="h-72 w-full">
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="metric" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              {selectedModels.map((m, idx) => (
                <Bar
                  key={m.model_key}
                  dataKey={m.label}
                  fill={colors[idx % colors.length]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
};
