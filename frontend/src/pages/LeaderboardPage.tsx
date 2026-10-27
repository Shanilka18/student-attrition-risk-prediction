import React, { useState, useMemo } from 'react';
import {
  Trophy,
  Search,
  Filter,
  ArrowUpDown,
  Sparkles,
  CheckCircle,
  HelpCircle,
  Clock,
  Zap,
  ArrowRight,
} from 'lucide-react';
import type { LeaderboardResponse, LeaderboardModel } from '../api/client';

interface LeaderboardPageProps {
  leaderboard: LeaderboardResponse | null;
  onSelectModel: (modelKey: string) => void;
}

export const LeaderboardPage: React.FC<LeaderboardPageProps> = ({
  leaderboard,
  onSelectModel,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [sortKey, setSortKey] = useState<keyof LeaderboardModel>('composite_rank');
  const [sortAscending, setSortAscending] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const categories = [
    { id: 'ALL', label: 'All Models (28)' },
    { id: 'A', label: 'Cat A: Linear Baselines' },
    { id: 'B', label: 'Cat B: Trees & Ensembles' },
    { id: 'C', label: 'Cat C: Boosting (Proposal Baseline)' },
    { id: 'D', label: 'Cat D: Tabular NN' },
    { id: 'E', label: 'Cat E: Deep Sequence (Proposal)' },
    { id: 'F', label: 'Cat F: Static Snapshot' },
  ];

  // Filter and sort models
  const filteredModels = useMemo(() => {
    if (!leaderboard?.models) return [];

    let list = [...leaderboard.models];

    // Filter by Category
    if (selectedCategory !== 'ALL') {
      list = list.filter(m => m.category === selectedCategory);
    }

    // Filter by Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        m => m.label.toLowerCase().includes(q) || m.model_key.toLowerCase().includes(q)
      );
    }

    // Sort
    list.sort((a, b) => {
      let aVal = a[sortKey] ?? 0;
      let bVal = b[sortKey] ?? 0;

      if (typeof aVal === 'string') aVal = (aVal as string).toLowerCase();
      if (typeof bVal === 'string') bVal = (bVal as string).toLowerCase();

      if (aVal < bVal) return sortAscending ? -1 : 1;
      if (aVal > bVal) return sortAscending ? 1 : -1;
      return 0;
    });

    return list;
  }, [leaderboard, selectedCategory, sortKey, sortAscending, searchQuery]);

  const handleSort = (key: keyof LeaderboardModel) => {
    if (sortKey === key) {
      setSortAscending(!sortAscending);
    } else {
      setSortKey(key);
      // For ranks or brier (lower is better), default ascending. For AUC/F1 (higher is better), default descending.
      setSortAscending(['composite_rank', 'overall_rank', 'test_brier_score', 'test_ece', 'inference_time_ms'].includes(key));
    }
  };

  const catChamps = leaderboard?.category_champions || {};

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
              <Trophy className="w-6 h-6 text-amber-400" />
              Comprehensive Model Benchmark Leaderboard
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Multi-objective evaluation across 28 registered models on held-out test partitions. Click any model to inspect full parameters and training curves.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-mono">
              Composite Weights: PR-AUC 25% | ROC 20% | F2 20% | Brier 15% | ECE 10% | Lead 10%
            </span>
          </div>
        </div>
      </div>

      {/* 2. Best in Each Category Cards Row */}
      <section className="space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Category Champions (Top Performer Per Architectural Paradigm)
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(catChamps).map(([catKey, champ]) => (
            <div
              key={catKey}
              onClick={() => onSelectModel(champ.model_key)}
              className="cursor-pointer rounded-xl border border-slate-800 bg-slate-900/60 p-4 hover:border-indigo-500/50 hover:bg-slate-900 transition-all flex items-center justify-between group"
            >
              <div>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  Category {catKey}
                </span>
                <h4 className="font-bold text-sm text-slate-100 group-hover:text-indigo-400 transition-colors mt-1">
                  {champ.label}
                </h4>
                <div className="flex items-center gap-3 text-xs text-slate-400 mt-2 font-mono">
                  <span>ROC-AUC: <strong className="text-slate-200">{champ.test_roc_auc?.toFixed(4)}</strong></span>
                  <span>PR-AUC: <strong className="text-slate-200">{champ.test_pr_auc?.toFixed(4)}</strong></span>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
            </div>
          ))}
        </div>
      </section>

      {/* 3. Controls & Filter Bar */}
      <section className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
        {/* Category Tabs */}
        <div className="flex gap-1.5 overflow-x-auto no-scrollbar pb-1">
          {categories.map(cat => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                selectedCategory === cat.id
                  ? 'bg-indigo-600 text-white'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative min-w-[240px]">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search model by name or key..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
        </div>
      </section>

      {/* 4. Leaderboard Table */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-lg shadow-black/20">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th
                  onClick={() => handleSort('composite_rank')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200"
                >
                  <span className="flex items-center gap-1">
                    Rank <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('label')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200"
                >
                  <span className="flex items-center gap-1">
                    Model Name <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('category')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200"
                >
                  <span className="flex items-center gap-1">
                    Cat <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_roc_auc')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    ROC-AUC <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_pr_auc')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    PR-AUC <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_f2_score')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    F2 Score <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_brier_score')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    Brier <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_ece')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    ECE <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('test_lead_time')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    Lead Time <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th
                  onClick={() => handleSort('inference_time_ms')}
                  className="py-3 px-3 cursor-pointer hover:text-slate-200 text-right"
                >
                  <span className="flex items-center justify-end gap-1">
                    Latency <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              {filteredModels.map((m, idx) => {
                const isChampion = m.model_key === leaderboard?.champion || m.composite_rank === 1.0;
                const isProposal = m.is_proposal_model;

                return (
                  <tr
                    key={m.model_key}
                    onClick={() => onSelectModel(m.model_key)}
                    className="hover:bg-slate-800/40 cursor-pointer transition-colors group"
                  >
                    <td className="py-3 px-3">
                      {isChampion ? (
                        <span className="inline-flex items-center gap-1 font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/30">
                          <Trophy className="w-3 h-3" /> #{idx + 1}
                        </span>
                      ) : (
                        <span className="text-slate-400 pl-2">#{idx + 1}</span>
                      )}
                    </td>

                    <td className="py-3 px-3 font-sans">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-100 group-hover:text-indigo-400 transition-colors">
                          {m.label}
                        </span>
                        {isChampion && (
                          <span className="text-[10px] uppercase font-bold px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            Overall Champion
                          </span>
                        )}
                        {isProposal && (
                          <span className="text-[10px] uppercase font-semibold px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                            Proposal Model
                          </span>
                        )}
                      </div>
                      <span className="text-[11px] text-slate-500 font-mono">{m.model_key}</span>
                    </td>

                    <td className="py-3 px-3">
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 text-[11px]">
                        Cat {m.category}
                      </span>
                    </td>

                    <td className="py-3 px-3 text-right font-bold text-slate-100">
                      {m.test_roc_auc?.toFixed(4)}
                    </td>

                    <td className="py-3 px-3 text-right text-indigo-300">
                      {m.test_pr_auc?.toFixed(4)}
                    </td>

                    <td className="py-3 px-3 text-right text-slate-200">
                      {m.test_f2_score?.toFixed(4)}
                    </td>

                    <td className="py-3 px-3 text-right text-slate-300">
                      {m.test_brier_score?.toFixed(4)}
                    </td>

                    <td className="py-3 px-3 text-right text-slate-400">
                      {m.test_ece?.toFixed(4)}
                    </td>

                    <td className="py-3 px-3 text-right text-emerald-400 font-semibold">
                      {m.test_lead_time?.toFixed(1)} sems
                    </td>

                    <td className="py-3 px-3 text-right text-slate-400">
                      {m.inference_time_ms ? `${m.inference_time_ms.toFixed(1)} ms` : '< 10 ms'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
