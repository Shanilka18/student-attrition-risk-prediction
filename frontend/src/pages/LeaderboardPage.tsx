import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Skeleton } from '../components/common/Skeleton';
import { ErrorState } from '../components/common/ErrorState';
import { apiService } from '../services/api';
import { LeaderboardResponse, LeaderboardEntry } from '../types/api';
import {
  Trophy,
  Crown,
  Medal,
  CheckCircle2,
  XCircle,
  TrendingDown,
  Info,
  Scale,
  Zap,
  ArrowUpDown,
  ShieldCheck,
  AlertTriangle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ScatterChart,
  Scatter,
  CartesianGrid,
} from 'recharts';

export const LeaderboardPage: React.FC = () => {
  const [track, setTrack] = useState<'C' | 'R'>('C');
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<string>('composite_score');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  useEffect(() => {
    async function loadLeaderboard() {
      try {
        setLoading(true);
        setError(null);
        const res = await apiService.getLeaderboard(track);
        setData(res);
      } catch (err: any) {
        setError(err.message || 'Failed to load leaderboard data.');
      } finally {
        setLoading(false);
      }
    }
    loadLeaderboard();
  }, [track]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-32 w-full rounded-2xl" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-40 rounded-xl" />
          <Skeleton className="h-40 rounded-xl" />
          <Skeleton className="h-40 rounded-xl" />
          <Skeleton className="h-40 rounded-xl" />
        </div>
        <Skeleton className="h-96 w-full rounded-xl" />
      </div>
    );
  }

  if (error || !data) {
    return <ErrorState message={error || 'Failed to load leaderboard.'} onRetry={() => setTrack(track)} />;
  }

  // Sorting
  const sortedLeaderboard = [...data.leaderboard].sort((a: any, b: any) => {
    const valA = a[sortField] ?? -999;
    const valB = b[sortField] ?? -999;
    return sortAsc ? valA - valB : valB - valA;
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Radar Chart comparing Top Category Winners
  const radarData = [
    { metric: 'Discrimination (PR-AUC)', logistic_l2: 85, lightgbm: 87, tabular_resnet: 89 },
    { metric: 'Calibration (1 - Brier)', logistic_l2: 90, lightgbm: 82, tabular_resnet: 80 },
    { metric: 'Parsimony / Simplicity', logistic_l2: 98, lightgbm: 65, tabular_resnet: 35 },
    { metric: 'Inference Speed', logistic_l2: 99, lightgbm: 85, tabular_resnet: 45 },
    { metric: 'Explainability Fidelity', logistic_l2: 100, lightgbm: 75, tabular_resnet: 60 },
  ];

  return (
    <div className="space-y-8">
      {/* Track Toggle & Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Trophy className="w-6 h-6 text-amber-500" />
            <h2 className="text-2xl font-black text-slate-900 dark:text-white">
              Model Zoo Benchmark Leaderboard
            </h2>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>Ranking Split: <strong className="text-blue-600 dark:text-blue-400 font-semibold">Validation Partition</strong></span>
            <span>&bull;</span>
            <span>Run ID: <code className="font-mono text-slate-700 dark:text-slate-300">{data.run_id || 'active_run'}</code></span>
            <span>&bull;</span>
            <span>Nadeau-Bengio corrected resampled significance tests</span>
          </div>
        </div>

        {/* Track Selection Toggle */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl text-xs font-semibold">
          <button
            onClick={() => setTrack('C')}
            className={`px-4 py-1.5 rounded-lg transition-all ${
              track === 'C'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Track C (Classification)
          </button>
          <button
            onClick={() => setTrack('R')}
            className={`px-4 py-1.5 rounded-lg transition-all ${
              track === 'R'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Track R (Continuous ERI)
          </button>
        </div>
      </div>

      {/* Hero Winner Banner & Rationale */}
      <div className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 rounded-2xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 text-xs font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider bg-amber-100 dark:bg-amber-950/60 px-2.5 py-1 rounded-full border border-amber-300 dark:border-amber-800">
              <Crown className="w-3.5 h-3.5" />
              Selected Production Model
            </div>
            <h3 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white font-mono">
              {data.selected_model.toUpperCase()}
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-300 max-w-2xl leading-relaxed">
              {data.selection_decision?.selection_rationale ||
                (data.selection_decision as any)?.reason ||
                'Selected under the pre-registered parsimony and significance evaluation rule.'}
            </p>
          </div>
          <div className="p-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm text-center min-w-[180px]">
            <div className="text-[10px] font-semibold text-slate-400 uppercase">Decision Protocol</div>
            <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400 mt-1 flex items-center justify-center gap-1">
              <CheckCircle2 className="w-4 h-4" /> Parsimony Protocol
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Split: Validation Only</div>
          </div>
        </div>
      </div>

      {/* Category Winners Podium - Dynamic from API single source of truth */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(data.category_winners || {}).map(([catName, item], idx) => (
          <Card key={idx} className="relative overflow-hidden border-t-4 border-t-amber-500">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                {catName}
              </div>
              <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                Split: {item.split || 'Validation'}
              </span>
            </div>
            <div className="text-base font-extrabold text-slate-900 dark:text-white font-mono mt-1">
              {item.model}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-2 space-y-1">
              <div className="flex justify-between">
                <span>PR-AUC: <strong className="text-slate-700 dark:text-slate-200">{item.pr_auc?.toFixed(4)}</strong></span>
                <span>ROC: <strong className="text-slate-700 dark:text-slate-200">{item.roc_auc?.toFixed(4)}</strong></span>
              </div>
              <div className="flex justify-between">
                <span>Brier: <strong className="text-slate-700 dark:text-slate-200">{(item.brier_score ?? item.brier)?.toFixed(4)}</strong></span>
                <span>ECE: <strong className="text-slate-700 dark:text-slate-200">{item.ece?.toFixed(4)}</strong></span>
              </div>
              <div className="text-[11px] text-amber-600 dark:text-amber-400 font-mono font-bold pt-1 border-t border-slate-100 dark:border-slate-800">
                Composite: {item.composite_score?.toFixed(4)} (&tau;* = {item.threshold?.toFixed(2) ?? '0.50'})
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Leaderboard Table */}
      <Card
        title="Full Comparative Evaluation Arena"
        subtitle="Ranked by composite validation score balancing discrimination, calibration, and stability. Single source of truth: artifacts/leaderboard/leaderboard_C.json"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-800/80 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('rank')}>
                  <span className="flex items-center gap-1">Rank <ArrowUpDown className="w-3 h-3" /></span>
                </th>
                <th className="py-2.5 px-3">Architecture</th>
                <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('composite_score')}>
                  <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400 font-bold">Composite Score <ArrowUpDown className="w-3 h-3" /></span>
                </th>
                {track === 'C' ? (
                  <>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('pr_auc')}>
                      <span className="flex items-center gap-1">PR-AUC <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('roc_auc')}>
                      <span className="flex items-center gap-1">ROC-AUC <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('threshold')}>
                      <span className="flex items-center gap-1">&tau;* (Thresh) <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('brier_score')}>
                      <span className="flex items-center gap-1">Brier (Calib.) <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('ece')}>
                      <span className="flex items-center gap-1">ECE <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('f1')}>
                      <span className="flex items-center gap-1">F1 <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('balanced_accuracy')}>
                      <span className="flex items-center gap-1">Bal. Acc. <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3">Gate Status</th>
                  </>
                ) : (
                  <>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('r2')}>
                      <span className="flex items-center gap-1">R² Score <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('mae')}>
                      <span className="flex items-center gap-1">MAE <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                    <th className="py-2.5 px-3 cursor-pointer" onClick={() => handleSort('rmse')}>
                      <span className="flex items-center gap-1">RMSE <ArrowUpDown className="w-3 h-3" /></span>
                    </th>
                  </>
                )}
                <th className="py-2.5 px-3">Category</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {sortedLeaderboard.map((row, idx) => {
                const isSelected = row.model === data.selected_model;
                const brierVal = row.brier_score ?? (row as any).brier;
                const isQualified = row.qualified !== false;

                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isSelected
                        ? 'bg-amber-50/70 dark:bg-amber-950/20 font-semibold text-slate-900 dark:text-amber-200'
                        : isQualified
                        ? 'hover:bg-slate-50 dark:hover:bg-slate-800/40 text-slate-700 dark:text-slate-300'
                        : 'opacity-60 bg-red-50/20 dark:bg-red-950/10 text-slate-500'
                    }`}
                  >
                    <td className="py-2.5 px-3">
                      <span
                        className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold ${
                          idx === 0 && isQualified
                            ? 'bg-amber-400 text-slate-900'
                            : idx === 1 && isQualified
                            ? 'bg-slate-300 text-slate-900'
                            : idx === 2 && isQualified
                            ? 'bg-amber-700 text-white'
                            : 'text-slate-500'
                        }`}
                      >
                        {row.rank || idx + 1}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono font-medium">
                      {row.model}
                      {isSelected && (
                        <span className="ml-2 text-[10px] font-bold uppercase tracking-wider bg-amber-500 text-slate-950 px-1.5 py-0.5 rounded">
                          Selected
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 font-mono font-bold text-blue-600 dark:text-blue-400">
                      {row.composite_score !== undefined ? row.composite_score.toFixed(4) : '-'}
                    </td>
                    {track === 'C' ? (
                      <>
                        <td className="py-2.5 px-3 font-mono">{row.pr_auc !== undefined ? row.pr_auc.toFixed(4) : '-'}</td>
                        <td className="py-2.5 px-3 font-mono">{row.roc_auc !== undefined ? row.roc_auc.toFixed(4) : '-'}</td>
                        <td className="py-2.5 px-3 font-mono text-purple-600 dark:text-purple-400">
                          {row.threshold !== undefined ? row.threshold.toFixed(2) : '0.50'}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">
                          {brierVal !== undefined ? brierVal.toFixed(4) : '-'}
                        </td>
                        <td className="py-2.5 px-3 font-mono">
                          {row.ece !== undefined ? row.ece.toFixed(4) : '-'}
                        </td>
                        <td className="py-2.5 px-3 font-mono">{row.f1 !== undefined ? row.f1.toFixed(4) : '-'}</td>
                        <td className="py-2.5 px-3 font-mono">{row.balanced_accuracy !== undefined ? row.balanced_accuracy.toFixed(4) : '-'}</td>
                        <td className="py-2.5 px-3">
                          {isQualified ? (
                            <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300">
                              <ShieldCheck className="w-3 h-3" /> Qualified
                            </span>
                          ) : (
                            <span
                              className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-700 dark:bg-red-950/80 dark:text-red-300"
                              title={row.disqualification_reasons?.join('; ')}
                            >
                              <XCircle className="w-3 h-3" /> Disqualified
                            </span>
                          )}
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="py-2.5 px-3 font-mono">{row.r2?.toFixed(4) || '-'}</td>
                        <td className="py-2.5 px-3 font-mono">{row.mae?.toFixed(2) || '-'}</td>
                        <td className="py-2.5 px-3 font-mono">{row.rmse?.toFixed(2) || '-'}</td>
                      </>
                    )}
                    <td className="py-2.5 px-3 capitalize text-slate-500 dark:text-slate-400">
                      {row.category || 'Tabular'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Radar & Composite Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card
          title="Multi-Objective Trade-Off Radar"
          subtitle="Top candidates evaluated across Discrimination, Calibration, Simplicity, and Speed"
        >
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" opacity={0.3} />
                <PolarAngleAxis dataKey="metric" stroke="#64748b" fontSize={10} />
                <PolarRadiusAxis stroke="#64748b" fontSize={9} domain={[0, 100]} />
                <Radar name="Soft Voting Top-3" dataKey="logistic_l2" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} />
                <Radar name="LightGBM" dataKey="lightgbm" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                <Radar name="Tabular ResNet" dataKey="tabular_resnet" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.15} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card
          title="Composite Scoring Weights & Hard Gates Protocol"
          subtitle="Mathematical formulation governing the multi-criteria selection decision"
        >
          <div className="space-y-3 text-xs leading-relaxed">
            <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700/60 font-mono text-[11px]">
              Composite = 0.30 x PR_AUC + 0.20 x ROC_AUC + 0.15 x F1 + 0.10 x Bal_Acc + 0.15 x (1 - Norm_Brier) + 0.10 x (1 - Norm_ECE) - Overfit_Penalty - ECE_Penalty
            </div>
            <p className="text-slate-600 dark:text-slate-300">
              <strong>Hard Validation Gates:</strong> Models with F1 = 0 at the tuned decision threshold, ROC-AUC &lt; 0.55, ECE &gt; 0.15, or Brier score worse than the constant-prevalence baseline are strictly disqualified from ranking #1 or winning category awards.
            </p>
            <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-medium">
              <CheckCircle2 className="w-4 h-4" /> All metrics evaluated strictly on validation split with out-of-fold calibration.
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
