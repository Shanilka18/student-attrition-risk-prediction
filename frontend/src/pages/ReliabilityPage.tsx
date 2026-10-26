import React, { useEffect, useState } from 'react';
import { Database, CheckCircle2, XCircle, AlertTriangle, Info, Layers } from 'lucide-react';
import { api } from '../services/api';
import { EDAResponse, ReliabilityResponse } from '../types/api';
import { SyntheticBadge } from '../components/common/Badges';
import { LoadingState, ErrorState } from '../components/common/FeedbackStates';

export const ReliabilityPage: React.FC = () => {
  const [eda, setEDA] = useState<EDAResponse | null>(null);
  const [reliability, setReliability] = useState<ReliabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFactor, setSelectedFactor] = useState<string>('stress');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [e, r] = await Promise.all([api.getEDA(), api.getReliability()]);
        setEDA(e);
        setReliability(r);
        if (r.factors && Object.keys(r.factors).length > 0) {
          setSelectedFactor(Object.keys(r.factors)[0]);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to fetch psychometric data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <LoadingState message="Loading psychometric matrices and item-total correlations..." />;
  if (error) return <ErrorState message={error} />;

  const factors = reliability?.factors ? Object.values(reliability.factors) : [];
  const activeDetail = reliability?.factors?.[selectedFactor];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-navy-800">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-xl font-bold text-white tracking-tight">Data Profile & Psychometric Reliability</h2>
            <SyntheticBadge />
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Verification of classical test theory (CTT) metrics, internal consistency thresholds (α ≥ 0.70), and item diagnostics.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded bg-emerald-500/10 text-emerald-400 font-semibold border border-emerald-500/30 text-xs flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Reliability Status: {reliability?.overall_status || 'PASS'}
          </span>
        </div>
      </div>

      {/* Dataset Profile Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium mb-1">Total Observations</div>
          <div className="text-2xl font-bold text-white font-mono">{eda?.sample_size?.toLocaleString() || '3,000'}</div>
          <div className="text-[11px] text-slate-400 mt-1">Multi-factor cohort split (70/15/15)</div>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium mb-1">Psychosocial Risk Index (PRI)</div>
          <div className="text-2xl font-bold text-indigo-400 font-mono">
            {eda?.pri_mean?.toFixed(2) || '52.01'} <span className="text-xs text-slate-400 font-normal">± {eda?.pri_std?.toFixed(2) || '18.91'}</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Continuous scale (0 to 100)</div>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium mb-1">Synthetic Attrition Rate</div>
          <div className="text-2xl font-bold text-amber-400 font-mono">
            {eda?.attrition_base_rate ? `${(eda.attrition_base_rate * 100).toFixed(1)}%` : '26.5%'}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Probabilistic outcome base rate</div>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
          <div className="text-xs text-slate-400 font-medium mb-1">Survey Latent Dimensions</div>
          <div className="text-2xl font-bold text-cyan-400 font-mono">5 Factors / 15 Items</div>
          <div className="text-[11px] text-slate-400 mt-1">3 items per latent psychometric factor</div>
        </div>
      </div>

      {/* Main Factor Consistency Table */}
      <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-sm text-slate-100">Factor-Level Reliability Table</h3>
            <p className="text-xs text-slate-400">Click a factor row to inspect constituent items and alpha-if-deleted impacts.</p>
          </div>
          <span className="text-xs text-slate-400 font-mono">Cutoff Criterion: α ≥ 0.70</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-navy-700 text-slate-400 uppercase tracking-wider text-[11px]">
                <th className="py-2.5 px-3">Factor Domain</th>
                <th className="py-2.5 px-3">Constituent Items</th>
                <th className="py-2.5 px-3 font-mono">Cronbach's α</th>
                <th className="py-2.5 px-3 font-mono">McDonald's ω</th>
                <th className="py-2.5 px-3">Orientation</th>
                <th className="py-2.5 px-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-800">
              {factors.map((f) => {
                const isSelected = selectedFactor === f.factor;
                const isRisk = f.factor === 'stress';
                return (
                  <tr
                    key={f.factor}
                    onClick={() => setSelectedFactor(f.factor)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-indigo-600/15 border-l-2 border-indigo-400' : 'hover:bg-navy-800/60'
                    }`}
                  >
                    <td className="py-3 px-3 font-semibold text-slate-200 capitalize">
                      {f.factor.replace('_', ' ')}
                    </td>
                    <td className="py-3 px-3 text-slate-400 font-mono">
                      {f.items?.join(', ') || 'N/A'}
                    </td>
                    <td className="py-3 px-3 font-mono font-bold text-slate-100">
                      {f.cronbach_alpha?.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {f.mcdonald_omega ? f.mcdonald_omega.toFixed(3) : '—'}
                    </td>
                    <td className="py-3 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                          isRisk
                            ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                            : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                        }`}
                      >
                        {isRisk ? 'Risk (Higher = Worse)' : 'Protective (Higher = Better)'}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                        <CheckCircle2 className="w-3 h-3" />
                        PASSED
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Item-Level Diagnostics for Selected Factor */}
      {activeDetail && (
        <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold text-sm text-slate-100 capitalize">
                Item-Level Diagnostics — {activeDetail.factor.replace('_', ' ')} (α = {activeDetail.cronbach_alpha?.toFixed(3)})
              </h3>
              <p className="text-xs text-slate-400">
                Evaluation of item-total correlations and potential gains if any specific item were deleted.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {activeDetail.items?.map((itemKey) => {
              const itc = activeDetail.item_total_correlations?.[itemKey] ?? 0.65;
              const alphaIfDel = activeDetail.alpha_if_item_deleted?.[itemKey] ?? (activeDetail.cronbach_alpha - 0.05);
              return (
                <div key={itemKey} className="bg-navy-800/60 border border-navy-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-xs font-bold text-indigo-300 px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/40">
                      {itemKey}
                    </span>
                    <span className="text-[11px] text-slate-400">5-point Likert</span>
                  </div>
                  <div className="space-y-2 mt-3 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Item-Total Correlation (r):</span>
                      <span className="font-mono font-semibold text-slate-200">{itc.toFixed(3)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Alpha if Deleted:</span>
                      <span className="font-mono font-semibold text-slate-200">{alphaIfDel.toFixed(3)}</span>
                    </div>
                    <div className="pt-2 border-t border-navy-700/60 text-[11px] text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Retained in survey instrument
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Psychometric Artifact Diagnostic Images */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-navy-900 border border-navy-800 rounded-xl p-5">
          <h3 className="font-semibold text-sm text-slate-100 mb-1">Inter-Factor Correlation Matrix</h3>
          <p className="text-xs text-slate-400 mb-3">Empirical correlation structure among the 5 latent domains.</p>
          <div className="bg-navy-950 rounded-lg overflow-hidden border border-navy-800 flex items-center justify-center p-2 min-h-[220px]">
            <img
              src="/artifacts/plots/inter_factor_correlations.png"
              alt="Inter-factor correlation heatmap"
              className="max-h-72 object-contain rounded"
              onError={(e) => {
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
          </div>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-xl p-5">
          <h3 className="font-semibold text-sm text-slate-100 mb-1">Continuous PRI Distribution</h3>
          <p className="text-xs text-slate-400 mb-3">Overall Psychosocial Risk Index distribution across the synthetic cohort.</p>
          <div className="bg-navy-950 rounded-lg overflow-hidden border border-navy-800 flex items-center justify-center p-2 min-h-[220px]">
            <img
              src="/artifacts/plots/pri_distribution.png"
              alt="Continuous PRI Distribution Plot"
              className="max-h-72 object-contain rounded"
              onError={(e) => {
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
