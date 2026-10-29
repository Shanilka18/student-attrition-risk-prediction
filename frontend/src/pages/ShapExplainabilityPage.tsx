import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Skeleton } from '../components/common/Skeleton';
import { ErrorState } from '../components/common/ErrorState';
import { ShapWaterfall, ERIGauge } from '../components/charts/ShapWaterfall';
import { apiService } from '../services/api';
import { ShapGlobalResponse, ShapStudentResponse, ShapStabilityResponse } from '../types/api';
import { Brain, AlertCircle, CheckCircle2, User, Search } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

export const ShapExplainabilityPage: React.FC = () => {
  const [globalShap, setGlobalShap] = useState<ShapGlobalResponse | null>(null);
  const [studentShap, setStudentShap] = useState<ShapStudentResponse | null>(null);
  const [stability, setStability] = useState<ShapStabilityResponse | null>(null);
  const [studentIdInput, setStudentIdInput] = useState<string>('P-00002');
  const [loading, setLoading] = useState(true);
  const [studentLoading, setStudentLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [gData, sData, stabData] = await Promise.all([
          apiService.getShapGlobal(),
          apiService.getShapStudent(studentIdInput),
          apiService.getShapStability(),
        ]);
        setGlobalShap(gData);
        setStudentShap(sData);
        setStability(stabData);
      } catch (err: any) {
        setError(err.message || 'Failed to load SHAP data.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const handleStudentSearch = async () => {
    if (!studentIdInput.trim()) return;
    try {
      setStudentLoading(true);
      const res = await apiService.getShapStudent(studentIdInput.trim());
      setStudentShap(res);
    } catch (err: any) {
      alert(`Student ID '${studentIdInput}' not found in validation pool.`);
    } finally {
      setStudentLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-24 w-full rounded-2xl" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <Skeleton className="h-80 rounded-xl" />
          <Skeleton className="h-80 rounded-xl" />
        </div>
      </div>
    );
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  }

  return (
    <div className="space-y-6">
      {/* Title & Permanent Non-Causal Note */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-500" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">
            Explainable AI (SHAP) & Local Student Attributions
          </h2>
        </div>
        <div className="mt-2.5 p-3 bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900/40 rounded-xl flex items-center gap-2 text-xs text-purple-900 dark:text-purple-300">
          <AlertCircle className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
          <span>
            <strong>Fundamental Principle:</strong> SHAP explains the model's score. It does not prove these factors caused the outcome.
          </span>
        </div>
      </div>

      {/* Cross-Fold Stability KPI Cards */}
      {stability && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Card className="border-l-4 border-l-purple-500">
            <div className="text-xs font-semibold text-slate-500 uppercase">Top-3 Feature Overlap</div>
            <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-1">
              {(stability.top3_overlap * 100).toFixed(0)}%
            </div>
            <div className="text-[11px] text-emerald-600 font-medium mt-1">Identical across CV folds</div>
          </Card>

          <Card className="border-l-4 border-l-blue-500">
            <div className="text-xs font-semibold text-slate-500 uppercase">Top-5 Feature Overlap</div>
            <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-1">
              {(stability.top5_overlap * 100).toFixed(0)}%
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Consistent feature ranking</div>
          </Card>

          <Card className="border-l-4 border-l-emerald-500">
            <div className="text-xs font-semibold text-slate-500 uppercase">Spearman Rank Correlation (ρ)</div>
            <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-1">
              {stability.spearman_rank_correlation.toFixed(3)}
            </div>
            <div className="text-[11px] text-emerald-600 font-medium mt-1">Rankings robust to resampling</div>
          </Card>
        </div>
      )}

      {/* Global SHAP Importance Bar Chart */}
      {globalShap && (
        <Card
          title="Global Feature Importance (Mean |SHAP|)"
          subtitle={`Average absolute contribution across all validation students (Base Log-Odds: ${globalShap.base_value.toFixed(3)})`}
          aboutText="Ranks features by overall impact on the model. Larger bars indicate variables that sway predicted attrition probabilities the most."
        >
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={globalShap.global_importance.slice(0, 8)}
                margin={{ top: 10, right: 30, left: 140, bottom: 5 }}
              >
                <XAxis type="number" stroke="#64748b" fontSize={11} label={{ value: 'Mean |SHAP Value|', position: 'insideBottom', offset: -5, fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" stroke="#64748b" fontSize={10} width={140} />
                <Tooltip
                  formatter={(val: any) => [Number(val).toFixed(4), 'Mean |SHAP|']}
                  contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '11px' }}
                />
                <Bar dataKey="mean_abs_shap" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Per-Student Explorer Section */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <User className="w-5 h-5 text-blue-500" />
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white">Individual Student Explorer</h3>
              <p className="text-[11px] text-slate-500">Query pseudonymous ID to view local additive attributions</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={studentIdInput}
              onChange={(e) => setStudentIdInput(e.target.value)}
              placeholder="e.g. P-00002"
              className="text-xs px-3 py-1.5 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-800 dark:text-slate-200 font-mono focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <button
              onClick={handleStudentSearch}
              className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white text-xs font-semibold rounded-lg flex items-center gap-1 transition-colors"
            >
              <Search className="w-3.5 h-3.5" />
              Inspect
            </button>
          </div>
        </div>

        {studentShap && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-1">
              <ERIGauge score={studentShap.eri_score} band={studentShap.risk_band} />
            </div>
            <div className="lg:col-span-2">
              <ShapWaterfall
                participantId={studentShap.participant_id}
                eriScore={studentShap.eri_score}
                riskBand={studentShap.risk_band}
                mainDrivers={studentShap.main_drivers}
                protectiveFactors={studentShap.protective_factors}
                narrative={studentShap.narrative}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
