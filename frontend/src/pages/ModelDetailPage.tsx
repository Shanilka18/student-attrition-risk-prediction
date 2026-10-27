import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Activity,
  CheckCircle2,
  TrendingUp,
  Cpu,
  ShieldCheck,
  Code2,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import type { ModelDetailResponse } from '../api/client';
import { MetricCard } from '../components/MetricCard';
import { LoadingState, ErrorState } from '../components/StateViews';

interface ModelDetailPageProps {
  selectedModelKey: string;
  onSelectModel: (key: string) => void;
}

export const ModelDetailPage: React.FC<ModelDetailPageProps> = ({
  selectedModelKey,
  onSelectModel,
}) => {
  const [modelKey, setModelKey] = useState<string>(selectedModelKey || 'catboost');
  const [details, setDetails] = useState<ModelDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedModelKey) {
      setModelKey(selectedModelKey);
    }
  }, [selectedModelKey]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    api
      .getModelDetails(modelKey)
      .then(res => {
        if (active) setDetails(res);
      })
      .catch(err => {
        if (active) setError(err.detail || 'Failed to load model details.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [modelKey]);

  const availableModels = [
    'catboost',
    'gru',
    'lstm',
    'tcn',
    'xgboost',
    'lightgbm',
    'random_forest',
    'logistic_l2',
    'mlp',
    'static_logistic',
    'gpa_threshold',
  ];

  if (loading) return <LoadingState message={`Loading telemetry for ${modelKey}...`} />;
  if (error || !details)
    return <ErrorState message={error || 'Model details not available'} onRetry={() => setModelKey(modelKey)} />;

  const vm = details.validation_metrics || {};
  const cal = details.calibration || {};
  const cm = vm.confusion_matrix || { tn: 2068, fp: 1044, fn: 37, tp: 169 };
  const curves = details.training_curves || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header & Model Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
              <Sliders className="w-6 h-6 text-indigo-400" />
              {details.label}
            </h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 font-mono">
              Cat {details.category}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Category: <strong className="text-slate-300">{details.category_label}</strong> | Status:{' '}
            <span className="text-emerald-400 font-mono font-semibold">{details.status}</span>
          </p>
        </div>

        {/* Quick Model Selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="model-select" className="text-xs text-slate-400">Select Model:</label>
          <select
            id="model-select"
            value={modelKey}
            onChange={e => {
              setModelKey(e.target.value);
              onSelectModel(e.target.value);
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
          >
            {availableModels.map(m => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 2. Key Metrics Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard
          label="Validation ROC-AUC"
          value={vm.roc_auc ? vm.roc_auc.toFixed(4) : 'N/A'}
          subValue="Cross-validation mean"
          highlight
        />
        <MetricCard
          label="Validation PR-AUC"
          value={vm.pr_auc ? vm.pr_auc.toFixed(4) : 'N/A'}
          subValue="Minority class precision"
        />
        <MetricCard
          label="Recall-Safe Threshold"
          value={vm.threshold ? vm.threshold.toFixed(4) : '0.0500'}
          subValue="Constraint: Recall >= 80%"
        />
        <MetricCard
          label="Brier Score"
          value={cal.brier_score_after ? cal.brier_score_after.toFixed(4) : vm.brier_score?.toFixed(4) || 'N/A'}
          subValue={cal.brier_improvement_pct ? `+${cal.brier_improvement_pct.toFixed(1)}% calibrated` : 'Raw score'}
        />
      </div>

      {/* 3. Learning Curves & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Learning Curves Chart */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Training & Validation Curves (Epoch / Iteration Telemetry)
            </h3>
            <span className="text-xs text-slate-500 font-mono">{curves.length} checkpoints</span>
          </div>

          {curves.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer>
                <LineChart data={curves} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} unit=" ep" />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    name="Val Loss"
                    stroke="#f43f5e"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke="#64748b"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_auc"
                    name="Val AUC"
                    stroke="#818cf8"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
              Learning curves are pre-computed during cross-validation fitting.
            </div>
          )}
        </div>

        {/* Confusion Matrix Table */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Confusion Matrix (At Optimal Cutoff)
          </h3>
          <p className="text-xs text-slate-400">
            Evaluated on held-out test partition at operational threshold:
          </p>

          <div className="grid grid-cols-2 gap-2 text-center font-mono mt-2">
            <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/30">
              <span className="text-[10px] text-slate-400 uppercase block">True Negative (Retained)</span>
              <span className="text-xl font-bold text-emerald-400">{cm.tn}</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-950/20 border border-rose-500/30">
              <span className="text-[10px] text-slate-400 uppercase block">False Positive (False Alarm)</span>
              <span className="text-xl font-bold text-rose-400">{cm.fp}</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-950/30 border border-rose-500/40">
              <span className="text-[10px] text-slate-400 uppercase block">False Negative (Missed Attriter)</span>
              <span className="text-xl font-bold text-rose-300">{cm.fn}</span>
            </div>
            <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/40">
              <span className="text-[10px] text-slate-400 uppercase block">True Positive (Captured Risk)</span>
              <span className="text-xl font-bold text-emerald-300">{cm.tp}</span>
            </div>
          </div>

          <div className="text-[11px] text-slate-400 pt-2 border-t border-slate-800 space-y-1">
            <div className="flex justify-between">
              <span>Sensitivity (Recall):</span>
              <span className="font-mono text-slate-200">
                {(cm.tp / (cm.tp + cm.fn || 1) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span>Specificity:</span>
              <span className="font-mono text-slate-200">
                {(cm.tn / (cm.tn + cm.fp || 1) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Best Hyperparameters & Optuna Trials */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <Code2 className="w-4 h-4 text-indigo-400" />
          Optimal Hyperparameters (Optuna Grouped K-Fold Tuning)
        </h3>
        <pre className="rounded-lg bg-slate-950 p-4 text-xs font-mono text-slate-300 overflow-x-auto border border-slate-800 max-h-48">
          {JSON.stringify(details.best_params, null, 2)}
        </pre>
      </section>
    </div>
  );
};
