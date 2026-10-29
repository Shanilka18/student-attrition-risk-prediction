import React from 'react';
import { Card } from '../common/Card';

interface MetricsSummaryTableProps {
  splits?: Record<string, any>[];
  track?: string;
}

export const MetricsSummaryTable: React.FC<MetricsSummaryTableProps> = ({
  splits = [],
  track = 'C',
}) => {
  const isRegression = track === 'R';

  return (
    <Card
      title="Multi-Split Benchmark Performance (Train / Val / Test)"
      subtitle="Evaluated across sealed data partitions to assess generalization and overfitting gap"
      aboutText="Compares in-fold training, tuning validation, and final held-out test set performance."
    >
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="bg-slate-100 dark:bg-slate-800/80 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
              <th className="py-2.5 px-3">Split</th>
              <th className="py-2.5 px-3">Samples</th>
              {!isRegression ? (
                <>
                  <th className="py-2.5 px-3">Accuracy</th>
                  <th className="py-2.5 px-3">Bal. Acc</th>
                  <th className="py-2.5 px-3">PR-AUC</th>
                  <th className="py-2.5 px-3">ROC-AUC</th>
                  <th className="py-2.5 px-3">F1 Score</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Recall</th>
                  <th className="py-2.5 px-3">Specificity</th>
                  <th className="py-2.5 px-3">Brier Score</th>
                  <th className="py-2.5 px-3">ECE</th>
                </>
              ) : (
                <>
                  <th className="py-2.5 px-3">R² Score</th>
                  <th className="py-2.5 px-3">Adj. R²</th>
                  <th className="py-2.5 px-3">MAE</th>
                  <th className="py-2.5 px-3">RMSE</th>
                  <th className="py-2.5 px-3">MAPE (%)</th>
                  <th className="py-2.5 px-3">Spearman ρ</th>
                </>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {splits.map((row, idx) => (
              <tr
                key={idx}
                className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors"
              >
                <td className="py-2 px-3 font-semibold text-slate-800 dark:text-slate-200 capitalize">
                  <span
                    className={`inline-block w-2 h-2 rounded-full mr-2 ${
                      row.split === 'test'
                        ? 'bg-rose-500'
                        : row.split === 'val'
                        ? 'bg-amber-500'
                        : 'bg-blue-500'
                    }`}
                  />
                  {row.split || `Split ${idx + 1}`}
                </td>
                <td className="py-2 px-3 font-mono">{row.samples?.toLocaleString() || '-'}</td>
                {!isRegression ? (
                  <>
                    <td className="py-2 px-3 font-mono">{row.accuracy ? (row.accuracy * 100).toFixed(1) + '%' : '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.balanced_accuracy?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono font-semibold text-blue-600 dark:text-blue-400">
                      {row.pr_auc?.toFixed(3) || '-'}
                    </td>
                    <td className="py-2 px-3 font-mono">{row.roc_auc?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.f1?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.precision?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.recall?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.specificity?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.brier_score?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.ece?.toFixed(3) || '-'}</td>
                  </>
                ) : (
                  <>
                    <td className="py-2 px-3 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                      {row.r2?.toFixed(3) || '-'}
                    </td>
                    <td className="py-2 px-3 font-mono">{row.adj_r2?.toFixed(3) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.mae?.toFixed(2) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.rmse?.toFixed(2) || '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.mape ? (row.mape * 100).toFixed(1) + '%' : '-'}</td>
                    <td className="py-2 px-3 font-mono">{row.spearman_rho?.toFixed(3) || '-'}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};
