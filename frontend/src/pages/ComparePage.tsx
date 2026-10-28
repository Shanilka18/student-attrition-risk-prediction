import { useMemo } from 'react';
import { useSearchParams, Link, useNavigate } from 'react-router-dom';
import type { EChartsOption } from 'echarts-for-react';
import { ArrowLeft, GitCompare } from 'lucide-react';
import { useCompareModels } from '@/api/queries';
import { Card } from '@/components/ui/Card';
import { ChartCard } from '@/components/ui/ChartCard';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { ErrorState } from '@/components/ui/ErrorState';
import { StatusTag } from '@/components/ui/Tag';
import { cn } from '@/utils/cn';

const COMPARE_COLORS = ['#1F4E46', '#2D5A27', '#8C7355', '#2C5282'];

export function ComparePage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const idsParam = searchParams.get('ids') || '';
  const track = (searchParams.get('track') || 'classification') as 'classification' | 'regression';
  const isClf = track === 'classification';

  const modelIds = useMemo(() => {
    return idsParam.split(',').map((id) => id.trim()).filter(Boolean).slice(0, 4);
  }, [idsParam]);

  const {
    data: compareData,
    isLoading,
    error,
  } = useCompareModels(modelIds, track);

  const models = compareData?.models || [];

  // Metrics to compare side-by-side
  const comparisonRows = useMemo(() => {
    if (models.length === 0) return [];

    if (isClf) {
      return [
        { label: 'Leaderboard Score (F1)', key: 'leaderboard_score', higherBetter: true },
        { label: 'AUC-ROC', key: 'val_auc_roc', higherBetter: true },
        { label: 'Balanced Accuracy', key: 'val_balanced_accuracy', higherBetter: true, format: (v: number) => `${(v * 100).toFixed(1)}%` },
        { label: 'Recall (At-Risk Caught)', key: 'val_recall', higherBetter: true, format: (v: number) => `${(v * 100).toFixed(1)}%` },
        { label: 'Precision (Alert Accuracy)', key: 'val_precision', higherBetter: true, format: (v: number) => `${(v * 100).toFixed(1)}%` },
        { label: 'Matthews Correlation (MCC)', key: 'val_mcc', higherBetter: true },
        { label: 'Generalisation Gap', key: 'generalization_gap', higherBetter: false },
        { label: 'Training Time (s)', key: 'training_time_s', higherBetter: false, format: (v: number) => `${v.toFixed(2)}s` },
        { label: 'Inference Latency (ms)', key: 'inference_time_ms', higherBetter: false, format: (v: number) => `${v.toFixed(1)}ms` },
      ];
    } else {
      return [
        { label: 'Leaderboard Score (R²)', key: 'leaderboard_score', higherBetter: true },
        { label: 'RMSE (Outlier Misses)', key: 'val_rmse', higherBetter: false, format: (v: number) => `${v.toFixed(1)} pts` },
        { label: 'MAE (Typical Error)', key: 'val_mae', higherBetter: false, format: (v: number) => `${v.toFixed(1)} pts` },
        { label: 'Spearman Rank Correlation', key: 'val_spearman_rho', higherBetter: true },
        { label: 'Generalisation Gap', key: 'generalization_gap', higherBetter: false },
        { label: 'Training Time (s)', key: 'training_time_s', higherBetter: false, format: (v: number) => `${v.toFixed(2)}s` },
      ];
    }
  }, [models, isClf]);

  // Compute best value per row for accessible highlighting (bold weight & underline)
  const bestValuePerRow = useMemo(() => {
    const map: Record<string, number> = {};
    comparisonRows.forEach(({ key, higherBetter }) => {
      const values = models
        .map((m) => {
          const v = m.metrics?.[key] ?? (m as any)[key];
          return typeof v === 'number' && !isNaN(v) ? v : null;
        })
        .filter((v): v is number => v !== null);

      if (values.length > 0) {
        map[key] = higherBetter ? Math.max(...values) : Math.min(...values);
      }
    });
    return map;
  }, [comparisonRows, models]);

  // Overlay Generalization Gap option
  const gapCompareOption: EChartsOption = useMemo(() => {
    if (models.length === 0) return {};
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: {
        data: ['Training Score', 'Validation Score'],
        top: 0,
        textStyle: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      grid: { top: 36, right: 20, bottom: 30, left: 44, containLabel: true },
      xAxis: {
        type: 'category',
        data: models.map((m) => m.display_name),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontFamily: 'Public Sans', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
      },
      series: [
        {
          name: 'Training Score',
          type: 'bar',
          data: models.map((m) => m.metrics?.train_f1_mean ?? 0.85),
          itemStyle: { color: '#5B6168' },
          barMaxWidth: 20,
        },
        {
          name: 'Validation Score',
          type: 'bar',
          data: models.map((m) => m.leaderboard_score),
          itemStyle: { color: '#1F4E46' },
          barMaxWidth: 20,
        },
      ],
    };
  }, [models]);

  if (modelIds.length === 0) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-4">
        <div>
          <Link
            to={`/leaderboard?track=${track}`}
            className="inline-flex items-center gap-1.5 text-ui text-muted hover:text-forest transition-colors font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to model leaderboard
          </Link>
        </div>
        <Card className="p-8 text-center space-y-3">
          <GitCompare className="w-8 h-8 text-muted mx-auto" />
          <h2 className="font-serif text-h2 text-ink font-semibold">No models selected for comparison</h2>
          <p className="text-body text-muted max-w-md mx-auto">
            Please navigate to the Model Leaderboard and check the boxes for up to 4 models you wish to inspect side-by-side.
          </p>
          <Button variant="primary" onClick={() => navigate(`/leaderboard?track=${track}`)}>
            Go to leaderboard
          </Button>
        </Card>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-6 max-w-6xl mx-auto p-4">
        <Skeleton className="h-20 w-full rounded" />
        <Skeleton className="h-96 w-full rounded" />
      </div>
    );
  }

  if (error || models.length === 0) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <ErrorState
          title="Comparison failed"
          message="Could not load side-by-side metrics for the selected model identifiers."
          onRetry={() => navigate(`/leaderboard?track=${track}`)}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto" data-testid="compare-page">
      <div>
        <Link
          to={`/leaderboard?track=${track}`}
          className="inline-flex items-center gap-1.5 text-ui text-muted hover:text-forest transition-colors font-medium"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to model leaderboard
        </Link>
      </div>

      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-line pb-4">
        <div>
          <h1 className="font-serif text-page-title text-ink font-semibold">
            Side-by-Side Model Comparison
          </h1>
          <p className="text-body text-muted mt-0.5">
            Comparative evaluation of {models.length} candidate models evaluated on identical 15-fold cross-validation splits.
          </p>
        </div>
        <div className="text-caption text-muted bg-paper px-3 py-1.5 rounded border border-line">
          Best metric in each row highlighted with <strong className="text-ink underline font-bold">bold underline</strong>.
        </div>
      </div>

      {/* Side-by-Side Comparison Table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-body divide-y divide-line border-collapse">
            <thead>
              <tr className="bg-paper text-caption font-semibold text-muted">
                <th className="p-3 w-56">Metric / Property</th>
                {models.map((m, idx) => (
                  <th key={m.model_id} className="p-3 min-w-[180px]">
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-1">
                        <span
                          className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                          style={{ backgroundColor: COMPARE_COLORS[idx % COMPARE_COLORS.length] }}
                        />
                        <span className="font-medium text-ink truncate">{m.display_name}</span>
                      </div>
                      <div className="text-[11px] text-muted capitalize">
                        {m.category.replace('_', ' ')} · <StatusTag status={m.overfit_status} />
                      </div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-ui bg-surface">
              {comparisonRows.map(({ label, key, format }) => {
                const bestVal = bestValuePerRow[key];

                return (
                  <tr key={key} className="hover:bg-paper/40 transition-colors">
                    <td className="p-3 font-medium text-ink bg-paper/20">{label}</td>
                    {models.map((m) => {
                      const rawVal = m.metrics?.[key] ?? (m as any)[key];
                      const isBest =
                        typeof rawVal === 'number' &&
                        typeof bestVal === 'number' &&
                        Math.abs(rawVal - bestVal) < 0.0001;

                      let displayVal = '—';
                      if (typeof rawVal === 'number' && !isNaN(rawVal)) {
                        displayVal = format ? format(rawVal) : rawVal.toFixed(3);
                      }

                      return (
                        <td
                          key={m.model_id}
                          className={cn(
                            'p-3 tabular-nums font-mono',
                            isBest && 'font-bold underline text-ink'
                          )}
                        >
                          {displayVal}
                          {isBest && (
                            <span className="ml-1 text-[10px] text-forest no-underline inline-block font-sans">
                              (Best)
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Overlaid Gap Visualization */}
      <ChartCard
        title="Comparative Generalisation Gaps"
        subtitle="Training fit versus cross-validation score across selected candidates"
        caption="Models maintaining a narrow margin between training and cross-validation scores offer the highest operational reliability."
        option={gapCompareOption}
        height={320}
      />
    </div>
  );
}
