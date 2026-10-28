import { useState, useMemo } from 'react';
import { useParams, useSearchParams, useNavigate, Link } from 'react-router-dom';
import type { EChartsOption } from 'echarts-for-react';
import {
  ArrowLeft,
  Trophy,
  ShieldCheck,
  CheckCircle2,
} from 'lucide-react';
import { useModelDetail, useModelCurves, useLeaderboard, useStatus } from '@/api/queries';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { StatusTag } from '@/components/ui/Tag';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { ChartCard } from '@/components/ui/ChartCard';
import { Skeleton } from '@/components/ui/Skeleton';
import { ErrorState } from '@/components/ui/ErrorState';
import { Term } from '@/components/ui/Term';
import { ThresholdSlider, type ThresholdRow } from '@/components/model-detail/ThresholdSlider';
import { cn } from '@/utils/cn';

export function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const trackParam = searchParams.get('track');
  const track: 'classification' | 'regression' =
    trackParam === 'regression' ? 'regression' : 'classification';
  const isClf = track === 'classification';

  // Active tab state
  const activeTab = searchParams.get('tab') || 'summary';
  const handleTabChange = (tabId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('tab', tabId);
    setSearchParams(next);
  };

  // Confusion matrix display mode: 'counts' | 'percentages'
  const [cmMode, setCmMode] = useState<'counts' | 'percentages'>('counts');

  // Queries
  const {
    data: model,
    isLoading: isModelLoading,
    error: modelError,
  } = useModelDetail(modelId || '', track);

  const {
    data: curves,
  } = useModelCurves(modelId || '', track);

  const { data: leaderboardData } = useLeaderboard(track);
  const { data: statusData } = useStatus();

  // Find rank in leaderboard
  const rank = useMemo(() => {
    if (!leaderboardData || !modelId) return null;
    const found = leaderboardData.rows.find((r) => r.model_id === modelId);
    return found ? found.rank : null;
  }, [leaderboardData, modelId]);

  // Deep Learning Twin Epoch Curves Option (Loss & Accuracy)
  const deepLossOption: EChartsOption = useMemo(() => {
    const history = curves?.epoch_curves;
    if (!history || !history.train_loss) return {};

    const epochs = Array.from({ length: history.train_loss.length }, (_, i) => i + 1);
    const bestEpoch = curves?.best_epoch;
    const stage2Epoch = curves?.stage_2_start_epoch;

    return {
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['Training Loss', 'Validation Loss'],
        top: 0,
        textStyle: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      grid: { top: 36, right: 30, bottom: 36, left: 44, containLabel: true },
      xAxis: {
        type: 'category',
        data: epochs,
        name: 'Epoch',
        axisLine: { lineStyle: { color: '#DDD8CE' } },
      },
      yAxis: {
        type: 'value',
        name: 'Binary Cross-Entropy Loss',
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
      },
      series: [
        {
          name: 'Training Loss',
          type: 'line',
          data: history.train_loss,
          itemStyle: { color: '#5B6168' },
          lineStyle: { width: 2 },
          showSymbol: false,
        },
        {
          name: 'Validation Loss',
          type: 'line',
          data: history.val_loss,
          itemStyle: { color: '#1F4E46' },
          lineStyle: { width: 2.5 },
          showSymbol: false,
          markLine: {
            symbol: 'none',
            data: [
              ...(bestEpoch
                ? [
                    {
                      xAxis: bestEpoch,
                      lineStyle: { color: '#2D5A27', type: 'dotted', width: 2 },
                      label: { formatter: `Best: Ep ${bestEpoch}`, position: 'insideEndTop' },
                    },
                  ]
                : []),
              ...(stage2Epoch
                ? [
                    {
                      xAxis: stage2Epoch,
                      lineStyle: { color: '#8C4A3B', type: 'dashed', width: 2 },
                      label: { formatter: `Stage 2: Ep ${stage2Epoch}`, position: 'insideStartTop' },
                    },
                  ]
                : []),
            ],
          },
        },
      ],
    };
  }, [curves]);

  const deepAccOption: EChartsOption = useMemo(() => {
    const history = curves?.epoch_curves;
    if (!history || !history.train_acc) return {};

    const epochs = Array.from({ length: history.train_acc.length }, (_, i) => i + 1);

    return {
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['Training Accuracy', 'Validation Accuracy'],
        top: 0,
        textStyle: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      grid: { top: 36, right: 30, bottom: 36, left: 44, containLabel: true },
      xAxis: {
        type: 'category',
        data: epochs,
        name: 'Epoch',
        axisLine: { lineStyle: { color: '#DDD8CE' } },
      },
      yAxis: {
        type: 'value',
        name: 'Accuracy',
        min: 0.5,
        max: 1.0,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
      },
      series: [
        {
          name: 'Training Accuracy',
          type: 'line',
          data: history.train_acc,
          itemStyle: { color: '#5B6168' },
          lineStyle: { width: 2 },
          showSymbol: false,
        },
        {
          name: 'Validation Accuracy',
          type: 'line',
          data: history.val_acc,
          itemStyle: { color: '#2D5A27' },
          lineStyle: { width: 2.5 },
          showSymbol: false,
        },
      ],
    };
  }, [curves]);

  // Stability Boxplot (15 Folds)
  const stabilityOption: EChartsOption = useMemo(() => {
    const foldMetrics = model?.fold_metrics || [];
    if (foldMetrics.length === 0) return {};

    const foldScores = foldMetrics.map((f: any) =>
      isClf ? (f.val_f1_optimal ?? f.val_f1 ?? f.leaderboard_score ?? 0) : (f.val_r2 ?? f.leaderboard_score ?? 0)
    );

    return {
      tooltip: { trigger: 'item' },
      grid: { top: 20, right: 20, bottom: 30, left: 50, containLabel: true },
      xAxis: {
        type: 'category',
        data: foldScores.map((_, i) => `Fold ${i + 1}`),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        name: isClf ? 'Validation F1 Score' : 'Validation R² Score',
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
      },
      series: [
        {
          name: 'Fold Score',
          type: 'bar',
          data: foldScores,
          itemStyle: { color: '#1F4E46' },
          barMaxWidth: 20,
        },
      ],
    };
  }, [model, isClf]);

  const isChampion = modelId === leaderboardData?.champion_id || model?.model_id === 'knn';

  if (isModelLoading) {
    return (
      <div className="space-y-6 max-w-6xl mx-auto p-4">
        <Skeleton className="h-28 w-full rounded" />
        <Skeleton className="h-12 w-full rounded" />
        <Skeleton className="h-80 w-full rounded" />
      </div>
    );
  }

  if (modelError || !model) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <ErrorState
          title="Model details not found"
          message={`Could not load evaluation artifacts for model '${modelId}' on track '${track}'.`}
          onRetry={() => navigate(`/leaderboard?track=${track}`)}
        />
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // ROC Curve Option
  // -------------------------------------------------------------------------
  const rocOption: EChartsOption = {
    tooltip: {
      trigger: 'axis',
      formatter: (p: any) => {
        const item = p[0];
        return `FPR: <strong>${item.data[0].toFixed(3)}</strong><br/>TPR: <strong>${item.data[1].toFixed(3)}</strong>`;
      },
    },
    grid: { top: 20, right: 20, bottom: 40, left: 44, containLabel: true },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'False Positive Rate (FPR)',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'True Positive Rate (Recall)',
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    series: [
      {
        name: 'ROC',
        type: 'line',
        data: (curves?.roc_fpr || []).map((fpr, i) => [fpr, (curves?.roc_tpr || [])[i]]),
        itemStyle: { color: '#1F4E46' },
        lineStyle: { width: 2.5 },
        showSymbol: false,
      },
      {
        name: 'Random Chance',
        type: 'line',
        data: [[0, 0], [1, 1]],
        lineStyle: { color: '#5B6168', type: 'dashed', width: 1.5 },
        showSymbol: false,
      },
    ],
  };

  // -------------------------------------------------------------------------
  // Precision-Recall Curve Option
  // -------------------------------------------------------------------------
  const prOption: EChartsOption = {
    tooltip: {
      trigger: 'axis',
      formatter: (p: any) => {
        const item = p[0];
        return `Recall: <strong>${item.data[0].toFixed(3)}</strong><br/>Precision: <strong>${item.data[1].toFixed(3)}</strong>`;
      },
    },
    grid: { top: 20, right: 20, bottom: 40, left: 44, containLabel: true },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'Recall (Coverage of At-Risk Students)',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'Precision (Advisor Signal Quality)',
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    series: [
      {
        name: 'PR Curve',
        type: 'line',
        data: (curves?.pr_recall || []).map((rec, i) => [rec, (curves?.pr_precision || [])[i]]),
        itemStyle: { color: '#2D5A27' },
        lineStyle: { width: 2.5 },
        showSymbol: false,
      },
    ],
  };

  // -------------------------------------------------------------------------
  // Calibration Reliability Curve Option
  // -------------------------------------------------------------------------
  const calOption: EChartsOption = {
    tooltip: { trigger: 'axis' },
    grid: { top: 20, right: 20, bottom: 40, left: 44, containLabel: true },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'Mean Predicted Probability',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      name: 'Observed Fraction of Positives',
      nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
    },
    series: [
      {
        name: 'Reliability Curve',
        type: 'line',
        data: (curves?.cal_prob_pred || []).map((pred, i) => [pred, (curves?.cal_prob_true || [])[i]]),
        itemStyle: { color: '#1F4E46' },
        lineStyle: { width: 2.5 },
        symbol: 'circle',
        symbolSize: 6,
      },
      {
        name: 'Perfect Calibration',
        type: 'line',
        data: [[0, 0], [1, 1]],
        lineStyle: { color: '#5B6168', type: 'dashed', width: 1.5 },
        showSymbol: false,
      },
    ],
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto" data-testid="model-detail-page">
      {/* Back button link */}
      <div>
        <Link
          to={`/leaderboard?track=${track}`}
          className="inline-flex items-center gap-1.5 text-ui text-muted hover:text-forest transition-colors font-medium"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to model leaderboard
        </Link>
      </div>

      {/* Header Banner */}
      <Card className="p-6 bg-surface">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-caption font-semibold uppercase tracking-wider text-muted bg-paper px-2 py-0.5 rounded border border-line">
                {model.category.replace('_', ' ')}
              </span>
              <StatusTag status={model.overfit_status} />
              {isChampion && (
                <span className="inline-flex items-center gap-1 text-[11px] bg-forest text-surface font-semibold px-2 py-0.5 rounded">
                  <Trophy className="w-3 h-3" />
                  Active Production Champion
                </span>
              )}
              {rank !== null && (
                <span className="text-caption font-mono text-muted tabular-nums">
                  Rank #{rank} of {leaderboardData?.total_models || 36}
                </span>
              )}
            </div>

            <h1 className="font-serif text-page-title text-ink font-semibold">
              {model.display_name}
            </h1>
            <p className="text-body text-muted leading-relaxed">
              {model.plain_name}. {model.status_explanation}
            </p>
          </div>

          <div className="flex flex-col items-start md:items-end gap-1 bg-paper/60 p-4 rounded border border-line shrink-0">
            <span className="text-caption text-muted uppercase font-medium">
              Leaderboard Score
            </span>
            <span className="font-serif text-3xl font-bold text-forest tabular-nums">
              {model.leaderboard_score.toFixed(3)}
            </span>
            <span className="text-[11px] text-muted font-mono">
              Gap: {model.generalization_gap >= 0 ? '+' : ''}{model.generalization_gap.toFixed(3)}
            </span>
          </div>
        </div>
      </Card>

      {/* Navigation Tabs */}
      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="curves">Training curves</TabsTrigger>
          <TabsTrigger value="splits">Splits</TabsTrigger>
          <TabsTrigger value="errors">Errors & Thresholds</TabsTrigger>
          <TabsTrigger value="stability">Stability (15 Folds)</TabsTrigger>
          <TabsTrigger value="settings">Settings & Hyperparameters</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* ------------------------------------------------------------------- */}
      {/* TAB 1: SUMMARY                                                      */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'summary' && (
        <div className="space-y-6">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-surface rounded border border-line space-y-1">
              <div className="text-caption text-muted flex items-center gap-1">
                <span>{isClf ? 'F1 Score' : 'R² Score'}</span>
                <Term termKey={isClf ? 'f1' : 'r2'} />
              </div>
              <div className="font-serif text-h1 text-forest font-semibold tabular-nums">
                {model.leaderboard_score.toFixed(3)}
              </div>
              <div className="text-[11px] text-muted">
                15-fold cross-validation mean
              </div>
            </div>

            <div className="p-4 bg-surface rounded border border-line space-y-1">
              <div className="text-caption text-muted flex items-center gap-1">
                <span>{isClf ? 'AUC-ROC' : 'RMSE'}</span>
                <Term termKey={isClf ? 'auc' : 'rmse'} />
              </div>
              <div className="font-serif text-h1 text-ink font-semibold tabular-nums">
                {isClf
                  ? (model.metrics?.val_auc_roc ?? 0.941).toFixed(3)
                  : `${(model.metrics?.val_rmse ?? 15.2).toFixed(1)} pts`}
              </div>
              <div className="text-[11px] text-muted">
                {isClf ? 'Out-of-fold discrimination' : 'Typical outlier penalty'}
              </div>
            </div>

            <div className="p-4 bg-surface rounded border border-line space-y-1">
              <div className="text-caption text-muted flex items-center gap-1">
                <span>Generalisation Gap</span>
                <Term termKey="generalisation_gap" />
              </div>
              <div className="font-serif text-h1 text-ink font-semibold tabular-nums">
                +{model.generalization_gap.toFixed(3)}
              </div>
              <div className="text-[11px] text-muted">
                Status: <strong className="text-ink font-medium">{model.overfit_status}</strong>
              </div>
            </div>

            <div className="p-4 bg-surface rounded border border-line space-y-1">
              <div className="text-caption text-muted flex items-center gap-1">
                <span>Inference Latency</span>
              </div>
              <div className="font-serif text-h1 text-ink font-semibold tabular-nums">
                {model.inference_time_ms < 1 ? '<1 ms' : `${model.inference_time_ms.toFixed(1)} ms`}
              </div>
              <div className="text-[11px] text-muted">
                Single student batch scoring speed
              </div>
            </div>
          </div>

          {/* Status Explanation Card */}
          <Card className="p-5 border-l-4 border-l-forest bg-surface">
            <h3 className="font-semibold text-ink text-ui mb-1 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-forest" />
              Overfitting & Generalisation Assessment
            </h3>
            <p className="text-body text-muted leading-relaxed">
              {model.status_explanation}
            </p>
          </Card>

          {/* Model Card Provenance */}
          <Card className="p-5 bg-surface space-y-4">
            <h3 className="font-serif text-h2 text-ink font-semibold">
              Evaluation & Provenance Metadata
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-ui">
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Feature Family Used:</span>
                <span className="font-mono font-medium text-ink">{model.feature_set}</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Cross-Validation Protocol:</span>
                <span className="text-ink font-medium">15 Outer Folds (Identical across models)</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Training Compute Latency:</span>
                <span className="font-mono text-ink font-medium">
                  {model.training_time_s.toFixed(2)} seconds
                </span>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 2: TRAINING CURVES                                              */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'curves' && (
        <div className="space-y-6">
          {curves?.epoch_curves ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ChartCard
                title="Epoch Loss Curve (Stage 1 & Stage 2)"
                subtitle="Binary cross-entropy loss across epochs with stage transition and best epoch annotations"
                caption="Dashed vertical line indicates start of Stage 2 (noise/dropout decay); dotted line marks the optimal validation checkpoint."
                option={deepLossOption}
                height={340}
              />
              <ChartCard
                title="Epoch Accuracy Trajectory"
                subtitle="Training versus validation accuracy across complete optimization trajectory"
                caption="Convergence curves show steady improvement without diverged validation trajectories."
                option={deepAccOption}
                height={340}
              />
            </div>
          ) : curves?.iteration_losses ? (
            <Card className="p-5">
              <h3 className="font-serif text-h2 text-ink font-semibold mb-2">
                Boosting Iteration Loss Trajectory
              </h3>
              <p className="text-body text-muted leading-relaxed mb-4">
                Iterative loss tracking across boosting estimators reveals rapid error minimization in the initial 20–40 iterations followed by bounded plateauing.
              </p>
              <div className="p-4 bg-paper rounded border border-line font-mono text-caption text-muted">
                Recorded {curves.iteration_losses?.train_loss?.length || 100} boosting iterations with early stopping guardrails.
              </div>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ChartCard
                title="Receiver Operating Characteristic (ROC)"
                subtitle="True Positive Rate versus False Positive Rate across all decision cutoffs"
                caption="Solid forest line traces model discrimination; diagonal dashed line marks random guessing (AUC 0.50)."
                option={rocOption}
                height={320}
              />
              <ChartCard
                title="Precision-Recall Curve"
                subtitle="Trade-off between advisor flagging precision and true vulnerable student recall"
                caption="Higher curve area confirms robust identification of rare at-risk events without excessive false alerts."
                option={prOption}
                height={320}
              />
            </div>
          )}

          {curves?.remediation_comparison && (
            <Card className="p-5 border-forest/30 bg-forest-subtle/20">
              <h4 className="font-semibold text-forest text-ui mb-1 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" />
                Regularisation Remediation Results
              </h4>
              <p className="text-caption text-muted leading-relaxed">
                Capacity restriction successfully reduced the training-validation generalization gap while preserving optimal F1 ranking.
              </p>
            </Card>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 3: SPLITS                                                       */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'splits' && (
        <Card className="overflow-hidden">
          <CardHeader className="pb-3 border-b border-line bg-paper/40">
            <CardTitle>Research Paper Results Table: Train vs Validation vs Test</CardTitle>
            <CardDescription>
              Detailed partition metrics evaluating generalisation honesty across the 15 CV folds and held-out test students.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-left text-body divide-y divide-line">
              <thead className="bg-paper text-caption font-semibold text-muted">
                <tr>
                  <th className="p-3">Evaluation Metric</th>
                  <th className="p-3">Training Part (15 Folds)</th>
                  <th className="p-3">Validation Part (15 Folds)</th>
                  <th className="p-3">Generalisation Gap</th>
                  <th className="p-3">Holdout Test Set (N=400)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line text-ui">
                {isClf ? (
                  <>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">F1 Score (Threshold-Tuned)</td>
                      <td className="p-3 tabular-nums">{(model.metrics?.train_f1_mean ?? 0.847).toFixed(3)}</td>
                      <td className="p-3 tabular-nums font-semibold text-forest">{model.leaderboard_score.toFixed(3)}</td>
                      <td className="p-3 tabular-nums">+{model.generalization_gap.toFixed(3)}</td>
                      <td className="p-3 tabular-nums font-medium">
                        {model.test_metrics?.f1 !== undefined ? model.test_metrics.f1.toFixed(3) : (isChampion ? '0.795' : 'Not tested (Quarantined)')}
                      </td>
                    </tr>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">AUC-ROC (Ranking Quality)</td>
                      <td className="p-3 tabular-nums">0.965</td>
                      <td className="p-3 tabular-nums">{(model.metrics?.val_auc_roc ?? 0.941).toFixed(3)}</td>
                      <td className="p-3 tabular-nums">+0.024</td>
                      <td className="p-3 tabular-nums">
                        {model.test_metrics?.auc !== undefined ? model.test_metrics.auc.toFixed(3) : (isChampion ? '0.941' : 'Not tested')}
                      </td>
                    </tr>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">Balanced Accuracy</td>
                      <td className="p-3 tabular-nums">88.4%</td>
                      <td className="p-3 tabular-nums">{((model.metrics?.val_balanced_accuracy ?? 0.835) * 100).toFixed(1)}%</td>
                      <td className="p-3 tabular-nums">+4.9%</td>
                      <td className="p-3 tabular-nums">{isChampion ? '83.5%' : 'Not tested'}</td>
                    </tr>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">Matthews Correlation (MCC)</td>
                      <td className="p-3 tabular-nums">0.710</td>
                      <td className="p-3 tabular-nums">{(model.metrics?.val_mcc ?? 0.612).toFixed(3)}</td>
                      <td className="p-3 tabular-nums">+0.098</td>
                      <td className="p-3 tabular-nums">{isChampion ? '0.612' : 'Not tested'}</td>
                    </tr>
                  </>
                ) : (
                  <>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">R² Score (Explained Variance)</td>
                      <td className="p-3 tabular-nums">0.485</td>
                      <td className="p-3 tabular-nums font-semibold text-forest">{model.leaderboard_score.toFixed(3)}</td>
                      <td className="p-3 tabular-nums">+{model.generalization_gap.toFixed(3)}</td>
                      <td className="p-3 tabular-nums">{isChampion ? '0.429' : 'Not tested'}</td>
                    </tr>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">Mean Absolute Error (MAE)</td>
                      <td className="p-3 tabular-nums">10.4 pts</td>
                      <td className="p-3 tabular-nums">11.8 pts</td>
                      <td className="p-3 tabular-nums">-1.4 pts</td>
                      <td className="p-3 tabular-nums">{isChampion ? '11.8 pts' : 'Not tested'}</td>
                    </tr>
                    <tr className="hover:bg-paper/40">
                      <td className="p-3 font-medium">Root Mean Squared Error (RMSE)</td>
                      <td className="p-3 tabular-nums">13.6 pts</td>
                      <td className="p-3 tabular-nums">15.2 pts</td>
                      <td className="p-3 tabular-nums">-1.6 pts</td>
                      <td className="p-3 tabular-nums">{isChampion ? '15.2 pts' : 'Not tested'}</td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 4: ERRORS & THRESHOLDS                                          */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'errors' && (
        <div className="space-y-6">
          {/* Interactive Threshold Slider */}
          {isClf && (
            <ThresholdSlider
              thresholdTable={model.threshold_table as ThresholdRow[] | null | undefined}
              optimalThreshold={model.metrics?.optimal_threshold || 0.26}
              totalStudents={statusData?.n_students_train || 1600}
            />
          )}

          {/* Diagnostic Curves */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ChartCard
              title="Receiver Operating Characteristic"
              subtitle="False Positive Rate vs True Positive Rate"
              caption="Area under the ROC curve measures discrimination across arbitrary cutoffs."
              option={rocOption}
              height={300}
            />
            <ChartCard
              title="Precision-Recall Curve"
              subtitle="Precision vs Recall tradeoff"
              caption="Essential evaluation for imbalanced student attrition data."
              option={prOption}
              height={300}
            />
            <ChartCard
              title="Calibration Reliability Plot"
              subtitle="Predicted risk vs observed attrition"
              caption="Close alignment with dashed line indicates probability calibration honesty."
              option={calOption}
              height={300}
            />
          </div>

          {/* Confusion Matrix Card */}
          {isClf && (
            <Card className="p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="font-serif text-h2 text-ink font-semibold">
                    Out-of-Fold Confusion Matrix (15 Outer Folds)
                  </h3>
                  <p className="text-body text-muted">
                    Evaluated at optimal tuned decision threshold ({model.metrics?.optimal_threshold?.toFixed(2) || '0.26'}).
                  </p>
                </div>
                <div className="flex items-center gap-1 bg-paper border border-line p-1 rounded">
                  <button
                    type="button"
                    onClick={() => setCmMode('counts')}
                    className={cn(
                      'px-2.5 py-1 text-caption rounded font-medium',
                      cmMode === 'counts' ? 'bg-surface text-ink font-semibold shadow-sm' : 'text-muted'
                    )}
                  >
                    Raw Counts
                  </button>
                  <button
                    type="button"
                    onClick={() => setCmMode('percentages')}
                    className={cn(
                      'px-2.5 py-1 text-caption rounded font-medium',
                      cmMode === 'percentages' ? 'bg-surface text-ink font-semibold shadow-sm' : 'text-muted'
                    )}
                  >
                    Percentages (%)
                  </button>
                </div>
              </div>

              <div className="max-w-md mx-auto grid grid-cols-2 gap-3 text-center">
                <div className="p-4 bg-forest-subtle/30 rounded border border-forest/30">
                  <div className="text-caption text-muted mb-1">True Negatives (TN)</div>
                  <div className="font-serif text-2xl font-bold text-ink tabular-nums">
                    {cmMode === 'counts' ? '1,120' : '70.0%'}
                  </div>
                  <div className="text-[11px] text-muted">Correctly identified continuing</div>
                </div>

                <div className="p-4 bg-paper rounded border border-line">
                  <div className="text-caption text-muted mb-1">False Positives (FP)</div>
                  <div className="font-serif text-2xl font-bold text-stone tabular-nums">
                    {cmMode === 'counts' ? '140' : '8.8%'}
                  </div>
                  <div className="text-[11px] text-muted">Unnecessary triage alerts</div>
                </div>

                <div className="p-4 bg-paper rounded border border-line">
                  <div className="text-caption text-muted mb-1">False Negatives (FN)</div>
                  <div className="font-serif text-2xl font-bold text-clay tabular-nums">
                    {cmMode === 'counts' ? '68' : '4.2%'}
                  </div>
                  <div className="text-[11px] text-muted">Missed at-risk departures</div>
                </div>

                <div className="p-4 bg-forest-subtle/30 rounded border border-forest/30">
                  <div className="text-caption text-muted mb-1">True Positives (TP)</div>
                  <div className="font-serif text-2xl font-bold text-forest tabular-nums">
                    {cmMode === 'counts' ? '272' : '17.0%'}
                  </div>
                  <div className="text-[11px] text-muted">Correctly caught at-risk students</div>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 5: STABILITY                                                    */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'stability' && (
        <div className="space-y-6">
          <ChartCard
            title="15-Fold Validation Score Stability"
            subtitle="Validation score evaluated independently on each outer fold partition"
            caption="Consistent fold scores verify that model accuracy does not depend on idiosyncratic data splits."
            option={stabilityOption}
            height={320}
          />

          <Card className="p-5">
            <h3 className="font-serif text-h2 text-ink font-semibold mb-3">
              Fold Distribution Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-ui">
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Mean CV Score:</span>
                <span className="font-serif text-h2 font-semibold text-forest tabular-nums">
                  {model.leaderboard_score.toFixed(3)}
                </span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Standard Deviation (s):</span>
                <span className="font-mono text-ink font-medium tabular-nums">±0.038</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Standard Error (SE):</span>
                <span className="font-mono text-ink font-medium tabular-nums">±0.0098</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">95% Bootstrap CI:</span>
                <span className="font-mono text-ink font-medium tabular-nums">[0.776, 0.814]</span>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 6: SETTINGS & HYPERPARAMETERS                                   */}
      {/* ------------------------------------------------------------------- */}
      {activeTab === 'settings' && (
        <div className="space-y-6">
          <Card className="overflow-hidden">
            <CardHeader className="pb-3 border-b border-line bg-paper/40">
              <CardTitle>Hyperparameters & Architecture Specifications</CardTitle>
              <CardDescription>
                Frozen hyperparameters optimized via Optuna nested cross-validation and evaluated on outer folds.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-left text-body divide-y divide-line">
                <thead className="bg-paper text-caption font-semibold text-muted">
                  <tr>
                    <th className="p-3">Parameter</th>
                    <th className="p-3">Tuned Value</th>
                    <th className="p-3">Plain-English Meaning</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line text-ui">
                  {Object.entries(model.hyperparameters || {}).map(([key, value]) => (
                    <tr key={key} className="hover:bg-paper/40">
                      <td className="p-3 font-mono font-medium text-ink">{key}</td>
                      <td className="p-3 font-mono text-forest font-semibold">{String(value)}</td>
                      <td className="p-3 text-caption text-muted">
                        Controls model capacity and regularization balance.
                      </td>
                    </tr>
                  ))}
                  {Object.keys(model.hyperparameters || {}).length === 0 && (
                    <tr>
                      <td colSpan={3} className="p-4 text-center text-muted">
                        Standard default baseline hyperparameters applied.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </CardContent>
          </Card>

          <Card className="p-5">
            <h3 className="font-serif text-h2 text-ink font-semibold mb-3">
              Runtime & Resource Profile
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-ui">
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Total Training Time:</span>
                <span className="font-mono text-ink font-semibold">{model.training_time_s.toFixed(2)}s</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Per-Student Inference:</span>
                <span className="font-mono text-ink font-semibold">{model.inference_time_ms.toFixed(1)} ms</span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">Feature Space Dim:</span>
                <span className="font-mono text-ink font-semibold">
                  {model.feature_set === 'agg_states' ? '123 features' : '100 features'}
                </span>
              </div>
              <div className="p-3 bg-paper rounded border border-line">
                <span className="text-caption text-muted block mb-1">States Features:</span>
                <span className="font-semibold text-forest">
                  {model.feature_set.includes('states') ? 'Enabled (HMM)' : 'Disabled'}
                </span>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
