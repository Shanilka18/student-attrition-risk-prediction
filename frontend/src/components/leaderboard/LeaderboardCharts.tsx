import { useMemo } from 'react';
import type { EChartsOption } from 'echarts-for-react';
import { ChartCard } from '@/components/ui/ChartCard';
import type { LeaderboardRow } from '@/api/types';

interface LeaderboardChartsProps {
  rows: LeaderboardRow[];
  track: 'classification' | 'regression';
  totalStudents?: number;
}

// Category palette mapping adhering to PROJECT_BRIEF §10
const CATEGORY_COLORS: Record<string, string> = {
  baseline: '#5B6168', // Muted
  linear: '#2C5282',   // Info
  classical: '#1F4E46', // Forest
  boosting: '#8C7355',  // Stone
  deep: '#2D5A27',      // Olive
  ensemble: '#4A5568',  // Slate
  hybrid: '#8C4A3B',    // Clay
};

export function LeaderboardCharts({
  rows,
  track,
  totalStudents = 1600,
}: LeaderboardChartsProps) {
  const isClf = track === 'classification';
  const scoreLabel = isClf ? 'CV F1 Score' : 'CV R² Score';

  // Sort rows descending by score
  const sortedRows = useMemo(() => {
    return [...rows].sort((a, b) => b.leaderboard_score - a.leaderboard_score);
  }, [rows]);

  // -------------------------------------------------------------------------
  // (a) Top 15 Models: Horizontal Bar Chart
  // -------------------------------------------------------------------------
  const top15 = useMemo(() => sortedRows.slice(0, 15).reverse(), [sortedRows]);

  const top15Option: EChartsOption = useMemo(() => {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const item = params[0];
          const row = top15[item.dataIndex];
          return `
            <div style="font-family:'Public Sans',sans-serif;font-size:12px;">
              <strong>${row.display_name}</strong><br/>
              Category: ${row.category}<br/>
              ${scoreLabel}: <strong>${row.leaderboard_score.toFixed(3)}</strong><br/>
              ${row.test_f1 !== null && row.test_f1 !== undefined ? `Test F1: <strong>${row.test_f1.toFixed(3)}</strong><br/>` : ''}
              ${row.test_r2 !== null && row.test_r2 !== undefined ? `Test R²: <strong>${row.test_r2.toFixed(3)}</strong><br/>` : ''}
              Gap: ${row.generalization_gap >= 0 ? '+' : ''}${row.generalization_gap.toFixed(3)} (${row.overfit_status})
            </div>
          `;
        },
      },
      grid: {
        top: 20,
        right: 40,
        bottom: 24,
        left: 170,
        containLabel: false,
      },
      xAxis: {
        type: 'value',
        min: 0,
        max: isClf ? 1.0 : Math.ceil(Math.max(...top15.map((r) => r.leaderboard_score), 0.5) * 10) / 10,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      yAxis: {
        type: 'category',
        data: top15.map((r) => r.display_name),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisTick: { show: false },
        axisLabel: {
          color: '#1A1D21',
          fontFamily: 'Public Sans',
          fontSize: 12,
          formatter: (value: string) => (value.length > 22 ? `${value.slice(0, 20)}…` : value),
        },
      },
      series: [
        {
          name: scoreLabel,
          type: 'bar',
          data: top15.map((r) => ({
            value: r.leaderboard_score,
            itemStyle: {
              color: CATEGORY_COLORS[r.category] || '#1F4E46',
            },
          })),
          label: {
            show: true,
            position: 'right',
            formatter: (p: any) => p.value.toFixed(3),
            color: '#1A1D21',
            fontFamily: 'Public Sans',
            fontSize: 11,
          },
        },
      ],
    };
  }, [top15, scoreLabel, isClf]);

  const top15Table = (
    <table className="w-full text-left text-caption divide-y divide-line">
      <thead className="bg-paper font-medium text-muted">
        <tr>
          <th className="p-2">Rank</th>
          <th className="p-2">Model</th>
          <th className="p-2">Category</th>
          <th className="p-2">{scoreLabel}</th>
          <th className="p-2">Gap</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line/60">
        {[...top15].reverse().map((r) => (
          <tr key={r.model_id}>
            <td className="p-2 tabular-nums font-mono">#{r.rank}</td>
            <td className="p-2 font-medium">{r.display_name}</td>
            <td className="p-2 capitalize">{r.category}</td>
            <td className="p-2 tabular-nums">{r.leaderboard_score.toFixed(3)}</td>
            <td className="p-2 tabular-nums">{r.generalization_gap.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // -------------------------------------------------------------------------
  // (b) Train vs Validation vs Test (Generalisation Gap View, Top 8)
  // -------------------------------------------------------------------------
  const top8 = useMemo(() => sortedRows.slice(0, 8), [sortedRows]);

  const generalisationOption: EChartsOption = useMemo(() => {
    const modelNames = top8.map((r) => r.display_name);
    const trainScores = top8.map((r) => (isClf ? r.train_f1_mean ?? 0 : r.train_r2_mean ?? 0));
    const valScores = top8.map((r) => r.leaderboard_score);
    const testScores = top8.map((r) => (isClf ? r.test_f1 ?? null : r.test_r2 ?? null));

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      legend: {
        data: ['Training (Mean)', 'Validation (15-Fold CV)', 'Holdout Test Set'],
        top: 0,
        textStyle: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      grid: {
        top: 36,
        right: 20,
        bottom: 50,
        left: 44,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: modelNames,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: {
          interval: 0,
          rotate: 22,
          color: '#1A1D21',
          fontFamily: 'Public Sans',
          fontSize: 11,
          formatter: (v: string) => (v.length > 16 ? `${v.slice(0, 14)}…` : v),
        },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      series: [
        {
          name: 'Training (Mean)',
          type: 'bar',
          data: trainScores,
          itemStyle: { color: '#5B6168' }, // Muted
          barMaxWidth: 16,
        },
        {
          name: 'Validation (15-Fold CV)',
          type: 'bar',
          data: valScores,
          itemStyle: { color: '#1F4E46' }, // Forest
          barMaxWidth: 16,
        },
        {
          name: 'Holdout Test Set',
          type: 'bar',
          data: testScores,
          itemStyle: { color: '#2D5A27' }, // Olive
          barMaxWidth: 16,
        },
      ],
    };
  }, [top8, isClf]);

  const generalisationTable = (
    <table className="w-full text-left text-caption divide-y divide-line">
      <thead className="bg-paper font-medium text-muted">
        <tr>
          <th className="p-2">Model</th>
          <th className="p-2">Training Score</th>
          <th className="p-2">CV Validation Score</th>
          <th className="p-2">Generalisation Gap</th>
          <th className="p-2">Test Holdout Score</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line/60">
        {top8.map((r) => (
          <tr key={r.model_id}>
            <td className="p-2 font-medium">{r.display_name}</td>
            <td className="p-2 tabular-nums">{(isClf ? r.train_f1_mean : r.train_r2_mean)?.toFixed(3) || '—'}</td>
            <td className="p-2 tabular-nums font-semibold text-forest">{r.leaderboard_score.toFixed(3)}</td>
            <td className="p-2 tabular-nums">+{r.generalization_gap.toFixed(3)}</td>
            <td className="p-2 tabular-nums">{(isClf ? r.test_f1 : r.test_r2)?.toFixed(3) || 'Not tested'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // -------------------------------------------------------------------------
  // (c) Metric Heatmap for Top 12 Models
  // -------------------------------------------------------------------------
  const top12 = useMemo(() => sortedRows.slice(0, 12).reverse(), [sortedRows]);

  const heatmapOption: EChartsOption = useMemo(() => {
    const metricsList = isClf
      ? ['F1 Score', 'AUC-ROC', 'Bal. Acc.', 'MCC', 'Precision', 'Recall', 'Low Gap']
      : ['R² Score', 'RMSE (inv)', 'MAE (inv)', 'Spearman', 'Within 10%', 'Low Gap'];

    const data: [number, number, number][] = [];

    top12.forEach((r, yIdx) => {
      if (isClf) {
        const f1 = r.val_f1_optimal ?? r.leaderboard_score;
        const auc = r.val_auc_roc ?? 0;
        const balAcc = r.val_balanced_accuracy ?? 0;
        const mcc = r.val_mcc ?? 0;
        const prec = r.val_precision ?? 0;
        const rec = r.val_recall ?? 0;
        const lowGap = Math.max(0, 1 - r.generalization_gap * 4); // Invert gap so higher is better

        const values = [f1, auc, balAcc, mcc, prec, rec, lowGap];
        values.forEach((v, xIdx) => {
          data.push([xIdx, yIdx, Math.round(v * 100) / 100]);
        });
      } else {
        const r2 = Math.max(0, r.val_r2 ?? r.leaderboard_score);
        const rmseInv = Math.max(0, 1 - (r.val_rmse ?? 20) / 40);
        const maeInv = Math.max(0, 1 - (r.val_mae ?? 15) / 30);
        const spearman = Math.max(0, r.val_spearman_rho ?? 0);
        const within10 = r.val_within_10_pct ?? 0.6;
        const lowGap = Math.max(0, 1 - r.generalization_gap * 4);

        const values = [r2, rmseInv, maeInv, spearman, within10, lowGap];
        values.forEach((v, xIdx) => {
          data.push([xIdx, yIdx, Math.round(v * 100) / 100]);
        });
      }
    });

    return {
      tooltip: {
        position: 'top',
        formatter: (p: any) => {
          const xLabel = metricsList[p.data[0]];
          const yLabel = top12[p.data[1]].display_name;
          return `${yLabel}<br/>${xLabel}: <strong>${p.data[2]}</strong>`;
        },
      },
      grid: {
        top: 24,
        right: 30,
        bottom: 40,
        left: 170,
        containLabel: false,
      },
      xAxis: {
        type: 'category',
        data: metricsList,
        splitArea: { show: true },
        axisLabel: { color: '#1A1D21', fontFamily: 'Public Sans', fontSize: 11 },
        axisLine: { lineStyle: { color: '#DDD8CE' } },
      },
      yAxis: {
        type: 'category',
        data: top12.map((r) => r.display_name),
        splitArea: { show: true },
        axisLabel: {
          color: '#1A1D21',
          fontFamily: 'Public Sans',
          fontSize: 11,
          formatter: (v: string) => (v.length > 22 ? `${v.slice(0, 20)}…` : v),
        },
        axisLine: { lineStyle: { color: '#DDD8CE' } },
      },
      visualMap: {
        min: 0.3,
        max: 1.0,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        itemWidth: 10,
        itemHeight: 80,
        inRange: {
          color: ['#F5F2EC', '#CBD5E1', '#8C7355', '#1F4E46'], // Paper to Stone to Forest
        },
        textStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 10 },
      },
      series: [
        {
          type: 'heatmap',
          data,
          label: {
            show: true,
            color: '#1A1D21',
            fontFamily: 'Public Sans',
            fontSize: 10,
            formatter: (p: any) => p.data[2].toFixed(2),
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 6,
              shadowColor: 'rgba(0, 0, 0, 0.2)',
            },
          },
        },
      ],
    };
  }, [top12, isClf]);

  const heatmapTable = (
    <table className="w-full text-left text-caption divide-y divide-line">
      <thead className="bg-paper font-medium text-muted">
        <tr>
          <th className="p-2">Model</th>
          <th className="p-2">F1 / R²</th>
          <th className="p-2">AUC / RMSE</th>
          <th className="p-2">Bal. Acc. / MAE</th>
          <th className="p-2">Low Gap Index</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line/60">
        {[...top12].reverse().map((r) => (
          <tr key={r.model_id}>
            <td className="p-2 font-medium">{r.display_name}</td>
            <td className="p-2 tabular-nums">{r.leaderboard_score.toFixed(3)}</td>
            <td className="p-2 tabular-nums">{(isClf ? r.val_auc_roc : r.val_rmse)?.toFixed(3) || '—'}</td>
            <td className="p-2 tabular-nums">{(isClf ? r.val_balanced_accuracy : r.val_mae)?.toFixed(3) || '—'}</td>
            <td className="p-2 tabular-nums">{(1 - r.generalization_gap).toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // -------------------------------------------------------------------------
  // (d) Training Time vs Score Scatter
  // -------------------------------------------------------------------------
  const scatterOption: EChartsOption = useMemo(() => {
    const data = sortedRows.map((r) => [
      Math.max(0.01, r.training_time_s),
      r.leaderboard_score,
      r.display_name,
      r.category,
      r.generalization_gap,
    ]);

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const [timeSec, score, name, cat, gap] = params.data;
          return `
            <div style="font-family:'Public Sans',sans-serif;font-size:12px;">
              <strong>${name}</strong><br/>
              Category: ${cat}<br/>
              ${scoreLabel}: <strong>${score.toFixed(3)}</strong><br/>
              Training time: <strong>${timeSec < 1 ? `${(timeSec * 1000).toFixed(0)} ms` : `${timeSec.toFixed(2)} s`}</strong><br/>
              Gap: ${gap.toFixed(3)}
            </div>
          `;
        },
      },
      grid: {
        top: 24,
        right: 30,
        bottom: 40,
        left: 48,
        containLabel: true,
      },
      xAxis: {
        type: 'log',
        name: 'Training Time (seconds, log scale)',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      yAxis: {
        type: 'value',
        name: scoreLabel,
        nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
        min: 0,
        max: isClf ? 1.0 : Math.ceil(Math.max(...sortedRows.map((r) => r.leaderboard_score), 0.5) * 10) / 10,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      series: [
        {
          type: 'scatter',
          symbolSize: 10,
          data,
          itemStyle: {
            color: (p: any) => CATEGORY_COLORS[p.data[3]] || '#1F4E46',
            borderColor: '#FFFFFF',
            borderWidth: 1,
          },
          label: {
            show: true,
            formatter: (p: any) => {
              // Only label top 5 models to avoid chart clutter
              const top5Names = sortedRows.slice(0, 5).map((r) => r.display_name);
              return top5Names.includes(p.data[2]) ? p.data[2] : '';
            },
            position: 'top',
            color: '#1A1D21',
            fontFamily: 'Public Sans',
            fontSize: 10,
          },
        },
      ],
    };
  }, [sortedRows, scoreLabel, isClf]);

  const scatterTable = (
    <table className="w-full text-left text-caption divide-y divide-line">
      <thead className="bg-paper font-medium text-muted">
        <tr>
          <th className="p-2">Model</th>
          <th className="p-2">Category</th>
          <th className="p-2">Training Time</th>
          <th className="p-2">{scoreLabel}</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line/60">
        {sortedRows.slice(0, 15).map((r) => (
          <tr key={r.model_id}>
            <td className="p-2 font-medium">{r.display_name}</td>
            <td className="p-2 capitalize">{r.category}</td>
            <td className="p-2 tabular-nums font-mono">{r.training_time_s.toFixed(2)}s</td>
            <td className="p-2 tabular-nums font-semibold">{r.leaderboard_score.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // -------------------------------------------------------------------------
  // (e) Category Comparison Box/Strip Plot
  // -------------------------------------------------------------------------
  const categoryOption: EChartsOption = useMemo(() => {
    const categories = ['baseline', 'linear', 'classical', 'boosting', 'deep', 'ensemble', 'hybrid'];
    const catLabels = ['Baseline', 'Linear', 'Classical', 'Boosting', 'Deep', 'Ensemble', 'Hybrid'];

    // Group scores by category
    const boxData: number[][] = [];
    categories.forEach((cat) => {
      const catScores = sortedRows
        .filter((r) => r.category === cat)
        .map((r) => r.leaderboard_score)
        .sort((a, b) => a - b);

      if (catScores.length === 0) {
        boxData.push([0, 0, 0, 0, 0]);
      } else if (catScores.length === 1) {
        const s = catScores[0];
        boxData.push([s, s, s, s, s]);
      } else {
        const min = catScores[0];
        const q1 = catScores[Math.floor(catScores.length * 0.25)];
        const median = catScores[Math.floor(catScores.length * 0.5)];
        const q3 = catScores[Math.floor(catScores.length * 0.75)];
        const max = catScores[catScores.length - 1];
        boxData.push([min, q1, median, q3, max]);
      }
    });

    return {
      tooltip: {
        trigger: 'item',
        formatter: (param: any) => {
          return `
            <div style="font-family:'Public Sans',sans-serif;font-size:12px;">
              <strong>${param.name}</strong><br/>
              Max: ${param.data[5].toFixed(3)}<br/>
              Q3: ${param.data[4].toFixed(3)}<br/>
              Median: ${param.data[3].toFixed(3)}<br/>
              Q1: ${param.data[2].toFixed(3)}<br/>
              Min: ${param.data[1].toFixed(3)}
            </div>
          `;
        },
      },
      grid: {
        top: 24,
        right: 20,
        bottom: 30,
        left: 44,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: catLabels,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontFamily: 'Public Sans', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        name: scoreLabel,
        nameTextStyle: { color: '#5B6168', fontFamily: 'Public Sans', fontSize: 11 },
        min: 0,
        max: 1.0,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        splitLine: { lineStyle: { color: '#E8E4DC', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontFamily: 'Public Sans' },
      },
      series: [
        {
          name: 'Score Distribution',
          type: 'boxplot',
          data: boxData,
          itemStyle: {
            color: '#F5F2EC',
            borderColor: '#1F4E46',
            borderWidth: 1.5,
          },
        },
      ],
    };
  }, [sortedRows, scoreLabel]);

  const categoryTable = (
    <table className="w-full text-left text-caption divide-y divide-line">
      <thead className="bg-paper font-medium text-muted">
        <tr>
          <th className="p-2">Category</th>
          <th className="p-2">Model Count</th>
          <th className="p-2">Best Score</th>
          <th className="p-2">Median Score</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line/60">
        {['classical', 'boosting', 'deep', 'ensemble', 'hybrid', 'linear', 'baseline'].map((cat) => {
          const m = sortedRows.filter((r) => r.category === cat);
          if (m.length === 0) return null;
          const scores = m.map((r) => r.leaderboard_score).sort((a, b) => a - b);
          const median = scores[Math.floor(scores.length / 2)];
          return (
            <tr key={cat}>
              <td className="p-2 font-medium capitalize">{cat}</td>
              <td className="p-2 tabular-nums">{m.length}</td>
              <td className="p-2 tabular-nums font-semibold text-forest">{scores[scores.length - 1].toFixed(3)}</td>
              <td className="p-2 tabular-nums">{median.toFixed(3)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );

  return (
    <div className="space-y-6 pt-4" data-testid="leaderboard-charts-section">
      <div>
        <h3 className="font-serif text-h2 text-ink font-semibold">
          Comparative Empirical Performance
        </h3>
        <p className="text-body text-muted mt-0.5">
          Research-grade visualizations of cross-validation leaderboards, generalisation trade-offs, and compute efficiency.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart (a): Top 15 Models */}
        <ChartCard
          title="Leaderboard Score by Model (Top 15)"
          subtitle="Cross-validation validation score across identical 15 folds, color-coded by architecture"
          sampleSize={totalStudents}
          caption="Higher bars indicate superior cross-validation ranking; models are strictly evaluated on held-out validation portions of the 15 outer folds."
          option={top15Option}
          height={380}
          tableAlternative={top15Table}
        />

        {/* Chart (b): Generalisation Gap (Top 8) */}
        <ChartCard
          title="Generalisation Gap: Training vs Validation vs Test"
          subtitle="Evaluation of the top 8 candidates comparing training fit to CV validation and holdout test set"
          sampleSize={totalStudents}
          caption="A small gap between training (grey) and validation (green) confirms robust generalisation without pathological over-fitting."
          option={generalisationOption}
          height={380}
          tableAlternative={generalisationTable}
        />

        {/* Chart (c): Metric Heatmap */}
        <ChartCard
          title="Multi-Metric Profile (Top 12 Models)"
          subtitle="Normalized balance across F1, discrimination AUC, balanced accuracy, MCC, and gap stability"
          sampleSize={totalStudents}
          caption="Darker forest green indicates superior metric performance; all metrics are calculated out-of-fold at the optimal decision threshold."
          option={heatmapOption}
          height={380}
          tableAlternative={heatmapTable}
        />

        {/* Chart (d): Compute Efficiency Scatter */}
        <ChartCard
          title="Training Latency vs Prediction Accuracy"
          subtitle="Trade-off between computational budget and cross-validation performance"
          sampleSize={totalStudents}
          caption="The top-left quadrant represents the Pareto frontier: models achieving high accuracy with sub-second execution overhead."
          option={scatterOption}
          height={380}
          tableAlternative={scatterTable}
        />
      </div>

      {/* Chart (e): Category Comparison Boxplot */}
      <ChartCard
        title="Category-Level Performance Distribution"
        subtitle="15-fold score variance across architectural model families"
        sampleSize={totalStudents}
        caption="Box plots represent minimum, first quartile, median, third quartile, and maximum scores observed within each architectural class."
        option={categoryOption}
        height={300}
        tableAlternative={categoryTable}
      />
    </div>
  );
}
