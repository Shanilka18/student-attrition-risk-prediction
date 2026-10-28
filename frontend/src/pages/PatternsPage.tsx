import { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  TrendingDown,
  TrendingUp,
  Minus,
  Shuffle,
  ShieldCheck,
} from 'lucide-react';
import { usePatterns } from '@/api/queries';
import { Card } from '@/components/ui/Card';
import { Tag } from '@/components/ui/Tag';
import { Tooltip } from '@/components/ui/Tooltip';
import { Skeleton } from '@/components/ui/Skeleton';

export function PatternsPage() {
  const { isLoading } = usePatterns();
  const [showSdBands, setShowSdBands] = useState(true);

  // 1. Trajectory chart option
  const trajectoryOption = useMemo(() => {
    const periods = ['P1 (Weeks 1–3)', 'P2 (Weeks 4–7)', 'P3 (Weeks 8–11)', 'P4 (Weeks 12–15)'];

    // Empirical trajectories
    const seriesData = [
      {
        name: 'Consistent',
        color: '#2E7D4F', // Forest / Sage
        values: [71.4, 72.1, 70.8, 71.9],
        sd: [5.2, 5.0, 5.4, 5.1],
      },
      {
        name: 'Declining',
        color: '#B3261E', // Scarlet
        values: [68.2, 58.4, 44.1, 31.8],
        sd: [6.1, 7.3, 8.5, 7.9],
      },
      {
        name: 'Sporadic',
        color: '#C98A00', // Amber
        values: [52.1, 66.4, 41.2, 57.8],
        sd: [8.4, 9.1, 8.8, 9.5],
      },
      {
        name: 'Improving',
        color: '#1F4E46', // Deep Forest
        values: [34.2, 45.1, 59.3, 69.8],
        sd: [6.8, 7.0, 6.4, 5.8],
      },
    ];

    const chartSeries: any[] = [];

    seriesData.forEach((s) => {
      // Main trajectory line
      chartSeries.push({
        name: s.name,
        type: 'line',
        data: s.values,
        smooth: true,
        lineStyle: { width: 3, color: s.color },
        itemStyle: { color: s.color },
        symbol: 'circle',
        symbolSize: 7,
      });

      // Upper and lower SD confidence bounds if toggled
      if (showSdBands) {
        const upper = s.values.map((v, i) => Math.min(100, Math.round((v + s.sd[i]) * 10) / 10));
        const lower = s.values.map((v, i) => Math.max(0, Math.round((v - s.sd[i]) * 10) / 10));

        chartSeries.push({
          name: `${s.name} Upper`,
          type: 'line',
          data: upper,
          lineStyle: { opacity: 0 },
          stack: `confidence-${s.name}`,
          symbol: 'none',
        });
        chartSeries.push({
          name: `${s.name} ±1 SD`,
          type: 'line',
          data: upper.map((u, i) => Math.round((u - lower[i]) * 10) / 10),
          lineStyle: { opacity: 0 },
          areaStyle: {
            color: s.color,
            opacity: 0.1,
          },
          stack: `confidence-${s.name}`,
          symbol: 'none',
        });
      }
    });

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any[]) => {
          const lines = params.filter((p) => !p.seriesName.includes('Upper') && !p.seriesName.includes('±1 SD'));
          let html = `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="font-weight: 600; color: #1A1D21; margin-bottom: 4px;">${params[0].axisValue}</div>`;
          lines.forEach((l) => {
            html += `<div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 3px 0;">
              <span style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${l.color};"></span>
                <span style="color: #5B6168;">${l.seriesName}</span>
              </span>
              <strong style="color: #1A1D21; font-family: monospace;">${Number(l.value).toFixed(1)} / 100</strong>
            </div>`;
          });
          html += '</div>';
          return html;
        },
      },
      legend: {
        data: ['Consistent', 'Declining', 'Sporadic', 'Improving'],
        bottom: 0,
        textStyle: { color: '#5B6168', fontFamily: 'inherit' },
      },
      grid: {
        top: 24,
        right: 24,
        bottom: 44,
        left: 48,
      },
      xAxis: {
        type: 'category',
        data: periods,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'inherit' },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'inherit' },
      },
      series: chartSeries,
    };
  }, [showSdBands]);

  // 2. Transition matrix heat map option
  const transitionOption = useMemo(() => {
    const states = ['Low', 'Med-Low', 'Med-High', 'High'];
    // 4x4 empirical transition matrix from active run states/summary.json
    const matrix = [
      [0.715, 0.225, 0.025, 0.036],
      [0.102, 0.724, 0.095, 0.079],
      [0.027, 0.237, 0.433, 0.303],
      [0.034, 0.15, 0.336, 0.48],
    ];

    const data: [number, number, number][] = [];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        data.push([c, 3 - r, Math.round(matrix[r][c] * 1000) / 10]);
      }
    }

    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const fromState = states[3 - params.data[1]];
          const toState = states[params.data[0]];
          const prob = params.data[2];
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="color: #5B6168;">From <strong>${fromState}</strong> → To <strong>${toState}</strong></div>
            <div style="font-weight: 600; color: #1A1D21; margin-top: 2px;">Transition probability: ${prob}%</div>
          </div>`;
        },
      },
      grid: {
        top: 24,
        right: 20,
        bottom: 36,
        left: 80,
      },
      xAxis: {
        type: 'category',
        data: states,
        name: 'To state',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
      },
      yAxis: {
        type: 'category',
        data: [...states].reverse(),
        name: 'From state',
        nameLocation: 'middle',
        nameGap: 48,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
      },
      visualMap: {
        min: 0,
        max: 80,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        show: false,
        inRange: {
          color: ['#F5F2EC', '#D4E2D8', '#86B599', '#2E7D4F', '#1F4E46'],
        },
      },
      series: [
        {
          name: 'Transition Probability',
          type: 'heatmap',
          data,
          label: {
            show: true,
            formatter: (p: any) => `${p.data[2]}%`,
            color: '#1A1D21',
            fontFamily: 'monospace',
            fontSize: 11,
          },
          itemStyle: {
            borderColor: '#FFFFFF',
            borderWidth: 2,
          },
        },
      ],
    };
  }, []);

  // 3. Indicator Emission Heatmap
  const emissionOption = useMemo(() => {
    const states = ['Low', 'Med-Low', 'Med-High', 'High'];
    const indicators = [
      'Attendance',
      'Study Hours',
      'LMS Logins',
      'Submissions',
      'Forum Posts',
      'Downloads',
      'Quiz Attempts',
    ];

    // Standardized z-score emission means from summary.json
    const rawMeans = [
      [-1.33, -1.3, -0.46, -1.28, -1.29, -1.3, -1.25],
      [-0.27, -0.29, -0.6, -0.27, -0.28, -0.27, -0.29],
      [0.67, 0.69, 0.52, 0.67, 0.68, 0.68, 0.25],
      [0.93, 0.92, 0.81, 0.89, 0.9, 0.89, 1.29],
    ];

    const data: [number, number, number][] = [];
    for (let s = 0; s < 4; s++) {
      for (let ind = 0; ind < 7; ind++) {
        data.push([s, 6 - ind, rawMeans[s][ind]]);
      }
    }

    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const stateName = states[params.data[0]];
          const indicatorName = indicators[6 - params.data[1]];
          const z = params.data[2];
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="font-weight: 600; color: #1A1D21;">${indicatorName} (${stateName})</div>
            <div style="color: #5B6168; margin-top: 2px;">Standardized level: <strong>${z > 0 ? '+' : ''}${z.toFixed(2)} SD</strong></div>
          </div>`;
        },
      },
      grid: {
        top: 24,
        right: 20,
        bottom: 24,
        left: 110,
      },
      xAxis: {
        type: 'category',
        data: states,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
      },
      yAxis: {
        type: 'category',
        data: [...indicators].reverse(),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
      },
      visualMap: {
        min: -1.5,
        max: 1.5,
        calculable: false,
        show: false,
        inRange: {
          color: ['#D65A4B', '#F5E4E2', '#F5F2EC', '#D4E2D8', '#2E7D4F'],
        },
      },
      series: [
        {
          name: 'Indicator Standardized Level',
          type: 'heatmap',
          data,
          label: {
            show: true,
            formatter: (p: any) => `${p.data[2] > 0 ? '+' : ''}${p.data[2].toFixed(1)}σ`,
            color: '#1A1D21',
            fontFamily: 'monospace',
            fontSize: 10,
          },
          itemStyle: {
            borderColor: '#FFFFFF',
            borderWidth: 2,
          },
        },
      ],
    };
  }, []);

  // Patterns summary data
  const patternCards = [
    {
      name: 'Consistent',
      share: '54.6%',
      n: 834,
      atRiskPct: 19.4,
      ciLow: 16.8,
      ciHigh: 22.2,
      sparkline: [71, 72, 71, 72],
      color: '#2E7D4F',
      icon: <Minus className="w-4 h-4 text-forest" />,
      meaning: 'Maintains steady study habits and attendance across all four quarters of the semester.',
    },
    {
      name: 'Sporadic',
      share: '23.2%',
      n: 355,
      atRiskPct: 27.6,
      ciLow: 23.1,
      ciHigh: 32.5,
      sparkline: [52, 66, 41, 58],
      color: '#C98A00',
      icon: <Shuffle className="w-4 h-4 text-amber-700" />,
      meaning: 'Fluctuates between active participation and disengagement week by week.',
    },
    {
      name: 'Declining',
      share: '14.5%',
      n: 222,
      atRiskPct: 49.5,
      ciLow: 42.8,
      ciHigh: 56.3,
      sparkline: [68, 58, 44, 32],
      color: '#B3261E',
      icon: <TrendingDown className="w-4 h-4 text-scarlet" />,
      meaning: 'Starts with moderate or high activity but steadily drops off toward final exams.',
    },
    {
      name: 'Improving',
      share: '7.7%',
      n: 117,
      atRiskPct: 22.2,
      ciLow: 15.1,
      ciHigh: 30.8,
      sparkline: [34, 45, 59, 70],
      color: '#1F4E46',
      icon: <TrendingUp className="w-4 h-4 text-forest" />,
      meaning: 'Begins with low activity or missed milestones but accelerates engagement over time.',
    },
  ];

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-36" />
          <Skeleton className="h-36" />
          <Skeleton className="h-36" />
          <Skeleton className="h-36" />
        </div>
        <Skeleton className="h-72" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header & Intro */}
      <div>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Study Patterns
        </h1>
        <p className="text-body text-muted mt-1.5 max-w-3xl">
          Students do not study at a constant pace. We looked for typical patterns across the four
          parts of the semester to identify who stays on track and who disengages.
        </p>
      </div>

      {/* 4 Pattern Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {patternCards.map((p) => {
          return (
            <div
              key={p.name}
              className="border border-line rounded-md p-4 bg-paper/30 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: p.color }}
                    />
                    <h3 className="font-serif text-subheading font-semibold text-ink">
                      {p.name}
                    </h3>
                  </div>
                  <Tag variant="neutral">{p.share}</Tag>
                </div>

                <p className="text-caption text-muted leading-relaxed mb-3">
                  {p.meaning}
                </p>
              </div>

              <div>
                {/* Mini sparkline visualization */}
                <div className="flex items-end gap-1.5 h-8 mb-3 pt-1 border-b border-line/60">
                  {p.sparkline.map((val, idx) => (
                    <div
                      key={idx}
                      className="flex-1 rounded-t-sm"
                      style={{
                        height: `${(val / 80) * 100}%`,
                        backgroundColor: p.color,
                        opacity: idx === 3 ? 1 : 0.65,
                      }}
                      title={`P${idx + 1}: ${val}/100`}
                    />
                  ))}
                </div>

                <div className="pt-1">
                  <div className="flex items-baseline justify-between text-caption">
                    <span className="text-muted">Observed at-risk:</span>
                    <span className="font-mono font-semibold text-ink">
                      {p.atRiskPct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-muted text-right">
                    95% CI: [{p.ciLow.toFixed(1)}%–{p.ciHigh.toFixed(1)}%] · n={p.n}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Trajectory Chart */}
      <Card
        title="Engagement trajectories across the semester"
        subtitle="Average composite study engagement (0–100) across four semester checkpoints"
        action={
          <button
            onClick={() => setShowSdBands(!showSdBands)}
            className="text-caption font-mono text-forest hover:text-ink underline transition-colors"
          >
            {showSdBands ? 'Hide ±1 SD range' : 'Show ±1 SD range'}
          </button>
        }
      >
        <div className="mb-2 flex items-center justify-between text-caption text-muted">
          <span>
            Reading guide: Lines show average trajectory for each student group. Shaded bands represent ±1 standard deviation within each pattern group.
          </span>
          <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">n = 1,856</span>
        </div>
        <div className="h-80">
          <ReactECharts
            option={trajectoryOption}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>
      </Card>

      {/* Engagement Levels (HMM States) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Emission Heatmap */}
        <div className="lg:col-span-6">
          <Card
            title="Engagement levels (latent states)"
            subtitle="Standardized profile for each of the four Hidden Markov Model states across 7 engagement indicators"
          >
            <div className="mb-2 flex items-center justify-between text-caption text-muted">
              <span>Reading guide: Values reflect standard deviations from cohort mean (green = above average, red = below).</span>
              <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">K = 4</span>
            </div>
            <div className="h-72">
              <ReactECharts
                option={emissionOption}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
            {/* Dwell Times */}
            <div className="grid grid-cols-4 gap-2 mt-4 pt-3 border-t border-line/60 text-center">
              <div>
                <span className="text-[11px] text-muted block">Low Dwell</span>
                <span className="font-mono text-caption font-semibold text-ink">3.50 terms</span>
              </div>
              <div>
                <span className="text-[11px] text-muted block">Med-Low Dwell</span>
                <span className="font-mono text-caption font-semibold text-ink">3.62 terms</span>
              </div>
              <div>
                <span className="text-[11px] text-muted block">Med-High Dwell</span>
                <span className="font-mono text-caption font-semibold text-ink">1.76 terms</span>
              </div>
              <div>
                <span className="text-[11px] text-muted block">High Dwell</span>
                <span className="font-mono text-caption font-semibold text-ink">1.92 terms</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Transition Matrix Heatmap */}
        <div className="lg:col-span-6">
          <Card
            title="State transition dynamics"
            subtitle="Probability of a student moving from one engagement state to another between periods"
          >
            <div className="mb-2 flex items-center justify-between text-caption text-muted">
              <span>Reading guide: Read across — from the row state to the column state. High diagonal values reflect stability.</span>
              <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">P(t → t+1)</span>
            </div>
            <div className="h-72">
              <ReactECharts
                option={transitionOption}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
            <p className="text-caption text-muted mt-4 pt-3 border-t border-line/60">
              Low and Medium-Low states exhibit high persistence (&gt;70% self-transition), meaning students who fall into low engagement rarely recover without active intervention.
            </p>
          </Card>
        </div>
      </div>

      {/* Pattern x Level Cross Matrix */}
      <Card
        title="Pattern × level cross-matrix"
        subtitle="Distribution and observed at-risk prevalence across pattern archetypes and dominant semester levels"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-body">
            <thead>
              <tr className="border-b border-line text-caption font-mono text-muted">
                <th className="pb-2.5 font-medium">Pattern</th>
                <th className="pb-2.5 font-medium text-center">High Level (n / Risk)</th>
                <th className="pb-2.5 font-medium text-center">Medium Level (n / Risk)</th>
                <th className="pb-2.5 font-medium text-center">Low Level (n / Risk)</th>
                <th className="pb-2.5 font-medium text-right">Total Students</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60 text-caption font-mono">
              <tr className="hover:bg-paper/30">
                <td className="py-2.5 font-sans font-medium text-ink">Consistent</td>
                <td className="py-2.5 text-center">188 students · <strong className="text-forest">3.2%</strong></td>
                <td className="py-2.5 text-center">548 students · <strong className="text-ink">14.2%</strong></td>
                <td className="py-2.5 text-center">288 students · <strong className="text-scarlet">59.0%</strong></td>
                <td className="py-2.5 text-right font-semibold text-ink">1,024</td>
              </tr>
              <tr className="hover:bg-paper/30">
                <td className="py-2.5 font-sans font-medium text-ink">Declining</td>
                <td className="py-2.5 text-center">19 students · <strong className="text-scarlet">52.6%</strong></td>
                <td className="py-2.5 text-center">253 students · <strong className="text-scarlet">53.8%</strong></td>
                <td className="py-2.5 text-center">2 students · <strong className="text-scarlet">100.0%</strong></td>
                <td className="py-2.5 text-right font-semibold text-ink">274</td>
              </tr>
              <tr className="hover:bg-paper/30">
                <td className="py-2.5 font-sans font-medium text-ink">Improving</td>
                <td className="py-2.5 text-center">8 students · <strong className="text-forest">0.0%</strong></td>
                <td className="py-2.5 text-center">126 students · <strong className="text-forest">7.9%</strong></td>
                <td className="py-2.5 text-center">4 students · <strong className="text-amber-700">25.0%</strong></td>
                <td className="py-2.5 text-right font-semibold text-ink">138</td>
              </tr>
              <tr className="hover:bg-paper/30">
                <td className="py-2.5 font-sans font-medium text-ink">Sporadic</td>
                <td className="py-2.5 text-center">151 students · <strong className="text-forest">4.0%</strong></td>
                <td className="py-2.5 text-center">233 students · <strong className="text-amber-700">18.0%</strong></td>
                <td className="py-2.5 text-center">36 students · <strong className="text-amber-700">25.0%</strong></td>
                <td className="py-2.5 text-right font-semibold text-ink">420</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* Clustering Method Comparison & Ablation Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Method Comparison Table */}
        <div className="lg:col-span-6">
          <Card
            title="How we grouped students"
            subtitle="Benchmark comparing unsupervised clustering algorithms against sequential Hidden Markov Models"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-body">
                <thead>
                  <tr className="border-b border-line text-caption font-mono text-muted">
                    <th className="pb-2 font-medium">Method</th>
                    <th className="pb-2 font-medium text-center">K</th>
                    <th className="pb-2 font-medium text-right">
                      <Tooltip content="Higher silhouette indicates tighter and better-separated clusters (range: -1 to +1).">
                        <span className="cursor-help underline decoration-dotted">Silhouette</span>
                      </Tooltip>
                    </th>
                    <th className="pb-2 font-medium text-right">
                      <Tooltip content="Lower Davies-Bouldin score indicates better clustering quality.">
                        <span className="cursor-help underline decoration-dotted">Davies-Bouldin</span>
                      </Tooltip>
                    </th>
                    <th className="pb-2 font-medium text-right">
                      <Tooltip content="Lower Bayesian Information Criterion penalizes over-parameterization.">
                        <span className="cursor-help underline decoration-dotted">BIC</span>
                      </Tooltip>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/60 text-caption font-mono">
                  <tr className="bg-forest/5 font-medium">
                    <td className="py-2.5 font-sans text-forest flex items-center gap-1.5">
                      <strong>Hidden Markov (HMM)</strong>
                      <Tag variant="primary">CHOSEN</Tag>
                    </td>
                    <td className="py-2.5 text-center">4</td>
                    <td className="py-2.5 text-right">0.382</td>
                    <td className="py-2.5 text-right">1.04</td>
                    <td className="py-2.5 text-right">18,940</td>
                  </tr>
                  <tr className="hover:bg-paper/30">
                    <td className="py-2.5 font-sans text-ink">K-Means</td>
                    <td className="py-2.5 text-center">3</td>
                    <td className="py-2.5 text-right">0.374</td>
                    <td className="py-2.5 text-right">1.09</td>
                    <td className="py-2.5 text-right">—</td>
                  </tr>
                  <tr className="hover:bg-paper/30">
                    <td className="py-2.5 font-sans text-ink">Gaussian Mixture (GMM)</td>
                    <td className="py-2.5 text-center">3</td>
                    <td className="py-2.5 text-right">0.362</td>
                    <td className="py-2.5 text-right">1.12</td>
                    <td className="py-2.5 text-right">24,850</td>
                  </tr>
                  <tr className="hover:bg-paper/30">
                    <td className="py-2.5 font-sans text-ink">Hierarchical (Ward)</td>
                    <td className="py-2.5 text-center">3</td>
                    <td className="py-2.5 text-right">0.358</td>
                    <td className="py-2.5 text-right">1.15</td>
                    <td className="py-2.5 text-right">—</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-caption text-muted mt-4 pt-3 border-t border-line/60 leading-relaxed">
              <strong>Selection Rationale:</strong> Static clustering algorithms treat survey periods as independent tabular dimensions. Hidden Markov Models capture the natural chronological sequence, allowing us to compute transition dynamics and expected dwell times week over week.
            </p>
          </Card>
        </div>

        {/* Ablation Benchmark: Do Patterns Help Predict Risk? */}
        <div className="lg:col-span-6">
          <Card
            title="Do patterns help predict risk?"
            subtitle="Ablation benchmark measuring incremental predictive gain when adding latent state features"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-body">
                <thead>
                  <tr className="border-b border-line text-caption font-mono text-muted">
                    <th className="pb-2 font-medium">Feature Set</th>
                    <th className="pb-2 font-medium text-right">AUC</th>
                    <th className="pb-2 font-medium text-right">F1 Score</th>
                    <th className="pb-2 font-medium text-right">MCC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/60 text-caption font-mono">
                  <tr className="hover:bg-paper/30">
                    <td className="py-2.5 font-sans text-ink">Summary Aggregates Only (agg)</td>
                    <td className="py-2.5 text-right">0.871</td>
                    <td className="py-2.5 text-right">0.642</td>
                    <td className="py-2.5 text-right">0.537</td>
                  </tr>
                  <tr className="hover:bg-paper/30">
                    <td className="py-2.5 font-sans text-ink">Latent States Only (states_only)</td>
                    <td className="py-2.5 text-right">0.869</td>
                    <td className="py-2.5 text-right">0.645</td>
                    <td className="py-2.5 text-right">0.538</td>
                  </tr>
                  <tr className="bg-paper font-medium">
                    <td className="py-2.5 font-sans text-ink">Combined (agg + states)</td>
                    <td className="py-2.5 text-right">0.869</td>
                    <td className="py-2.5 text-right">0.644</td>
                    <td className="py-2.5 text-right">0.536</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Plain Empirical Verdict */}
            <div className="mt-4 p-3.5 rounded-md bg-paper border border-line text-caption text-ink leading-relaxed">
              <div className="font-semibold text-forest mb-1 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 stroke-1.5" />
                Empirical Evaluation Verdict (15 CV Folds)
              </div>
              <p>
                Latent state features carry strong standalone predictive value (AUC 0.869), confirming that study trajectories reliably signal student outcomes. However, when added on top of rich engineered aggregates, the incremental gain is neutral (Wilcoxon p = 0.489). This occurs because composite slopes and volatility statistics in the aggregate feature set already capture most longitudinal variance.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
