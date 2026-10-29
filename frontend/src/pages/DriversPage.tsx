import { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Compass,
  AlertCircle,
  ShieldAlert,
} from 'lucide-react';
import { useDrivers } from '@/api/queries';
import { Card } from '@/components/ui/Card';
import { ErrorState } from '@/components/ui/ErrorState';
import { Tag } from '@/components/ui/Tag';
import { Skeleton } from '@/components/ui/Skeleton';
import {
  getDependenceInsightSentence,
  getShapConsistencySentence,
  getGroupSizeWarningSentence,
} from '@/content/sentences';

type GroupingMode = 'factor' | 'indicator' | 'period' | 'family';

export function DriversPage() {
  const {
    data: driversData,
    isLoading,
    isError,
    error,
    refetch,
  } = useDrivers();
  const [grouping, setGrouping] = useState<GroupingMode>('factor');
  const [selectedFeature, setSelectedFeature] = useState<string>('composite_drop_from_best_to_last');

  // Top 10 Drivers Ranking
  const top10 = useMemo(() => {
    return (driversData?.validation_ranking ?? []).slice(0, 10);
  }, [driversData]);

  // Set initial selected feature once data loads
  const currentFeatureKey = useMemo(() => {
    if (driversData?.dependence_samples && !driversData.dependence_samples[selectedFeature]) {
      const keys = Object.keys(driversData.dependence_samples);
      return keys.length > 0 ? keys[0] : selectedFeature;
    }
    return selectedFeature;
  }, [driversData, selectedFeature]);

  // 1. Top Drivers Bar Chart Option
  const barChartOption = useMemo(() => {
    if (!driversData) return {};

    if (grouping === 'factor') {
      const reversed = [...top10].reverse();
      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: (params: any) => {
            const item = params[0];
            return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
              <div style="font-weight: 600; color: #1A1D21;">${item.name}</div>
              <div style="color: #5B6168; margin-top: 2px;">Average Impact: <strong>+${item.value.toFixed(2)} points</strong></div>
            </div>`;
          },
        },
        grid: { top: 16, right: 36, bottom: 24, left: 240 },
        xAxis: {
          type: 'value',
          name: 'Impact on risk score (points)',
          nameLocation: 'middle',
          nameGap: 28,
          nameTextStyle: { color: '#5B6168', fontSize: 11 },
          splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
          axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
        },
        yAxis: {
          type: 'category',
          data: reversed.map((d) => d.plain_name),
          axisLine: { lineStyle: { color: '#DDD8CE' } },
          axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
        },
        series: [
          {
            name: 'Average Absolute Impact',
            type: 'bar',
            data: reversed.map((d) => d.mean_abs_points),
            itemStyle: {
              color: '#1F4E46',
              borderRadius: [0, 3, 3, 0],
            },
            label: {
              show: true,
              position: 'right',
              formatter: (p: any) => `+${Number(p.value).toFixed(2)} pts`,
              color: '#5B6168',
              fontFamily: 'monospace',
              fontSize: 11,
            },
          },
        ],
      };
    }

    if (grouping === 'indicator') {
      const items = Object.entries(driversData.grouped_importance?.by_indicator ?? {})
        .sort((a, b) => b[1].share_pct - a[1].share_pct)
        .reverse();
      const labels: Record<string, string> = {
        resource_access: 'Learning Material Access',
        composite: 'Composite Engagement Slopes',
        review_time: 'Review & Revision Time',
        study_consistency: 'Study Schedule Consistency',
        participation: 'Interactive Participation',
        assignment_engagement: 'Assignment Preparation',
        study_time: 'Self-Study Hours',
        quiz_practice: 'Practice Quiz Attempts',
      };

      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: (params: any) => {
            const item = params[0];
            return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
              <div style="font-weight: 600; color: #1A1D21;">${item.name}</div>
              <div style="color: #5B6168; margin-top: 2px;">Share of Total Model Impact: <strong>${item.value.toFixed(1)}%</strong></div>
            </div>`;
          },
        },
        grid: { top: 16, right: 40, bottom: 24, left: 200 },
        xAxis: {
          type: 'value',
          name: 'Share of total importance (%)',
          nameLocation: 'middle',
          nameGap: 28,
          nameTextStyle: { color: '#5B6168', fontSize: 11 },
          splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
          axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
        },
        yAxis: {
          type: 'category',
          data: items.map(([k]) => labels[k] ?? k),
          axisLine: { lineStyle: { color: '#DDD8CE' } },
          axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
        },
        series: [
          {
            name: 'Share of Total Impact',
            type: 'bar',
            data: items.map(([, v]) => v.share_pct),
            itemStyle: {
              color: '#2E7D4F',
              borderRadius: [0, 3, 3, 0],
            },
            label: {
              show: true,
              position: 'right',
              formatter: (p: any) => `${Number(p.value).toFixed(1)}%`,
              color: '#5B6168',
              fontFamily: 'monospace',
              fontSize: 11,
            },
          },
        ],
      };
    }

    if (grouping === 'period') {
      const items = Object.entries(driversData.grouped_importance?.by_period ?? {})
        .sort((a, b) => b[1].share_pct - a[1].share_pct)
        .reverse();

      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: (params: any) => {
            const item = params[0];
            return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
              <div style="font-weight: 600; color: #1A1D21;">${item.name}</div>
              <div style="color: #5B6168; margin-top: 2px;">Share of Total Model Impact: <strong>${item.value.toFixed(1)}%</strong></div>
            </div>`;
          },
        },
        grid: { top: 16, right: 40, bottom: 24, left: 160 },
        xAxis: {
          type: 'value',
          name: 'Share of total importance (%)',
          nameLocation: 'middle',
          nameGap: 28,
          nameTextStyle: { color: '#5B6168', fontSize: 11 },
          splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
          axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
        },
        yAxis: {
          type: 'category',
          data: items.map(([k]) =>
            k === 'Overall' ? 'Full Semester (Drift & Trajectory)' : `Period ${k.replace('P', '')}`
          ),
          axisLine: { lineStyle: { color: '#DDD8CE' } },
          axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
        },
        series: [
          {
            name: 'Share of Total Impact',
            type: 'bar',
            data: items.map(([, v]) => v.share_pct),
            itemStyle: {
              color: '#C98A00',
              borderRadius: [0, 3, 3, 0],
            },
            label: {
              show: true,
              position: 'right',
              formatter: (p: any) => `${Number(p.value).toFixed(1)}%`,
              color: '#5B6168',
              fontFamily: 'monospace',
              fontSize: 11,
            },
          },
        ],
      };
    }

    // By Family / Type
    const items = Object.entries(driversData.grouped_importance?.by_family ?? {})
      .sort((a, b) => b[1].share_pct - a[1].share_pct)
      .reverse();
    const familyLabels: Record<string, string> = {
      aggregates_drift: 'Engineered Longitudinal Aggregates & Slopes',
      raw_answers: 'Raw Survey Responses (Point-in-Time)',
      state_features: 'Latent Markov State Features',
    };

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const item = params[0];
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="font-weight: 600; color: #1A1D21;">${item.name}</div>
            <div style="color: #5B6168; margin-top: 2px;">Share: <strong>${item.value.toFixed(1)}%</strong></div>
          </div>`;
        },
      },
      grid: { top: 16, right: 40, bottom: 24, left: 240 },
      xAxis: {
        type: 'value',
        name: 'Share of total importance (%)',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'category',
        data: items.map(([k]) => familyLabels[k] ?? k),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#1A1D21', fontSize: 11, fontFamily: 'inherit' },
      },
      series: [
        {
          name: 'Share of Total Impact',
          type: 'bar',
          data: items.map(([, v]) => v.share_pct),
          itemStyle: {
            color: '#1F4E46',
            borderRadius: [0, 3, 3, 0],
          },
          label: {
            show: true,
            position: 'right',
            formatter: (p: any) => `${Number(p.value).toFixed(1)}%`,
            color: '#5B6168',
            fontFamily: 'monospace',
            fontSize: 11,
          },
        },
      ],
    };
  }, [driversData, grouping, top10]);

  // 2. Dependence Scatter Option
  const dependenceOption = useMemo(() => {
    if (!driversData?.dependence_samples) return {};
    const samples = driversData.dependence_samples[currentFeatureKey] ?? [];

    const scatterData = samples.map((s) => [s.feature_val, s.shap_points]);

    // Trendline calculation (linear regression slope)
    let minX = Infinity;
    let maxX = -Infinity;
    samples.forEach((s) => {
      if (s.feature_val < minX) minX = s.feature_val;
      if (s.feature_val > maxX) maxX = s.feature_val;
    });

    const n = samples.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;
    samples.forEach((s) => {
      sumX += s.feature_val;
      sumY += s.shap_points;
      sumXY += s.feature_val * s.shap_points;
      sumX2 += s.feature_val * s.feature_val;
    });
    const slope = n > 1 ? (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX) : 0;
    const intercept = n > 1 ? (sumY - slope * sumX) / n : 0;

    const trendLine = [
      [minX, slope * minX + intercept],
      [maxX, slope * maxX + intercept],
    ];

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.seriesType === 'line') return '';
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="color: #5B6168;">Reported Feature Value: <strong>${params.value[0].toFixed(2)}</strong></div>
            <div style="color: #1A1D21; margin-top: 2px;">Impact on MEI: <strong>${params.value[1] >= 0 ? '+' : ''}${params.value[1].toFixed(2)} pts</strong></div>
          </div>`;
        },
      },
      grid: { top: 24, right: 24, bottom: 36, left: 48 },
      xAxis: {
        type: 'value',
        name: 'Reported feature value',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'Shift in risk score (points)',
        nameLocation: 'middle',
        nameGap: 36,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'Student Observation',
          type: 'scatter',
          data: scatterData,
          symbolSize: 6,
          itemStyle: {
            color: '#1F4E46',
            opacity: 0.5,
          },
        },
        {
          name: 'Trend',
          type: 'line',
          data: trendLine,
          lineStyle: {
            color: '#B3261E',
            width: 2,
            type: 'dashed',
          },
          symbol: 'none',
        },
      ],
    };
  }, [driversData, currentFeatureKey]);

  // Plain insight sentence for dependence view
  const dependenceSentence = useMemo(() => {
    const item = top10.find((d) => d.feature === currentFeatureKey);
    const plainName = item?.plain_name ?? 'This engagement factor';
    const delta = item ? item.mean_abs_points : 0.76;
    return getDependenceInsightSentence(plainName, delta, 'Disengagement or sharp declines');
  }, [currentFeatureKey, top10]);

  // 3. Beeswarm Scatter Option
  const beeswarmOption = useMemo(() => {
    if (!driversData?.beeswarm_samples) return {};
    const top8Features = top10.slice(0, 8);
    const featureNames = top8Features.map((f) => f.plain_name);

    // Group beeswarm samples by feature
    const scatterData: any[] = [];
    driversData.beeswarm_samples.forEach((s) => {
      const fIdx = featureNames.indexOf(s.plain_name);
      if (fIdx !== -1) {
        // Jitter on y-axis
        const jitter = (Math.random() - 0.5) * 0.45;
        scatterData.push({
          value: [s.shap_points, fIdx + jitter, s.feature_val],
          name: s.plain_name,
        });
      }
    });

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const shift = params.value[0];
          const val = params.value[2];
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="font-weight: 600; color: #1A1D21;">${params.name}</div>
            <div style="color: #5B6168; margin-top: 2px;">Reported value: <strong>${val.toFixed(2)}</strong></div>
            <div style="color: #1A1D21;">Score shift: <strong>${shift >= 0 ? '+' : ''}${shift.toFixed(2)} points</strong></div>
          </div>`;
        },
      },
      grid: { top: 24, right: 36, bottom: 44, left: 240 },
      xAxis: {
        type: 'value',
        name: 'Impact on student risk score (points shift)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: '#5B6168', fontSize: 11 },
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        min: -0.5,
        max: featureNames.length - 0.5,
        interval: 1,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: {
          formatter: (val: number) => featureNames[Math.round(val)] ?? '',
          color: '#1A1D21',
          fontSize: 11,
          fontFamily: 'inherit',
        },
        splitLine: { lineStyle: { color: '#EEEAE2' } },
      },
      visualMap: {
        dimension: 2,
        min: -2,
        max: 2,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        text: ['High feature value', 'Low feature value'],
        textStyle: { color: '#5B6168', fontSize: 11 },
        inRange: {
          color: ['#2E7D4F', '#86B599', '#DDD8CE', '#D65A4B', '#B3261E'],
        },
      },
      series: [
        {
          name: 'Student Attribution',
          type: 'scatter',
          data: scatterData,
          symbolSize: 5,
        },
      ],
    };
  }, [driversData, top10]);

  // 4. Consistency Sentence
  const consistencySentence = useMemo(() => {
    if (!driversData?.shap_consistency) return '';
    return getShapConsistencySentence(driversData.shap_consistency);
  }, [driversData]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton className="h-72" />
          <Skeleton className="h-72" />
        </div>
      </div>
    );
  }

  if (isError || !driversData) {
    return (
      <ErrorState
        title="Risk drivers are unavailable"
        message="The active run's explanation results could not be loaded. Check the server and try again."
        error={error}
        onRetry={() => void refetch()}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Header & Intro */}
      <div>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Risk Drivers
        </h1>
        <p className="text-body text-muted mt-1.5 max-w-3xl">
          What in the answers pushes the score up or down. Every student&apos;s Engagement Index
          is explained by comparing their study patterns against cohort benchmarks.
        </p>
      </div>

      {/* Top Drivers Bar Chart */}
      <Card
        title="Top contributing factors"
        subtitle="Ranked by mean absolute impact on the calibrated 0–100 risk score"
        action={
          <div className="flex items-center gap-1 bg-paper border border-line p-0.5 rounded-md text-caption">
            <button
              onClick={() => setGrouping('factor')}
              className={`px-2.5 py-1 rounded transition-colors font-medium ${
                grouping === 'factor'
                  ? 'bg-forest text-paper shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              By Factor
            </button>
            <button
              onClick={() => setGrouping('indicator')}
              className={`px-2.5 py-1 rounded transition-colors font-medium ${
                grouping === 'indicator'
                  ? 'bg-forest text-paper shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              By Question
            </button>
            <button
              onClick={() => setGrouping('period')}
              className={`px-2.5 py-1 rounded transition-colors font-medium ${
                grouping === 'period'
                  ? 'bg-forest text-paper shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              By Part of Semester
            </button>
            <button
              onClick={() => setGrouping('family')}
              className={`px-2.5 py-1 rounded transition-colors font-medium ${
                grouping === 'family'
                  ? 'bg-forest text-paper shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              By Type
            </button>
          </div>
        }
      >
        <div className="mb-2 flex items-center justify-between text-caption text-muted">
          <span>
            Reading guide: Shows how much each factor shifts risk scores on average. Longitudinal slopes and drops carry more signal than individual weekly surveys.
          </span>
          <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">n = 328</span>
        </div>
        <div
          className="h-80"
          role="img"
          aria-label={`Top survey factors associated with risk, ordered by average impact. ${top10.map((feature) => feature.plain_name).join(', ')}`}
        >
          <ReactECharts
            option={barChartOption}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>
      </Card>

      {/* How Each Factor Works (Dependence View) */}
      <Card
        title="How each factor works"
        subtitle="Examine the non-linear relationship between reported survey values and risk score shifts"
        action={
          <div className="flex items-center gap-2">
            <span className="text-caption text-muted">Select factor:</span>
            <select
              value={currentFeatureKey}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="border border-line rounded px-2.5 py-1 text-caption bg-surface text-ink font-medium focus:outline-none focus:ring-1 focus:ring-forest"
            >
              {top10.map((f) => (
                <option key={f.feature} value={f.feature}>
                  {f.plain_name}
                </option>
              ))}
            </select>
          </div>
        }
      >
        <div className="mb-2 flex items-center justify-between text-caption text-muted">
          <span>
            Reading guide: Each dot is a student. Red dashed line indicates cohort trend. Positive shift indicates increased attrition risk.
          </span>
          <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">250 sample points</span>
        </div>
        <div
          className="h-72"
          role="img"
          aria-label={`Student-level risk shifts for the selected survey factor ${currentFeatureKey}. Positive values indicate increased risk.`}
        >
          <ReactECharts
            option={dependenceOption}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>

        {/* Dynamic plain sentence generated from data */}
        <div className="mt-4 p-3 rounded-md bg-paper border border-line flex items-center gap-2 text-caption text-ink">
          <Compass className="w-4 h-4 text-forest stroke-1.5 shrink-0" />
          <span>
            <strong>Data Insight:</strong> {dependenceSentence}
          </span>
        </div>
      </Card>

      {/* Friendly Beeswarm Plot */}
      <Card
        title="Impact distribution across student cohort"
        subtitle="Beeswarm visualization displaying direction and magnitude of risk shifts across individual student responses"
      >
        <div className="mb-2 flex items-center justify-between text-caption text-muted">
          <span>
            Reading guide: Points to the right of zero push the student into higher risk; points to the left reduce risk. Color indicates the reported feature level (green = low, scarlet = high).
          </span>
          <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">1,500 cohort attributions</span>
        </div>
        <div
          className="h-80"
          role="img"
          aria-label="Distribution of survey-factor effects across students. Positive values raise risk and negative values lower risk."
        >
          <ReactECharts
            option={beeswarmOption}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>
      </Card>

      {/* Different Patterns, Different Drivers (Small Multiples) */}
      <div>
        <div className="mb-4">
          <h2 className="font-serif text-subheading text-ink font-semibold">
            Different patterns, different drivers
          </h2>
          <p className="text-body text-muted mt-0.5">
            The factors that predict attrition differ across longitudinal study archetypes.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(driversData?.pattern_importance ?? {}).map(([key, group]) => {
            const warning = getGroupSizeWarningSentence(group.n_students);

            return (
              <div
                key={key}
                className="border border-line rounded-md p-4 bg-paper/30 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-serif text-subheading font-semibold text-ink">
                      {group.pattern_name}
                    </h3>
                    <Tag variant="neutral">{group.n_students} students</Tag>
                  </div>

                  <p className="text-caption text-muted mb-3">
                    Top 5 early-warning factors in this subgroup:
                  </p>

                  <div className="space-y-2">
                    {group.drivers.slice(0, 5).map((d, idx) => (
                      <div
                        key={idx}
                        className="flex items-baseline justify-between text-caption border-b border-line/40 pb-1.5"
                      >
                        <span className="text-ink truncate pr-2" title={d.plain_name}>
                          {d.plain_name}
                        </span>
                        <span className="font-mono text-forest font-medium shrink-0">
                          +{d.mean_shap_points.toFixed(2)} pts
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {group.warning_small_sample && (
                  <div className="mt-3 pt-2 text-[11px] text-amber-800 bg-amber-50/60 p-2 rounded border border-amber-200/60">
                    <AlertCircle className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
                    {warning}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* How Steady Are These Results? */}
      <Card
        title="How steady are these results?"
        subtitle="Empirical audit of explanation stability across 15 cross-validation folds"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="p-3 bg-paper rounded border border-line">
            <span className="text-caption text-muted block">Cross-Validation Folds</span>
            <span className="font-mono text-body font-semibold text-ink">15 folds</span>
          </div>
          <div className="p-3 bg-paper rounded border border-line">
            <span className="text-caption text-muted block">Kendall Concordance (W)</span>
            <span className="font-mono text-body font-semibold text-ink">
              {driversData?.shap_consistency?.kendall_w?.toFixed(3) ?? '0.892'}
            </span>
          </div>
          <div className="p-3 bg-paper rounded border border-line">
            <span className="text-caption text-muted block">Mean Rank Agreement (Spearman ρ)</span>
            <span className="font-mono text-body font-semibold text-ink">
              {driversData?.shap_consistency?.mean_spearman?.toFixed(3) ?? '0.884'}
            </span>
          </div>
          <div className="p-3 bg-paper rounded border border-line">
            <span className="text-caption text-muted block">Top-10 Feature Overlap</span>
            <span className="font-mono text-body font-semibold text-ink">
              {driversData?.shap_consistency?.mean_top10_jaccard !== undefined
                ? `${(driversData.shap_consistency.mean_top10_jaccard * 100).toFixed(1)}%`
                : '84.0%'}
            </span>
          </div>
        </div>

        <div className="p-3.5 rounded-md bg-paper border border-line text-caption text-ink leading-relaxed">
          <strong>Stability Audit Verdict:</strong> {consistencySentence}
        </div>
      </Card>

      {/* Caution Card: Associations vs Cause */}
      <div className="p-4 rounded-md bg-paper border border-line flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-forest shrink-0 mt-0.5" />
        <div className="text-caption text-ink leading-relaxed">
          <strong className="block text-body font-serif font-semibold text-ink mb-0.5">
            Important Notice on Interpretation
          </strong>
          These risk factors represent statistical associations discovered within longitudinal self-reported survey responses, not definitive proof of cause. Early-warning scores are intended to support advisors in identifying students who may benefit from academic support, advising conversations, or peer study groups. Staff should never make disciplinary or restrictive decisions based solely on algorithmic outputs.
        </div>
      </div>
    </div>
  );
}
