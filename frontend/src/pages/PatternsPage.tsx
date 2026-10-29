import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { usePatterns } from '@/api/queries';
import { Card } from '@/components/ui/Card';
import { ErrorState } from '@/components/ui/ErrorState';
import { Skeleton } from '@/components/ui/Skeleton';
import { Tag } from '@/components/ui/Tag';
import { Tooltip } from '@/components/ui/Tooltip';

const periods = [
  { key: 'composite_p1', label: 'Period 1' },
  { key: 'composite_p2', label: 'Period 2' },
  { key: 'composite_p3', label: 'Period 3' },
  { key: 'composite_p4', label: 'Period 4' },
];

const indicators = [
  'Attendance',
  'Study hours',
  'LMS logins',
  'Submissions',
  'Forum posts',
  'Downloads',
  'Quiz attempts',
];

const patternDetails = [
  {
    name: 'Consistent',
    color: '#2E7D4F',
    meaning: 'Engagement remains relatively steady across the four survey periods.',
  },
  {
    name: 'Declining',
    color: '#B3261E',
    meaning: 'Engagement falls across the four survey periods.',
  },
  {
    name: 'Improving',
    color: '#1F4E46',
    meaning: 'Engagement rises across the four survey periods.',
  },
  {
    name: 'Sporadic',
    color: '#C98A00',
    meaning: 'Engagement varies across the four survey periods.',
  },
];

function formatMetric(value: unknown, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : 'Not available';
}

export function PatternsPage() {
  const { data, isLoading, isError, error, refetch } = usePatterns();
  const stateEntries = useMemo(
    () =>
      Object.entries(data?.state_names ?? {}).sort(
        ([left], [right]) => Number(left) - Number(right)
      ),
    [data?.state_names]
  );
  const stateLabels = stateEntries.map(([, label]) => label);
  const riskRates = data?.risk_rates_by_pattern ?? {};
  const totalStudents = Object.values(riskRates).reduce(
    (total, pattern) => total + pattern.n_students,
    0
  );
  const patternRows = patternDetails.flatMap((details) => {
    const risk = riskRates[details.name];
    if (!risk) return [];
    return [
      {
        ...details,
        ...risk,
        sharePct: totalStudents ? (risk.n_students / totalStudents) * 100 : 0,
      },
    ];
  });

  const emissionMeans = data?.emission_means;
  const emissionMatrix = useMemo(() => {
    if (Array.isArray(emissionMeans)) return emissionMeans;
    return stateEntries.map(([stateId]) => emissionMeans?.[stateId] ?? []);
  }, [emissionMeans, stateEntries]);

  const trajectoryOption = useMemo(() => {
    const series = patternRows.flatMap((pattern) => {
      const trajectory = data?.mean_trajectories_by_pattern[pattern.name];
      if (!trajectory) return [];
      return [
        {
          name: pattern.name,
          type: 'line',
          smooth: true,
          data: periods.map(({ key }) => trajectory[key] ?? null),
          lineStyle: { width: 3, color: pattern.color },
          itemStyle: { color: pattern.color },
          symbol: 'circle',
          symbolSize: 7,
        },
      ];
    });

    return {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0, textStyle: { color: '#5B6168' } },
      grid: { top: 24, right: 24, bottom: 48, left: 52 },
      xAxis: {
        type: 'category',
        data: periods.map(({ label }) => label),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#5B6168' },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: 'Mean engagement (0–100)',
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168' },
      },
      series,
    };
  }, [data?.mean_trajectories_by_pattern, patternRows]);

  const transitionOption = useMemo(() => {
    const matrix = data?.transition_matrix ?? [];
    const values: [number, number, number][] = [];
    matrix.forEach((row, rowIndex) => {
      row.forEach((probability, columnIndex) => {
        values.push([
          columnIndex,
          matrix.length - 1 - rowIndex,
          probability * 100,
        ]);
      });
    });

    return {
      tooltip: {
        position: 'top',
        formatter: (params: { data: [number, number, number] }) => {
          const [column, reversedRow, probability] = params.data;
          const from = stateLabels[matrix.length - 1 - reversedRow] ?? 'Unknown';
          const to = stateLabels[column] ?? 'Unknown';
          return `${from} to ${to}: ${probability.toFixed(1)}%`;
        },
      },
      grid: { top: 16, right: 16, bottom: 32, left: 96 },
      xAxis: {
        type: 'category',
        data: stateLabels,
        name: 'Next period',
        axisLabel: { color: '#1A1D21' },
      },
      yAxis: {
        type: 'category',
        data: [...stateLabels].reverse(),
        name: 'Current period',
        axisLabel: { color: '#1A1D21' },
      },
      visualMap: {
        min: 0,
        max: 100,
        show: false,
        inRange: { color: ['#F5F2EC', '#D4E2D8', '#86B599', '#1F4E46'] },
      },
      series: [
        {
          name: 'Transition probability',
          type: 'heatmap',
          data: values,
          label: {
            show: true,
            formatter: (params: { data: [number, number, number] }) =>
              `${params.data[2].toFixed(0)}%`,
            color: '#1A1D21',
          },
          itemStyle: { borderColor: '#FFFFFF', borderWidth: 2 },
        },
      ],
    };
  }, [data?.transition_matrix, stateLabels]);

  const emissionOption = useMemo(() => {
    const values: [number, number, number][] = [];
    emissionMatrix.forEach((row, stateIndex) => {
      row.forEach((mean, indicatorIndex) => {
        values.push([stateIndex, indicators.length - 1 - indicatorIndex, mean]);
      });
    });

    return {
      tooltip: {
        position: 'top',
        formatter: (params: { data: [number, number, number] }) => {
          const [stateIndex, reversedIndicator, value] = params.data;
          const indicator = indicators[indicators.length - 1 - reversedIndicator];
          const state = stateLabels[stateIndex] ?? 'Unknown';
          return `${indicator} (${state}): ${formatMetric(value, 2)} SD`;
        },
      },
      grid: { top: 16, right: 16, bottom: 24, left: 112 },
      xAxis: {
        type: 'category',
        data: stateLabels,
        axisLabel: { color: '#1A1D21' },
      },
      yAxis: {
        type: 'category',
        data: [...indicators].reverse(),
        axisLabel: { color: '#1A1D21' },
      },
      visualMap: {
        min: -1.5,
        max: 1.5,
        show: false,
        inRange: { color: ['#D65A4B', '#F5F2EC', '#D4E2D8', '#2E7D4F'] },
      },
      series: [
        {
          name: 'Standardized emission mean',
          type: 'heatmap',
          data: values,
          label: {
            show: true,
            formatter: (params: { data: [number, number, number] }) =>
              `${params.data[2] > 0 ? '+' : ''}${params.data[2].toFixed(1)}`,
            color: '#1A1D21',
          },
          itemStyle: { borderColor: '#FFFFFF', borderWidth: 2 },
        },
      ],
    };
  }, [emissionMatrix, stateLabels]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((item) => (
            <Skeleton key={item} className="h-36" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <ErrorState
        title="Study patterns are unavailable"
        message="The active run's state analysis could not be loaded. Check the server and try again."
        error={error}
        onRetry={() => void refetch()}
      />
    );
  }

  const ablationResults = Object.entries(data.ablation_benchmark?.results ?? {});
  const clusteringRows = data.clustering_comparison_table ?? [];
  const aucPValue =
    data.ablation_benchmark?.wilcoxon_tests?.logistic_regression?.auc?.p_value;
  const isAucSignificant =
    typeof aucPValue === 'number' && aucPValue < 0.05;
  const chartDescription = patternRows
    .map((pattern) => {
      const values = periods
        .map(({ key }) => data.mean_trajectories_by_pattern[pattern.name]?.[key])
        .filter((value): value is number => typeof value === 'number');
      return `${pattern.name}: ${values.map((value) => value.toFixed(1)).join(', ')}`;
    })
    .join('; ');

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Study Patterns
        </h1>
        <p className="text-body text-muted mt-1.5 max-w-3xl">
          These patterns describe how survey engagement changed over four periods.
          They are summaries of the active run, not predictions about individual students.
        </p>
        <div className="flex flex-wrap gap-2 mt-3">
          <Tag variant="neutral">Development students: {totalStudents.toLocaleString()}</Tag>
          <Tag variant="neutral">HMM states: {data.hmm_k}</Tag>
          <Tag variant="neutral">Clustering groups: {data.clustering_k}</Tag>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {patternRows.map((pattern) => (
          <Card key={pattern.name} className="flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-serif text-section-title font-semibold text-ink">
                  {pattern.name}
                </h2>
                <Tag variant="neutral">{pattern.sharePct.toFixed(1)}%</Tag>
              </div>
              <p className="text-caption text-muted mt-2">{pattern.meaning}</p>
            </div>
            <div className="mt-4">
              <div className="text-caption text-muted">
                {pattern.n_students.toLocaleString()} development students
              </div>
              <div className="flex items-baseline justify-between mt-2 text-caption">
                <span className="text-muted">Observed at-risk rate</span>
                <span className="font-mono font-semibold text-ink">
                  {pattern.at_risk_pct.toFixed(1)}%
                </span>
              </div>
              <p className="text-[11px] font-mono text-muted text-right mt-1">
                95% CI: [{pattern.ci95_low.toFixed(1)}%–{pattern.ci95_high.toFixed(1)}%]
              </p>
            </div>
          </Card>
        ))}
      </div>

      <Card
        title="Engagement across the four periods"
        subtitle="Mean composite engagement for each pattern in the development data."
      >
        <div
          role="img"
          aria-label={`Mean engagement by pattern across four periods. ${chartDescription}`}
        >
          <ReactECharts
            option={trajectoryOption}
            style={{ height: 320, width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>
        <p className="sr-only">
          Pattern trajectory values by period: {chartDescription}
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card
          title="State transitions"
          subtitle="Each cell shows the chance of moving from the row state to the column state."
        >
          <div
            role="img"
            aria-label={`Transition matrix for ${stateLabels.length} engagement states.`}
          >
            <ReactECharts
              option={transitionOption}
              style={{ height: 320, width: '100%' }}
              opts={{ renderer: 'canvas' }}
            />
          </div>
          <p className="sr-only">
            Transition rows are current states and columns are next-period states.
          </p>
        </Card>

        <Card
          title="Engagement levels within each state"
          subtitle="Values are standardized emission means; positive values are above the cohort mean."
        >
          <div
            role="img"
            aria-label={`Standardized mean profile for ${stateLabels.length} states across ${indicators.length} engagement indicators.`}
          >
            <ReactECharts
              option={emissionOption}
              style={{ height: 320, width: '100%' }}
              opts={{ renderer: 'canvas' }}
            />
          </div>
          <p className="sr-only">
            Rows are engagement indicators and columns are hidden states.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4 pt-3 border-t border-line/60">
            {stateEntries.map(([stateId, stateName]) => (
              <div key={stateId} className="text-center">
                <span className="text-[11px] text-muted block">{stateName} dwell</span>
                <span className="font-mono text-caption font-semibold text-ink">
                  {formatMetric(data.expected_dwell_times[stateId], 2)} periods
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card
        title="Pattern risk rates"
        subtitle="Observed rates and confidence intervals use development data only."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-body">
            <thead>
              <tr className="border-b border-line text-caption text-muted">
                <th className="pb-2 pr-4 font-medium">Pattern</th>
                <th className="pb-2 px-4 text-right font-medium">Students</th>
                <th className="pb-2 px-4 text-right font-medium">Share</th>
                <th className="pb-2 px-4 text-right font-medium">At-risk rate</th>
                <th className="pb-2 pl-4 text-right font-medium">95% CI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60 text-caption">
              {patternRows.map((pattern) => (
                <tr key={pattern.name}>
                  <td className="py-2.5 pr-4 font-medium text-ink">{pattern.name}</td>
                  <td className="py-2.5 px-4 text-right font-mono">
                    {pattern.n_students.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono">
                    {pattern.sharePct.toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono">
                    {pattern.at_risk_pct.toFixed(1)}%
                  </td>
                  <td className="py-2.5 pl-4 text-right font-mono">
                    [{pattern.ci95_low.toFixed(1)}%, {pattern.ci95_high.toFixed(1)}%]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card
          title="Clustering comparison"
          subtitle={`Candidate clustering methods; selected cluster count is K = ${data.clustering_k}.`}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-body">
              <thead>
                <tr className="border-b border-line text-caption text-muted">
                  <th className="pb-2 pr-3 font-medium">Method</th>
                  <th className="pb-2 px-3 text-right font-medium">K</th>
                  <th className="pb-2 px-3 text-right font-medium">
                    <Tooltip content="Higher silhouette values indicate better-separated clusters.">
                      <span className="cursor-help underline decoration-dotted">Silhouette</span>
                    </Tooltip>
                  </th>
                  <th className="pb-2 px-3 text-right font-medium">Davies–Bouldin</th>
                  <th className="pb-2 pl-3 text-right font-medium">BIC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/60 text-caption font-mono">
                {clusteringRows.map((row, index) => (
                  <tr key={`${row.model_key ?? row.algorithm}-${row.k}-${index}`}>
                    <td className="py-2 pr-3 font-sans text-ink">
                      {String(row.algorithm ?? 'Unknown')}
                    </td>
                    <td className="py-2 px-3 text-right">{row.k ?? '—'}</td>
                    <td className="py-2 px-3 text-right">
                      {formatMetric(row.silhouette)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {formatMetric(row.davies_bouldin)}
                    </td>
                    <td className="py-2 pl-3 text-right">
                      {formatMetric(row.bic)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card
          title="Do state features improve prediction?"
          subtitle="Cross-validation ablation results for each model and feature set."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-body">
              <thead>
                <tr className="border-b border-line text-caption text-muted">
                  <th className="pb-2 pr-3 font-medium">Model / feature set</th>
                  <th className="pb-2 px-3 text-right font-medium">AUC</th>
                  <th className="pb-2 px-3 text-right font-medium">F1</th>
                  <th className="pb-2 pl-3 text-right font-medium">MCC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/60 text-caption font-mono">
                {ablationResults.flatMap(([model, featureSets]) =>
                  Object.entries(featureSets as Record<string, Record<string, number>>).map(
                    ([featureSet, metrics]) => (
                      <tr key={`${model}-${featureSet}`}>
                        <td className="py-2 pr-3 font-sans text-ink">
                          {model.replace(/_/g, ' ')} / {featureSet}
                        </td>
                        <td className="py-2 px-3 text-right">
                          {formatMetric(metrics.auc_mean)}
                        </td>
                        <td className="py-2 px-3 text-right">
                          {formatMetric(metrics.f1_mean)}
                        </td>
                        <td className="py-2 pl-3 text-right">
                          {formatMetric(metrics.mcc_mean)}
                        </td>
                      </tr>
                    )
                  )
                )}
              </tbody>
            </table>
          </div>
          <p className="mt-4 p-3 rounded-md bg-paper border border-line text-caption text-ink">
            {typeof aucPValue === 'number'
              ? `The AUC ablation p-value is ${aucPValue.toFixed(3)}. The difference is ${
                  isAucSignificant ? 'statistically significant' : 'not statistically significant'
                } at the 0.05 level.`
              : 'The AUC significance test is not available in this run.'}
          </p>
        </Card>
      </div>
    </div>
  );
}
