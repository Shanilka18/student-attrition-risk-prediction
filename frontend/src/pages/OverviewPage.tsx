import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  ArrowRight,
  ShieldCheck,
  FileSpreadsheet,
  RefreshCw,
  Info,
} from 'lucide-react';
import { useStatus, useLeaderboard, useCategoriesBest } from '@/api/queries';
import { Stat } from '@/components/ui/Stat';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Tag } from '@/components/ui/Tag';
import { Term } from '@/components/ui/Term';
import { Skeleton } from '@/components/ui/Skeleton';
import { ErrorState } from '@/components/ui/ErrorState';
import { getRiskBandDefinitionLabel } from '@/content/riskBands';

export function OverviewPage() {
  const navigate = useNavigate();
  const {
    data: status,
    isLoading: isStatusLoading,
    isError: isStatusError,
    error: statusError,
    refetch: refetchStatus,
  } = useStatus();
  const { data: leaderboard, isLoading: isLeaderboardLoading } =
    useLeaderboard('classification');
  const { data: categoriesBest, isLoading: isCategoriesLoading } = useCategoriesBest('classification');

  const bands = status?.bands?.test_holdout_verification?.bands ?? [];
  const definitions = status?.bands?.definitions;
  const totalStudents = status?.n_students_total;
  const trainStudents = status?.n_students_train;
  const testStudents = status?.n_students_test;
  const totalModels = leaderboard?.total_models;
  const champion = status?.champion;

  const highRiskShare = useMemo(() => {
    if (!testStudents) return null;
    const modStudents =
      bands.find((band) => band.band_key === 'moderate')?.n_students ?? 0;
    const highStudents =
      bands.find((band) => band.band_key === 'high')?.n_students ?? 0;
    const sum = modStudents + highStudents;
    return ((sum / testStudents) * 100).toFixed(1);
  }, [bands, testStudents]);

  const histogramOption = useMemo(() => {
    const series = bands.map((band) => ({
      value: band.n_students,
      itemStyle: {
        color: definitions?.[band.band_key]?.hex_color ?? '#5B6168',
        borderRadius: [3, 3, 0, 0],
      },
    }));
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any[]) => {
          const item = params[0];
          const band = bands[item.dataIndex];
          const pct = testStudents
            ? ((item.value / testStudents) * 100).toFixed(1)
            : 'N/A';
          return `<div style="font-family: inherit; font-size: 13px; padding: 4px;">
            <div style="font-weight: 600; color: #1A1D21;">${item.name}</div>
            <div style="color: #5B6168; margin-top: 2px;">Students: <strong>${item.value}</strong> (${pct}%)</div>
            <div style="color: #5B6168;">Observed at-risk rate: <strong>${band.observed_at_risk_pct.toFixed(1)}%</strong></div>
          </div>`;
        },
      },
      grid: { top: 24, right: 16, bottom: 36, left: 48 },
      xAxis: {
        type: 'category',
        data: bands.map((band) =>
          getRiskBandDefinitionLabel(
            definitions?.[band.band_key],
            band.band_label
          )
        ),
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisTick: { alignWithLabel: true },
        axisLabel: { color: '#5B6168', fontSize: 11, interval: 0, fontFamily: 'inherit' },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#EEEAE2', type: 'dashed' } },
        axisLabel: { color: '#5B6168', fontSize: 11, fontFamily: 'inherit' },
      },
      series: [{ name: 'Students', type: 'bar', barWidth: '45%', data: series }],
    };
  }, [bands, definitions, testStudents]);

  if (isStatusLoading && isLeaderboardLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (isStatusError) {
    return (
      <ErrorState
        title="Project results are unavailable"
        message="The active model records could not be loaded. Check that the server is running and try again."
        error={statusError}
        onRetry={() => void refetchStatus()}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Title & Editorial Description */}
      <div>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Engagement Index
        </h1>
        <p className="text-body text-muted mt-1.5 max-w-3xl">
          A score from 0 to 100 that shows how much a student&apos;s study engagement signals a risk
          of leaving. Higher means higher risk.
        </p>
      </div>

      {/* Row of 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat
          label="Students in the data"
          value={totalStudents?.toLocaleString() ?? 'Unavailable'}
          note={`${trainStudents?.toLocaleString() ?? 'Unavailable'} development · ${testStudents?.toLocaleString() ?? 'Unavailable'} held-out test`}
        />
        <Stat
          label="Models compared"
          value={totalModels ?? 'Unavailable'}
          note="Candidate models in the active classification leaderboard"
        />
        <Stat
          label="Model in use"
          value={champion?.display_name ?? 'Unavailable'}
          note={champion?.plain_name ?? 'Active model details are unavailable'}
        />
        <Stat
          label="Share flagged as higher risk"
          value={highRiskShare === null ? 'Unavailable' : `${highRiskShare}%`}
          note="Held-out students in the Moderate or High bands"
        />
      </div>

      {/* How the Score is Made: 4-Step Horizontal Diagram */}
      <Card
        title="How the score is made"
        subtitle="A transparent four-stage pipeline from student survey responses to calibrated early-warning signals"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
          {/* Step 1 */}
          <div className="border border-line rounded-md p-4 bg-paper/40 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="w-6 h-6 rounded-full bg-forest text-paper text-xs font-mono font-semibold flex items-center justify-center">
                  1
                </span>
                <span className="text-caption font-mono text-muted">Step 1</span>
              </div>
              <h4 className="font-serif text-subheading text-ink font-semibold mb-1.5">
                Students answer a short questionnaire
              </h4>
              <p className="text-caption text-muted leading-relaxed">
                Students self-report study habits, attendance, peer collaboration, and academic sentiment across 4 semester checkpoints.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-line/60">
              <span className="text-caption font-mono font-medium text-forest block">
                {totalStudents?.toLocaleString() ?? 'Unavailable'} records in the active run
              </span>
              <span className="text-caption text-muted">35 survey columns recorded</span>
            </div>
          </div>

          {/* Step 2 */}
          <div className="border border-line rounded-md p-4 bg-paper/40 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="w-6 h-6 rounded-full bg-forest text-paper text-xs font-mono font-semibold flex items-center justify-center">
                  2
                </span>
                <span className="text-caption font-mono text-muted">Step 2</span>
              </div>
              <h4 className="font-serif text-subheading text-ink font-semibold mb-1.5">
                Answers are checked and cleaned
              </h4>
              <p className="text-caption text-muted leading-relaxed">
                Automated validation detects anomalous response times, checks <Term termKey="likert" customLabel="Likert scales" />, and verifies cohort completeness.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-line/60">
              <span className="text-caption font-mono font-medium text-forest block">
                {trainStudents?.toLocaleString() ?? 'Unavailable'} development records
              </span>
              <span className="text-caption text-muted">Cleaning and missingness are reported in the run record</span>
            </div>
          </div>

          {/* Step 3 */}
          <div className="border border-line rounded-md p-4 bg-paper/40 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="w-6 h-6 rounded-full bg-forest text-paper text-xs font-mono font-semibold flex items-center justify-center">
                  3
                </span>
                <span className="text-caption font-mono text-muted">Step 3</span>
              </div>
              <h4 className="font-serif text-subheading text-ink font-semibold mb-1.5">
                Study patterns are found across the semester
              </h4>
              <p className="text-caption text-muted leading-relaxed">
                <Term termKey="hmm_state" customLabel="Hidden Markov Models" /> discover temporal engagement trajectories: Consistent, Declining, Sporadic, or Improving.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-line/60">
              <span className="text-caption font-mono font-medium text-forest block">
                4 latent study patterns
              </span>
              <span className="text-caption text-muted">3 discrete engagement levels</span>
            </div>
          </div>

          {/* Step 4 */}
          <div className="border border-line rounded-md p-4 bg-paper/40 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="w-6 h-6 rounded-full bg-forest text-paper text-xs font-mono font-semibold flex items-center justify-center">
                  4
                </span>
                <span className="text-caption font-mono text-muted">Step 4</span>
              </div>
              <h4 className="font-serif text-subheading text-ink font-semibold mb-1.5">
                The best model gives each student a score
              </h4>
              <p className="text-caption text-muted leading-relaxed">
                The rigorously evaluated champion model assigns a calibrated 0–100 risk score and identifies the top 3 contributing drivers.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-line/60">
              <span className="text-caption font-mono font-medium text-forest block">
                {totalModels ?? 'Unavailable'} models in this track
              </span>
              <span className="text-caption text-muted">0–100 score + plain reasons</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Score Distribution & Risk Band Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Score Distribution Histogram */}
        <div className="lg:col-span-6">
          <Card
            title="Score distribution"
            subtitle="Held-out students grouped by the active model's saved risk-band definitions"
          >
            <div className="mb-2 flex items-center justify-between text-caption text-muted">
              <span>Bars show the number of held-out students and their observed at-risk rate.</span>
              <span className="font-mono text-caption text-ink font-medium shrink-0 ml-2">n = {testStudents?.toLocaleString() ?? 'Unavailable'}</span>
            </div>
            <div
              className="h-64"
              role="img"
              aria-label="Score distribution by risk band. Bars show held-out student counts and observed at-risk rates."
            >
              <ReactECharts
                option={histogramOption}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
            <div className="flex items-center justify-center gap-6 mt-3 text-caption text-muted pt-2 border-t border-line/60">
              {bands.map((band) => (
                <div key={band.band_key} className="flex items-center gap-1.5">
                  <span
                    className="w-3 h-3 rounded-sm"
                    style={{
                      backgroundColor:
                        definitions?.[band.band_key]?.hex_color ?? '#5B6168',
                    }}
                  />
                  <span>
                    {getRiskBandDefinitionLabel(
                      definitions?.[band.band_key],
                      band.band_label
                    )}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Risk Bands Table */}
        <div className="lg:col-span-6">
          <Card
            title="Observed risk by band"
            subtitle="Holdout test verification showing actual attrition rates across defined MEI risk tiers"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-body">
                <thead>
                  <tr className="border-b border-line text-caption font-mono text-muted">
                    <th className="pb-2.5 font-medium">Risk Band</th>
                    <th className="pb-2.5 font-medium text-center">Score Range</th>
                    <th className="pb-2.5 font-medium text-right">Students</th>
                    <th className="pb-2.5 font-medium text-right">Observed Risk</th>
                    <th className="pb-2.5 font-medium">Recommended Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/60">
                  {bands.map((b) => (
                    <tr key={b.band_key} className="hover:bg-paper/30 transition-colors">
                      <td className="py-3 font-medium text-ink flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{
                            backgroundColor:
                              definitions?.[b.band_key]?.hex_color ?? '#5B6168',
                          }}
                        />
                        {b.band_label}
                      </td>
                      <td className="py-3 text-center font-mono text-caption text-ink">
                        {b.score_range}
                      </td>
                      <td className="py-3 text-right font-mono text-caption text-ink">
                        {b.n_students}
                      </td>
                      <td className="py-3 text-right font-mono text-caption font-semibold text-ink">
                        {b.observed_at_risk_pct.toFixed(1)}%
                      </td>
                      <td className="py-3 text-caption text-muted leading-snug">
                        {definitions?.[b.band_key]?.action ?? 'Action not recorded'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Monotonicity Verification Status */}
            <div className="mt-4 p-3 rounded-md bg-paper border border-line flex items-center justify-between text-caption">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-forest stroke-1.5 shrink-0" />
                <span className="text-ink">
                  <strong>Strict Monotonicity:</strong>{' '}
                  {status?.bands?.test_holdout_verification?.observed_rates
                    ?.map((rate) => `${rate.toFixed(1)}%`)
                    .join(' < ')}
                </span>
              </div>
              <Tag
                variant={
                  status?.bands?.test_holdout_verification?.is_strictly_monotonic
                    ? 'good-fit'
                    : 'overfit'
                }
              >
                {status?.bands?.test_holdout_verification?.monotonicity_status ??
                  'NOT RECORDED'}
              </Tag>
            </div>
          </Card>
        </div>
      </div>

      {/* Shortcuts & Category Best Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Quick Action Shortcuts */}
        <div className="lg:col-span-5 flex flex-col justify-between p-5 bg-paper rounded-md border border-line">
          <div>
            <h3 className="font-serif text-subheading text-ink font-semibold mb-1">
              Operational Actions
            </h3>
            <p className="text-caption text-muted leading-relaxed mb-4">
              Deploy the calibrated model onto a new student roster or retrain the pipeline when fresh semester data arrives.
            </p>
            <div className="space-y-2.5">
              <Button
                variant="primary"
                onClick={() => navigate('/score')}
                icon={<FileSpreadsheet className="w-4 h-4" />}
                className="w-full justify-start"
              >
                Score student cohort
              </Button>
              <Button
                variant="secondary"
                onClick={() => navigate('/train')}
                icon={<RefreshCw className="w-4 h-4" />}
                className="w-full justify-start"
              >
                Train on new survey data
              </Button>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-line/60 flex items-start gap-2 text-caption text-muted">
            <Info className="w-4 h-4 text-forest stroke-1.5 shrink-0 mt-0.5" />
            <span>
              All scores and driver attributions are calculated on local university hardware with no cloud data transmission.
            </span>
          </div>
        </div>

        {/* Best in Each Category Mini-List */}
        <div className="lg:col-span-7">
          <Card
            title="Best model in each category"
            subtitle="Top-performing algorithm per model family evaluated on 15 cross-validation folds"
            action={
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate('/leaderboard')}
                icon={<ArrowRight className="w-3.5 h-3.5" />}
              >
                View full leaderboard
              </Button>
            }
          >
            {isCategoriesLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-10" />
                <Skeleton className="h-10" />
                <Skeleton className="h-10" />
              </div>
            ) : (
              <div className="divide-y divide-line/60">
                {Object.entries(categoriesBest ?? {}).map(([cat, row]) => {
                  const isChamp = row.is_champion;
                  return (
                    <div
                      key={cat}
                      onClick={() => navigate(`/models/${row.model_id}?track=classification`)}
                      className="py-2.5 flex items-center justify-between hover:bg-paper/40 px-2 rounded cursor-pointer transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <Tag variant={isChamp ? 'primary' : 'neutral'}>
                          {cat.toUpperCase()}
                        </Tag>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-body text-ink">
                              {row.display_name}
                            </span>
                            {isChamp && (
                              <Tag variant="primary">CHAMPION</Tag>
                            )}
                          </div>
                          <div className="text-caption text-muted">
                            {row.plain_name}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 text-right">
                        <div>
                          <div className="font-mono text-body font-semibold text-ink">
                            {row.leaderboard_score.toFixed(3)}
                          </div>
                          <div className="text-caption font-mono text-muted">
                            Rank #{row.rank}
                          </div>
                        </div>
                        <ArrowRight className="w-4 h-4 text-muted/60" />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
