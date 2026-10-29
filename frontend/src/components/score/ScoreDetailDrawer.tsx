import { useState } from 'react';
import { X, TrendingUp, TrendingDown, ShieldCheck, Activity } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { Badge } from '@/components/ui/Badge';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useStudentDetail } from '@/api/queries';

interface ScoreDetailDrawerProps {
  uploadId: string;
  studentId: string | null;
  onClose: () => void;
}

const INDICATOR_NAMES: Record<string, string> = {
  resource_access: 'LMS Access',
  study_time: 'Study Time',
  review_time: 'Review Time',
  study_consistency: 'Consistency',
  participation: 'Participation',
  assignment_engagement: 'Assignments',
  quiz_practice: 'Practice & Quizzes',
};

export function ScoreDetailDrawer({ uploadId, studentId, onClose }: ScoreDetailDrawerProps) {
  const { data: detail, isLoading, error } = useStudentDetail(uploadId, studentId);
  const [selectedIndicator, setSelectedIndicator] = useState<string>('composite');

  if (!studentId) return null;

  // Chart configuration for ECharts
  const getTrajectoryOption = () => {
    if (!detail) return {};

    const periods = ['Weeks 1–4', 'Weeks 5–8', 'Weeks 9–12', 'Weeks 13–16'];
    const isComposite = selectedIndicator === 'composite';

    const seriesData = isComposite
      ? detail.trajectories.composite
      : detail.trajectories.indicators[selectedIndicator] || [3, 3, 3, 3];

    return {
      grid: {
        top: 25,
        right: 25,
        bottom: 30,
        left: 40,
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any[]) => {
          const p = params[0];
          return `<div class="text-caption font-sans">
            <div class="font-semibold text-ink">${p.name}</div>
            <div class="text-small text-muted mt-0.5">Rating: <span class="font-mono font-semibold text-forest">${p.value} / 5.0</span></div>
          </div>`;
        },
      },
      xAxis: {
        type: 'category',
        data: periods,
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#6B665E', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        min: 1,
        max: 5,
        interval: 1,
        splitLine: { lineStyle: { color: '#EDE8DF', type: 'dashed' } },
        axisLabel: { color: '#6B665E', fontSize: 11 },
      },
      series: [
        {
          name: isComposite ? 'Composite Pace' : INDICATOR_NAMES[selectedIndicator] || selectedIndicator,
          type: 'line',
          data: seriesData,
          smooth: true,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: {
            color: isComposite ? '#1F4E46' : '#C85A32',
            width: 3,
          },
          itemStyle: {
            color: isComposite ? '#1F4E46' : '#C85A32',
          },
          areaStyle: {
            color: isComposite ? 'rgba(31, 78, 70, 0.08)' : 'rgba(200, 90, 50, 0.08)',
          },
        },
      ],
    };
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-ink/30 backdrop-blur-none flex justify-end">
      <div className="relative w-full max-w-xl bg-surface border-l border-border shadow-xl h-full flex flex-col">
        {/* Drawer Header */}
        <div className="p-6 border-b border-border bg-paper flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-forest-subtle flex items-center justify-center text-forest">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-muted uppercase tracking-wider block">
                Student Profile
              </span>
              <h3 className="font-serif text-section font-semibold text-ink">
                {studentId}
              </h3>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 text-muted hover:text-ink rounded-md hover:bg-paper transition-colors"
            aria-label="Close drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading && (
            <div className="py-20 text-center text-caption text-muted">
              <div className="w-6 h-6 border-2 border-forest border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              Loading student trajectories and explanation...
            </div>
          )}

          {error && (
            <div className="p-4 bg-crimson-subtle border border-crimson/30 rounded text-caption text-crimson">
              Failed loading detail: {error.message}
            </div>
          )}

          {detail && (
            <>
              {/* Score Headline Card */}
              <div className="p-5 bg-paper rounded-lg border border-border flex items-center justify-between">
                <div>
                  <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
                    Multimodal Engagement Index
                  </span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="font-serif text-hero font-bold text-ink tabular-nums">
                      {detail.mei}
                    </span>
                    <span className="text-body text-muted">/ 100</span>
                  </div>
                </div>

                <div className="flex flex-col items-end gap-1.5">
                  <Badge
                    variant={
                      detail.band === 'High'
                        ? 'crimson'
                        : detail.band === 'Moderate'
                        ? 'amber'
                        : 'forest'
                    }
                    size="md"
                  >
                    {detail.band} Risk
                  </Badge>
                  <Badge variant={detail.flag === 'Yes' ? 'crimson' : 'neutral'} size="sm">
                    {detail.flag === 'Yes' ? 'At-Risk Flag' : 'Normal'}
                  </Badge>
                </div>
              </div>

              {/* 4-Period Longitudinal Trajectory Section */}
              <Card className="border-border">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-caption font-semibold uppercase tracking-wider text-ink">
                      4-Period Engagement Trajectory
                    </CardTitle>
                    <span className="text-[11px] text-muted">1 (Low) to 5 (High)</span>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Trajectory Selector Pills */}
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => setSelectedIndicator('composite')}
                      className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors ${
                        selectedIndicator === 'composite'
                          ? 'bg-forest text-surface font-semibold'
                          : 'bg-paper text-ink hover:bg-border/60 border border-border'
                      }`}
                    >
                      Composite Average
                    </button>
                    {Object.entries(INDICATOR_NAMES).map(([key, label]) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setSelectedIndicator(key)}
                        className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                          selectedIndicator === key
                            ? 'bg-forest text-surface font-semibold'
                            : 'bg-paper text-ink hover:bg-border/60 border border-border'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {/* Chart Container */}
                  <div
                    className="h-48 w-full bg-paper rounded border border-border p-2"
                    role="img"
                    aria-label={`Four-period engagement trajectory for ${selectedIndicator}.`}
                  >
                    <ReactECharts
                      option={getTrajectoryOption()}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </div>
                </CardContent>
              </Card>

              {/* Additive Waterfall Explanation Section */}
              <Card className="border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-caption font-semibold uppercase tracking-wider text-ink">
                    Waterfall Point Attribution
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2.5">
                  {/* Baseline */}
                  <div className="flex items-center justify-between p-2.5 bg-paper rounded border border-border text-caption">
                    <span className="text-muted font-medium">Cohort Baseline Average</span>
                    <span className="font-mono text-ink font-semibold">
                      {detail.baseline_mei.toFixed(1)} pts
                    </span>
                  </div>

                  {/* Drivers */}
                  {detail.reasons.map((r, i) => {
                    const isRisk = r.direction === 'risk_increasing';
                    return (
                      <div
                        key={i}
                        className={`flex items-center justify-between p-2.5 rounded border text-caption ${
                          isRisk
                            ? 'bg-crimson-subtle/40 border-crimson/20'
                            : 'bg-forest-subtle/40 border-forest/20'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {isRisk ? (
                            <TrendingUp className="w-4 h-4 text-crimson shrink-0" />
                          ) : (
                            <TrendingDown className="w-4 h-4 text-forest shrink-0" />
                          )}
                          <span className="text-ink">{r.text}</span>
                        </div>
                        <span
                          className={`font-mono font-semibold shrink-0 ${
                            isRisk ? 'text-crimson' : 'text-forest'
                          }`}
                        >
                          {r.points > 0 ? `+${r.points}` : r.points} pts
                        </span>
                      </div>
                    );
                  })}

                  {/* Final MEI */}
                  <div className="flex items-center justify-between p-2.5 bg-forest-subtle border border-forest/30 rounded text-caption font-semibold">
                    <span className="text-ink">Calculated MEI Score</span>
                    <span className="font-mono text-forest text-body">{detail.mei} pts</span>
                  </div>
                </CardContent>
              </Card>

              {/* Longitudinal Pattern & Level Explanations */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-caption">
                <div className="p-3.5 bg-paper rounded border border-border space-y-1">
                  <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
                    Trajectory Pattern
                  </span>
                  <span className="font-serif font-bold text-ink block">{detail.pattern}</span>
                  <p className="text-small text-muted leading-relaxed">
                    {detail.plain_meanings.pattern}
                  </p>
                </div>

                <div className="p-3.5 bg-paper rounded border border-border space-y-1">
                  <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
                    Engagement Intensity
                  </span>
                  <span className="font-serif font-bold text-ink block">{detail.level}</span>
                  <p className="text-small text-muted leading-relaxed">
                    {detail.plain_meanings.level}
                  </p>
                </div>
              </div>

              {/* Ethical usage note */}
              <div className="p-4 bg-paper rounded-md border border-border flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-forest shrink-0 mt-0.5" />
                <div className="text-small text-muted leading-relaxed">
                  <span className="font-semibold text-ink">What this does not mean:</span> {detail.plain_meanings.disclaimer}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
