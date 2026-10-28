import { useState, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Download,
  Search,
  CheckCircle2,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Eye,
  FileSpreadsheet,
  RotateCcw,
  Award,
} from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  useScoringResults,
  getDownloadResultsUrl,
} from '@/api/queries';
import type { RunScoringResponse, ScoredStudentRow } from '@/api/types';
import { ScoreDetailDrawer } from './ScoreDetailDrawer';

interface ScoreResultsStepProps {
  uploadId: string;
  scoringResponse: RunScoringResponse;
  onReset: () => void;
}

export function ScoreResultsStep({
  uploadId,
  scoringResponse,
  onReset,
}: ScoreResultsStepProps) {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortField, setSortField] = useState<'mei' | 'student_id'>('mei');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [filterBand, setFilterBand] = useState<string>('all');
  const [filterFlag, setFilterFlag] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [searchParams] = useSearchParams();
  // Selected student for drawer
  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(
    () => searchParams.get('student_id') || null
  );

  const { data: resultsData, isLoading } = useScoringResults({
    uploadId,
    page,
    pageSize,
    sort: sortField,
    direction: sortDir,
    filterBand,
    filterFlag,
  });

  const dist = scoringResponse.risk_distribution;
  const distPct = scoringResponse.risk_distribution_pct;
  const totalScored = scoringResponse.scored_count;
  const outcome = scoringResponse.outcome_check;

  // Filter client-side search query
  const displayedRows = useMemo(() => {
    if (!resultsData?.rows) return [];
    if (!searchQuery.trim()) return resultsData.rows;
    const q = searchQuery.toLowerCase().trim();
    return resultsData.rows.filter((r) => r.student_id.toLowerCase().includes(q));
  }, [resultsData?.rows, searchQuery]);

  // Histogram option for risk distribution
  const getDistributionChartOption = () => {
    return {
      grid: {
        top: 25,
        right: 25,
        bottom: 25,
        left: 45,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any[]) => {
          const p = params[0];
          return `<div class="text-caption font-sans">
            <div class="font-semibold text-ink">${p.name} Risk Band</div>
            <div class="text-small text-muted mt-0.5">Students: <span class="font-mono font-semibold text-ink">${p.value} (${(
            (p.value / (totalScored || 1)) *
            100
          ).toFixed(1)}%)</span></div>
          </div>`;
        },
      },
      xAxis: {
        type: 'category',
        data: ['Low Risk (0–29)', 'Moderate Risk (30–69)', 'High Risk (70–100)'],
        axisLine: { lineStyle: { color: '#DDD8CE' } },
        axisLabel: { color: '#6B665E', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: '#EDE8DF', type: 'dashed' } },
        axisLabel: { color: '#6B665E', fontSize: 11 },
      },
      series: [
        {
          name: 'Students',
          type: 'bar',
          data: [
            { value: dist.Low, itemStyle: { color: '#1F4E46' } },
            { value: dist.Moderate, itemStyle: { color: '#C47D2B' } },
            { value: dist.High, itemStyle: { color: '#B33929' } },
          ],
          barWidth: '45%',
          label: {
            show: true,
            position: 'top',
            color: '#24211D',
            fontSize: 11,
            formatter: '{c}',
          },
        },
      ],
    };
  };

  const handleSort = (field: 'mei' | 'student_id') => {
    if (sortField === field) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir(field === 'mei' ? 'desc' : 'asc');
    }
    setPage(1);
  };

  return (
    <div className="space-y-6">
      {/* Top Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-serif text-page-title text-ink font-semibold">
            Cohort Scoring Results
          </h2>
          <p className="text-body text-muted mt-1">
            Multimodal Engagement Index computed for {totalScored.toLocaleString()} students using the frozen champion model.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <Button variant="secondary" size="sm" onClick={onReset} icon={<RotateCcw className="w-3.5 h-3.5" />}>
            Score Another File
          </Button>

          <a
            href={getDownloadResultsUrl(uploadId, 'csv')}
            download={`mei_scored_${uploadId}.csv`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-caption font-medium border border-border rounded hover:bg-paper text-ink transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download CSV
          </a>

          <a
            href={getDownloadResultsUrl(uploadId, 'xlsx')}
            download={`mei_scored_${uploadId}.xlsx`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-caption font-medium bg-forest text-surface rounded hover:bg-forest/90 transition-colors shadow-none"
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            Download Excel Report (.xlsx)
          </a>
        </div>
      </div>

      {/* Summary Stat Tiles */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Total Scored */}
        <Card className="border-border">
          <CardContent className="p-5">
            <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
              Students Scored
            </span>
            <div className="font-serif text-hero font-bold text-ink mt-1 tabular-nums">
              {totalScored.toLocaleString()}
            </div>
            <span className="text-small text-forest font-medium mt-1 inline-flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              100% Validated
            </span>
          </CardContent>
        </Card>

        {/* Low Risk */}
        <Card className="border-border">
          <CardContent className="p-5">
            <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
              Low Risk (0–29)
            </span>
            <div className="font-serif text-hero font-bold text-forest mt-1 tabular-nums">
              {dist.Low.toLocaleString()}
            </div>
            <span className="text-small text-muted font-medium mt-1 block">
              {distPct.Low}% of cohort
            </span>
          </CardContent>
        </Card>

        {/* Moderate Risk */}
        <Card className="border-border">
          <CardContent className="p-5">
            <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
              Moderate Risk (30–69)
            </span>
            <div className="font-serif text-hero font-bold text-amber-text mt-1 tabular-nums">
              {dist.Moderate.toLocaleString()}
            </div>
            <span className="text-small text-muted font-medium mt-1 block">
              {distPct.Moderate}% of cohort
            </span>
          </CardContent>
        </Card>

        {/* High Risk */}
        <Card className="border-border">
          <CardContent className="p-5">
            <span className="text-[11px] text-muted uppercase font-semibold tracking-wider block">
              High Risk (70–100)
            </span>
            <div className="font-serif text-hero font-bold text-crimson mt-1 tabular-nums">
              {dist.High.toLocaleString()}
            </div>
            <span className="text-small text-muted font-medium mt-1 block">
              {distPct.High}% of cohort
            </span>
          </CardContent>
        </Card>
      </div>

      {/* Outcome Check Card (if file included outcomes) */}
      {outcome && (
        <Card className="border-forest/40 bg-forest-subtle/30">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="w-5 h-5 text-forest" />
                <CardTitle className="text-section">
                  How Well Did the Model Perform on Your File?
                </CardTitle>
              </div>
              <Badge variant="forest">Computed on This File Only</Badge>
            </div>
            <CardDescription>
              Your survey spreadsheet contained actual outcome labels ({outcome.total_outcome_rows} students). The model caught {outcome.actual_at_risk_count} at-risk students.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pt-2 text-center">
              <div className="p-3 bg-surface rounded border border-border">
                <span className="text-[11px] text-muted uppercase font-semibold block">Recall</span>
                <span className="font-serif text-section font-bold text-ink tabular-nums">
                  {outcome.recall !== null && outcome.recall !== undefined ? outcome.recall.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="p-3 bg-surface rounded border border-border">
                <span className="text-[11px] text-muted uppercase font-semibold block">Precision</span>
                <span className="font-serif text-section font-bold text-ink tabular-nums">
                  {outcome.precision !== null && outcome.precision !== undefined ? outcome.precision.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="p-3 bg-surface rounded border border-border">
                <span className="text-[11px] text-muted uppercase font-semibold block">F1 Score</span>
                <span className="font-serif text-section font-bold text-forest tabular-nums">
                  {outcome.f1_score !== null && outcome.f1_score !== undefined ? outcome.f1_score.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="p-3 bg-surface rounded border border-border">
                <span className="text-[11px] text-muted uppercase font-semibold block">AUC-ROC</span>
                <span className="font-serif text-section font-bold text-ink tabular-nums">
                  {outcome.auc_roc !== null && outcome.auc_roc !== undefined ? outcome.auc_roc.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="p-3 bg-surface rounded border border-border">
                <span className="text-[11px] text-muted uppercase font-semibold block">Confusion Matrix</span>
                <span className="text-small font-mono text-ink block mt-1">
                  TP:{outcome.confusion_matrix.tp} FP:{outcome.confusion_matrix.fp} / FN:{outcome.confusion_matrix.fn} TN:{outcome.confusion_matrix.tn}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Distribution Histogram Card */}
      <Card className="border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-section">Cohort Risk Band Distribution</CardTitle>
          <CardDescription>
            Categorization by calibrated threshold boundaries: Low Risk (0–29), Moderate Risk (30–69), and High Risk (70–100).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-56 w-full bg-paper rounded border border-border p-2">
            <ReactECharts
              option={getDistributionChartOption()}
              style={{ height: '100%', width: '100%' }}
            />
          </div>
        </CardContent>
      </Card>

      {/* Interactive Results Table Card */}
      <Card className="border-border">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <CardTitle className="text-section">Scored Students</CardTitle>
              <CardDescription>
                Click any student row to view full 4-period trajectories and the waterfall explanation.
              </CardDescription>
            </div>

            {/* Filters and Search Bar */}
            <div className="flex flex-wrap items-center gap-2.5">
              {/* Search */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-muted absolute left-2.5 top-2.5" />
                <input
                  type="text"
                  aria-label="Search Student ID"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 pr-3 py-1.5 text-caption bg-surface border border-border rounded font-mono text-ink focus:outline-none focus:border-forest w-48"
                />
              </div>

              {/* Band Filter */}
              <select
                value={filterBand}
                onChange={(e) => {
                  setFilterBand(e.target.value);
                  setPage(1);
                }}
                className="px-2.5 py-1.5 text-caption bg-surface border border-border rounded text-ink font-medium focus:outline-none focus:border-forest"
              >
                <option value="all">All Risk Bands</option>
                <option value="Low">Low Risk</option>
                <option value="Moderate">Moderate Risk</option>
                <option value="High">High Risk</option>
              </select>

              {/* Flag Filter */}
              <select
                value={filterFlag}
                onChange={(e) => {
                  setFilterFlag(e.target.value);
                  setPage(1);
                }}
                className="px-2.5 py-1.5 text-caption bg-surface border border-border rounded text-ink font-medium focus:outline-none focus:border-forest"
              >
                <option value="all">All Statuses</option>
                <option value="Yes">Flagged At-Risk</option>
                <option value="No">Not Flagged</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-caption">
              <thead>
                <tr className="border-b border-border bg-paper text-muted font-semibold text-[11px] uppercase tracking-wider">
                  <th
                    className="py-3 px-4 cursor-pointer hover:text-ink select-none"
                    onClick={() => handleSort('student_id')}
                  >
                    <div className="flex items-center gap-1.5">
                      Student ID
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="py-3 px-4 cursor-pointer hover:text-ink select-none"
                    onClick={() => handleSort('mei')}
                  >
                    <div className="flex items-center gap-1.5">
                      Engagement Index (MEI)
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-4">Risk Band</th>
                  <th className="py-3 px-4">Flagged</th>
                  <th className="py-3 px-4">Pattern</th>
                  <th className="py-3 px-4">Level</th>
                  <th className="py-3 px-4">Primary Reasons</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {isLoading ? (
                  <tr>
                    <td colSpan={8} className="py-12 text-center text-muted">
                      <div className="w-5 h-5 border-2 border-forest border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                      Loading cohort scoring table...
                    </td>
                  </tr>
                ) : displayedRows.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-12 text-center text-muted">
                      No student records match the active search or filters.
                    </td>
                  </tr>
                ) : (
                  displayedRows.map((row: ScoredStudentRow) => {
                    const isHigh = row.band === 'High';
                    const isMod = row.band === 'Moderate';

                    // Parse reasons: "1. Reason A; 2. Reason B; 3. Reason C"
                    const reasonsParts = row.top_reasons
                      ? row.top_reasons.split('; ')
                      : [];
                    const firstReason = reasonsParts[0] || 'Standard monitoring';
                    const extraReasonsCount = Math.max(0, reasonsParts.length - 1);

                    return (
                      <tr
                        key={row.student_id}
                        onClick={() => setSelectedStudentId(row.student_id)}
                        className="hover:bg-forest-subtle/30 cursor-pointer transition-colors"
                      >
                        {/* Student ID */}
                        <td className="py-3 px-4 font-mono font-medium text-ink">
                          {row.student_id}
                        </td>

                        {/* MEI Bar + Number */}
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-3">
                            <span className="font-serif font-bold text-ink w-8 tabular-nums">
                              {row.mei}
                            </span>
                            <div className="w-24 h-2 bg-paper rounded-full overflow-hidden border border-border">
                              <div
                                className={`h-full rounded-full ${
                                  isHigh
                                    ? 'bg-crimson'
                                    : isMod
                                    ? 'bg-amber'
                                    : 'bg-forest'
                                }`}
                                style={{ width: `${Math.min(100, Math.max(5, row.mei))}%` }}
                              />
                            </div>
                          </div>
                        </td>

                        {/* Band Badge */}
                        <td className="py-3 px-4">
                          <Badge
                            variant={isHigh ? 'crimson' : isMod ? 'amber' : 'forest'}
                            size="sm"
                          >
                            {row.band}
                          </Badge>
                        </td>

                        {/* Flag Badge */}
                        <td className="py-3 px-4">
                          <Badge
                            variant={row.flag === 'Yes' ? 'crimson' : 'neutral'}
                            size="sm"
                          >
                            {row.flag === 'Yes' ? 'Flagged' : 'Normal'}
                          </Badge>
                        </td>

                        {/* Pattern */}
                        <td className="py-3 px-4 font-medium text-ink">{row.pattern}</td>

                        {/* Level */}
                        <td className="py-3 px-4 text-muted">{row.level}</td>

                        {/* Primary Reasons */}
                        <td className="py-3 px-4 max-w-xs">
                          <div className="flex items-center gap-1.5">
                            <span className="truncate text-ink/80" title={row.top_reasons}>
                              {firstReason}
                            </span>
                            {extraReasonsCount > 0 && (
                              <span
                                className="px-1.5 py-0.5 text-[10px] bg-paper rounded border border-border text-muted shrink-0"
                                title={reasonsParts.slice(1).join('\n')}
                              >
                                +{extraReasonsCount}
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Action View */}
                        <td className="py-3 px-4 text-right">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedStudentId(row.student_id);
                            }}
                            className="p-1 text-forest hover:bg-forest/10 rounded transition-colors"
                            aria-label={`View detail for ${row.student_id}`}
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {resultsData && resultsData.total_pages > 1 && (
            <div className="p-4 border-t border-border bg-paper flex flex-wrap items-center justify-between gap-4 text-caption">
              <div className="text-muted">
                Showing{' '}
                <span className="font-semibold text-ink">
                  {(page - 1) * pageSize + 1}–{Math.min(resultsData.total, page * pageSize)}
                </span>{' '}
                of <span className="font-semibold text-ink">{resultsData.total.toLocaleString()}</span> students
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  icon={<ChevronLeft className="w-3.5 h-3.5" />}
                >
                  Previous
                </Button>
                <span className="text-caption font-mono text-ink px-2">
                  Page {page} of {resultsData.total_pages}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(resultsData.total_pages, p + 1))}
                  disabled={page >= resultsData.total_pages}
                  icon={<ChevronRight className="w-3.5 h-3.5" />}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Side Drawer Component */}
      <ScoreDetailDrawer
        uploadId={uploadId}
        studentId={selectedStudentId}
        onClose={() => setSelectedStudentId(null)}
      />
    </div>
  );
}
