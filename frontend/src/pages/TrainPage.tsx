/**
 * /train — Real Survey Retraining Pipeline & Model Governance
 *
 * 6-step wizard:
 *   1. Get template
 *   2. Upload survey file
 *   3. Check data (traffic light, class balance, small-data advice)
 *   4. Choose mode (Quick vs Full)
 *   5. Live training progress
 *   6. Review & decide (side-by-side comparison, activate or keep)
 *
 * Below stepper: Previous Runs table with rollback action.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Download,
  Info,
  Play,
  RefreshCw,
  Shield,
  StopCircle,
  XCircle,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { FileDropzone } from '@/components/ui/FileDropzone';
import { Stepper } from '@/components/ui/Stepper';
import { Tag } from '@/components/ui/Tag';
import {
  getTemplateDownloadUrl,
  useActivateRun,
  useCancelRetraining,
  useRuns,
  useRunsCompare,
  useRetrainJobStatus,
  useStartRetraining,
  useValidateRetrainFile,
} from '@/api/queries';
import type { RetrainValidateResponse, RunSummary } from '@/api/types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function trafficColor(light: string): string {
  if (light === 'GREEN') return 'text-[#2E7D4F]';
  if (light === 'AMBER') return 'text-[#C98A00]';
  return 'text-[#B3261E]';
}

function trafficBg(light: string): string {
  if (light === 'GREEN') return 'bg-green-50 border-green-200';
  if (light === 'AMBER') return 'bg-amber-50 border-amber-200';
  return 'bg-red-50 border-red-200';
}

function fmt(n: number | undefined | null, digits = 4): string {
  if (n === undefined || n === null) return '—';
  return n.toFixed(digits);
}

// ---------------------------------------------------------------------------
// Step 1 — Get Template
// ---------------------------------------------------------------------------
function StepGetTemplate({ onNext }: { onNext: () => void }) {
  return (
    <div className="space-y-6">
      <p className="text-body text-ink">
        Download the survey template and fill it in with your collected student
        responses. Each row is one student. Column order does not matter — the
        system will recognise the questions automatically.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <a
          href={getTemplateDownloadUrl('xlsx')}
          className="flex items-center gap-3 rounded-lg border border-border bg-surface p-4 hover:bg-surface/80 transition-colors"
        >
          <Download className="w-5 h-5 text-forest shrink-0" />
          <div>
            <p className="text-sm font-medium text-ink">Download template (.xlsx)</p>
            <p className="text-xs text-muted">Excel workbook with column headers and instructions</p>
          </div>
        </a>
        <a
          href={getTemplateDownloadUrl('csv')}
          className="flex items-center gap-3 rounded-lg border border-border bg-surface p-4 hover:bg-surface/80 transition-colors"
        >
          <Download className="w-5 h-5 text-forest shrink-0" />
          <div>
            <p className="text-sm font-medium text-ink">Download template (.csv)</p>
            <p className="text-xs text-muted">Comma-separated with identical columns</p>
          </div>
        </a>
      </div>

      <div className="rounded-lg bg-[#F5F2EC] border border-[#E0DDD6] p-4 space-y-2">
        <p className="text-sm font-medium text-ink">What the file needs to contain</p>
        <ul className="text-sm text-muted space-y-1 list-disc list-inside">
          <li>7 engagement questions × 4 periods = 28 columns (required)</li>
          <li>3 attrition intent questions A1, A2, A3 (required for retraining)</li>
          <li>At least 150 student rows</li>
          <li>Likert scale 1–5 as numbers <em>or</em> as the full answer text</li>
          <li>CSV, XLSX, or XLS — up to 20 MB</li>
        </ul>
      </div>

      <div className="flex justify-end">
        <Button variant="primary" onClick={onNext} icon={<ArrowRight className="w-4 h-4" />}>
          Next — Upload file
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 2 — Upload Survey File
// ---------------------------------------------------------------------------
function StepUploadFile({
  onValidated,
}: {
  onValidated: (result: RetrainValidateResponse) => void;
}) {
  const validate = useValidateRetrainFile();

  const handleFile = useCallback(
    (file: File) => {
      validate.mutate(file, {
        onSuccess: (res) => onValidated(res),
      });
    },
    [validate, onValidated],
  );

  return (
    <div className="space-y-6">
      <p className="text-body text-ink">
        Upload your completed survey file. The system checks that all required
        questions are present and that there are enough students to fit reliable
        models.
      </p>

      <FileDropzone
        onFileSelect={handleFile}
        maxSizeMB={20}
        disabled={validate.isPending}
      />

      {validate.isError && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 p-4">
          <XCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{(validate.error as Error)?.message ?? 'Upload failed.'}</p>
        </div>
      )}

      <p className="text-xs text-muted">
        Files are processed on this computer, are not shared outside this system,
        and are removed automatically after 24 hours.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 3 — Check Data
// ---------------------------------------------------------------------------
function StepCheckData({
  validation,
  onBack,
  onNext,
}: {
  validation: RetrainValidateResponse;
  onBack: () => void;
  onNext: () => void;
}) {
  const light = validation.traffic_light;
  const guards = validation.guards;
  const balance = validation.class_balance;

  return (
    <div className="space-y-6">
      {/* Traffic light summary */}
      <div className={`rounded-lg border p-4 space-y-2 ${trafficBg(light)}`}>
        <div className="flex items-center gap-2">
          {light === 'GREEN' ? (
            <CheckCircle2 className="w-5 h-5 text-[#2E7D4F]" />
          ) : light === 'AMBER' ? (
            <AlertTriangle className="w-5 h-5 text-[#C98A00]" />
          ) : (
            <XCircle className="w-5 h-5 text-[#B3261E]" />
          )}
          <span className={`font-semibold text-sm ${trafficColor(light)}`}>
            {light === 'GREEN' ? 'Data looks good' : light === 'AMBER' ? 'Data meets minimum requirements — small-data guards active' : 'Cannot proceed — see issues below'}
          </span>
        </div>
        {validation.is_refusal && validation.refusal_reason && (
          <p className="text-sm text-red-700">{validation.refusal_reason}</p>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Rows found', value: validation.total_rows.toLocaleString() },
          { label: 'Can be scored', value: validation.usable_rows.toLocaleString() },
          { label: 'At-risk students', value: `${balance.at_risk_count} (${balance.at_risk_pct}%)` },
          { label: 'Missing answers', value: `${validation.missingness.overall_missing_pct}%` },
        ].map(({ label, value }) => (
          <div key={label} className="rounded-lg bg-surface border border-border p-3">
            <p className="text-xs text-muted">{label}</p>
            <p className="text-base font-semibold text-ink mt-0.5">{value}</p>
          </div>
        ))}
      </div>

      {/* Cross-validation scheme */}
      {guards && (
        <div className="rounded-lg bg-[#F5F2EC] border border-[#E0DDD6] p-4 space-y-2">
          <p className="text-sm font-medium text-ink">How models will be evaluated</p>
          <p className="text-sm text-muted">
            Cross-validation: <strong>{guards.cv_scheme}</strong> ({guards.cv_folds_count} folds)
            {guards.is_small_cohort && ' — small-cohort mode (N < 500)'}
            {guards.is_imbalanced && ' · class weights enabled (positive rate < 15 %)'}
          </p>
          {guards.disabled_models.length > 0 && (
            <p className="text-xs text-muted">
              Disabled models in this cohort size: {guards.disabled_models.join(', ')}
            </p>
          )}
        </div>
      )}

      {/* Warnings */}
      {validation.warnings.length > 0 && (
        <div className="space-y-2">
          {validation.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2 rounded-lg bg-amber-50 border border-amber-200 p-3">
              <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800">{w}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between">
        <Button variant="secondary" onClick={onBack}>Back</Button>
        <Button
          variant="primary"
          onClick={onNext}
          disabled={validation.is_refusal}
          icon={<ArrowRight className="w-4 h-4" />}
        >
          Next — Choose mode
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 4 — Choose Mode
// ---------------------------------------------------------------------------
function StepChooseMode({
  estimatedQuick,
  estimatedFull,
  mode,
  onModeChange,
  onBack,
  onStart,
  isStarting,
}: {
  estimatedQuick: number;
  estimatedFull: number;
  mode: 'quick' | 'full';
  onModeChange: (m: 'quick' | 'full') => void;
  onBack: () => void;
  onStart: () => void;
  isStarting: boolean;
}) {
  return (
    <div className="space-y-6">
      <p className="text-body text-ink">
        Choose how thorough the training run should be. Both modes evaluate the
        same quality indicators — Quick skips the slowest model families so it
        finishes sooner.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {(
          [
            {
              id: 'quick' as const,
              label: 'Quick run',
              icon: <Zap className="w-5 h-5" />,
              time: `≈ ${estimatedQuick} min`,
              description:
                'Tests the most effective model families — logistic regression, boosted trees, random forest, and nearest-neighbour. Skips deep-sequence networks and stacking. Good for a first check on new survey data.',
            },
            {
              id: 'full' as const,
              label: 'Full run',
              icon: <Shield className="w-5 h-5" />,
              time: `≈ ${estimatedFull} min`,
              description:
                'Runs all model families including deep-sequence networks where the cohort is large enough. Produces the most complete leaderboard and the most reliable champion selection.',
            },
          ] as const
        ).map(({ id, label, icon, time, description }) => (
          <button
            key={id}
            onClick={() => onModeChange(id)}
            className={`text-left rounded-lg border p-4 space-y-2 transition-colors ${
              mode === id
                ? 'border-forest bg-[#EAF1EF]'
                : 'border-border bg-surface hover:bg-[#F5F2EC]'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className={`flex items-center gap-2 font-semibold text-sm ${mode === id ? 'text-forest' : 'text-ink'}`}>
                {icon}
                {label}
              </div>
              <span className="flex items-center gap-1 text-xs text-muted">
                <Clock className="w-3 h-3" />
                {time}
              </span>
            </div>
            <p className="text-sm text-muted">{description}</p>
            {mode === id && (
              <div className="flex items-center gap-1 text-xs text-forest font-medium">
                <CheckCircle2 className="w-3 h-3" />
                Selected
              </div>
            )}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <Button variant="secondary" onClick={onBack}>Back</Button>
        <Button
          variant="primary"
          onClick={onStart}
          loading={isStarting}
          icon={<Play className="w-4 h-4" />}
        >
          Start {mode === 'quick' ? 'quick' : 'full'} retraining
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 5 — Live Training Progress
// ---------------------------------------------------------------------------
function StepProgress({
  jobId,
  onCancel,
  onComplete,
}: {
  jobId: string;
  onCancel: () => void;
  onComplete: (runId: string) => void;
}) {
  const cancel = useCancelRetraining();
  const { data: job } = useRetrainJobStatus(jobId, true);

  useEffect(() => {
    if (job?.status === 'finished' && job.result?.run_id) {
      onComplete(job.result.run_id as string);
    }
  }, [job, onComplete]);

  const progress = job?.progress ?? 0;
  const pct = Math.round(progress * 100);
  const isDone = job?.status === 'finished';
  const isFailed = job?.status === 'failed';
  const isCancelled = job?.status === 'cancelled';

  return (
    <div className="space-y-6">
      <div className="rounded-lg bg-surface border border-border p-4 space-y-4">
        {/* Progress bar */}
        <div>
          <div className="flex justify-between text-xs text-muted mb-1.5">
            <span>{job?.message ?? 'Preparing…'}</span>
            <span>{pct}%</span>
          </div>
          <div className="h-2 rounded-full bg-[#E0DDD6]">
            <div
              className="h-2 rounded-full bg-forest transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {/* Status badges */}
        <div className="flex items-center gap-2 flex-wrap">
          <Tag variant={isDone ? 'risk-low' : isFailed || isCancelled ? 'risk-high' : 'risk-moderate'}>
            {job?.status ?? 'queued'}
          </Tag>
          {job?.result?.champion_name && (
            <span className="text-xs text-muted">
              Champion so far: <strong className="text-ink">{job.result.champion_name as string}</strong>
            </span>
          )}
        </div>
      </div>

      {isFailed && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 p-4">
          <XCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{job?.error ?? 'Retraining failed. Check the server log for details.'}</p>
        </div>
      )}

      {isCancelled && (
        <div className="flex items-start gap-2 rounded-lg bg-amber-50 border border-amber-200 p-4">
          <StopCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-800">Retraining was cancelled. You can start a new run from step 1.</p>
        </div>
      )}

      {!isDone && !isFailed && !isCancelled && (
        <div className="flex justify-end">
          <Button
            variant="secondary"
            onClick={() => cancel.mutate(jobId, { onSuccess: onCancel })}
            icon={<StopCircle className="w-4 h-4" />}
          >
            Cancel run
          </Button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 6 — Review & Decide
// ---------------------------------------------------------------------------
function StepReview({
  newRunId,
  activeRunId,
  onActivate,
  onKeep,
}: {
  newRunId: string;
  activeRunId: string;
  onActivate: () => void;
  onKeep: () => void;
}) {
  const activate = useActivateRun();
  const { data: comp, isLoading } = useRunsCompare(
    activeRunId !== newRunId ? activeRunId : null,
    newRunId,
  );
  const [showRanks, setShowRanks] = useState(false);

  const comparison = comp?.comparison;

  const handleActivate = () => {
    activate.mutate(
      { run_id: newRunId, actor: 'Web UI Retraining Wizard' },
      { onSuccess: onActivate },
    );
  };

  if (isLoading) {
    return <p className="text-sm text-muted">Comparing runs…</p>;
  }

  return (
    <div className="space-y-6">
      {/* Verdict */}
      {comparison && (
        <div
          className={`rounded-lg border p-4 ${comparison.is_recommended ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}
        >
          <div className="flex items-start gap-2">
            {comparison.is_recommended ? (
              <CheckCircle2 className="w-4 h-4 text-[#2E7D4F] shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-[#C98A00] shrink-0 mt-0.5" />
            )}
            <p className="text-sm text-ink">{comparison.verdict_summary}</p>
          </div>
        </div>
      )}

      {/* Metric diff table */}
      {comparison && (
        <div className="rounded-lg bg-surface border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[#F5F2EC] border-b border-border">
                <th className="text-left px-4 py-2 text-muted font-medium">Metric</th>
                <th className="text-right px-4 py-2 text-muted font-medium">Active run</th>
                <th className="text-right px-4 py-2 text-muted font-medium">New run</th>
                <th className="text-right px-4 py-2 text-muted font-medium">Change</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  label: 'Validation score',
                  a: comp!.run_a.champion?.primary_score as number | undefined,
                  b: comp!.run_b.champion?.primary_score as number | undefined,
                  diff: comparison.cv_score_diff,
                },
                {
                  label: 'Test F1',
                  a: comp!.run_a.champion?.test_f1 as number | undefined,
                  b: comp!.run_b.champion?.test_f1 as number | undefined,
                  diff: comparison.test_f1_diff,
                },
              ].map(({ label, a, b, diff }) => (
                <tr key={label} className="border-b border-border last:border-0">
                  <td className="px-4 py-2 text-ink">{label}</td>
                  <td className="px-4 py-2 text-right text-muted">{fmt(a)}</td>
                  <td className="px-4 py-2 text-right font-medium text-ink">{fmt(b)}</td>
                  <td
                    className={`px-4 py-2 text-right font-medium ${diff > 0 ? 'text-[#2E7D4F]' : diff < 0 ? 'text-[#B3261E]' : 'text-muted'}`}
                  >
                    {diff > 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                  </td>
                </tr>
              ))}
              <tr>
                <td className="px-4 py-2 text-ink">Champion model</td>
                <td className="px-4 py-2 text-right text-muted">{comparison.champion_a}</td>
                <td className="px-4 py-2 text-right font-medium text-ink">
                  {comparison.champion_b}
                  {comparison.champion_changed && (
                    <span className="ml-1 text-[0.65rem] rounded bg-amber-100 text-amber-700 px-1">changed</span>
                  )}
                </td>
                <td className="px-4 py-2 text-right" />
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Rank changes (collapsible) */}
      {comparison && comparison.rank_changes.length > 0 && (
        <div>
          <button
            onClick={() => setShowRanks((v) => !v)}
            className="flex items-center gap-1 text-sm text-forest font-medium"
          >
            {showRanks ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            Rank changes in new leaderboard
          </button>
          {showRanks && (
            <div className="mt-2 rounded-lg bg-surface border border-border overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-[#F5F2EC] border-b border-border">
                    <th className="text-left px-3 py-2 text-muted font-medium">Model</th>
                    <th className="text-right px-3 py-2 text-muted font-medium">Old rank</th>
                    <th className="text-right px-3 py-2 text-muted font-medium">New rank</th>
                    <th className="text-right px-3 py-2 text-muted font-medium">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rank_changes.slice(0, 10).map((r) => (
                    <tr key={r.model_id} className="border-b border-border last:border-0">
                      <td className="px-3 py-1.5 text-ink">{r.display_name}</td>
                      <td className="px-3 py-1.5 text-right text-muted">{r.old_rank ?? '—'}</td>
                      <td className="px-3 py-1.5 text-right font-medium text-ink">{r.new_rank}</td>
                      <td className="px-3 py-1.5 text-right text-muted">{r.score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Note */}
      <div className="rounded-lg bg-[#F5F2EC] border border-[#E0DDD6] p-3">
        <p className="text-xs text-muted">
          The previous model is never deleted. Choosing "Keep current model" leaves the
          active champion unchanged. You can activate the new run later from the Previous
          Runs table below.
        </p>
      </div>

      <div className="flex items-center gap-3 justify-end">
        <Button variant="secondary" onClick={onKeep}>
          Keep current model
        </Button>
        <Button
          variant="primary"
          onClick={handleActivate}
          loading={activate.isPending}
          icon={<CheckCircle2 className="w-4 h-4" />}
        >
          Use the new model
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Previous Runs table
// ---------------------------------------------------------------------------
function PreviousRunsTable() {
  const { data: runsList } = useRuns();
  const activate = useActivateRun();

  if (!runsList || runsList.runs.length === 0) return null;

  return (
    <Card title="Previous runs">
      <p className="mb-2 text-xs text-muted sm:hidden">
        Scroll the table sideways to see every column.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-[#F5F2EC]">
              <th className="text-left px-4 py-2 text-muted font-medium">Run ID</th>
              <th className="text-left px-4 py-2 text-muted font-medium">Source</th>
              <th className="text-right px-4 py-2 text-muted font-medium">Students</th>
              <th className="text-left px-4 py-2 text-muted font-medium">Champion</th>
              <th className="text-right px-4 py-2 text-muted font-medium">Test F1</th>
              <th className="text-left px-4 py-2 text-muted font-medium">Trained</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {runsList.runs.map((run: RunSummary) => (
              <tr key={run.run_id} className="border-b border-border last:border-0">
                <td className="px-4 py-2 font-mono text-xs text-ink">
                  {run.run_id.slice(-16)}
                  {run.is_active && (
                    <span className="ml-1.5 rounded bg-[#1F4E46] text-white text-[0.6rem] px-1 py-0.5">
                      active
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-muted capitalize">{run.data_source}</td>
                <td className="px-4 py-2 text-right text-muted">{run.n_students}</td>
                <td className="px-4 py-2 text-muted">{run.champion_model_id}</td>
                <td className="px-4 py-2 text-right text-muted">
                  {run.champion_f1 != null ? run.champion_f1.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-2 text-muted text-xs">
                  {run.trained_at ? new Date(run.trained_at).toLocaleDateString() : '—'}
                </td>
                <td className="px-4 py-2 text-right">
                  {!run.is_active && (
                    <button
                      className="text-xs text-forest font-medium hover:underline disabled:opacity-40"
                      disabled={activate.isPending}
                      onClick={() =>
                        activate.mutate({ run_id: run.run_id, actor: 'Web UI Rollback' })
                      }
                    >
                      <RefreshCw className="w-3 h-3 inline mr-1" />
                      Use this one
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main TrainPage
// ---------------------------------------------------------------------------
export function TrainPage() {
  const [step, setStep] = useState(0);
  const [validation, setValidation] = useState<RetrainValidateResponse | null>(null);
  const [mode, setMode] = useState<'quick' | 'full'>('quick');
  const [jobId, setJobId] = useState<string | null>(null);
  const [newRunId, setNewRunId] = useState<string | null>(null);

  const startRetrain = useStartRetraining();

  // Derive activeRunId from status (stored in runs list)
  const { data: runsList } = useRuns();
  const activeRunId = runsList?.active_run_id ?? '';

  const steps = [
    { title: 'Get template' },
    { title: 'Upload file' },
    { title: 'Check data' },
    { title: 'Choose mode' },
    { title: 'Training' },
    { title: 'Review & decide' },
  ];

  const handleStartTraining = () => {
    if (!validation?.upload_id) return;
    startRetrain.mutate(
      { upload_id: validation.upload_id, mode, seed: 42 },
      {
        onSuccess: (res) => {
          setJobId(res.job_id);
          setStep(4);
        },
      },
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Train on New Data
        </h1>
        <p className="text-body text-muted mt-1">
          Run the full retraining pipeline on your university's validated survey
          responses, compare the result to the current model, and choose whether
          to switch.
        </p>
      </div>

      {/* Stepper */}
      <Card>
        <Stepper steps={steps} activeStep={step} />

        <div className="mt-6">
          {step === 0 && <StepGetTemplate onNext={() => setStep(1)} />}

          {step === 1 && (
            <StepUploadFile
              onValidated={(res) => {
                setValidation(res);
                setStep(2);
              }}
            />
          )}

          {step === 2 && validation && (
            <StepCheckData
              validation={validation}
              onBack={() => setStep(1)}
              onNext={() => setStep(3)}
            />
          )}

          {step === 3 && validation && (
            <StepChooseMode
              estimatedQuick={validation.estimated_time_min}
              estimatedFull={Math.round(validation.estimated_time_min * 3.5)}
              mode={mode}
              onModeChange={setMode}
              onBack={() => setStep(2)}
              onStart={handleStartTraining}
              isStarting={startRetrain.isPending}
            />
          )}

          {step === 4 && jobId && (
            <StepProgress
              jobId={jobId}
              onCancel={() => setStep(0)}
              onComplete={(runId) => {
                setNewRunId(runId);
                setStep(5);
              }}
            />
          )}

          {step === 5 && newRunId && (
            <StepReview
              newRunId={newRunId}
              activeRunId={activeRunId}
              onActivate={() => {
                setStep(0);
                setValidation(null);
                setJobId(null);
                setNewRunId(null);
              }}
              onKeep={() => {
                setStep(0);
                setValidation(null);
                setJobId(null);
                setNewRunId(null);
              }}
            />
          )}
        </div>
      </Card>

      {/* Previous runs */}
      <PreviousRunsTable />
    </div>
  );
}
