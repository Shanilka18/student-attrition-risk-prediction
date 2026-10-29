import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Check,
  RotateCcw,
  Database,
  Layers,
  ArrowRight,
  TrendingDown,
  RefreshCw,
  Plus,
  Trash2,
  HelpCircle,
  FileText,
  ExternalLink,
  Code2,
  X,
  ShieldCheck,
  Clock,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { api } from '../api/client';
import type {
  BatchUploadResult,
  StudentPredictionResult,
  ValidationErrorDetail,
  RetrainInspectResponse,
  RetrainJobStatusResponse,
  RetrainJobResult,
  RetrainRunItem,
  MetaResponse,
} from '../api/client';
import { RiskBadge } from '../components/RiskBadge';
import { TrajectoryChart } from '../components/TrajectoryChart';
import type { TrajectoryPoint } from '../components/TrajectoryChart';
import { LoadingState } from '../components/StateViews';

interface PredictPageProps {
  meta?: MetaResponse | null;
  onRefreshMeta?: () => void;
}

export const PredictPage: React.FC<PredictPageProps> = ({ meta, onRefreshMeta }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'simulate'>('upload');

  // File Upload State
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadErrors, setUploadErrors] = useState<ValidationErrorDetail[] | null>(null);
  const [batchResult, setBatchResult] = useState<BatchUploadResult | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<StudentPredictionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Retraining State
  const [inspectLoading, setInspectLoading] = useState<boolean>(false);
  const [inspectResult, setInspectResult] = useState<RetrainInspectResponse | null>(null);
  const [isRetrainModalOpen, setIsRetrainModalOpen] = useState<boolean>(false);
  const [retrainBudget, setRetrainBudget] = useState<'quick' | 'standard' | 'full'>('quick');
  const [retrainScope, setRetrainScope] = useState<'champions' | 'full'>('champions');
  const [activeJobId, setActiveJobId] = useState<string | null>(() => {
    return localStorage.getItem('academic_intelligence_active_retrain_job');
  });
  const [jobStatus, setJobStatus] = useState<RetrainJobStatusResponse | null>(null);
  const [retrainResult, setRetrainResult] = useState<RetrainJobResult | null>(null);
  const [retrainError, setRetrainError] = useState<string | null>(null);

  // Phase 2: Active Run & Past Runs Management
  const [pastRuns, setPastRuns] = useState<RetrainRunItem[]>([]);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [activatingJobId, setActivatingJobId] = useState<string | null>(null);
  const [isRollingBack, setIsRollingBack] = useState<boolean>(false);
  const [showPastRuns, setShowPastRuns] = useState<boolean>(false);

  // Single Student Simulator State
  const [simStudentId, setSimStudentId] = useState<string>('STU_MANUAL_01');
  const [simSemesters, setSimSemesters] = useState<Array<{
    semester: number;
    gpa: number;
    exam_marks: number;
    attendance_pct: number;
    assignment_delay_days: number;
    late_assignments_count: number;
  }>>([
    { semester: 1, gpa: 3.4, exam_marks: 75.0, attendance_pct: 90.0, assignment_delay_days: 0.5, late_assignments_count: 0 },
    { semester: 2, gpa: 2.8, exam_marks: 62.0, attendance_pct: 74.0, assignment_delay_days: 3.5, late_assignments_count: 2 },
    { semester: 3, gpa: 2.1, exam_marks: 48.0, attendance_pct: 55.0, assignment_delay_days: 6.0, late_assignments_count: 4 },
  ]);
  const [simLoading, setSimLoading] = useState<boolean>(false);
  const [simResult, setSimResult] = useState<StudentPredictionResult | null>(null);

  // Polling Effect for Background Retraining Job
  useEffect(() => {
    if (!activeJobId) return;

    let isMounted = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const status = await api.getJobStatus(activeJobId);
        if (!isMounted) return;
        setJobStatus(status);

        if (status.state === 'succeeded') {
          if (status.result) {
            setRetrainResult(status.result);
          }
          localStorage.removeItem('academic_intelligence_active_retrain_job');
          setActiveJobId(null);
          fetchPastRuns();
        } else if (status.state === 'failed') {
          setRetrainError(status.error || 'Retraining job encountered an error.');
          localStorage.removeItem('academic_intelligence_active_retrain_job');
          setActiveJobId(null);
        } else {
          // Queued or running: poll every 2 seconds
          timer = setTimeout(poll, 2000);
        }
      } catch (err: any) {
        if (!isMounted) return;
        if (err.status === 404) {
          localStorage.removeItem('academic_intelligence_active_retrain_job');
          setActiveJobId(null);
        } else {
          timer = setTimeout(poll, 3000);
        }
      }
    };

    poll();

    return () => {
      isMounted = false;
      if (timer) clearTimeout(timer);
    };
  }, [activeJobId]);

  // File Drag & Drop Handlers
  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = async (selectedFile: File) => {
    setFile(selectedFile);
    setUploadErrors(null);
    setBatchResult(null);
    setSelectedStudent(null);
    setInspectResult(null);
    setInspectLoading(true);

    try {
      const res = await api.inspectDataset(selectedFile);
      setInspectResult(res);
    } catch (err: any) {
      console.error('Inspection failed:', err);
    } finally {
      setInspectLoading(false);
    }
  };

  const handleStartRetrain = async () => {
    if (!file) return;
    setIsRetrainModalOpen(false);
    setRetrainError(null);
    setRetrainResult(null);

    try {
      const res = await api.startRetrain(file, retrainBudget, retrainScope);
      setActiveJobId(res.job_id);
      localStorage.setItem('academic_intelligence_active_retrain_job', res.job_id);
      setJobStatus({
        job_id: res.job_id,
        state: 'queued',
        phase: 'validate',
        progress_pct: 5,
        message: 'Retraining job queued. Initializing isolated directory workspace...',
        models_done: 0,
        models_total: 0,
        elapsed_s: 0.0,
      });
    } catch (err: any) {
      const detailMsg = err.detail?.message || err.detail || 'Failed to submit retraining job.';
      setRetrainError(String(detailMsg));
    }
  };

  const fetchPastRuns = async () => {
    try {
      setLoadingRuns(true);
      const res = await api.getRetrainRuns();
      setPastRuns(res.runs || []);
    } catch (e: any) {
      console.warn('Could not load past runs:', e);
    } finally {
      setLoadingRuns(false);
    }
  };

  useEffect(() => {
    fetchPastRuns();
  }, []);

  const handleActivateRun = async (jobId: string) => {
    try {
      setActivatingJobId(jobId);
      setActionMessage(null);
      setActionError(null);
      const res = await api.activateRun(jobId);
      setActionMessage(res.message);
      onRefreshMeta?.();
      await fetchPastRuns();
    } catch (e: any) {
      setActionError(e?.detail || 'Failed to activate retrained model for scoring.');
    } finally {
      setActivatingJobId(null);
    }
  };

  const handleRollback = async () => {
    try {
      setIsRollingBack(true);
      setActionMessage(null);
      setActionError(null);
      const res = await api.rollbackRun();
      setActionMessage(res.message);
      onRefreshMeta?.();
      await fetchPastRuns();
    } catch (e: any) {
      setActionError(e?.detail || 'Failed to roll back to synthetic benchmark.');
    } finally {
      setIsRollingBack(false);
    }
  };

  const handleDeleteRun = async (jobId: string) => {
    if (!window.confirm(`Are you sure you want to delete retrained run '${jobId}'? This cannot be undone.`)) {
      return;
    }
    try {
      setActionMessage(null);
      setActionError(null);
      const res = await api.deleteRetrainRun(jobId);
      setActionMessage(res.message);
      onRefreshMeta?.();
      await fetchPastRuns();
      if (retrainResult?.job_id === jobId) {
        setRetrainResult(null);
      }
    } catch (e: any) {
      setActionError(e?.detail || 'Failed to delete retrained run.');
    }
  };

  const executeUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadErrors(null);

    try {
      const res = await api.uploadPredict(file);
      setBatchResult(res);
      if (res.predictions.length > 0) {
        setSelectedStudent(res.predictions[0]);
      }
    } catch (err: any) {
      if (Array.isArray(err.detail)) {
        setUploadErrors(err.detail);
      } else {
        setUploadErrors([{ error_type: 'UPLOAD_ERROR', message: String(err.detail || 'Upload processing failed.') }]);
      }
    } finally {
      setUploading(false);
    }
  };

  // Single Student Simulator Handlers
  const addSemester = () => {
    if (simSemesters.length >= 8) return;
    const nextSem = simSemesters.length + 1;
    const last = simSemesters[simSemesters.length - 1] || {
      gpa: 3.0,
      exam_marks: 65,
      attendance_pct: 80,
      assignment_delay_days: 1.0,
      late_assignments_count: 0,
    };
    setSimSemesters([
      ...simSemesters,
      {
        semester: nextSem,
        gpa: Math.max(0.0, Math.round((last.gpa - 0.2) * 100) / 100),
        exam_marks: Math.max(0.0, Math.round(last.exam_marks - 5)),
        attendance_pct: Math.max(0.0, Math.round(last.attendance_pct - 5)),
        assignment_delay_days: Math.round(last.assignment_delay_days + 1.5),
        late_assignments_count: last.late_assignments_count + 1,
      },
    ]);
  };

  const removeSemester = (idx: number) => {
    if (simSemesters.length <= 1) return;
    const updated = simSemesters.filter((_, i) => i !== idx).map((s, i) => ({ ...s, semester: i + 1 }));
    setSimSemesters(updated);
  };

  const executeSimulation = async () => {
    setSimLoading(true);
    try {
      const res = await api.predictStudent({
        student_id: simStudentId,
        semesters: simSemesters,
      });
      setSimResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setSimLoading(false);
    }
  };

  // Band distribution data for batch chart
  const bandChartData = batchResult?.summary?.risk_band_counts
    ? Object.entries(batchResult.summary.risk_band_counts).map(([band, count]) => ({ band, count }))
    : [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
            <UploadCloud className="w-6 h-6 text-indigo-400" />
            Operational Prediction & Risk Triage Console
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Run batch institutional cohort scoring via CSV/XLSX upload, or simulate custom longitudinal student profiles.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              activeTab === 'upload' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Batch Upload (CSV/Excel)
          </button>
          <button
            onClick={() => setActiveTab('simulate')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              activeTab === 'simulate' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Single-Student Simulator
          </button>
        </div>
      </div>

      {/* Global Action Alerts */}
      {actionMessage && (
        <div className="rounded-xl border border-emerald-500/40 bg-emerald-950/20 p-3.5 flex items-center justify-between text-xs text-emerald-300">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{actionMessage}</span>
          </div>
          <button
            onClick={() => setActionMessage(null)}
            className="text-slate-400 hover:text-slate-200 p-1 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {actionError && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/20 p-3.5 flex items-center justify-between text-xs text-rose-300">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{actionError}</span>
          </div>
          <button
            onClick={() => setActionError(null)}
            className="text-slate-400 hover:text-slate-200 p-1 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Active Model Scoring Provenance Banner */}
      <div className={`rounded-xl border p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs ${
        meta?.data_mode === 'REAL'
          ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
          : 'bg-slate-900/60 border-slate-800 text-slate-300'
      }`}>
        <div className="flex items-start md:items-center gap-3">
          <div className={`p-2 rounded-lg shrink-0 ${
            meta?.data_mode === 'REAL'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
          }`}>
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase ${
                meta?.data_mode === 'REAL'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}>
                {meta?.data_mode === 'REAL' ? 'Active: REAL USER DATA' : 'Active: SYNTHETIC BENCHMARK'}
              </span>
              <span className="font-bold text-white text-sm">
                Scoring Champion: {meta?.champion_model?.label || 'CatBoost'}
              </span>
              {meta?.active_run_id && (
                <span className="text-slate-400 font-mono text-[11px] hidden sm:inline">
                  (Run: {meta.active_run_id})
                </span>
              )}
            </div>
            <p className="text-slate-400 mt-0.5 text-xs">
              {meta?.data_mode === 'REAL' ? (
                <>
                  Predictions use retrained weights: ROC-AUC <strong>{meta.champion_model?.test_roc_auc}</strong> | PR-AUC <strong>{meta.champion_model?.test_pr_auc}</strong> | F2 <strong>{meta.champion_model?.test_f2_score}</strong> | Lead Time <strong>{meta.champion_model?.lead_time_semesters} sems</strong>.
                </>
              ) : (
                <>
                  Inference is evaluated using the canonical thesis benchmark (CatBoost, ROC-AUC {meta?.champion_model?.test_roc_auc ?? 0.7815}).
                </>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0 self-end md:self-auto">
          {meta?.data_mode === 'REAL' && (
            <button
              onClick={handleRollback}
              disabled={isRollingBack}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors disabled:opacity-50"
              title="Deactivate real user model scoring and roll back to canonical thesis synthetic benchmark"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${isRollingBack ? 'animate-spin' : ''}`} />
              Rollback to Synthetic
            </button>
          )}

          <button
            onClick={() => setShowPastRuns(!showPastRuns)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-300 text-xs font-medium border border-slate-700 transition-colors"
          >
            <Clock className="w-3.5 h-3.5" />
            {showPastRuns ? 'Hide Past Runs' : `Past Retrained Models (${pastRuns.length})`}
          </button>
        </div>
      </div>

      {/* Past Runs Drawer / Table */}
      {showPastRuns && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                Retrained Models Registry (Isolated Benchmarks)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                All models are stored under isolated directories (<code>artifacts/user_dataset/&lt;job_id&gt;/</code>). Activate any model to switch the live scoring engine.
              </p>
            </div>
            <button
              onClick={fetchPastRuns}
              disabled={loadingRuns}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Refresh list"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingRuns ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {pastRuns.length === 0 ? (
            <div className="text-center py-6 text-xs text-slate-500 font-mono">
              No retrained runs found in <code>artifacts/user_dataset/</code>. Upload data with outcome columns to retrain.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                    <th className="py-2.5 px-3">Run ID / Date</th>
                    <th className="py-2.5 px-3">Champion Model</th>
                    <th className="py-2.5 px-3">Cohort Size</th>
                    <th className="py-2.5 px-3">Test ROC-AUC</th>
                    <th className="py-2.5 px-3">Test PR-AUC</th>
                    <th className="py-2.5 px-3">Test F2</th>
                    <th className="py-2.5 px-3">Lead Time</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {pastRuns.map((r) => (
                    <tr key={r.job_id} className={`hover:bg-slate-800/30 transition-colors ${r.is_active ? 'bg-emerald-950/20' : ''}`}>
                      <td className="py-3 px-3">
                        <div className="font-bold text-white">{r.job_id}</div>
                        <div className="text-[10px] text-slate-500">
                          {r.trained_at ? new Date(r.trained_at).toLocaleString() : 'N/A'}
                        </div>
                      </td>
                      <td className="py-3 px-3">
                        <div className="font-semibold text-amber-300">{r.champion_label}</div>
                        <div className="text-[10px] text-slate-500 capitalize">{r.budget} budget &bull; {r.scope}</div>
                      </td>
                      <td className="py-3 px-3 text-slate-300">
                        {r.n_students} students
                        <div className="text-[10px] text-slate-500">({r.n_attrited} attrited)</div>
                      </td>
                      <td className="py-3 px-3 text-amber-400 font-bold">{r.test_roc_auc.toFixed(4)}</td>
                      <td className="py-3 px-3 text-indigo-400">{r.test_pr_auc.toFixed(4)}</td>
                      <td className="py-3 px-3 text-emerald-400">{r.test_f2_score.toFixed(4)}</td>
                      <td className="py-3 px-3 text-cyan-400">{r.lead_time_semesters.toFixed(1)} sems</td>
                      <td className="py-3 px-3">
                        {r.is_active ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                            ACTIVE
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-slate-800 text-slate-400 border border-slate-700">
                            Inactive
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          {!r.is_active ? (
                            <button
                              onClick={() => handleActivateRun(r.job_id)}
                              disabled={activatingJobId === r.job_id}
                              className="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-[11px] font-semibold transition-colors disabled:opacity-50"
                            >
                              {activatingJobId === r.job_id ? 'Activating...' : 'Activate'}
                            </button>
                          ) : (
                            <button
                              onClick={handleRollback}
                              disabled={isRollingBack}
                              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium transition-colors disabled:opacity-50"
                            >
                              Rollback
                            </button>
                          )}
                          <button
                            onClick={() => handleDeleteRun(r.job_id)}
                            className="p-1 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/30 transition-colors"
                            title="Delete run artifacts"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── MODE 1: BATCH UPLOAD ────────────────────────────────────────────── */}
      {activeTab === 'upload' && (
        <div className="space-y-6">
          {/* Upload Zone */}
          <section className="rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/30 p-8 text-center hover:border-indigo-500/50 transition-colors">
            <input
              type="file"
              ref={fileInputRef}
              onChange={e => e.target.files && handleFileSelected(e.target.files[0])}
              accept=".csv, .xlsx, .xls"
              className="hidden"
            />

            <div
              onDragOver={e => e.preventDefault()}
              onDrop={handleFileDrop}
              className="space-y-4 max-w-md mx-auto"
            >
              <div className="w-12 h-12 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto">
                <FileSpreadsheet className="w-6 h-6" />
              </div>

              <div>
                <h3 className="text-sm font-bold text-white">
                  {file ? file.name : 'Drag & drop student observation file'}
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Supports <strong>CSV</strong> or <strong>Excel (.xlsx)</strong> formatted in either canonical long format or wide format. Max 10 MB.
                </p>

                {/* Outcome Detection Chip */}
                {inspectLoading && (
                  <div className="mt-2.5 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-slate-800 text-indigo-300 border border-slate-700">
                    <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />
                    Inspecting dataset for outcome labels...
                  </div>
                )}

                {!inspectLoading && inspectResult && (
                  <div className="mt-2.5 flex items-center justify-center">
                    {inspectResult.has_outcomes ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30">
                        <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />
                        Outcome labels detected: {inspectResult.n_students} students, {(inspectResult.attrition_rate * 100).toFixed(1)}% attrited
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium font-mono bg-slate-800 text-slate-400 border border-slate-700">
                        <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                        No outcome column found — inference only
                      </span>
                    )}
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors"
                >
                  Browse Computer
                </button>
                <a
                  href={api.getTemplateUrl(false)}
                  download="academic_observation_template.csv"
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-indigo-300 text-xs font-medium border border-slate-800 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Template (Inference)
                </a>
                <a
                  href={api.getTemplateUrl(true)}
                  download="academic_observation_template_with_outcomes.csv"
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-amber-300 text-xs font-medium border border-amber-900/40 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Template (With Outcomes)
                </a>
              </div>

              {file && (
                <div className="pt-3 flex flex-col sm:flex-row items-center gap-3">
                  <button
                    onClick={executeUpload}
                    disabled={uploading || Boolean(activeJobId)}
                    className="flex-1 w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md shadow-indigo-600/30 disabled:opacity-50"
                  >
                    {uploading ? 'Processing Zero-Leakage Pipeline & TreeSHAP...' : 'Execute Model Inference & Explainability'}
                  </button>

                  {inspectResult?.has_outcomes && (
                    <button
                      type="button"
                      onClick={() => setIsRetrainModalOpen(true)}
                      disabled={!inspectResult.can_retrain || Boolean(activeJobId)}
                      title={
                        !inspectResult.can_retrain
                          ? inspectResult.blocking_issues.join(' | ')
                          : activeJobId
                          ? 'A retraining job is currently in progress.'
                          : 'Retrain champion and candidate models on your uploaded cohort'
                      }
                      className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-all shadow-md shadow-amber-600/30 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 whitespace-nowrap"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${activeJobId ? 'animate-spin' : ''}`} />
                      Retrain on My Data
                    </button>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* Dataset Retrain Issues & Warnings Box */}
          {inspectResult && (inspectResult.blocking_issues.length > 0 || inspectResult.warnings.length > 0) && (
            <div className="space-y-3">
              {inspectResult.blocking_issues.length > 0 && (
                <div className="rounded-xl border border-rose-500/40 bg-rose-950/20 p-4 space-y-2">
                  <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase tracking-wider font-mono">
                    <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                    Retraining Blocked ({inspectResult.blocking_issues.length} Criteria Not Met)
                  </div>
                  <ul className="space-y-1 text-xs text-rose-300/90 pl-6 list-disc">
                    {inspectResult.blocking_issues.map((issue, idx) => (
                      <li key={idx}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}

              {inspectResult.warnings.length > 0 && (
                <div className="rounded-xl border border-amber-500/40 bg-amber-950/20 p-4 space-y-2">
                  <div className="flex items-center gap-2 text-amber-400 text-xs font-bold uppercase tracking-wider font-mono">
                    <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
                    Advisory Dataset Warnings
                  </div>
                  <ul className="space-y-1 text-xs text-amber-300/90 pl-6 list-disc">
                    {inspectResult.warnings.map((warn, idx) => (
                      <li key={idx}>{warn}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Live Retraining Progress Banner */}
          {activeJobId && jobStatus && (jobStatus.state === 'queued' || jobStatus.state === 'running') && (
            <div className="rounded-xl border border-amber-500/40 bg-slate-900/90 p-5 space-y-3 shadow-lg shadow-amber-950/20">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 text-amber-400 font-bold text-sm">
                  <RefreshCw className="w-4 h-4 animate-spin text-amber-400" />
                  <span>Model Retraining in Progress ({jobStatus.phase || 'initializing'})</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                  {jobStatus.current_model && (
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-amber-300 border border-amber-500/30">
                      Model: {jobStatus.current_model} ({jobStatus.models_done}/{jobStatus.models_total})
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    {jobStatus.elapsed_s.toFixed(1)}s elapsed
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-amber-500 h-2.5 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${Math.max(5, jobStatus.progress_pct)}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-mono">{jobStatus.message}</span>
                <span className="text-amber-400 font-mono font-bold">{jobStatus.progress_pct}%</span>
              </div>
            </div>
          )}

          {/* Retraining Success Result Card */}
          {retrainResult && (
            <div className="rounded-2xl border border-amber-500/50 bg-gradient-to-br from-amber-950/30 via-slate-900 to-slate-900/90 p-6 space-y-5 shadow-xl shadow-amber-950/20">
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-0.5 rounded-full text-xs font-bold font-mono bg-amber-500/20 text-amber-400 border border-amber-500/40">
                      Trained on user-provided data
                    </span>
                    <span className="text-xs text-slate-400 font-mono">
                      Job ID: {retrainResult.job_id}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2 mt-1">
                    <Sparkles className="w-5 h-5 text-amber-400" />
                    Retrained Champion: {retrainResult.champion_label}
                  </h3>
                  <p className="text-xs text-slate-400">
                    Evaluated across {retrainResult.n_students} students ({retrainResult.n_attrited} attrited cases) under {retrainResult.budget} budget.
                  </p>
                </div>

                <button
                  onClick={() => setRetrainResult(null)}
                  className="text-slate-500 hover:text-slate-300 p-1 rounded-lg transition-colors"
                  title="Dismiss card"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <span className="text-xs text-slate-400 uppercase">Test ROC-AUC</span>
                  <div className="text-xl font-bold text-amber-400 mt-1">
                    {retrainResult.test_roc_auc.toFixed(4)}
                  </div>
                  <span className="text-xs text-slate-500">Discrimination</span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <span className="text-xs text-slate-400 uppercase">Test PR-AUC</span>
                  <div className="text-xl font-bold text-indigo-400 mt-1">
                    {retrainResult.test_pr_auc.toFixed(4)}
                  </div>
                  <span className="text-xs text-slate-500">Imbalance-Robust</span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <span className="text-xs text-slate-400 uppercase">Test F2-Score</span>
                  <div className="text-xl font-bold text-emerald-400 mt-1">
                    {retrainResult.test_f2_score.toFixed(4)}
                  </div>
                  <span className="text-xs text-slate-500">Recall-Oriented</span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <span className="text-xs text-slate-400 uppercase">Early Lead Time</span>
                  <div className="text-xl font-bold text-cyan-400 mt-1">
                    {retrainResult.lead_time_semesters.toFixed(1)} Sems
                  </div>
                  <span className="text-xs text-slate-500">Pre-Attrition Lead</span>
                </div>
              </div>

              {/* Card Actions: Scoring Activation & Isolation Provenance */}
              <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800">
                <div className="flex items-center gap-2">
                  {meta?.active_run_id === retrainResult.job_id ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                      <Check className="w-4 h-4 text-emerald-400" />
                      Active Scoring Model
                    </span>
                  ) : (
                    <button
                      onClick={() => handleActivateRun(retrainResult.job_id)}
                      disabled={activatingJobId === retrainResult.job_id}
                      className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-bold shadow-md shadow-amber-500/20 transition-all disabled:opacity-50"
                    >
                      {activatingJobId === retrainResult.job_id ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Activating Model...
                        </>
                      ) : (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          Use this model for scoring
                        </>
                      )}
                    </button>
                  )}

                  {meta?.data_mode === 'REAL' && (
                    <button
                      onClick={handleRollback}
                      disabled={isRollingBack}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-colors disabled:opacity-50"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      Rollback to Synthetic
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                  <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Isolated: Saved to <code>artifacts/user_dataset/{retrainResult.job_id}/</code></span>
                </div>
              </div>
            </div>
          )}

          {/* Retraining Failure Alert */}
          {retrainError && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-950/20 p-4 space-y-2 flex items-start justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase tracking-wider font-mono">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                  Retraining Job Failed
                </div>
                <p className="text-xs text-rose-300">{retrainError}</p>
                <p className="text-xs text-slate-400 mt-1">
                  Existing thesis models and benchmarks were not changed.
                </p>
              </div>
              <button
                onClick={() => setRetrainError(null)}
                className="text-slate-500 hover:text-slate-300 p-1 rounded-lg transition-colors"
                title="Dismiss error"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Validation Errors Box */}
          {uploadErrors && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-950/20 p-5 space-y-3">
              <div className="flex items-center gap-2 text-rose-400 text-sm font-bold">
                <AlertTriangle className="w-4 h-4" />
                Schema Validation Report ({uploadErrors.length} Issue{uploadErrors.length > 1 ? 's' : ''} Identified)
              </div>
              <p className="text-xs text-slate-300">
                Uploaded file failed validation against <code>/data/schema/schema.json</code>. Correct the highlighted rows and re-upload:
              </p>
              <div className="max-h-48 overflow-y-auto space-y-1.5 font-mono text-xs">
                {uploadErrors.map((err, idx) => (
                  <div key={idx} className="p-2 rounded bg-slate-950/80 border border-slate-800 text-rose-300 flex items-start gap-2">
                    <span className="text-rose-400 font-bold">
                      {err.row ? `[Row ${err.row}]` : `[Schema]`}:
                    </span>
                    <span>{err.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Batch Results View */}
          {batchResult && (
            <div className="space-y-6">
              {/* Summary Stats Row */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                  <span className="text-xs text-slate-400 uppercase font-mono">Cohort Evaluated</span>
                  <div className="text-2xl font-bold font-mono text-white mt-1">
                    {batchResult.total_students} Students
                  </div>
                  <span className="text-xs text-slate-500 font-mono">
                    {batchResult.total_records} Longitudinal Records
                  </span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                  <span className="text-xs text-slate-400 uppercase font-mono">Average Risk Index</span>
                  <div className="text-2xl font-bold font-mono text-indigo-400 mt-1">
                    {batchResult.summary.average_risk_index.toFixed(1)} / 100
                  </div>
                  <span className="text-xs text-slate-500 font-mono">
                    Cohort Mean Calibrated Risk
                  </span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                  <span className="text-xs text-slate-400 uppercase font-mono">Inference Latency</span>
                  <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                    {batchResult.inference_time_per_student_ms.toFixed(1)} ms
                  </div>
                  <span className="text-xs text-emerald-400 font-mono">
                    ✓ Proposal NFR1 Compliant (&lt; 2s)
                  </span>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 flex flex-col justify-between">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400 uppercase font-mono">Actions</span>
                    {inspectResult?.has_outcomes && (
                      <span className="text-[10px] text-amber-400 font-mono font-semibold">Outcomes Detected</span>
                    )}
                  </div>
                  <div className="flex gap-2 mt-2">
                    <a
                      href={api.getDownloadUrl(batchResult.download_token, 'csv')}
                      className="flex-1 py-1.5 text-center text-xs font-semibold rounded bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
                    >
                      Export CSV
                    </a>
                    <a
                      href={api.getDownloadUrl(batchResult.download_token, 'json')}
                      className="flex-1 py-1.5 text-center text-xs font-semibold rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                    >
                      Export JSON
                    </a>
                  </div>
                  {inspectResult?.has_outcomes && (
                    <button
                      type="button"
                      onClick={() => setIsRetrainModalOpen(true)}
                      disabled={!inspectResult.can_retrain || Boolean(activeJobId)}
                      title={
                        !inspectResult.can_retrain
                          ? inspectResult.blocking_issues.join(' | ')
                          : activeJobId
                          ? 'A retraining job is currently in progress.'
                          : 'Retrain champion and candidate models on this uploaded cohort'
                      }
                      className="mt-2 w-full py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-all shadow-sm shadow-amber-600/30 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${activeJobId ? 'animate-spin' : ''}`} />
                      Retrain on This Dataset
                    </button>
                  )}
                </div>
              </div>

              {/* Student Results Table */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white">Student Attrition Triage Table</h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Evaluated with Champion <strong>CatBoost (Category C)</strong> with exact TreeSHAP attributions. Click any row to inspect trajectory.
                    </p>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    {batchResult.predictions.length} Predictions
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
                        <th className="py-2.5 px-3">Student ID</th>
                        <th className="py-2.5 px-3">Window (t)</th>
                        <th className="py-2.5 px-3">Academic Risk Index</th>
                        <th className="py-2.5 px-3">Risk Band</th>
                        <th className="py-2.5 px-3">Top Driving Factors</th>
                        <th className="py-2.5 px-3">Clinical Explanation Narrative</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-sans text-slate-300">
                      {batchResult.predictions.map(p => {
                        const isSelected = selectedStudent?.student_id === p.student_id;
                        return (
                          <tr
                            key={p.student_id}
                            onClick={() => setSelectedStudent(p)}
                            className={`cursor-pointer transition-colors ${
                              isSelected ? 'bg-indigo-950/30' : 'hover:bg-slate-800/30'
                            }`}
                          >
                            <td className="py-3 px-3 font-mono font-bold text-indigo-300">
                              {p.student_id}
                            </td>
                            <td className="py-3 px-3 font-mono text-slate-400">
                              Sem {p.window_semester}
                            </td>
                            <td className="py-3 px-3 font-mono font-bold text-slate-100">
                              {p.academic_risk_index.toFixed(1)} / 100
                            </td>
                            <td className="py-3 px-3">
                              <RiskBadge band={p.risk_band} score={p.academic_risk_index} />
                            </td>
                            <td className="py-3 px-3 text-[11px] text-slate-400 max-w-xs">
                              {p.top_factors.map(f => f.display_name).join(', ')}
                            </td>
                            <td className="py-3 px-3 text-xs text-slate-200 max-w-md line-clamp-2">
                              {p.explanation_text}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Selected Student Expanded View */}
              {selectedStudent && (
                <div className="rounded-xl border border-indigo-500/40 bg-slate-900/60 p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div>
                      <h4 className="text-sm font-bold text-white flex items-center gap-2">
                        Student Diagnostic Detail: <span className="font-mono text-indigo-400">{selectedStudent.student_id}</span>
                      </h4>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Calibrated Attrition Risk: <strong className="text-white">{selectedStudent.academic_risk_index.toFixed(1)}/100</strong> ({selectedStudent.risk_band} Band)
                      </p>
                    </div>
                    <RiskBadge band={selectedStudent.risk_band} size="lg" />
                  </div>

                  <div className="rounded-lg bg-indigo-950/20 border border-indigo-500/20 p-3 text-xs text-slate-200">
                    <strong className="text-indigo-400 block mb-1">Plain-Language Diagnostic Sentence:</strong>
                    "{selectedStudent.explanation_text}"
                  </div>

                  {/* Top Driving Factors */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {selectedStudent.top_factors.map((f, idx) => (
                      <div key={idx} className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                        <span className="text-[10px] uppercase font-bold text-slate-400 block">
                          Factor #{idx + 1} ({f.direction === 'RISK_INCREASING' ? 'Risk Elevator' : 'Stabilizer'})
                        </span>
                        <div className="font-semibold text-slate-200 mt-0.5">{f.display_name}</div>
                        <div className="flex justify-between items-baseline mt-2 font-mono text-[11px]">
                          <span className="text-slate-400">Value: {f.feature_value} {f.unit}</span>
                          <span className={f.direction === 'RISK_INCREASING' ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                            {f.direction === 'RISK_INCREASING' ? '+' : ''}{f.contribution.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── MODE 2: SINGLE STUDENT SIMULATOR ─────────────────────────────────── */}
      {activeTab === 'simulate' && (
        <div className="space-y-6">
          <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white">Manual Semester Trajectory Builder</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Adjust student signals across semesters to observe live risk index response and SHAP explanations.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={simStudentId}
                  onChange={e => setSimStudentId(e.target.value)}
                  className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                  placeholder="Student ID"
                />
                <button
                  onClick={addSemester}
                  disabled={simSemesters.length >= 8}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors disabled:opacity-50"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add Semester
                </button>
              </div>
            </div>

            {/* Semester Inputs Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase font-semibold">
                    <th className="py-2 px-2">Sem</th>
                    <th className="py-2 px-2">GPA (0.0-4.0)</th>
                    <th className="py-2 px-2">Exam Marks (%)</th>
                    <th className="py-2 px-2">Attendance (%)</th>
                    <th className="py-2 px-2">Delay (Days)</th>
                    <th className="py-2 px-2">Late Count</th>
                    <th className="py-2 px-2 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {simSemesters.map((s, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/20">
                      <td className="py-2 px-2 font-bold text-indigo-400">Sem {s.semester}</td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="0.05"
                          min="0.0"
                          max="4.0"
                          value={s.gpa}
                          onChange={e => {
                            const val = parseFloat(e.target.value);
                            const updated = [...simSemesters];
                            updated[idx].gpa = val;
                            setSimSemesters(updated);
                          }}
                          className="w-20 px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="1"
                          min="0"
                          max="100"
                          value={s.exam_marks}
                          onChange={e => {
                            const val = parseFloat(e.target.value);
                            const updated = [...simSemesters];
                            updated[idx].exam_marks = val;
                            setSimSemesters(updated);
                          }}
                          className="w-20 px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="1"
                          min="0"
                          max="100"
                          value={s.attendance_pct}
                          onChange={e => {
                            const val = parseFloat(e.target.value);
                            const updated = [...simSemesters];
                            updated[idx].attendance_pct = val;
                            setSimSemesters(updated);
                          }}
                          className="w-20 px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="45"
                          value={s.assignment_delay_days}
                          onChange={e => {
                            const val = parseFloat(e.target.value);
                            const updated = [...simSemesters];
                            updated[idx].assignment_delay_days = val;
                            setSimSemesters(updated);
                          }}
                          className="w-20 px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="1"
                          min="0"
                          max="15"
                          value={s.late_assignments_count}
                          onChange={e => {
                            const val = parseInt(e.target.value, 10);
                            const updated = [...simSemesters];
                            updated[idx].late_assignments_count = val;
                            setSimSemesters(updated);
                          }}
                          className="w-16 px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
                        />
                      </td>
                      <td className="py-2 px-2 text-right">
                        <button
                          onClick={() => removeSemester(idx)}
                          disabled={simSemesters.length <= 1}
                          className="p-1 text-slate-500 hover:text-rose-400 disabled:opacity-30"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="pt-2">
              <button
                onClick={executeSimulation}
                disabled={simLoading}
                className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md shadow-indigo-600/30 disabled:opacity-50"
              >
                {simLoading ? 'Scoring Sequence...' : 'Compute Academic Risk Index & Generate Diagnosis'}
              </button>
            </div>
          </section>

          {/* Simulation Output */}
          {simResult && (
            <div className="rounded-xl border border-indigo-500/40 bg-slate-900/60 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    Simulation Assessment Result: <span className="font-mono text-indigo-400">{simResult.student_id}</span>
                  </h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Evaluated through Champion CatBoost Model (window t={simResult.window_semester})
                  </p>
                </div>
                <RiskBadge band={simResult.risk_band} score={simResult.academic_risk_index} size="lg" />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                <TrajectoryChart
                  data={simSemesters}
                  title={`Simulated Trajectory (${simSemesters.length} Semesters)`}
                  height={240}
                />

                <div className="space-y-3">
                  <div className="rounded-lg bg-indigo-950/20 border border-indigo-500/20 p-3 text-xs text-slate-200">
                    <strong className="text-indigo-400 block mb-1">Diagnostic Sentence:</strong>
                    "{simResult.explanation_text}"
                  </div>

                  <div className="space-y-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                      Top Contributing Factors:
                    </span>
                    {simResult.top_factors.map((f, i) => (
                      <div key={i} className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800 text-xs">
                        <span className="text-slate-200">{f.display_name}</span>
                        <span className={f.direction === 'RISK_INCREASING' ? 'text-rose-400 font-mono font-bold' : 'text-emerald-400 font-mono font-bold'}>
                          {f.direction === 'RISK_INCREASING' ? '+' : ''}{f.contribution.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Integration & Data Resources ── */}
      <section className="border-t border-slate-800/80 pt-8 mt-10">
        <div className="mb-4">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Download className="w-4 h-4 text-indigo-400" />
            Data Templates & Integration Resources
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Download observation templates, review model performance benchmarks, or access system API endpoints.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Card 1: Upload Template (CSV) */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
              <Code2 className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-white">Upload Template (CSV)</h4>
              <p className="text-xs text-slate-400 mt-1">
                Canonical schema-compliant CSV template containing pre-formatted semester observation examples.
              </p>
            </div>
            <a
              href={api.getTemplateUrl()}
              download="academic_observation_template.csv"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download Template CSV
            </a>
          </div>

          {/* Card 2: Full Leaderboard (CSV) */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-white">Full Leaderboard (CSV)</h4>
              <p className="text-xs text-slate-400 mt-1">
                Test set evaluation metrics, composite ranks, and inference latency for all registered models.
              </p>
            </div>
            <a
              href="/leaderboard"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              View Leaderboard JSON
            </a>
          </div>

          {/* Card 3: OpenAPI Documentation */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
            <div className="w-10 h-10 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center">
              <ExternalLink className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-white">OpenAPI Documentation</h4>
              <p className="text-xs text-slate-400 mt-1">
                Interactive Swagger UI and ReDoc endpoints for system integration and contract validation.
              </p>
            </div>
            <div className="flex gap-2">
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
              >
                Swagger UI
              </a>
              <a
                href="http://127.0.0.1:8000/redoc"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
              >
                ReDoc
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ── Retraining Confirmation Modal ── */}
      {isRetrainModalOpen && inspectResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
          <div className="relative w-full max-w-lg rounded-2xl border border-slate-700 bg-slate-900 p-6 space-y-5 shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center">
                  <RefreshCw className="w-4 h-4 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">Retrain on My Data</h3>
                  <p className="text-xs text-slate-400">Configure background model training run</p>
                </div>
              </div>
              <button
                onClick={() => setIsRetrainModalOpen(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dataset Summary */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-2">
              <span className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider block">
                Dataset Summary
              </span>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                  <div className="text-base font-bold font-mono text-white">{inspectResult.n_students}</div>
                  <div className="text-[11px] text-slate-400">Total Students</div>
                </div>
                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                  <div className="text-base font-bold font-mono text-white">{inspectResult.n_records}</div>
                  <div className="text-[11px] text-slate-400">Observations</div>
                </div>
                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                  <div className="text-base font-bold font-mono text-amber-400">
                    {inspectResult.n_attrited} ({(inspectResult.attrition_rate * 100).toFixed(1)}%)
                  </div>
                  <div className="text-[11px] text-slate-400">Attrited</div>
                </div>
              </div>
            </div>

            {/* Hyperparameter Optimization Budget */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 block">
                Training Budget (Hyperparameter Optimization)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['quick', 'standard', 'full'] as const).map(b => (
                  <button
                    key={b}
                    type="button"
                    onClick={() => setRetrainBudget(b)}
                    className={`py-2 px-3 rounded-lg text-xs font-semibold border transition-all ${
                      retrainBudget === b
                        ? 'bg-amber-600/20 border-amber-500 text-amber-300 shadow-sm shadow-amber-600/30'
                        : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="capitalize">{b}</span>
                    <span className="block text-[10px] font-normal opacity-70">
                      {b === 'quick' ? 'Fast (1 trial)' : b === 'standard' ? 'Standard (10 trials)' : 'Full (30 trials)'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Scope */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 block">
                Model Scope
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setRetrainScope('champions')}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold border text-left transition-all ${
                    retrainScope === 'champions'
                      ? 'bg-amber-600/20 border-amber-500 text-amber-300 shadow-sm shadow-amber-600/30'
                      : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="font-bold">Champions Only</div>
                  <div className="text-[10px] font-normal opacity-70">GRU, CatBoost, Logistic, RF, LightGBM</div>
                </button>
                <button
                  type="button"
                  onClick={() => setRetrainScope('full')}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold border text-left transition-all ${
                    retrainScope === 'full'
                      ? 'bg-amber-600/20 border-amber-500 text-amber-300 shadow-sm shadow-amber-600/30'
                      : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="font-bold">All Models</div>
                  <div className="text-[10px] font-normal opacity-70">Full 11-model zoo (Categories A-F)</div>
                </button>
              </div>
            </div>

            {/* Isolation Guarantee Callout */}
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-3.5 flex items-start gap-3">
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <p className="text-xs text-emerald-200/90 leading-relaxed">
                <strong>Filesystem Isolation:</strong> Results and checkpoints are saved separately in{' '}
                <code>artifacts/user_dataset/</code> and do not replace or overwrite the synthetic thesis benchmark.
              </p>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 pt-2 border-t border-slate-800">
              <button
                type="button"
                onClick={() => setIsRetrainModalOpen(false)}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleStartRetrain}
                className="px-5 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold transition-all shadow-md shadow-amber-600/30 flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Confirm & Start Retraining
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
