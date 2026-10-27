import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  AlertTriangle,
  CheckCircle2,
  Layers,
  ArrowRight,
  TrendingDown,
  RefreshCw,
  Plus,
  Trash2,
  HelpCircle,
  FileText,
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
import type { BatchUploadResult, StudentPredictionResult, ValidationErrorDetail } from '../api/client';
import { RiskBadge } from '../components/RiskBadge';
import { TrajectoryChart } from '../components/TrajectoryChart';
import type { TrajectoryPoint } from '../components/TrajectoryChart';
import { LoadingState } from '../components/StateViews';

export const PredictPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'simulate'>('upload');

  // File Upload State
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadErrors, setUploadErrors] = useState<ValidationErrorDetail[] | null>(null);
  const [batchResult, setBatchResult] = useState<BatchUploadResult | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<StudentPredictionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // File Drag & Drop Handlers
  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (selectedFile: File) => {
    setFile(selectedFile);
    setUploadErrors(null);
    setBatchResult(null);
    setSelectedStudent(null);
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
              </div>

              <div className="flex items-center justify-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors"
                >
                  Browse Computer
                </button>
                <a
                  href={api.getTemplateUrl()}
                  download="academic_observation_template.csv"
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-indigo-300 text-xs font-medium border border-slate-800 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Template CSV
                </a>
              </div>

              {file && (
                <div className="pt-3">
                  <button
                    onClick={executeUpload}
                    disabled={uploading}
                    className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md shadow-indigo-600/30 disabled:opacity-50"
                  >
                    {uploading ? 'Processing Zero-Leakage Pipeline & TreeSHAP...' : 'Execute Model Inference & Explainability'}
                  </button>
                </div>
              )}
            </div>
          </section>

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
                  <span className="text-xs text-slate-400 uppercase font-mono">Export Predictions</span>
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
    </div>
  );
};
