import React, { useState } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Play,
  Download,
  RotateCcw,
  Users,
  Eye,
  X,
  Sparkles,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { api } from '../services/api';
import {
  UploadResponse,
  CohortPredictionResponse,
  StudentPrediction,
  JobStatusResponse,
} from '../types/api';
import { ChartCard } from '../components/common/ChartCard';
import { LoadingState } from '../components/common/FeedbackStates';
import { useCohort } from '../context/CohortContext';

const RISK_COLORS: Record<string, string> = {
  Low: '#10b981',
  Moderate: '#f59e0b',
  High: '#ef4444',
};

export const ScoreDataPage: React.FC = () => {
  const {
    file,
    setFile,
    fileName,
    setFileName,
    uploadResult,
    setUploadResult,
    mappings,
    setMappings,
    predictionResult,
    setPredictionResult,
    selectedStudent,
    setSelectedStudent,
    setCurrentPage,
    resetCohort,
  } = useCohort();

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  // Single student detail drawer
  const [selectedStudent, setSelectedStudent] = useState<StudentPrediction | null>(null);

  // Retraining state
  const [retrainingJob, setRetrainingJob] = useState<JobStatusResponse | null>(null);
  const [retrainingError, setRetrainingError] = useState<string | null>(null);

  // Drag and drop handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const res = await api.uploadFile(file);
      setUploadResult(res);
      setFileName(file.name);
      setMappings(res.suggested_mappings || {});
    } catch (err: any) {
      setUploadError(err.message || 'File upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleRunInference = async () => {
    if (!uploadResult) return;
    setPredicting(true);
    try {
      const res = await api.runPrediction(uploadResult.upload_id, mappings);
      setPredictionResult(res);
    } catch (err: any) {
      alert(`Inference failed: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const handleRetrain = async () => {
    if (!uploadResult) return;
    try {
      const job = await api.triggerRetraining(uploadResult.upload_id);
      setRetrainingJob(job);
      // Poll job status
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.getJobStatus(job.job_id);
          setRetrainingJob(status);
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
          }
        } catch {
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch (err: any) {
      setRetrainingError(err.message || 'Retraining initiation failed');
    }
  };

  // Pie chart data
  const distributionData = predictionResult?.cohort_summary?.risk_distribution
    ? Object.entries(predictionResult.cohort_summary.risk_distribution).map(([tier, count]) => ({
        name: tier,
        value: count,
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-2 border-b border-navy-800">
        <h2 className="text-xl font-bold text-white tracking-tight">Score Cohort Dataset</h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload student survey responses (.csv, .xlsx, .xls) to validate schema, map columns, run inference, and export downstream Fusion contracts.
        </p>
      </div>

      {/* Step 1: Upload Card */}
      {!uploadResult && (
        <div className="bg-navy-900 border border-navy-800 rounded-xl p-6 shadow-lg text-center">
          <div className="max-w-md mx-auto space-y-4">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mx-auto text-indigo-400">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Upload Questionnaire Responses</h3>
              <p className="text-xs text-slate-400 mt-1">
                CSV or Excel format containing student study IDs and Likert items (S1-S3, M1-M3, B1-B3, SE1-SE3, PS1-PS3).
              </p>
            </div>

            <input
              type="file"
              id="survey-upload"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
              className="hidden"
            />
            <label
              htmlFor="survey-upload"
              className="inline-block px-4 py-2 rounded-lg bg-navy-800 border border-navy-700 text-xs font-semibold text-slate-200 hover:bg-navy-750 cursor-pointer transition-colors"
            >
              {file ? file.name : 'Select File from Device'}
            </label>

            {file && (
              <div className="pt-2">
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-md shadow-indigo-600/30 disabled:opacity-50 transition-all"
                >
                  {uploading ? 'Validating Schema...' : 'Upload & Validate'}
                </button>
              </div>
            )}

            {uploadError && (
              <div className="p-3 bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs rounded-lg">
                {uploadError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Step 2: Validation Report & Column Mapping */}
      {uploadResult && !predictionResult && (
        <div className="space-y-6">
          <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-navy-800">
              <div>
                <span className="text-[10px] text-slate-400 font-mono">Session ID: {uploadResult.upload_id}</span>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-indigo-400" />
                  {uploadResult.filename} ({uploadResult.row_count} rows, {uploadResult.column_count} columns)
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`px-2.5 py-1 rounded text-xs font-bold border flex items-center gap-1.5 ${
                    uploadResult.validation_report.status === 'PASSED'
                      ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                      : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                  }`}
                >
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Schema: {uploadResult.validation_report.status}
                </span>
                <button
                  onClick={() => {
                    setUploadResult(null);
                    setFile(null);
                  }}
                  className="px-2 py-1 rounded bg-navy-800 text-slate-400 hover:text-white text-xs border border-navy-700"
                >
                  Replace File
                </button>
              </div>
            </div>

            {/* Column Mapping Table */}
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-slate-300 mb-2">Column Mapping Verification</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-xs">
                {Object.entries(mappings).map(([orig, target]) => {
                  const conf = uploadResult.mapping_confidences?.[orig] ?? 1.0;
                  return (
                    <div key={orig} className="p-2.5 bg-navy-950 rounded-lg border border-navy-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between text-slate-400 font-mono text-[10px]">
                        <span className="truncate" title={orig}>{orig}</span>
                        <span className="text-indigo-400 font-bold">{(conf * 100).toFixed(0)}%</span>
                      </div>
                      <div className="mt-1.5 flex items-center gap-1">
                        <span className="text-slate-400">→</span>
                        <input
                          type="text"
                          value={target}
                          onChange={(e) =>
                            setMappings({ ...mappings, [orig]: e.target.value })
                          }
                          className="w-full bg-navy-900 border border-navy-700 rounded px-1.5 py-0.5 text-xs text-slate-100 font-mono focus:border-indigo-500 focus:outline-none"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Action Bar */}
            <div className="mt-6 flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-navy-800">
              <div className="text-xs text-slate-400">
                Default champion model: <strong className="text-indigo-300">HistGradientBoosting / Ridge</strong>
              </div>
              <div className="flex items-center gap-3">
                {uploadResult.has_outcomes && (
                  <button
                    onClick={handleRetrain}
                    className="px-4 py-2 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/40 text-xs font-bold hover:bg-amber-500/30 transition-colors flex items-center gap-1.5"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Retrain on My Data
                  </button>
                )}
                <button
                  onClick={handleRunInference}
                  disabled={predicting}
                  className="px-6 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-md shadow-indigo-600/30 flex items-center gap-2 transition-all disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  {predicting ? 'Scoring Cohort...' : 'Score Cohort'}
                </button>
              </div>
            </div>

            {/* Retraining Job Progress Banner */}
            {retrainingJob && (
              <div className="mt-4 p-4 bg-navy-950 border border-amber-500/30 rounded-lg text-xs space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-amber-300">Background Retraining: {retrainingJob.job_id}</span>
                  <span className="font-mono text-slate-300">{retrainingJob.status.toUpperCase()} ({retrainingJob.progress_pct}%)</span>
                </div>
                <div className="w-full bg-navy-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-amber-400 h-full transition-all duration-300"
                    style={{ width: `${retrainingJob.progress_pct}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-400">{retrainingJob.message}</p>
                {retrainingJob.results && (
                  <div className="p-2 bg-navy-900 rounded border border-navy-700 text-emerald-300 font-mono text-[11px]">
                    Retrained Champion: {retrainingJob.results.champion_model} ({retrainingJob.results.champion_performance})
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Step 3: Prediction Results Dashboard */}
      {predictionResult && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 font-medium mb-1">Total Students Scored</div>
              <div className="text-2xl font-bold text-white font-mono">
                {predictionResult.cohort_summary.valid_records ??
                  predictionResult.cohort_summary.total_submitted ??
                  predictionResult.cohort_summary.total_students ??
                  predictionResult.students?.length ??
                  0}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Model: {predictionResult.model_used}</div>
            </div>

            <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 font-medium mb-1">Average Cohort PRI</div>
              <div className="text-2xl font-bold text-indigo-400 font-mono">
                {(
                  predictionResult.cohort_summary.pri_mean ??
                  predictionResult.cohort_summary.average_pri ??
                  0
                ).toFixed(1)}{' '}
                / 100
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Mean Psychosocial Risk</div>
            </div>

            <div className="bg-navy-900 border border-navy-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 font-medium mb-1">High-Risk Flagged</div>
              <div className="text-2xl font-bold text-rose-400 font-mono">
                {predictionResult.cohort_summary.high_risk_count} students
              </div>
              <div className="text-[11px] text-rose-400/80 mt-1">Requiring immediate outreach</div>
            </div>

            <div className="bg-navy-900 border border-navy-800 rounded-xl p-4 flex flex-col justify-between">
              <div className="text-xs text-slate-400 font-medium mb-1">Export Contracts</div>
              <div className="flex gap-2 mt-2">
                <a
                  href={`/api/export/${predictionResult.upload_id}.csv`}
                  download
                  className="flex-1 py-1.5 text-center rounded bg-navy-800 hover:bg-navy-750 text-slate-200 text-xs font-semibold border border-navy-700 flex items-center justify-center gap-1"
                >
                  <Download className="w-3.5 h-3.5" /> CSV
                </a>
                <a
                  href={`/api/export/fusion/${predictionResult.upload_id}.json`}
                  download
                  className="flex-1 py-1.5 text-center rounded bg-indigo-600/30 hover:bg-indigo-600/40 text-indigo-200 text-xs font-semibold border border-indigo-500/40 flex items-center justify-center gap-1"
                >
                  <Download className="w-3.5 h-3.5" /> Fusion JSON
                </a>
              </div>
            </div>
          </div>

          {/* Risk Donut & Student Table */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ChartCard title="Cohort Risk Category Breakdown" className="lg:col-span-1">
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={distributionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {distributionData.map((entry) => (
                        <Cell key={`cell-${entry.name}`} fill={RISK_COLORS[entry.name] || '#6366f1'} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                      formatter={(val: number) => [val, 'Students']}
                    />
                    <Legend
                      verticalAlign="bottom"
                      height={36}
                      formatter={(val: string) => <span className="text-xs text-slate-300 font-medium">{val}</span>}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            {/* Scored Students Table */}
            <div className="lg:col-span-2 bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-sm text-slate-100">Scored Student Cohort</h3>
                  <p className="text-xs text-slate-400">Click a student to view their 5-factor radar and local SHAP decomposition.</p>
                </div>
              </div>

              <div className="flex-1 overflow-x-auto max-h-80 overflow-y-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="sticky top-0 bg-navy-950">
                    <tr className="border-b border-navy-700 text-slate-400 uppercase tracking-wider text-[11px]">
                      <th className="py-2.5 px-3">Study ID</th>
                      <th className="py-2.5 px-3 font-mono">PRI Score</th>
                      <th className="py-2.5 px-3">Risk Tier</th>
                      <th className="py-2.5 px-3">Top Risk Indicator</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-navy-800">
                    {predictionResult.students.map((stu) => {
                      const topDriver = stu.top_risk_contributors?.[0]?.feature?.replace('factor_', '') || 'Stress';
                      return (
                        <tr
                          key={stu.study_id}
                          onClick={() => setSelectedStudent(stu)}
                          className="hover:bg-navy-800/60 cursor-pointer transition-colors"
                        >
                          <td className="py-2.5 px-3 font-mono font-semibold text-slate-200">
                            {stu.study_id}
                          </td>
                          <td className="py-2.5 px-3 font-mono font-bold text-slate-100">
                            {stu.pri_score.toFixed(1)}
                          </td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                                stu.risk_category === 'High'
                                  ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                                  : stu.risk_category === 'Moderate'
                                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                                  : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                              }`}
                            >
                              {stu.risk_category}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-slate-400 capitalize">{topDriver}</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedStudent(stu);
                              }}
                              className="px-2 py-1 rounded text-[11px] bg-navy-800 hover:bg-navy-750 text-slate-300 border border-navy-700"
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Per-Student Detail Drawer */}
      {selectedStudent && (
        <div className="fixed inset-y-0 right-0 w-full max-w-xl bg-navy-900 border-l border-navy-700 shadow-2xl z-50 p-6 overflow-y-auto no-print">
          <div className="flex items-center justify-between pb-4 border-b border-navy-700">
            <div>
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Student Diagnostic</span>
              <h3 className="text-lg font-bold text-white font-mono">{selectedStudent.study_id}</h3>
            </div>
            <button
              onClick={() => setSelectedStudent(null)}
              className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-navy-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-6 mt-5 text-xs">
            {/* Score & Tier Banner */}
            <div className="p-4 bg-navy-950 border border-navy-800 rounded-xl flex items-center justify-between">
              <div>
                <span className="text-slate-400 text-[11px]">Psychosocial Risk Index</span>
                <div className="text-3xl font-extrabold text-white font-mono mt-0.5">
                  {selectedStudent.pri_score.toFixed(1)} <span className="text-xs text-slate-400 font-normal">/ 100</span>
                </div>
              </div>
              <span
                className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${
                  selectedStudent.risk_category === 'High'
                    ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                    : selectedStudent.risk_category === 'Moderate'
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                    : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                }`}
              >
                {selectedStudent.risk_category} Risk
              </span>
            </div>

            {/* Plain-Language Narrative Box */}
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-2">
              <h4 className="font-semibold text-indigo-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                Plain-Language Narrative Explanation
              </h4>
              <p className="text-slate-200 leading-relaxed">{selectedStudent.narrative_headline}</p>
              {selectedStudent.narrative_drivers && (
                <p className="text-slate-300 text-[11px]"><strong>Risk Drivers:</strong> {selectedStudent.narrative_drivers}</p>
              )}
              {selectedStudent.narrative_buffers && (
                <p className="text-slate-300 text-[11px]"><strong>Protective Buffers:</strong> {selectedStudent.narrative_buffers}</p>
              )}
            </div>

            {/* 5-Factor Radar Chart */}
            <div>
              <h4 className="font-semibold text-slate-200 mb-2">5-Factor Psychometric Radar</h4>
              <div className="h-60 bg-navy-950 rounded-xl border border-navy-800 p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart
                    data={[
                      { subject: 'Stress', val: selectedStudent.factor_scores.stress },
                      { subject: 'Motivation', val: selectedStudent.factor_scores.motivation },
                      { subject: 'Belonging', val: selectedStudent.factor_scores.belongingness },
                      { subject: 'Self-Efficacy', val: selectedStudent.factor_scores.self_efficacy },
                      { subject: 'Peer Support', val: selectedStudent.factor_scores.peer_support },
                    ]}
                  >
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="subject" stroke="#94a3b8" fontSize={11} />
                    <PolarRadiusAxis angle={30} domain={[1, 5]} stroke="#64748b" fontSize={10} />
                    <Radar name="Student Score" dataKey="val" stroke="#818cf8" fill="#6366f1" fillOpacity={0.4} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
