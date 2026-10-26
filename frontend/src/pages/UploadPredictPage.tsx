import React, { useState } from 'react';
import { Card } from '../components/common/Card';
import { ShapWaterfall, ERIGauge } from '../components/charts/ShapWaterfall';
import { apiService } from '../services/api';
import { UploadPredictResponse, PredictionItem } from '../types/api';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  Eye,
  X,
  FileText,
  Clock,
  Sparkles,
} from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

export const UploadPredictPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<PredictionItem | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUploadAndScore = async () => {
    if (!file) return;
    try {
      setLoading(true);
      setError(null);
      const res = await apiService.uploadAndPredict(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to score uploaded file.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSample = () => {
    const csvContent =
      'household_income_bracket,financial_stress,commute_time_min,commute_cost_lkr_week,housing_type,internet_reliability,internet_affordability,device_type,device_quality,quiet_study_space,part_time_employment,work_hours_per_week,scholarship_bursary,dependents_count,district_type,year_of_study,faculty,baseline_gpa_band,baseline_engagement_score\n' +
      '"Below 50,000",4,75,2500,"Rented room",2,2,"Smartphone only",2,1,1,25,0,2,"Rural",2,"Computing","2.0-2.5",45.0\n' +
      '"100,000 - 150,000",2,30,800,"With parents",4,4,"Own laptop",4,4,0,0,0,0,"Urban",1,"Engineering","3.0-3.3",72.5\n' +
      '"50,000 - 100,000",3,50,1500,"Boarding house",3,3,"Own laptop",3,2,1,15,1,1,"Semi-Urban",3,"Business","2.7-3.0",58.0\n';

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_student_questionnaire.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const donutColors: Record<string, string> = {
    Low: '#10b981',
    Moderate: '#3b82f6',
    High: '#f59e0b',
    Critical: '#ef4444',
  };

  const donutData = result?.cohort_summary?.risk_distribution
    ? Object.entries(result.cohort_summary.risk_distribution).map(([k, v]) => ({
        name: k,
        value: v,
        color: donutColors[k] || '#64748b',
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-white">
          Upload Survey Cohort & Automated Risk Scoring
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Batch ingestion supporting .csv and .xlsx. Automatically scrubs student identifiers, matches synonyms, and derives calibrated ERI scores.
        </p>
      </div>

      {/* Privacy Guarantee Box */}
      <div className="p-3.5 bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/40 rounded-xl flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 text-blue-900 dark:text-blue-200">
          <ShieldCheck className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0" />
          <span>
            <strong>Zero PII Guarantee:</strong> Names, national IDs, contact numbers, and emails are purged before ingestion. Cryptographic pseudonyms (P-XXXXX) are assigned automatically.
          </span>
        </div>
        <button
          onClick={handleDownloadSample}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-white dark:bg-slate-900 border border-blue-300 dark:border-blue-800 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-50 whitespace-nowrap transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          Download Sample Template
        </button>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="md:col-span-2">
          <div className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-2xl p-8 text-center bg-white dark:bg-slate-900 hover:border-blue-500 transition-colors">
            <UploadCloud className="w-12 h-12 text-blue-500 mx-auto mb-3 animate-bounce" />
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Drag & Drop Survey Questionnaire File
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supports .csv, .xlsx, or .xls format (Up to 50 MB)
            </p>
            <div className="mt-4 flex items-center justify-center gap-3">
              <label className="cursor-pointer px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-sm transition-colors">
                Browse Files
                <input
                  type="file"
                  accept=".csv, .xlsx, .xls"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
              {file && (
                <span className="text-xs font-mono text-slate-700 dark:text-slate-300 font-medium">
                  {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Action Panel */}
        <div className="flex flex-col justify-center space-y-3 bg-slate-50 dark:bg-slate-800/40 p-5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <div className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
            Scoring Engine Status
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400">
            Selected Model: <span className="font-mono font-bold text-blue-600">logistic_l2</span>
            <br />
            Probability Calibrator: <span className="font-mono font-bold text-emerald-600">Isotonic</span>
          </div>
          <button
            onClick={handleUploadAndScore}
            disabled={!file || loading}
            className={`w-full py-2.5 rounded-xl text-xs font-bold shadow-md transition-all flex items-center justify-center gap-2 ${
              file && !loading
                ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                : 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
            }`}
          >
            {loading ? (
              <>
                <Clock className="w-4 h-4 animate-spin" />
                Scoring Student Cohort...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Execute Production Prediction
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/50 rounded-xl text-xs text-rose-800 dark:text-rose-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Schema Ingestion Report */}
          <Card
            title="Schema Ingestion & Privacy Validation Report"
            subtitle={`Matched ${result.schema_report.columns_matched_count} columns across ${result.schema_report.total_rows_ingested} student records`}
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 bg-emerald-50 dark:bg-emerald-950/20 rounded-lg border border-emerald-200 dark:border-emerald-900/40">
                <span className="text-emerald-700 dark:text-emerald-300 font-semibold block">Matched Columns</span>
                <span className="font-mono font-bold text-lg text-emerald-900 dark:text-emerald-100">
                  {result.schema_report.columns_matched_count}
                </span>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-900/40">
                <span className="text-blue-700 dark:text-blue-300 font-semibold block">PII Columns Dropped</span>
                <span className="font-mono font-bold text-lg text-blue-900 dark:text-blue-100">
                  {result.schema_report.dropped_pii_columns.length}
                </span>
              </div>
              <div className="p-3 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-900/40">
                <span className="text-amber-700 dark:text-amber-300 font-semibold block">Missing Fields Imputed</span>
                <span className="font-mono font-bold text-lg text-amber-900 dark:text-amber-100">
                  {result.schema_report.missing_features_imputed.length}
                </span>
              </div>
              <div className="p-3 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-900/40">
                <span className="text-purple-700 dark:text-purple-300 font-semibold block">Pseudonyms Issued</span>
                <span className="font-mono font-bold text-lg text-purple-900 dark:text-purple-100">
                  {result.schema_report.total_rows_ingested}
                </span>
              </div>
            </div>
          </Card>

          {/* Cohort Summary & Donut */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <Card title="Cohort Risk Triage Distribution" subtitle="Proportion of students across ERI bands">
              <div className="h-56 w-full flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={donutData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={4}>
                      {donutData.map((entry, idx) => (
                        <Cell key={`cell-${idx}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '11px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title="Cohort Key Statistics" subtitle="Aggregate Environmental Risk Index metrics">
              <div className="space-y-3 text-xs pt-3">
                <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500">Students Evaluated:</span>
                  <span className="font-mono font-bold">{result.cohort_summary.total_students}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500">Mean ERI:</span>
                  <span className="font-mono font-bold text-blue-600">{result.cohort_summary.mean_eri.toFixed(1)} / 100</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500">Median ERI:</span>
                  <span className="font-mono font-bold">{result.cohort_summary.median_eri.toFixed(1)} / 100</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500">Causal Summary Gate:</span>
                  <span className="font-mono text-slate-700 dark:text-slate-300">{result.cohort_summary.causal_summary_status}</span>
                </div>
              </div>
            </Card>

            <Card title="Download Export Bundle" subtitle="Generated deliverables for counseling & fusion">
              <div className="grid grid-cols-2 gap-2 pt-2">
                <a
                  href={result.download_urls.csv}
                  download
                  className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 transition-colors"
                >
                  <FileText className="w-4 h-4 text-blue-500" /> CSV
                </a>
                <a
                  href={result.download_urls.xlsx}
                  download
                  className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 transition-colors"
                >
                  <FileSpreadsheet className="w-4 h-4 text-emerald-500" /> Excel (.xlsx)
                </a>
                <a
                  href={result.download_urls.json}
                  download
                  className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 transition-colors"
                >
                  <Download className="w-4 h-4 text-purple-500" /> Fusion JSON
                </a>
                <a
                  href={result.download_urls.html}
                  download
                  className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5 text-slate-700 dark:text-slate-200 transition-colors"
                >
                  <Eye className="w-4 h-4 text-rose-500" /> HTML Report
                </a>
              </div>
            </Card>
          </div>

          {/* Student Roster Table */}
          <Card title="Scored Student Cohort Sample" subtitle="Top records with deterministic explanations">
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-800 font-semibold text-slate-700 dark:text-slate-300 border-b border-slate-200 dark:border-slate-700">
                    <th className="py-2.5 px-3">Participant ID</th>
                    <th className="py-2.5 px-3">Calibrated ERI</th>
                    <th className="py-2.5 px-3">Risk Tier</th>
                    <th className="py-2.5 px-3">Prescribed SLA</th>
                    <th className="py-2.5 px-3">Quality Note</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {result.scored_students_sample.map((stu, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                      <td className="py-2 px-3 font-mono font-bold text-slate-800 dark:text-slate-200">{stu.participant_id}</td>
                      <td className="py-2 px-3 font-mono font-bold text-blue-600 dark:text-blue-400">{stu.eri.toFixed(1)}</td>
                      <td className="py-2 px-3">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                            stu.risk_band === 'Critical'
                              ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                              : stu.risk_band === 'High'
                              ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300'
                              : stu.risk_band === 'Moderate'
                              ? 'bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300'
                              : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'
                          }`}
                        >
                          {stu.risk_band}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-600 dark:text-slate-400">{stu.intervention_sla}</td>
                      <td className="py-2 px-3 text-[11px] text-slate-500">{stu.confidence_note}</td>
                      <td className="py-2 px-3 text-right">
                        <button
                          onClick={() => setSelectedStudent(stu)}
                          className="px-2.5 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 text-slate-700 dark:text-slate-300 rounded text-[11px] font-semibold transition-colors"
                        >
                          Waterfall
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Slide-Out Detail Modal / Drawer */}
      {selectedStudent && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl relative space-y-4">
            <div className="flex justify-between items-center border-b border-slate-100 dark:border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  Student Attribution Profile: {selectedStudent.participant_id}
                </h3>
                <span className="text-xs text-slate-500">
                  ERI Score: {selectedStudent.eri.toFixed(1)}/100 | Tier: {selectedStudent.risk_band}
                </span>
              </div>
              <button
                onClick={() => setSelectedStudent(null)}
                className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs space-y-2">
              <div className="font-semibold text-slate-700 dark:text-slate-300">Narrative Summary:</div>
              <p className="text-slate-600 dark:text-slate-400">{selectedStudent.narrative}</p>
            </div>

            <div className="flex justify-between items-center pt-2">
              <span className="text-xs text-slate-400">Intervention SLA: {selectedStudent.intervention_sla}</span>
              <button
                onClick={() => setSelectedStudent(null)}
                className="px-4 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-semibold"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
