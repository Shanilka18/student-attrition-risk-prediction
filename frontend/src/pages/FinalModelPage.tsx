import React from 'react';
import { Card } from '../components/common/Card';
import { ShieldAlert, CheckCircle2, Lock, Scale, AlertCircle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export const FinalModelPage: React.FC = () => {
  const eriDistribution = [
    { band: '0 - 15', count: 120, label: 'Minimal Risk' },
    { band: '15 - 25', count: 85, label: 'Low Risk' },
    { band: '25 - 40', count: 45, label: 'Moderate Risk' },
    { band: '40 - 60', count: 24, label: 'Elevated Risk' },
    { band: '60 - 80', count: 10, label: 'High Risk' },
    { band: '80 - 100', count: 2, label: 'Critical Risk' },
  ];

  const riskBands = [
    { band: 'Low', range: '0.0 - 24.9', sla: 'Within 30 days', action: 'Routine academic advising & environmental check-in' },
    { band: 'Moderate', range: '25.0 - 49.9', sla: 'Within 14 days', action: 'Targeted bursary review & digital resource subsidy' },
    { band: 'High', range: '50.0 - 74.9', sla: 'Within 5 days', action: 'Direct counselor intervention & emergency financial grant' },
    { band: 'Critical', range: '75.0 - 100.0', sla: 'Within 48 hours', action: 'Immediate Dean triage & comprehensive retention case review' },
  ];

  const fairnessAudit = [
    { subgroup: 'Gender: Female', n: 135, roc_auc: 0.742, pr_auc: 0.384, brier: 0.205, fair: 'Passed' },
    { subgroup: 'Gender: Male', n: 151, roc_auc: 0.735, pr_auc: 0.378, brier: 0.210, fair: 'Passed' },
    { subgroup: 'Origin: Rural', n: 110, roc_auc: 0.748, pr_auc: 0.391, brier: 0.202, fair: 'Passed' },
    { subgroup: 'Origin: Urban', n: 176, roc_auc: 0.732, pr_auc: 0.374, brier: 0.212, fair: 'Passed' },
    { subgroup: 'Faculty: Computing', n: 142, roc_auc: 0.739, pr_auc: 0.381, brier: 0.207, fair: 'Passed' },
    { subgroup: 'Faculty: Engineering', n: 144, roc_auc: 0.737, pr_auc: 0.379, brier: 0.209, fair: 'Passed' },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-white">
          Production Model: Calibrated Environmental Risk Index (ERI)
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Model card specification, probability calibration efficacy, operational triage tiers, and demographic fairness audits.
        </p>
      </div>

      {/* Model Card Specification */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-l-4 border-l-blue-600">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Architecture</div>
          <div className="text-xl font-bold text-slate-900 dark:text-white font-mono mt-1">Logistic Regression (L2)</div>
          <div className="text-xs text-slate-500 mt-1">Regularization C = 0.5, Liblinear solver</div>
        </Card>

        <Card className="border-l-4 border-l-purple-600">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Calibrator Pipeline</div>
          <div className="text-xl font-bold text-slate-900 dark:text-white font-mono mt-1">Isotonic Regression</div>
          <div className="text-xs text-purple-600 dark:text-purple-400 font-medium mt-1">ECE dropped from 0.089 to 0.041</div>
        </Card>

        <Card className="border-l-4 border-l-emerald-600">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Held-Out Test Set Report</div>
          <div className="text-xl font-bold text-slate-900 dark:text-white font-mono mt-1 flex items-center gap-2">
            ROC 0.738 <Lock className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-xs text-slate-500 mt-1">Evaluated once; zero leakage observed</div>
        </Card>
      </div>

      {/* ERI Distribution & Calibration Impact */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card
          title="Environmental Risk Index (ERI) Score Distribution"
          subtitle="Continuous score mapped from 0 to 100 on the validation cohort (N = 286)"
          aboutText="Displays the spread of calibrated attrition risk. Skewed toward lower tiers reflecting realistic baseline retention rates."
        >
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eriDistribution} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                <XAxis dataKey="band" stroke="#64748b" fontSize={11} label={{ value: 'ERI Score Band (0-100)', position: 'insideBottom', offset: -5, fontSize: 10 }} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  formatter={(val: any, name: any, item: any) => [`${val} Students (${item.payload.label})`, 'Students']}
                  contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '11px' }}
                />
                <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Operational Triage SLA Table */}
        <Card
          title="Operational Risk Bands & Action SLAs"
          subtitle="Standard operating procedures mapped to ERI risk thresholds"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 font-semibold text-slate-500">
                  <th className="py-2 px-2">Risk Band</th>
                  <th className="py-2 px-2">ERI Range</th>
                  <th className="py-2 px-2">Target SLA</th>
                  <th className="py-2 px-2">Prescribed Intervention</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {riskBands.map((band, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 px-2 font-bold">
                      <span
                        className={`inline-block w-2 h-2 rounded-full mr-2 ${
                          band.band === 'Critical'
                            ? 'bg-rose-500'
                            : band.band === 'High'
                            ? 'bg-amber-500'
                            : band.band === 'Moderate'
                            ? 'bg-blue-500'
                            : 'bg-emerald-500'
                        }`}
                      />
                      {band.band}
                    </td>
                    <td className="py-2.5 px-2 font-mono">{band.range}</td>
                    <td className="py-2.5 px-2 font-medium text-slate-700 dark:text-slate-300">{band.sla}</td>
                    <td className="py-2.5 px-2 text-slate-600 dark:text-slate-400">{band.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Subgroup Demographic Fairness Audit */}
      <Card
        title="Subgroup Fairness & Demographic Parity Audit"
        subtitle="Evaluating potential performance disparities across protected and sensitive student demographics"
        aboutText="Audits the model across gender, origin, and faculty cohorts. Equalized opportunity metrics show no disparate harm."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-800/80 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
                <th className="py-2.5 px-3">Demographic Subgroup</th>
                <th className="py-2.5 px-3">Subgroup Size (N)</th>
                <th className="py-2.5 px-3">ROC-AUC</th>
                <th className="py-2.5 px-3">PR-AUC</th>
                <th className="py-2.5 px-3">Brier Score</th>
                <th className="py-2.5 px-3 text-right">Fairness Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {fairnessAudit.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="py-2 px-3 font-semibold text-slate-800 dark:text-slate-200">{row.subgroup}</td>
                  <td className="py-2 px-3 font-mono">{row.n}</td>
                  <td className="py-2 px-3 font-mono">{row.roc_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono">{row.pr_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono">{row.brier.toFixed(3)}</td>
                  <td className="py-2 px-3 text-right font-medium text-emerald-600 dark:text-emerald-400 flex items-center justify-end gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    {row.fair}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};
