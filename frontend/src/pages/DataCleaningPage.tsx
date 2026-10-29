import React from 'react';
import { Card } from '../components/common/Card';
import { CheckCircle2, ShieldCheck, Database, FileSpreadsheet, AlertCircle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

export const DataCleaningPage: React.FC = () => {
  const missingnessData = [
    { feature: 'commute_cost_lkr_week', pct: 6.2 },
    { feature: 'work_hours_per_week', pct: 4.8 },
    { feature: 'internet_affordability', pct: 3.5 },
    { feature: 'quiet_study_space', pct: 2.1 },
    { feature: 'financial_stress', pct: 1.4 },
    { feature: 'household_income_bracket', pct: 0.9 },
  ];

  const classData = [
    { label: 'Retained Students', count: 1649, pct: '82.45%', color: '#10b981' },
    { label: 'Institutional Attrition', count: 351, pct: '17.55%', color: '#f43f5e' },
  ];

  const cleaningLog = [
    { rule: 'PII Scrubbing', action: 'Names, NIC, Phone, Email columns removed prior to dataframe parsing', status: 'Passed' },
    { rule: 'Commute Minutes', action: 'Clipping values outside 5-300 min range; imputed with district median', status: 'Passed' },
    { rule: 'Income Brackets', action: 'Standardized to ordinal bands; mapped unknown categories to Mode', status: 'Passed' },
    { rule: 'Part-Time Work Hours', action: 'Zero-filled when part_time_employment == 0; non-negative clipped', status: 'Passed' },
    { rule: 'GPA Bands', action: 'Categorized into 5 ordinal tiers (Below 2.0 to 3.7-4.0)', status: 'Passed' },
  ];

  const leakageRules = [
    { title: 'In-Fold Preprocessing', desc: 'All transformations (scaling, encoding, imputation) are fitted strictly on training folds.' },
    { title: 'Outcome Separation', desc: 'Outcome, consent, attention-check, identifier, and timestamp fields are excluded from model features.' },
    { title: 'Group-Aware Stratification', desc: 'Folds maintain exact class balance (17.55%) without student cross-contamination.' },
    { title: 'Sealed Held-Out Test Set', desc: 'Test partition (286 students) remained frozen and was evaluated exactly once for finalists.' },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-white">
          Data Architecture, Cleaning & Partition Audits
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Detailed inspection of raw survey ingestion, missingness imputation, class balance, and strict leakage guardrails.
        </p>
      </div>

      {/* Partition Splits */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-t-4 border-t-blue-500">
          <div className="flex justify-between items-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Training Set</span>
            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300">
              71.4%
            </span>
          </div>
          <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-2">
            1,428 <span className="text-xs font-normal text-slate-400">Students</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Stratified 5-fold cross-validation arena</div>
        </Card>

        <Card className="border-t-4 border-t-amber-500">
          <div className="flex justify-between items-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Validation Set</span>
            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300">
              14.3%
            </span>
          </div>
          <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-2">
            286 <span className="text-xs font-normal text-slate-400">Students</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Tuning, calibration fitting, and SHAP stability</div>
        </Card>

        <Card className="border-t-4 border-t-rose-500">
          <div className="flex justify-between items-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Held-Out Test Set</span>
            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-300">
              14.3%
            </span>
          </div>
          <div className="text-2xl font-black text-slate-900 dark:text-white font-mono mt-2">
            286 <span className="text-xs font-normal text-slate-400">Students</span>
          </div>
          <div className="text-[11px] text-rose-600 dark:text-rose-400 mt-1 font-medium">Sealed until model freeze (ROC 0.738)</div>
        </Card>
      </div>

      {/* Visualizers: Missingness & Class Balance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card
          title="Missingness Distribution Across Key Covariates"
          subtitle="Percentage of missing values handled via in-fold median and mode imputation"
          aboutText="Assesses non-response bias. Low missingness across covariates preserves statistical power."
        >
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={missingnessData} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
                <XAxis type="number" unit="%" stroke="#64748b" fontSize={11} domain={[0, 10]} />
                <YAxis type="category" dataKey="feature" stroke="#64748b" fontSize={10} width={130} />
                <Tooltip
                  formatter={(val: any) => [`${val}% Missing`, 'Missingness']}
                  contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '11px' }}
                />
                <Bar dataKey="pct" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card
          title="Cohort Class Imbalance Audit"
          subtitle="Prevalence of primary target outcome (institutional attrition vs retained)"
          aboutText="Shows severe class imbalance (17.55% events). Headlining metrics prioritize PR-AUC and Brier over naive accuracy."
        >
          <div className="h-60 flex flex-col justify-center space-y-4 px-4">
            {classData.map((item, i) => (
              <div key={i} className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-700 dark:text-slate-300 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                    {item.label}
                  </span>
                  <span className="font-mono text-slate-900 dark:text-white font-bold">
                    {item.count} ({item.pct})
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: item.pct,
                      backgroundColor: item.color,
                    }}
                  />
                </div>
              </div>
            ))}
            <div className="p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 rounded-lg text-xs text-amber-800 dark:text-amber-300">
              <span className="font-semibold">Imbalance Ratio: ~4.7 : 1.</span> Class-weighted loss functions and in-fold SMOTE resampling were applied to protect minority class recall.
            </div>
          </div>
        </Card>
      </div>

      {/* Cleaning Log Table & Leakage Checklist */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card title="Data Cleaning & Quality Pipeline Log" subtitle="Rule-based transformation sequence">
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 font-semibold text-slate-500">
                  <th className="py-2">Quality Rule</th>
                  <th className="py-2">Operation Performed</th>
                  <th className="py-2 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                {cleaningLog.map((log, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 font-semibold text-slate-800 dark:text-slate-200">{log.rule}</td>
                    <td className="py-2.5 text-slate-600 dark:text-slate-400">{log.action}</td>
                    <td className="py-2.5 text-right font-medium text-emerald-600 dark:text-emerald-400 flex items-center justify-end gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      {log.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Data Leakage Prevention Guardrails" subtitle="Architectural protocols enforced across models">
          <div className="space-y-3">
            {leakageRules.map((r, idx) => (
              <div key={idx} className="flex gap-3 p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-700/60">
                <ShieldCheck className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="text-xs font-bold text-slate-800 dark:text-slate-200">{r.title}</div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{r.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};
