import React from 'react';
import {
  Clock,
  TrendingDown,
  ShieldAlert,
  Zap,
  Target,
  CheckCircle,
  HelpCircle,
  Calendar,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import { MetricCard } from '../components/MetricCard';

export const EarlyWarningPage: React.FC = () => {
  // Lead time distribution data: Semesters before actual dropout when model first crosses operational threshold
  const leadTimeData = [
    { leadSemesters: '1 Semester (6 mo)', students: 38, percentage: 18.5 },
    { leadSemesters: '2 Semesters (12 mo)', students: 86, percentage: 41.7 },
    { leadSemesters: '3 Semesters (18 mo)', students: 54, percentage: 26.2 },
    { leadSemesters: '4+ Semesters (24+ mo)', students: 28, percentage: 13.6 },
  ];

  // Risk index distribution data (Attrited vs Retained separation)
  const distributionData = [
    { riskBin: '0–5 (Low)', retained: 78.4, attrited: 1.2 },
    { riskBin: '5–12 (Moderate)', retained: 14.8, attrited: 14.6 },
    { riskBin: '12–22 (High)', retained: 5.2, attrited: 42.1 },
    { riskBin: '22–100 (Critical)', retained: 1.6, attrited: 42.1 },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* 1. Header */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <Clock className="w-6 h-6 text-indigo-400" />
          Early-Warning Lead Time & Prognostic Window Analysis
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Evaluating the operational intervention runway provided to academic advisors prior to student academic probation or withdrawal.
        </p>
      </div>

      {/* 2. Headline Lead Time Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Mean Warning Runway"
          value="2.0 – 2.5 Sems"
          subValue="~12 to 15 months of lead time"
          highlight
          tooltip="Average number of semesters before academic termination when the student is first flagged."
        />
        <MetricCard
          label="Pre-Probation Catch Rate"
          value="82.4%"
          subValue="Identified before GPA < 2.0"
          tooltip="Proportion of attriting students flagged while their cross-sectional GPA was still above the 2.0 probation threshold."
        />
        <MetricCard
          label="Recall @ Top 10%"
          value="64.2%"
          subValue="Advising capacity capture"
          tooltip="Proportion of true future attriters captured in the highest 10% risk decile."
        />
        <MetricCard
          label="Recall @ Top 20%"
          value="81.5%"
          subValue="High triage safety"
          tooltip="Proportion of true future attriters captured in the top 20% risk cohort."
        />
      </div>

      {/* 3. Lead Time Histogram & Runway */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lead Time Distribution Bar Chart */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Calendar className="w-4 h-4 text-indigo-400" />
              Lead Time Distribution (Semesters Before Termination)
            </h3>
            <span className="text-xs text-slate-400 font-mono">Test Cohort Attriters</span>
          </div>
          <p className="text-xs text-slate-400">
            Over <strong>81.5% of at-risk students</strong> are flagged at least 2 semesters (1 year) prior to formal withdrawal.
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer>
              <BarChart data={leadTimeData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="leadSemesters" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="percentage" name="% of Attrited Students" fill="#818cf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Risk Index Separation Density */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Target className="w-4 h-4 text-emerald-400" />
              Risk Index Distribution (Separation Quality)
            </h3>
            <span className="text-xs text-slate-400 font-mono">Retained vs Attrited</span>
          </div>
          <p className="text-xs text-slate-400">
            Density distribution across clinical tiers. Demonstrates clear separation between retained persisters and future attriters.
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer>
              <BarChart data={distributionData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="riskBin" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="retained" name="Retained Cohort (%)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="attrited" name="Attrited Cohort (%)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* 4. Comparative Runway Table */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          Proactive vs Reactive Early-Warning System Comparison
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider bg-slate-950/60">
                <th className="py-2.5 px-3">EWS Mechanism</th>
                <th className="py-2.5 px-3">Trigger Condition</th>
                <th className="py-2.5 px-3">Average Lead Time</th>
                <th className="py-2.5 px-3">Intervention Feasibility</th>
                <th className="py-2.5 px-3">Operational Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr className="hover:bg-slate-800/20">
                <td className="py-3 px-3 font-semibold text-rose-400">
                  Conventional Static EWS
                </td>
                <td className="py-3 px-3 font-mono">Last GPA &lt; 2.00</td>
                <td className="py-3 px-3 font-mono text-rose-300">0.5 Semesters (3–6 wks)</td>
                <td className="py-3 px-3 text-slate-400">
                  Very low. By the time GPA falls below 2.0, student is already on probation with mounting credit backlogs.
                </td>
                <td className="py-3 px-3 text-rose-300">Reactive & Punitive</td>
              </tr>
              <tr className="hover:bg-slate-800/20 bg-indigo-950/20">
                <td className="py-3 px-3 font-semibold text-indigo-300">
                  Academic Intelligence Framework
                </td>
                <td className="py-3 px-3 font-mono">Longitudinal Trend + Sequence (ARI &ge; 5.0)</td>
                <td className="py-3 px-3 font-mono font-bold text-emerald-400">
                  2.0 – 2.5 Semesters (12–15 mo)
                </td>
                <td className="py-3 px-3 text-slate-300">
                  High. Detects attendance decay and assignment delays well before cumulative GPA collapses.
                </td>
                <td className="py-3 px-3 text-emerald-400 font-semibold">Proactive & Supportive</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
