import React from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { Card } from '../common/Card';

interface LovePlotProps {
  balanceData?: Record<string, any>;
  exposureName?: string;
}

export const LovePlot: React.FC<LovePlotProps> = ({
  balanceData = {},
  exposureName = 'financial_hardship',
}) => {
  const expData = balanceData[exposureName] || {};
  const covariates = expData.covariates || [];
  const rawSMD = expData.smd_raw || [];
  const weightedSMD = expData.smd_weighted || [];

  const chartData = covariates.map((cov: string, i: number) => ({
    covariate: cov.replace(/_/g, ' '),
    yIndex: i,
    raw: rawSMD[i] !== undefined ? Math.abs(rawSMD[i]) : 0,
    weighted: weightedSMD[i] !== undefined ? Math.abs(weightedSMD[i]) : 0,
  }));

  return (
    <Card
      title={`Covariate Balance Love Plot: ${exposureName.replace(/_/g, ' ').toUpperCase()}`}
      subtitle="Standardized Mean Difference (SMD) before vs after IPW weighting"
      aboutText="Assesses whether propensity weighting eliminated baseline confounding imbalances. SMD below 0.10 indicates adequate covariate balance."
    >
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 30, left: 100, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
            <XAxis
              type="number"
              dataKey="smd"
              domain={[0, 0.5]}
              stroke="#64748b"
              fontSize={11}
              label={{ value: 'Absolute Standardized Mean Difference (|SMD|)', position: 'insideBottom', offset: -5, fontSize: 10 }}
            />
            <YAxis
              type="category"
              dataKey="covariate"
              stroke="#64748b"
              fontSize={10}
              width={120}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="bg-slate-900 text-white p-2 rounded text-xs">
                      <p className="font-semibold text-slate-200">{d.covariate}</p>
                      <p className="text-rose-400">Unadjusted |SMD|: {d.raw?.toFixed(3)}</p>
                      <p className="text-emerald-400 font-bold">Weighted |SMD|: {d.weighted?.toFixed(3)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <ReferenceLine
              x={0.1}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label={{ value: 'Balance Threshold (0.10)', position: 'top', fill: '#ef4444', fontSize: 10 }}
            />
            <Scatter
              name="Unadjusted Cohort"
              data={chartData.map((d: any) => ({ ...d, smd: d.raw }))}
              fill="#f43f5e"
              shape="circle"
            />
            <Scatter
              name="IPW Weighted Cohort"
              data={chartData.map((d: any) => ({ ...d, smd: d.weighted }))}
              fill="#10b981"
              shape="diamond"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};
