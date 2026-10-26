import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Card } from '../common/Card';
import { ShapDriverItem } from '../../types/api';

interface ShapWaterfallProps {
  participantId: string;
  eriScore: number;
  riskBand: string;
  mainDrivers: ShapDriverItem[];
  protectiveFactors: ShapDriverItem[];
  narrative?: string;
}

export const ShapWaterfall: React.FC<ShapWaterfallProps> = ({
  participantId,
  eriScore,
  riskBand,
  mainDrivers = [],
  protectiveFactors = [],
  narrative,
}) => {
  // Combine positive drivers and protective factors
  const combined = [
    ...mainDrivers.map((d) => ({ ...d, sign: 'positive', val: Math.abs(d.points) })),
    ...protectiveFactors.map((d) => ({ ...d, sign: 'protective', val: -Math.abs(d.points) })),
  ].sort((a, b) => b.val - a.val);

  return (
    <Card
      title={`Individual Attribution Waterfall: ${participantId}`}
      subtitle={`Calibrated ERI Score: ${eriScore.toFixed(1)}/100 | Tier: ${riskBand}`}
      aboutText="Decomposes the student's ERI score into additive point contributions. Positive points elevate attrition risk; negative points are protective."
    >
      <div className="space-y-4">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={combined}
              margin={{ top: 10, right: 30, left: 100, bottom: 5 }}
            >
              <XAxis
                type="number"
                fontSize={11}
                stroke="#64748b"
                label={{ value: 'ERI Points (+ Risk / - Protective)', position: 'insideBottom', offset: -5, fontSize: 10 }}
              />
              <YAxis
                type="category"
                dataKey="description"
                fontSize={10}
                stroke="#64748b"
                width={120}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-2.5 rounded text-xs shadow-lg">
                        <p className="font-semibold text-slate-200">{d.description}</p>
                        <p className="text-slate-400">Feature: <span className="font-mono">{d.feature}</span></p>
                        <p className={d.val >= 0 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                          Impact: {d.val > 0 ? `+${d.val}` : d.val} ERI pts
                        </p>
                        <p className="text-[10px] text-slate-400 font-mono">Raw SHAP: {d.shap_raw?.toFixed(4)}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine x={0} stroke="#475569" />
              <Bar dataKey="val" radius={[3, 3, 3, 3]}>
                {combined.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.val >= 0 ? '#f59e0b' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {narrative && (
          <div className="bg-slate-50 dark:bg-slate-800/60 p-3.5 rounded-lg border border-slate-200 dark:border-slate-700/60 text-xs">
            <span className="font-semibold text-slate-700 dark:text-slate-300">Deterministic Narrative: </span>
            <span className="text-slate-600 dark:text-slate-400">{narrative}</span>
          </div>
        )}
      </div>
    </Card>
  );
};

export const ERIGauge: React.FC<{ score: number; band: string }> = ({ score, band }) => {
  const getBandColor = (b: string) => {
    switch (b?.toLowerCase()) {
      case 'critical':
        return 'text-rose-500 bg-rose-500/10 border-rose-500/30';
      case 'high':
        return 'text-amber-500 bg-amber-500/10 border-amber-500/30';
      case 'moderate':
        return 'text-blue-500 bg-blue-500/10 border-blue-500/30';
      default:
        return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30';
    }
  };

  return (
    <div className="text-center p-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
      <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400 font-semibold mb-1">
        Environmental Risk Index (ERI)
      </div>
      <div className="text-4xl font-black text-slate-900 dark:text-white font-mono tracking-tight my-2">
        {score.toFixed(1)}
        <span className="text-xs font-normal text-slate-400 ml-1">/ 100</span>
      </div>
      <div
        className={`inline-block px-3 py-1 rounded-full text-xs font-bold border capitalize ${getBandColor(
          band
        )}`}
      >
        {band} Risk Tier
      </div>
    </div>
  );
};
