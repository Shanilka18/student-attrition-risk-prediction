import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { Card } from '../common/Card';

interface DualCurveProps {
  modelName: string;
  epochs: (number | string)[];
  metrics: Record<string, any>;
  bestEpoch?: number;
  track?: string;
}

export const LossAccuracyDualCurve: React.FC<DualCurveProps> = ({
  modelName,
  epochs,
  metrics,
  bestEpoch,
  track = 'C',
}) => {
  // Transform metric series into an array of row objects
  const chartData = epochs.map((ep, idx) => {
    const row: Record<string, any> = { epoch: ep };
    for (const [key, val] of Object.entries(metrics)) {
      if (Array.isArray(val) && val.length > idx) {
        row[key] = typeof val[idx] === 'number' ? Number(val[idx].toFixed(4)) : val[idx];
      }
    }
    return row;
  });

  const optimalEpoch = bestEpoch || (epochs.length > 0 ? epochs[Math.floor(epochs.length / 2)] : 1);
  const isRegression = track === 'R';

  // Helper custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 text-white p-2.5 rounded-lg text-xs shadow-xl border border-slate-700">
          <p className="font-semibold text-slate-300 mb-1">Epoch / Iteration: {label}</p>
          {payload.map((p: any, i: number) => (
            <p key={i} style={{ color: p.color }} className="flex justify-between gap-4">
              <span>{p.name}:</span>
              <span className="font-mono font-medium">{p.value}</span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      {/* 1. Loss Curve */}
      <Card
        title="Training vs Validation Loss"
        subtitle={`Optimization convergence with early-stopping marker (optimal epoch: ${optimalEpoch})`}
        aboutText="Traces the objective loss reduction across training iterations. A divergence between train and validation indicates overfitting."
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              {optimalEpoch && (
                <ReferenceLine
                  x={optimalEpoch}
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  label={{
                    value: `Best: ${optimalEpoch}`,
                    position: 'top',
                    fill: '#ef4444',
                    fontSize: 10,
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey="train_loss"
                name="Train Loss"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={chartData.length < 25}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="Validation Loss"
                stroke="#f97316"
                strokeWidth={2}
                dot={chartData.length < 25}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* 2. Accuracy or R2 Curve */}
      <Card
        title={isRegression ? 'Training vs Validation R²' : 'Training vs Validation Accuracy'}
        subtitle={`Evaluation metric trajectory across iterations (optimal epoch: ${optimalEpoch})`}
        aboutText={
          isRegression
            ? 'Monitors continuous variance explanation (R²) across epochs.'
            : 'Traces classification sample accuracy. Plateau indicates model convergence.'
        }
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              {optimalEpoch && (
                <ReferenceLine
                  x={optimalEpoch}
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  label={{
                    value: `Best: ${optimalEpoch}`,
                    position: 'top',
                    fill: '#ef4444',
                    fontSize: 10,
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey={isRegression ? 'train_r2' : 'train_acc'}
                name={isRegression ? 'Train R²' : 'Train Accuracy'}
                stroke="#10b981"
                strokeWidth={2}
                dot={chartData.length < 25}
              />
              <Line
                type="monotone"
                dataKey={isRegression ? 'val_r2' : 'val_acc'}
                name={isRegression ? 'Val R²' : 'Validation Accuracy'}
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={chartData.length < 25}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};
