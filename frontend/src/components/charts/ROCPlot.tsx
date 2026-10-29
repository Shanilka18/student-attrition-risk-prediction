import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { Card } from '../common/Card';

interface ROCPlotProps {
  fpr?: number[];
  tpr?: number[];
  auc?: number;
}

export const ROCPlot: React.FC<ROCPlotProps> = ({ fpr = [], tpr = [], auc }) => {
  const data = fpr.map((f, i) => ({
    fpr: Number(f.toFixed(3)),
    tpr: Number((tpr[i] || 0).toFixed(3)),
  }));

  return (
    <Card
      title={`ROC Curve ${auc ? `(AUC = ${auc.toFixed(3)})` : ''}`}
      subtitle="True Positive Rate vs False Positive Rate trade-off"
      aboutText="Measures the classifier's ability to discriminate between retained students and attrition events across classification thresholds."
    >
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
            <XAxis
              dataKey="fpr"
              stroke="#64748b"
              fontSize={11}
              domain={[0, 1]}
              label={{ value: 'False Positive Rate (1 - Specificity)', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#64748b' }}
            />
            <YAxis
              stroke="#64748b"
              fontSize={11}
              domain={[0, 1]}
              label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#64748b' }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-slate-900 text-white p-2 rounded text-xs">
                      <p>FPR: {payload[0]?.payload.fpr}</p>
                      <p className="text-blue-400 font-semibold">TPR: {payload[0]?.payload.tpr}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine x={0} y={0} stroke="#94a3b8" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="tpr"
              stroke="#2563eb"
              strokeWidth={2.5}
              dot={false}
              name="ROC Curve"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export const CalibrationPlot: React.FC<{ probPred?: number[]; probTrue?: number[] }> = ({
  probPred = [],
  probTrue = [],
}) => {
  const data = probPred.map((p, i) => ({
    pred: Number(p.toFixed(3)),
    true: Number((probTrue[i] || 0).toFixed(3)),
  }));

  return (
    <Card
      title="Reliability Calibration Diagram"
      subtitle="Predicted probability vs Empirical observed attrition rate"
      aboutText="A perfectly calibrated risk score aligns exactly along the 45-degree diagonal: when the model predicts a 30% risk, exactly 30% of those students drop out."
    >
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
            <XAxis
              dataKey="pred"
              stroke="#64748b"
              fontSize={11}
              domain={[0, 1]}
              label={{ value: 'Mean Predicted Risk', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#64748b' }}
            />
            <YAxis
              stroke="#64748b"
              fontSize={11}
              domain={[0, 1]}
              label={{ value: 'Observed Fraction', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#64748b' }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-slate-900 text-white p-2 rounded text-xs">
                      <p>Predicted: {payload[0]?.payload.pred}</p>
                      <p className="text-emerald-400 font-semibold">Observed: {payload[0]?.payload.true}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Line
              type="monotone"
              dataKey="true"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 4, fill: '#10b981' }}
              name="Model Reliability"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};
