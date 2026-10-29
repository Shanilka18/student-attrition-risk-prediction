import React, { useState } from 'react';
import { Card } from '../components/common/Card';
import { apiService } from '../services/api';
import { FusionScoreResponse } from '../types/api';
import { Share2, Play, CheckCircle2, Clock, Code2, AlertCircle } from 'lucide-react';

export const FusionIntegrationPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<FusionScoreResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sampleIds = ['P-00002', 'P-00003', 'P-00004'];

  const handleTestFusionApi = async () => {
    try {
      setLoading(true);
      setError(null);
      const start = performance.now();
      const res = await apiService.scoreFusionPayload({ participant_ids: sampleIds });
      const elapsed = performance.now() - start;
      setLatencyMs(Number(elapsed.toFixed(1)));
      setResponse(res);
    } catch (err: any) {
      setError(err.message || 'Fusion API call failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <div className="flex items-center gap-2">
          <Share2 className="w-6 h-6 text-indigo-500" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">
            Group Fusion Framework REST Contract (KND-CDAP-IT-05)
          </h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Decoupled score-level API integrating the Environmental & Financial Index with the other academic and wellbeing components.
        </p>
      </div>

      {/* Contract Specification Card */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card
          title="Score-Level Integration Schema"
          subtitle="Strictly typed payload format emitted to central group fusion"
        >
          <div className="space-y-3 text-xs">
            <p className="text-slate-600 dark:text-slate-300">
              The central fusion architecture integrates component probabilities into an overall institutional attrition risk score. To maintain clean modular boundaries, our component exposes calibrated ERI on a 0–100 scale:
            </p>
            <pre className="p-3 bg-slate-900 text-slate-100 rounded-lg font-mono text-[11px] overflow-x-auto border border-slate-800">
{`{
  "total_scored": 3,
  "explanation_type": "model_explanation",
  "scores": [
    {
      "participant_id": "P-00002",
      "eri": 12.38,
      "band": "Low",
      "model_version": "1.0.0",
      "top_drivers": ["work_hours_per_week", "internet_reliability"]
    }
  ]
}`}
            </pre>
            <div className="p-2.5 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/40 rounded-lg text-blue-900 dark:text-blue-300 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-blue-600 flex-shrink-0" />
              <span>
                Immutable tag: <code>explanation_type: "model_explanation"</code> prevents semantic confusion with causal estimates.
              </span>
            </div>
          </div>
        </Card>

        {/* Live Interactive Tester */}
        <Card
          title="Interactive Live Endpoint Test (POST /fusion/score)"
          subtitle="Test payload delivery directly against the running FastAPI microservice"
        >
          <div className="space-y-4">
            <div className="text-xs text-slate-600 dark:text-slate-300">
              Test Request Payload:
              <span className="font-mono bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded ml-1">
                {JSON.stringify({ participant_ids: sampleIds })}
              </span>
            </div>

            <button
              onClick={handleTestFusionApi}
              disabled={loading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold shadow-sm flex items-center gap-2 transition-colors cursor-pointer"
            >
              {loading ? <Clock className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Send Test Request
            </button>

            {latencyMs !== null && (
              <div className="text-xs text-emerald-600 dark:text-emerald-400 font-mono flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> 200 OK — Round-trip latency: {latencyMs} ms
              </div>
            )}

            {error && (
              <div className="p-3 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/50 rounded-lg text-xs text-rose-800 dark:text-rose-300 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {response && (
              <div className="space-y-1">
                <span className="text-[11px] font-semibold text-slate-400">Response Payload:</span>
                <pre className="p-3 bg-slate-900 text-emerald-400 rounded-lg font-mono text-[11px] max-h-56 overflow-y-auto border border-slate-800">
                  {JSON.stringify(response, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
