import {
  HealthResponse,
  MetaResponse,
  LeaderboardResponse,
  ModelHistoryResponse,
  ModelMetricsResponse,
  AllSplitsResponse,
  ShapGlobalResponse,
  ShapStudentResponse,
  ShapStabilityResponse,
  CausalSummaryResponse,
  CausalDiagnosticsResponse,
  CausalHeterogeneityResponse,
  DagResponse,
  UploadPredictResponse,
  FusionScoreResponse,
} from '../types/api';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, options);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error ${response.status}: ${errorText || response.statusText}`);
  }
  return response.json();
}

export const apiService = {
  // System
  getHealth: () => fetchJson<HealthResponse>('/health'),
  getMeta: () => fetchJson<MetaResponse>('/meta'),

  // Models & Leaderboard
  getLeaderboard: (track: 'C' | 'R' = 'C') =>
    fetchJson<LeaderboardResponse>(`/leaderboard?track=${track}`),
  getModelHistory: (modelName: string) =>
    fetchJson<ModelHistoryResponse>(`/models/${modelName}/history`),
  getModelMetrics: (modelName: string) =>
    fetchJson<ModelMetricsResponse>(`/models/${modelName}/metrics`),
  getAllSplitsMetrics: () => fetchJson<AllSplitsResponse>('/metrics/all-splits'),

  // Explainability (SHAP)
  getShapGlobal: () => fetchJson<ShapGlobalResponse>('/shap/global'),
  getShapStudent: (studentId: string) =>
    fetchJson<ShapStudentResponse>(`/shap/student/${studentId}`),
  getShapStability: () => fetchJson<ShapStabilityResponse>('/shap/stability'),

  // Causal Inference
  getCausalSummary: () => fetchJson<CausalSummaryResponse>('/causal/summary'),
  getCausalDiagnostics: () => fetchJson<CausalDiagnosticsResponse>('/causal/diagnostics'),
  getCausalHeterogeneity: () => fetchJson<CausalHeterogeneityResponse>('/causal/heterogeneity'),
  getDagSpecification: () => fetchJson<DagResponse>('/dag'),

  // Prediction & Batch Upload
  uploadAndPredict: async (file: File): Promise<UploadPredictResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/predict/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Upload error' }));
      throw new Error(err.detail || 'Failed to process file');
    }
    return response.json();
  },

  // Fusion Framework Integration
  scoreFusionPayload: (payload: { participant_ids?: string[]; students?: any[] }) =>
    fetchJson<FusionScoreResponse>('/fusion/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};
