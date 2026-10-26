import {
  HealthResponse,
  MetaResponse,
  LeaderboardResponse,
  ChampionResponse,
  ReliabilityResponse,
  EDAResponse,
  GlobalShapResponse,
  UploadResponse,
  CohortPredictionResponse,
  ModelDetailsResponse,
  JobStatusResponse,
  DocItem,
} from '../types/api';

const API_BASE = '';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    let errorDetail = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>('/health'),
  getMeta: () => fetchJson<MetaResponse>('/api/meta'),
  getLeaderboard: (track?: string, category?: string, featureSet?: string) => {
    const params = new URLSearchParams();
    if (track) params.append('track', track);
    if (category && category !== 'All') params.append('category', category);
    if (featureSet && featureSet !== 'All') params.append('feature_set', featureSet);
    const qs = params.toString() ? `?${params.toString()}` : '';
    return fetchJson<LeaderboardResponse>(`/api/leaderboard${qs}`);
  },
  getChampion: () => fetchJson<ChampionResponse>('/api/champion'),
  getReliability: () => fetchJson<ReliabilityResponse>('/api/reliability'),
  getEDA: () => fetchJson<EDAResponse>('/api/eda'),
  getGlobalShap: (track: string = 'regression') =>
    fetchJson<GlobalShapResponse>(`/api/shap/global?track=${track}`),
  getModelDetails: (name: string, track?: string) => {
    const qs = track ? `?track=${track}` : '';
    return fetchJson<ModelDetailsResponse>(`/api/models/${name}${qs}`);
  },
  uploadFile: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson<UploadResponse>('/api/upload', {
      method: 'POST',
      body: formData,
    });
  },
  runPrediction: async (
    uploadId: string,
    confirmedMapping?: Record<string, string>,
    modelName?: string | null
  ): Promise<CohortPredictionResponse> => {
    return fetchJson<CohortPredictionResponse>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: uploadId,
        confirmed_mapping: confirmedMapping || null,
        model_name: modelName || null,
      }),
    });
  },
  triggerRetraining: async (uploadId: string, targetColumn: string = 'attrition_label') => {
    return fetchJson<JobStatusResponse>('/api/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: uploadId,
        target_column: targetColumn,
      }),
    });
  },
  getJobStatus: (jobId: string) => fetchJson<JobStatusResponse>(`/api/jobs/${jobId}`),
  getDocsList: () => fetchJson<DocItem[]>('/api/docs'),
  getDocContent: (filename: string) =>
    fetchJson<{ filename: string; content: string }>(`/api/docs/${filename}`),
};
