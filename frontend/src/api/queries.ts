import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from './client';
import type {
  StatusResponse,
  LeaderboardRow,
  LeaderboardResponse,
  ModelDetailResponse,
  ModelCurvesResponse,
  CompareModelsResponse,
  EngagementPatternsResponse,
  RiskDriversResponse,
  DictionaryResponse,
  RunsListResponse,
  JobStatusResponse,
  ValidateUploadResponse,
  RunScoringRequest,
  RunScoringResponse,
  PaginatedScoredRows,
  StudentDetailResponse,
  SingleScoreRequest,
  SingleScoreResponse,
} from './types';

export const queryKeys = {
  status: ['status'] as const,
  health: ['health'] as const,
  leaderboard: (track: string) => ['leaderboard', track] as const,
  categoriesBest: (track: string) => ['categories-best', track] as const,
  modelDetail: (id: string, track: string) => ['model', id, track] as const,
  modelCurves: (id: string, track: string) => ['curves', id, track] as const,
  compare: (ids: string[], track: string) => ['compare', ids.join(','), track] as const,
  patterns: ['patterns'] as const,
  drivers: ['drivers'] as const,
  dictionary: ['dictionary'] as const,
  runs: ['runs'] as const,
  jobs: ['jobs'] as const,
  scoringResults: (uploadId: string, page: number, pageSize: number, sort: string, dir: string, band?: string, flag?: string) =>
    ['scoring-results', uploadId, page, pageSize, sort, dir, band ?? '', flag ?? ''] as const,
  studentDetail: (uploadId: string, studentId: string) => ['student-detail', uploadId, studentId] as const,
};

export function useStatus() {
  return useQuery<StatusResponse>({
    queryKey: queryKeys.status,
    queryFn: () => apiFetch<StatusResponse>('/api/status'),
    staleTime: 60_000,
  });
}

export function useLeaderboard(track: 'classification' | 'regression' = 'classification') {
  return useQuery<LeaderboardResponse>({
    queryKey: queryKeys.leaderboard(track),
    queryFn: () => apiFetch<LeaderboardResponse>(`/api/leaderboard?track=${track}`),
    staleTime: 60_000,
  });
}

export function useCategoriesBest(track: 'classification' | 'regression' = 'classification') {
  return useQuery<Record<string, LeaderboardRow>>({
    queryKey: queryKeys.categoriesBest(track),
    queryFn: () => apiFetch<Record<string, LeaderboardRow>>(`/api/categories/best?track=${track}`),
    staleTime: 60_000,
  });
}

export function useModelDetail(modelId: string, track: 'classification' | 'regression' = 'classification') {
  return useQuery<ModelDetailResponse>({
    queryKey: queryKeys.modelDetail(modelId, track),
    queryFn: () => apiFetch<ModelDetailResponse>(`/api/models/${modelId}?track=${track}`),
    enabled: Boolean(modelId),
    staleTime: 60_000,
  });
}

export function useModelCurves(modelId: string, track: 'classification' | 'regression' = 'classification') {
  return useQuery<ModelCurvesResponse>({
    queryKey: queryKeys.modelCurves(modelId, track),
    queryFn: () => apiFetch<ModelCurvesResponse>(`/api/models/${modelId}/curves?track=${track}`),
    enabled: Boolean(modelId),
    staleTime: 60_000,
  });
}

export function useCompareModels(modelIds: string[], track: 'classification' | 'regression' = 'classification') {
  return useQuery<CompareModelsResponse>({
    queryKey: queryKeys.compare(modelIds, track),
    queryFn: () => apiFetch<CompareModelsResponse>(`/api/compare?ids=${modelIds.join(',')}&track=${track}`),
    enabled: modelIds.length > 0,
    staleTime: 60_000,
  });
}

export function usePatterns() {
  return useQuery<EngagementPatternsResponse>({
    queryKey: queryKeys.patterns,
    queryFn: () => apiFetch<EngagementPatternsResponse>('/api/patterns'),
    staleTime: 300_000,
  });
}

export function useDrivers() {
  return useQuery<RiskDriversResponse>({
    queryKey: queryKeys.drivers,
    queryFn: () => apiFetch<RiskDriversResponse>('/api/drivers'),
    staleTime: 300_000,
  });
}

export function useDictionary() {
  return useQuery<DictionaryResponse>({
    queryKey: queryKeys.dictionary,
    queryFn: () => apiFetch<DictionaryResponse>('/api/dictionary'),
    staleTime: 600_000,
  });
}

export function useRuns() {
  return useQuery<RunsListResponse>({
    queryKey: queryKeys.runs,
    queryFn: () => apiFetch<RunsListResponse>('/api/runs'),
    staleTime: 30_000,
  });
}

export function useJobs() {
  return useQuery<JobStatusResponse[]>({
    queryKey: queryKeys.jobs,
    queryFn: () => apiFetch<JobStatusResponse[]>('/api/jobs'),
    refetchInterval: 5000,
  });
}

// ---------------------------------------------------------------------------
// Student Scoring & Uploads (Phase 15)
// ---------------------------------------------------------------------------

export function useValidateUpload() {
  return useMutation<ValidateUploadResponse, Error, File>({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return apiFetch<ValidateUploadResponse>('/api/score/validate', {
        method: 'POST',
        body: formData,
      });
    },
  });
}

export function useRunScoring() {
  const queryClient = useQueryClient();
  return useMutation<RunScoringResponse, Error, RunScoringRequest>({
    mutationFn: async (req: RunScoringRequest) => {
      return apiFetch<RunScoringResponse>('/api/score/run', {
        method: 'POST',
        body: JSON.stringify(req),
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['scoring-results', data.upload_id] });
    },
  });
}

export interface ScoringResultsParams {
  uploadId: string;
  page?: number;
  pageSize?: number;
  sort?: string;
  direction?: 'asc' | 'desc';
  filterBand?: string;
  filterFlag?: string;
}

export function useScoringResults(params: ScoringResultsParams) {
  const {
    uploadId,
    page = 1,
    pageSize = 50,
    sort = 'mei',
    direction = 'desc',
    filterBand,
    filterFlag,
  } = params;

  let url = `/api/score/${uploadId}/results?page=${page}&page_size=${pageSize}&sort=${sort}&direction=${direction}`;
  if (filterBand && filterBand !== 'all') {
    url += `&filter_band=${encodeURIComponent(filterBand)}`;
  }
  if (filterFlag && filterFlag !== 'all') {
    url += `&filter_flag=${encodeURIComponent(filterFlag)}`;
  }

  return useQuery<PaginatedScoredRows>({
    queryKey: queryKeys.scoringResults(uploadId, page, pageSize, sort, direction, filterBand, filterFlag),
    queryFn: () => apiFetch<PaginatedScoredRows>(url),
    enabled: Boolean(uploadId),
    staleTime: 60_000,
  });
}

export function useStudentDetail(uploadId: string, studentId: string | null) {
  return useQuery<StudentDetailResponse>({
    queryKey: queryKeys.studentDetail(uploadId, studentId ?? ''),
    queryFn: () => apiFetch<StudentDetailResponse>(`/api/score/${uploadId}/student/${encodeURIComponent(studentId!)}`),
    enabled: Boolean(uploadId && studentId),
    staleTime: 120_000,
  });
}

export function useScoreSingleStudent() {
  return useMutation<SingleScoreResponse, Error, SingleScoreRequest>({
    mutationFn: async (req: SingleScoreRequest) => {
      return apiFetch<SingleScoreResponse>('/api/score/single', {
        method: 'POST',
        body: JSON.stringify(req),
      });
    },
  });
}

export function getDownloadResultsUrl(uploadId: string, format: 'xlsx' | 'csv' = 'xlsx'): string {
  const base = import.meta.env.VITE_API_BASE || '';
  return `${base}/api/score/${uploadId}/download?format=${format}`;
}

export function getDownloadProblemsUrl(uploadId: string, format: 'xlsx' | 'csv' = 'csv'): string {
  const base = import.meta.env.VITE_API_BASE || '';
  return `${base}/api/score/${uploadId}/problems?format=${format}`;
}

export function getTemplateDownloadUrl(format: 'xlsx' | 'csv' = 'xlsx'): string {
  const base = import.meta.env.VITE_API_BASE || '';
  return `${base}/api/template?format=${format}`;
}
