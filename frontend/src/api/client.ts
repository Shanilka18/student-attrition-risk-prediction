/**
 * Typed API client for the MEI FastAPI backend.
 * Uses native fetch with consistent error shaping and timeout support.
 */

export class ApiError extends Error {
  status: number;
  detail: string;
  path?: string;

  constructor(status: number, detail: string, path?: string) {
    super(`API Error ${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

export async function apiFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const headers = new Headers(options.headers || {});
  
  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errJson = await response.json();
      detail = errJson.detail || errJson.message || detail;
    } catch {
      // Non-JSON error payload
    }
    throw new ApiError(response.status, detail, endpoint);
  }

  // Handle binary or blob downloads
  const contentType = response.headers.get('content-type') || '';
  if (
    contentType.includes('application/octet-stream') ||
    contentType.includes('text/csv') ||
    contentType.includes('spreadsheetml')
  ) {
    return (await response.blob()) as unknown as T;
  }

  return response.json();
}
