export interface SessionInfo {
  id: string
  dataset: string
  label: string | null
  fs: number
  duration_s: number
  n_subcarriers: number
  started_at: string | null
  room: string | null
}

export interface SessionMeta {
  id: string
  fs: number
  duration_s: number
  n_subcarriers: number
  meta: Record<string, unknown>
}

export interface BreathingPoint {
  t: number
  bpm: number
  confidence: number
}

export interface Psd {
  freqs: number[]
  psd: number[]
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  sessions: (signal?: AbortSignal) => getJson<SessionInfo[]>('/api/sessions', signal),
  session: (id: string, signal?: AbortSignal) =>
    getJson<SessionMeta>(`/api/sessions/${id}`, signal),
  breathing: (id: string, signal?: AbortSignal) =>
    getJson<{ points: BreathingPoint[] }>(`/api/sessions/${id}/breathing`, signal),
  psd: (id: string, subcarrier: number, signal?: AbortSignal) =>
    getJson<Psd>(`/api/sessions/${id}/psd?subcarrier=${subcarrier}`, signal),
}
