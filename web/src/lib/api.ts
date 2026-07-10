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

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  sessions: () => getJson<SessionInfo[]>('/api/sessions'),
  session: (id: string) => getJson<SessionMeta>(`/api/sessions/${id}`),
  breathing: (id: string) => getJson<{ points: BreathingPoint[] }>(`/api/sessions/${id}/breathing`),
  psd: (id: string, subcarrier: number) =>
    getJson<Psd>(`/api/sessions/${id}/psd?subcarrier=${subcarrier}`),
}
