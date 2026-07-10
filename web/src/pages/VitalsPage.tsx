import { useEffect, useState } from 'react'
import { api, getJson, type SessionInfo } from '../lib/api'
import { LineChart } from '../components/LineChart'

interface VitalsPoint {
  t: number
  bpm: number
  confidence: number
}

interface ActivityPoint {
  t: number
  state: string
  presence: number
  motion: number
}

interface VitalsResponse {
  breathing: VitalsPoint[]
  heart: VitalsPoint[]
  activity: ActivityPoint[]
  summary: {
    duration_s: number
    breathing_median_bpm?: number
    breathing_confidence?: number
    heart_median_bpm?: number
    heart_confidence?: number
    presence_fraction?: number
    motion_fraction?: number
  }
}

export function VitalsPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [sessionId, setSessionId] = useState('')
  const [vitals, setVitals] = useState<VitalsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.sessions().then((list) => {
      setSessions(list)
      if (list.length > 0) setSessionId(list[0].id)
    }).catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (!sessionId) return
    setVitals(null)
    setError(null)
    const ctrl = new AbortController()
    getJson<VitalsResponse>(`/api/sessions/${sessionId}/vitals`, ctrl.signal)
      .then(setVitals)
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e))
      })
    return () => ctrl.abort() // a slow response for a deselected session must not land
  }, [sessionId])

  if (sessions.length === 0) return <div className="empty">{error ?? 'No sessions recorded yet.'}</div>

  const s = vitals?.summary

  return (
    <div>
      <div className="page-head">
        <h1>Vitals</h1>
        <label className="stat-row">
          session{' '}
          <select value={sessionId} onChange={(e) => setSessionId(e.target.value)}>
            {sessions.map((x) => <option key={x.id} value={x.id}>{x.id}</option>)}
          </select>
        </label>
      </div>

      {error ? (
        <div className="empty">{error}</div>
      ) : vitals === null ? (
        <div className="empty">estimating...</div>
      ) : (
        <>
          <div className="card-grid">
            <div className="stat-card">
              <div className="stat-card-label">breathing (median)</div>
              <div className="stat-card-value">
                {s?.breathing_median_bpm?.toFixed(1) ?? '--'} <small>bpm</small>
              </div>
              <div className="stat-card-sub">confidence {s?.breathing_confidence ?? '--'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">
                heart (median) <span className="tag-experimental">experimental</span>
              </div>
              <div className="stat-card-value">
                {s?.heart_median_bpm?.toFixed(0) ?? '--'} <small>bpm</small>
              </div>
              <div className="stat-card-sub">confidence {s?.heart_confidence ?? '--'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">presence</div>
              <div className="stat-card-value">
                {s?.presence_fraction !== undefined ? `${Math.round(s.presence_fraction * 100)}%` : '--'}
              </div>
              <div className="stat-card-sub">of the session</div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">restlessness</div>
              <div className="stat-card-value">
                {s?.motion_fraction !== undefined ? `${Math.round(s.motion_fraction * 100)}%` : '--'}
              </div>
              <div className="stat-card-sub">time spent moving</div>
            </div>
          </div>

          {vitals.breathing.length > 0 && (
            <>
              <h2>Breathing rate</h2>
              <LineChart
                x={vitals.breathing.map((p) => p.t)}
                series={[{ label: 'breaths/min', values: vitals.breathing.map((p) => p.bpm), color: '#38bdf8' }]}
                xLabel="time (s)"
                height={220}
              />
            </>
          )}

          {vitals.heart.length > 0 && (
            <>
              <h2>Heart rate (experimental - stationary subject only)</h2>
              <LineChart
                x={vitals.heart.map((p) => p.t)}
                series={[{ label: 'beats/min', values: vitals.heart.map((p) => p.bpm), color: '#f472b6' }]}
                xLabel="time (s)"
                height={220}
              />
            </>
          )}

          {vitals.activity.length > 0 && (
            <>
              <h2>Presence and motion</h2>
              <LineChart
                x={vitals.activity.map((p) => p.t)}
                series={[
                  { label: 'presence', values: vitals.activity.map((p) => p.presence), color: '#4ade80' },
                  { label: 'motion', values: vitals.activity.map((p) => p.motion), color: '#facc15' },
                ]}
                xLabel="time (s)"
                height={160}
              />
            </>
          )}

          <p className="hint">
            All values are spectral estimates from CSI amplitude: breathing 0.1-0.7 Hz, heart 0.8-2.2 Hz,
            motion 0.7-5 Hz. Heart rate needs a still subject and honest skepticism until validated against
            a reference sensor. CSI cannot assess cardiac rhythm (that requires ECG); rhythm-regularity
            statistics may come later once beat detection is validated on real hardware.
          </p>
        </>
      )}
    </div>
  )
}
