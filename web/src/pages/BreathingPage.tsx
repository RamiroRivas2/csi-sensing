import { useEffect, useState } from 'react'
import { api, type SessionInfo, type BreathingPoint } from '../lib/api'
import { LineChart } from '../components/LineChart'

export function BreathingPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [sessionId, setSessionId] = useState('')
  const [points, setPoints] = useState<BreathingPoint[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.sessions().then((list) => {
      setSessions(list)
      if (list.length > 0) setSessionId(list[0].id)
    }).catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (!sessionId) return
    setPoints(null)
    api.breathing(sessionId).then((r) => setPoints(r.points)).catch((e) => setError(String(e)))
  }, [sessionId])

  if (sessions.length === 0) return <div className="empty">{error ?? 'No sessions recorded yet.'}</div>

  const median = points && points.length > 0
    ? [...points.map((p) => p.bpm)].sort((a, b) => a - b)[Math.floor(points.length / 2)]
    : null

  return (
    <div>
      <div className="page-head">
        <h1>Breathing</h1>
        <div className="stat-row">
          <label>
            session{' '}
            <select value={sessionId} onChange={(e) => setSessionId(e.target.value)}>
              {sessions.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
            </select>
          </label>
          {median !== null && <span className="big-stat">{median.toFixed(1)} <small>bpm median</small></span>}
        </div>
      </div>
      {points === null ? (
        <div className="empty">estimating...</div>
      ) : points.length === 0 ? (
        <div className="empty">Session too short for a 30 s breathing window.</div>
      ) : (
        <>
          <LineChart
            x={points.map((p) => p.t)}
            series={[{ label: 'breaths/min', values: points.map((p) => p.bpm), color: '#38bdf8' }]}
            xLabel="time (s)"
            height={260}
          />
          <h2>Estimate confidence</h2>
          <LineChart
            x={points.map((p) => p.t)}
            series={[{ label: 'confidence', values: points.map((p) => p.confidence), color: '#64748b' }]}
            xLabel="time (s)"
            height={120}
          />
          <p className="hint">
            30 s windows, 5 s hop, PCA across subcarriers, band-pass 0.1-0.7 Hz, Welch peak.
            Low confidence means motion or an empty bed, not a trustworthy rate.
          </p>
        </>
      )}
    </div>
  )
}
