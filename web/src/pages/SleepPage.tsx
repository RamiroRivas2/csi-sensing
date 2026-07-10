import { useEffect, useState } from 'react'
import { api, type SessionInfo } from '../lib/api'
import { LineChart } from '../components/LineChart'

interface SleepReport {
  duration_h: number
  present_fraction: number
  sleep_efficiency: number
  awakenings: number
  restlessness: number
  longest_still_h: number
  breathing_median_bpm: number | null
  breathing_iqr_bpm: number | null
}

interface Night extends SleepReport {
  id: string
  started_at: string | null
}

interface Wellbeing {
  nights: Night[]
  baseline_ready: boolean
  flags: { metric: string; message: string; baseline: number; recent: number }[]
}

export function SleepPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [sessionId, setSessionId] = useState('')
  const [report, setReport] = useState<SleepReport | null>(null)
  const [wellbeing, setWellbeing] = useState<Wellbeing | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.sessions().then((list) => {
      setSessions(list)
      if (list.length > 0) setSessionId(list[0].id)
    }).catch((e) => setError(String(e)))
    fetch('/api/wellbeing').then((r) => r.json()).then(setWellbeing).catch(() => {})
  }, [])

  useEffect(() => {
    if (!sessionId) return
    setReport(null)
    fetch(`/api/sessions/${sessionId}/sleep`)
      .then((r) => r.json())
      .then(setReport)
      .catch((e) => setError(String(e)))
  }, [sessionId])

  if (sessions.length === 0) return <div className="empty">{error ?? 'No sessions recorded yet.'}</div>

  return (
    <div>
      <div className="page-head">
        <h1>Sleep</h1>
        <label className="stat-row">
          night{' '}
          <select value={sessionId} onChange={(e) => setSessionId(e.target.value)}>
            {sessions.map((x) => <option key={x.id} value={x.id}>{x.id}</option>)}
          </select>
        </label>
      </div>

      {report && (
        <div className="card-grid">
          <div className="stat-card">
            <div className="stat-card-label">sleep efficiency</div>
            <div className="stat-card-value">{Math.round(report.sleep_efficiency * 100)}%</div>
            <div className="stat-card-sub">still while in bed</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">awakenings</div>
            <div className="stat-card-value">{report.awakenings}</div>
            <div className="stat-card-sub">motion bouts over 30 s</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">longest still stretch</div>
            <div className="stat-card-value">
              {report.longest_still_h.toFixed(1)} <small>h</small>
            </div>
            <div className="stat-card-sub">of {report.duration_h.toFixed(1)} h recorded</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">breathing</div>
            <div className="stat-card-value">
              {report.breathing_median_bpm ?? '--'} <small>bpm</small>
            </div>
            <div className="stat-card-sub">
              spread {report.breathing_iqr_bpm ?? '--'} bpm over the night
            </div>
          </div>
        </div>
      )}

      <h2>Trends across nights</h2>
      {wellbeing === null ? (
        <div className="empty">loading...</div>
      ) : wellbeing.nights.length < 2 ? (
        <div className="empty">
          Trends appear after a few recorded nights; baseline deviation flags need 7+.
          Tonight's recording is night one.
        </div>
      ) : (
        <>
          {!wellbeing.baseline_ready && (
            <p className="hint">
              {wellbeing.nights.length} nights recorded - deviation flags unlock at 7.
            </p>
          )}
          {wellbeing.flags.map((f) => (
            <div key={f.metric} className="fall-banner">
              <div>
                <b>{f.message}</b>
                <span className="fall-banner-sub">
                  {' '}- recent {f.recent} vs baseline {f.baseline}. A sustained change worth
                  noticing, not a diagnosis.
                </span>
              </div>
            </div>
          ))}
          <LineChart
            x={wellbeing.nights.map((_, i) => i + 1)}
            series={[
              {
                label: 'sleep efficiency',
                values: wellbeing.nights.map((n) => n.sleep_efficiency),
                color: '#38bdf8',
              },
              {
                label: 'restlessness',
                values: wellbeing.nights.map((n) => n.restlessness),
                color: '#facc15',
              },
            ]}
            xLabel="night"
            height={200}
          />
          <LineChart
            x={wellbeing.nights.map((_, i) => i + 1)}
            series={[
              {
                label: 'awakenings',
                values: wellbeing.nights.map((n) => n.awakenings),
                color: '#f472b6',
              },
            ]}
            xLabel="night"
            height={140}
          />
        </>
      )}

      <p className="hint">
        Why track this: the sleep-research literature links sustained drops in sleep efficiency,
        rising restlessness, and more frequent awakenings to mood changes, including depression
        risk. These are wellbeing indicators computed from radio reflections - they can flag that
        a pattern changed, and nothing more. If the trends here worry you, that is a conversation
        for a professional, not a chart.
      </p>
    </div>
  )
}
