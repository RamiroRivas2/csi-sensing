import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type SessionInfo } from '../lib/api'

export function SessionsPage() {
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.sessions().then(setSessions).catch((e) => setError(String(e)))
  }, [])

  if (error) return <div className="empty">{error}</div>
  if (!sessions) return <div className="empty">loading...</div>
  if (sessions.length === 0)
    return (
      <div className="empty">
        <p>No sessions yet. Record one with the collector, or generate a synthetic one:</p>
        <pre>
          uv run python scripts/make_synthetic_log.py{'\n'}
          uv run python -m collector.collector --replay data/raw/synthetic_15bpm.log
        </pre>
      </div>
    )

  return (
    <div>
      <div className="page-head"><h1>Sessions</h1></div>
      <table className="session-table">
        <thead>
          <tr><th>id</th><th>dataset</th><th>label</th><th>room</th><th>duration</th><th>fs</th><th>subcarriers</th></tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.id}>
              <td><Link to={`/sessions/${encodeURIComponent(s.id)}`}>{s.id}</Link></td>
              <td>{s.dataset}</td>
              <td>{s.label ?? '-'}</td>
              <td>{s.room ?? '-'}</td>
              <td>{s.duration_s}s</td>
              <td>{s.fs.toFixed(1)} Hz</td>
              <td>{s.n_subcarriers}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
