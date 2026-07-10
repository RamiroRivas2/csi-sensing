import { useEffect, useMemo, useState } from 'react'
import { api, type SessionInfo, type Psd } from '../lib/api'
import { fetchFloat32 } from '../lib/binary'
import { LineChart } from '../components/LineChart'

export function DspInspectorPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [sessionId, setSessionId] = useState('')
  const [subcarrier, setSubcarrier] = useState(0)
  const [low, setLow] = useState(0.1)
  const [high, setHigh] = useState(0.7)
  const [signal, setSignal] = useState<{ raw: number[]; filtered: number[]; fs: number } | null>(null)
  const [psd, setPsd] = useState<Psd | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.sessions().then((list) => {
      setSessions(list)
      if (list.length > 0) setSessionId(list[0].id)
    }).catch((e) => setError(String(e)))
  }, [])

  const selected = sessions.find((s) => s.id === sessionId)

  useEffect(() => {
    if (!sessionId || !selected) return
    setError(null)
    fetchFloat32(
      `/api/sessions/${sessionId}/signal?subcarrier=${subcarrier}&low=${low}&high=${high}`,
    )
      .then(({ data, shape }) => {
        const t = shape[1]
        setSignal({
          raw: Array.from(data.subarray(0, t)),
          filtered: Array.from(data.subarray(t, 2 * t)),
          fs: selected.fs,
        })
      })
      .catch((e) => setError(String(e)))
    api.psd(sessionId, subcarrier).then(setPsd).catch((e) => setError(String(e)))
  }, [sessionId, subcarrier, low, high, selected])

  const times = useMemo(
    () => (signal ? signal.raw.map((_, i) => i / signal.fs) : []),
    [signal],
  )

  if (sessions.length === 0) return <div className="empty">{error ?? 'No sessions recorded yet.'}</div>

  return (
    <div>
      <div className="page-head">
        <h1>DSP inspector</h1>
        <div className="stat-row">
          <label>
            session{' '}
            <select value={sessionId} onChange={(e) => setSessionId(e.target.value)}>
              {sessions.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
            </select>
          </label>
          <label>
            subcarrier{' '}
            <input
              type="number" min={0} max={(selected?.n_subcarriers ?? 1) - 1} value={subcarrier}
              onChange={(e) => setSubcarrier(Number(e.target.value))}
            />
          </label>
          <label>
            band{' '}
            <input type="number" step={0.05} min={0.01} value={low}
              onChange={(e) => setLow(Number(e.target.value))} />
            {' - '}
            <input type="number" step={0.05} value={high}
              onChange={(e) => setHigh(Number(e.target.value))} />
            {' Hz'}
          </label>
        </div>
      </div>
      {error && <div className="empty">{error}</div>}
      {signal && (
        <>
          <h2>Raw vs filtered</h2>
          <LineChart
            x={times}
            series={[
              { label: 'raw', values: signal.raw, color: '#5b6b8f' },
              { label: `filtered ${low}-${high} Hz`, values: signal.filtered, color: '#38bdf8' },
            ]}
            xLabel="time (s)"
          />
        </>
      )}
      {psd && (
        <>
          <h2>Welch PSD</h2>
          <LineChart
            x={psd.freqs}
            series={[{ label: 'psd', values: psd.psd, color: '#a3e635' }]}
            xLabel="frequency (Hz)"
          />
          <p className="hint">A stationary person breathing shows a peak between 0.2 and 0.5 Hz.</p>
        </>
      )}
    </div>
  )
}
