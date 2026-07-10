import { useEffect, useRef, useState } from 'react'
import { CsiHeatmap, type CsiHeatmapHandle } from '../components/CsiHeatmap'
import { LineChart } from '../components/LineChart'

type Status = 'connecting' | 'streaming' | 'no-collector' | 'closed'

interface Vitals {
  breathing_bpm: number
  breathing_confidence: number
  heart_bpm: number
  heart_confidence: number
  state: string
  presence: number
  motion: number
}

const HISTORY_LIMIT = 150 // ~5 min of estimates at one every 2 s

export function DashboardPage() {
  const heatmapRef = useRef<CsiHeatmapHandle>(null)
  const frameTimesRef = useRef<number[]>([])
  const [status, setStatus] = useState<Status>('connecting')
  const [vitals, setVitals] = useState<Vitals | null>(null)
  const [rssi, setRssi] = useState<number | null>(null)
  const [fps, setFps] = useState<number | null>(null)
  const [history, setHistory] = useState<{ t: number; breathing: number; heart: number }[]>([])

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`)
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'frame') {
        setStatus('streaming')
        setRssi(msg.rssi)
        heatmapRef.current?.appendColumn(msg.amp)
        const times = frameTimesRef.current
        times.push(msg.t)
        if (times.length > 60) times.shift()
        if (times.length > 10) {
          setFps((times.length - 1) / (times[times.length - 1] - times[0]))
        }
      } else if (msg.type === 'vitals') {
        setVitals(msg)
        setHistory((h) =>
          [...h, { t: Date.now() / 1000, breathing: msg.breathing_bpm, heart: msg.heart_bpm }].slice(
            -HISTORY_LIMIT,
          ),
        )
      } else if (msg.type === 'error') {
        setStatus('no-collector')
      }
    }
    ws.onclose = () => setStatus((s) => (s === 'streaming' ? 'closed' : s))
    return () => ws.close()
  }, [])

  const heartUsable = vitals !== null && vitals.state === 'still' && vitals.heart_confidence > 0.1
  const stateColor =
    vitals?.state === 'still' ? 'ok' : vitals?.state === 'moving' ? 'warn' : 'muted'

  return (
    <div>
      <div className="page-head">
        <h1>Dashboard</h1>
        <div className="stat-row">
          <span className={`pill ${status === 'streaming' ? 'pill-ok' : 'pill-warn'}`}>
            {status === 'streaming' ? 'streaming' : status === 'no-collector' ? 'collector not running' : status}
          </span>
          {rssi !== null && <span className="pill">RSSI {rssi} dBm</span>}
          {fps !== null && <span className="pill">{fps.toFixed(1)} frames/s</span>}
        </div>
      </div>

      <div className="card-grid">
        <div className="stat-card">
          <div className="stat-card-label">breathing</div>
          <div className="stat-card-value">
            {vitals ? vitals.breathing_bpm.toFixed(1) : '--'} <small>bpm</small>
          </div>
          <div className="stat-card-sub">
            {vitals ? `confidence ${vitals.breathing_confidence.toFixed(2)}` : 'waiting for 20 s of signal'}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label">
            heart rate <span className="tag-experimental">experimental</span>
          </div>
          <div className="stat-card-value">
            {heartUsable ? vitals.heart_bpm.toFixed(0) : '--'} <small>bpm</small>
          </div>
          <div className="stat-card-sub">
            {vitals
              ? heartUsable
                ? `confidence ${vitals.heart_confidence.toFixed(2)}`
                : vitals.state !== 'still'
                  ? 'needs a stationary subject'
                  : 'signal too weak'
              : 'waiting for signal'}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label">room state</div>
          <div className={`stat-card-value state-${stateColor}`}>{vitals?.state ?? '--'}</div>
          <div className="stat-card-sub">
            {vitals ? `presence ${vitals.presence.toFixed(2)} - motion ${vitals.motion.toFixed(2)}` : ''}
          </div>
        </div>
      </div>

      {status === 'no-collector' ? (
        <div className="empty">
          <p>The collector is not running. Start it against hardware or a replay:</p>
          <pre>
            uv run python scripts/make_synthetic_log.py{'\n'}
            uv run python -m collector.collector --replay data/raw/synthetic_15bpm.log
          </pre>
        </div>
      ) : (
        <>
          <h2>Live CSI</h2>
          <CsiHeatmap ref={heatmapRef} liveColumns={600} height={240} />
          <p className="hint">Rows are subcarriers, color is amplitude. Breathing shows as a slow coherent shimmer; motion as vertical ripples.</p>
        </>
      )}

      {history.length > 2 && (
        <>
          <h2>Rate history</h2>
          <LineChart
            x={history.map((h) => h.t - history[0].t)}
            series={[
              { label: 'breathing bpm', values: history.map((h) => h.breathing), color: '#38bdf8' },
              { label: 'heart bpm (experimental)', values: history.map((h) => h.heart), color: '#f472b6' },
            ]}
            xLabel="elapsed (s)"
            height={180}
          />
        </>
      )}
    </div>
  )
}
