import { useEffect, useRef, useState } from 'react'
import { CsiHeatmap, type CsiHeatmapHandle } from '../components/CsiHeatmap'

type Status = 'connecting' | 'streaming' | 'no-collector' | 'closed'

export function LivePage() {
  const heatmapRef = useRef<CsiHeatmapHandle>(null)
  const [status, setStatus] = useState<Status>('connecting')
  const [bpm, setBpm] = useState<number | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [rssi, setRssi] = useState<number | null>(null)

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`)
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'frame') {
        setStatus('streaming')
        setRssi(msg.rssi)
        heatmapRef.current?.appendColumn(msg.amp)
      } else if (msg.type === 'breathing') {
        setBpm(msg.bpm)
        setConfidence(msg.confidence)
      } else if (msg.type === 'error') {
        setStatus('no-collector')
      }
    }
    ws.onclose = () => setStatus((s) => (s === 'streaming' ? 'closed' : s))
    return () => ws.close()
  }, [])

  return (
    <div>
      <div className="page-head">
        <h1>Live</h1>
        <div className="stat-row">
          <span className={`pill ${status === 'streaming' ? 'pill-ok' : 'pill-warn'}`}>
            {status === 'streaming' ? 'streaming' : status === 'no-collector' ? 'collector not running' : status}
          </span>
          {rssi !== null && <span className="pill">RSSI {rssi} dBm</span>}
          <span className="big-stat">
            {bpm !== null ? bpm.toFixed(1) : '--'} <small>bpm</small>
          </span>
          {confidence !== null && <span className="pill">confidence {confidence.toFixed(2)}</span>}
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
        <CsiHeatmap ref={heatmapRef} liveColumns={600} height={300} />
      )}
      <p className="hint">Rows are subcarriers, color is amplitude. Breathing shows as a slow coherent shimmer; motion as vertical ripples.</p>
    </div>
  )
}
