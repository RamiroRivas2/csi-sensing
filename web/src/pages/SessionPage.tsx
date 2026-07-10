import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api, type SessionMeta } from '../lib/api'
import { fetchFloat32, type Float32Matrix } from '../lib/binary'
import { CsiHeatmap } from '../components/CsiHeatmap'

export function SessionPage() {
  const { id = '' } = useParams()
  const sessionId = decodeURIComponent(id)
  const [meta, setMeta] = useState<SessionMeta | null>(null)
  const [matrix, setMatrix] = useState<Float32Matrix | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.session(sessionId).then(setMeta).catch((e) => setError(String(e)))
    fetchFloat32(`/api/sessions/${sessionId}/csi?max_cols=2000`)
      .then(setMatrix)
      .catch((e) => setError(String(e)))
  }, [sessionId])

  if (error) return <div className="empty">{error}</div>
  return (
    <div>
      <div className="page-head">
        <h1>{sessionId}</h1>
        {meta && (
          <div className="stat-row">
            <span className="pill">{meta.duration_s}s</span>
            <span className="pill">{meta.fs.toFixed(1)} Hz</span>
            <span className="pill">{meta.n_subcarriers} subcarriers</span>
            {typeof meta.meta.label === 'string' && <span className="pill pill-ok">{meta.meta.label}</span>}
          </div>
        )}
      </div>
      {matrix ? (
        <CsiHeatmap data={matrix.data} shape={matrix.shape} height={320} />
      ) : (
        <div className="empty">loading heatmap...</div>
      )}
      <p className="hint">
        Full-session amplitude heatmap (time x subcarrier, downsampled to 2000 columns max).
        Use the DSP inspector and Breathing pages for per-subcarrier analysis.
      </p>
      {meta && (
        <pre className="meta-block">{JSON.stringify(meta.meta, null, 2)}</pre>
      )}
    </div>
  )
}
