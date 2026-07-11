import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api, type SessionMeta } from '../lib/api'
import { fetchFloat32, type Float32Matrix } from '../lib/binary'
import { CsiHeatmap } from '../components/CsiHeatmap'

export function SessionPage() {
  // useParams values are already decoded by the router; decoding again would
  // corrupt (or throw on) any id containing a literal %
  const { id: sessionId = '' } = useParams()
  const [meta, setMeta] = useState<SessionMeta | null>(null)
  const [matrix, setMatrix] = useState<Float32Matrix | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setMeta(null)
    setMatrix(null)
    setError(null)
    const ctrl = new AbortController()
    api.session(sessionId, ctrl.signal).then((m) => {
      if (!ctrl.signal.aborted) setMeta(m)
    }).catch((e) => {
      if (!ctrl.signal.aborted) setError(String(e))
    })
    fetchFloat32(`/api/sessions/${sessionId}/csi?max_cols=2000`, ctrl.signal)
      .then((m) => {
        if (!ctrl.signal.aborted) setMatrix(m)
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e))
      })
    return () => ctrl.abort() // a slow response for a previous session must not land
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
