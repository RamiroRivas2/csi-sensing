import { useEffect, useState } from 'react'
import { getJson } from '../lib/api'
import { ConfusionMatrix } from '../components/ConfusionMatrix'

interface ModelMetrics {
  accuracy: number
  macro_f1: number
  per_class_f1: Record<string, number>
  confusion: number[][]
  classes: string[]
}

interface Experiment {
  id: string
  metrics: Record<string, ModelMetrics>
  config: Record<string, unknown>
}

export function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const ctrl = new AbortController()
    getJson<Experiment[]>('/api/experiments', ctrl.signal)
      .then((list) => {
        if (!ctrl.signal.aborted) setExperiments(list)
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e))
      })
    return () => ctrl.abort()
  }, [])

  if (error) return <div className="empty">{error}</div>
  if (experiments === null) return <div className="empty">loading...</div>

  if (experiments.length === 0) {
    return (
      <div>
        <div className="page-head"><h1>Experiments</h1></div>
        <div className="empty">
          <p>No experiments yet. Run the UT-HAR baseline:</p>
          <pre>
            uv run python -c "from csi.io.uthar import download_uthar; download_uthar()"{'\n'}
            uv run python -m csi.ml.train --config experiments/exp01_uthar_rf/config.json
          </pre>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="page-head"><h1>Experiments</h1></div>
      {experiments.map((exp) => {
        const models = Object.entries(exp.metrics)
        const best = models.reduce((a, b) => (b[1].accuracy > a[1].accuracy ? b : a))
        const classes = best[1].classes
        return (
          <div key={exp.id} className="exp-block">
            <h2>{exp.id}</h2>
            <div className="card-grid">
              {models.map(([name, m]) => (
                <div className="stat-card" key={name}>
                  <div className="stat-card-label">{name.replace(/_/g, ' ')}</div>
                  <div className="stat-card-value">{(m.accuracy * 100).toFixed(1)}<small>% acc</small></div>
                  <div className="stat-card-sub">macro F1 {m.macro_f1.toFixed(3)}</div>
                </div>
              ))}
            </div>

            <h2>Per-class F1</h2>
            <table className="session-table">
              <thead>
                <tr>
                  <th>class</th>
                  {models.map(([name]) => <th key={name}>{name.replace(/_/g, ' ')}</th>)}
                </tr>
              </thead>
              <tbody>
                {classes.map((cls) => (
                  <tr key={cls}>
                    <td>{cls}</td>
                    {models.map(([name, m]) => <td key={name}>{m.per_class_f1[cls].toFixed(3)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>

            <h2>Confusion matrix - {best[0].replace(/_/g, ' ')}</h2>
            <ConfusionMatrix confusion={best[1].confusion} classes={classes} />
          </div>
        )
      })}
    </div>
  )
}
