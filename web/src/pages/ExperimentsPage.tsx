import { useEffect, useState } from 'react'

interface Experiment {
  id: string
  [key: string]: unknown
}

export function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])

  useEffect(() => {
    fetch('/api/experiments').then((r) => r.json()).then(setExperiments).catch(() => {})
  }, [])

  if (experiments.length === 0) {
    return (
      <div>
        <div className="page-head"><h1>Experiments</h1></div>
        <div className="empty">
          <p>No experiments yet. This page fills in when the UT-HAR baseline (exp01) lands:</p>
          <pre>uv run python -m csi.ml.train --config experiments/exp01_uthar_rf/config.json</pre>
        </div>
      </div>
    )
  }
  return (
    <div>
      <div className="page-head"><h1>Experiments</h1></div>
      <pre className="meta-block">{JSON.stringify(experiments, null, 2)}</pre>
    </div>
  )
}
