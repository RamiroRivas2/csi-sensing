import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

interface Props {
  x: number[]
  series: { label: string; values: number[]; color: string }[]
  xLabel?: string
  height?: number
}

export function LineChart({ x, series, xLabel, height = 220 }: Props) {
  const hostRef = useRef<HTMLDivElement>(null)
  const plotRef = useRef<uPlot | null>(null)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const opts: uPlot.Options = {
      width: host.clientWidth || 800,
      height,
      scales: { x: { time: false } },
      axes: [
        { label: xLabel, stroke: '#8fa1c7', grid: { stroke: '#1e293f' }, ticks: { stroke: '#1e293f' } },
        { stroke: '#8fa1c7', grid: { stroke: '#1e293f' }, ticks: { stroke: '#1e293f' } },
      ],
      legend: { show: series.length > 1 },
      series: [
        {},
        ...series.map((s) => ({ label: s.label, stroke: s.color, width: 1.5 })),
      ],
    }
    const data = [x, ...series.map((s) => s.values)] as uPlot.AlignedData
    plotRef.current?.destroy()
    plotRef.current = new uPlot(opts, data, host)
    const onResize = () => plotRef.current?.setSize({ width: host.clientWidth, height })
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      plotRef.current?.destroy()
      plotRef.current = null
    }
  }, [x, series, xLabel, height])

  return <div ref={hostRef} />
}
