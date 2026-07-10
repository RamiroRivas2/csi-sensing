import { useEffect, useMemo, useRef } from 'react'
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

  // Structural identity of the chart: recreate uPlot only when the axes or the set
  // of series (count/labels/colors) change - NOT when the data values update. This
  // keeps the live dashboard from tearing down and rebuilding the plot ~20x/second.
  const structureKey = useMemo(
    () => `${xLabel ?? ''}|${height}|${series.map((s) => `${s.label}:${s.color}`).join(',')}`,
    [xLabel, height, series],
  )

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
      series: [{}, ...series.map((s) => ({ label: s.label, stroke: s.color, width: 1.5 }))],
    }
    const data = [x, ...series.map((s) => s.values)] as uPlot.AlignedData
    plotRef.current = new uPlot(opts, data, host)
    const onResize = () => plotRef.current?.setSize({ width: host.clientWidth, height })
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      plotRef.current?.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structureKey])

  // Data-only updates reuse the existing plot instance.
  useEffect(() => {
    const data = [x, ...series.map((s) => s.values)] as uPlot.AlignedData
    plotRef.current?.setData(data)
  }, [x, series])

  return <div ref={hostRef} />
}
