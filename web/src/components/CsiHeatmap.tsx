import { useEffect, useImperativeHandle, useRef, forwardRef } from 'react'
import { VIRIDIS, lutIndex } from '../lib/colormap'

/**
 * Canvas heatmap: rows = subcarriers, columns = time. Renders by writing an
 * ImageData at the matrix's native size then letting CSS scale it - one blit,
 * fast enough for live scrolling.
 */

export interface CsiHeatmapHandle {
  /** Append one column (length = subcarrier count) and scroll left. */
  appendColumn: (amp: number[]) => void
}

interface Props {
  /** Row-major (time, subcarrier) matrix for static rendering. */
  data?: Float32Array
  shape?: number[]
  /** For live mode: fixed size of the scrolling buffer. */
  liveColumns?: number
  liveRows?: number
  height?: number
}

function render(
  canvas: HTMLCanvasElement,
  matrix: Float32Array,
  nTime: number,
  nSub: number,
) {
  canvas.width = nTime
  canvas.height = nSub
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  let lo = Infinity
  let hi = -Infinity
  for (const v of matrix) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  const img = ctx.createImageData(nTime, nSub)
  for (let t = 0; t < nTime; t++) {
    for (let s = 0; s < nSub; s++) {
      const idx = lutIndex(matrix[t * nSub + s], lo, hi) * 3
      // draw subcarrier 0 at the bottom
      const p = ((nSub - 1 - s) * nTime + t) * 4
      img.data[p] = VIRIDIS[idx]
      img.data[p + 1] = VIRIDIS[idx + 1]
      img.data[p + 2] = VIRIDIS[idx + 2]
      img.data[p + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

export const CsiHeatmap = forwardRef<CsiHeatmapHandle, Props>(function CsiHeatmap(
  { data, shape, liveColumns = 600, liveRows = 64, height = 260 },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bufferRef = useRef<Float32Array | null>(null)
  const filledRef = useRef(0)

  useEffect(() => {
    if (data && shape && canvasRef.current) {
      render(canvasRef.current, data, shape[0], shape[1])
    }
  }, [data, shape])

  useImperativeHandle(ref, () => ({
    appendColumn(amp: number[]) {
      const nSub = amp.length || liveRows
      if (!bufferRef.current || bufferRef.current.length !== liveColumns * nSub) {
        bufferRef.current = new Float32Array(liveColumns * nSub)
        filledRef.current = 0
      }
      const buf = bufferRef.current
      if (filledRef.current < liveColumns) {
        for (let s = 0; s < nSub; s++) buf[filledRef.current * nSub + s] = amp[s]
        filledRef.current++
      } else {
        buf.copyWithin(0, nSub)
        for (let s = 0; s < nSub; s++) buf[(liveColumns - 1) * nSub + s] = amp[s]
      }
      if (canvasRef.current) {
        render(canvasRef.current, buf.subarray(0, filledRef.current * nSub), filledRef.current, nSub)
      }
    },
  }))

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height,
        imageRendering: 'pixelated',
        borderRadius: 8,
        border: '1px solid #2a3550',
        background: '#101828',
        display: 'block',
      }}
    />
  )
})
