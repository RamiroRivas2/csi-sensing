/** Viridis lookup table (256 RGB entries) for canvas heatmaps. */

// 9 anchor points of matplotlib viridis, linearly interpolated to 256 entries
const ANCHORS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [253, 231, 37],
]

function buildLut(): Uint8Array {
  const lut = new Uint8Array(256 * 3)
  const segments = ANCHORS.length - 1
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * segments
    const seg = Math.min(segments - 1, Math.floor(pos))
    const frac = pos - seg
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = Math.round(ANCHORS[seg][c] * (1 - frac) + ANCHORS[seg + 1][c] * frac)
    }
  }
  return lut
}

export const VIRIDIS = buildLut()

/** Normalize v in [lo, hi] to a LUT index. */
export function lutIndex(v: number, lo: number, hi: number): number {
  if (hi <= lo) return 0
  const t = (v - lo) / (hi - lo)
  return Math.max(0, Math.min(255, Math.round(t * 255)))
}
