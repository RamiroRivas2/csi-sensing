/** The night as a horizontal band: still (blue), moving (amber), empty (gray). */

interface Segment {
  t: number
  state: string
}

const COLORS: Record<string, string> = {
  still: '#2563b8',
  moving: '#d3a51e',
  empty: '#333d52',
}

export function Hypnogram({ activity, durationS }: { activity: Segment[]; durationS: number }) {
  if (activity.length === 0) return null
  const fmt = (s: number) =>
    durationS >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`

  return (
    <div>
      <div className="hypnogram">
        {activity.map((a, i) => {
          const next = activity[i + 1]?.t ?? durationS
          const width = ((next - a.t) / durationS) * 100
          return (
            <div
              key={a.t}
              className="hypnogram-seg"
              style={{ width: `${width}%`, background: COLORS[a.state] ?? COLORS.empty }}
              title={`${a.state} at ${fmt(a.t)}`}
            />
          )
        })}
      </div>
      <div className="hypnogram-footer">
        <span>0</span>
        <span className="hypnogram-key">
          <i style={{ background: COLORS.still }} /> asleep/still
          <i style={{ background: COLORS.moving }} /> restless
          <i style={{ background: COLORS.empty }} /> out of bed
        </span>
        <span>{fmt(durationS)}</span>
      </div>
    </div>
  )
}
