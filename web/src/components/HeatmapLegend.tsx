/** Viridis colorbar with low/high labels, matching the canvas heatmap LUT. */

const VIRIDIS_STOPS =
  'linear-gradient(to right, #440154, #482878, #3e4a89, #31688e, #26828e, #1f9e89, #35b779, #6dcd59, #fde725)'

export function HeatmapLegend() {
  return (
    <div className="heatmap-legend">
      <span>low</span>
      <div className="heatmap-legend-bar" style={{ background: VIRIDIS_STOPS }} />
      <span>high amplitude</span>
    </div>
  )
}
