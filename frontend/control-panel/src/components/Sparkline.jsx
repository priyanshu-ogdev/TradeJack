/** Minimal, dependency-free sparkline. Deliberately not a charting library --
 * this is the only chart in the panel, so a small hand-built SVG keeps the
 * dependency surface (and therefore what can silently break) as small as
 * the actual need. */
export default function Sparkline({ points, width = 240, height = 56, color = '#4EA89E' }) {
  if (!points || points.length < 2) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center text-panel-500 text-xs font-mono"
      >
        awaiting data
      </div>
    )
  }

  const values = points.map((p) => p.equity)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const coords = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - ((v - min) / range) * height
    return [x, y]
  })

  const path = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const areaPath = `${path} L${width},${height} L0,${height} Z`

  const last = values[values.length - 1]
  const first = values[0]
  const trendUp = last >= first

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={trendUp ? '#4EA89E' : '#E0524F'} stopOpacity="0.25" />
          <stop offset="100%" stopColor={trendUp ? '#4EA89E' : '#E0524F'} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill="url(#sparkFill)" />
      <path d={path} fill="none" stroke={trendUp ? '#4EA89E' : '#E0524F'} strokeWidth="1.5" />
    </svg>
  )
}
