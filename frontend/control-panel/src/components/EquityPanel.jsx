import Sparkline from './Sparkline.jsx'

function formatCurrency(v) {
  if (v == null) return '—'
  return v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 })
}

export default function EquityPanel({ status, equityPoints }) {
  const equity = status?.equity ?? status?.risk?.current_equity
  const points = equityPoints ?? []
  const first = points[0]?.equity
  const changePct = first && equity != null ? ((equity - first) / first) * 100 : null

  return (
    <section className="rounded-none border border-panel-700 bg-panel-800 p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">Equity</h2>
        <span className="font-mono text-xs text-panel-500 tabular">
          {status?.trade_count ?? 0} trades
        </span>
      </div>

      <div className="mt-3 flex items-end justify-between">
        <div>
          <div className="font-mono text-3xl font-medium text-panel-100 tabular">
            {formatCurrency(equity)}
          </div>
          {changePct != null && (
            <div
              className={`mt-1 font-mono text-sm tabular ${
                changePct >= 0 ? 'text-nominal' : 'text-danger'
              }`}
            >
              {changePct >= 0 ? '+' : ''}
              {changePct.toFixed(2)}% this session
            </div>
          )}
        </div>
        <Sparkline points={points} />
      </div>
    </section>
  )
}
