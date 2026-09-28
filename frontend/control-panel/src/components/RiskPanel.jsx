function BudgetBar({ fraction }) {
  const pct = Math.min(Math.max((fraction ?? 0) * 100, 0), 100)
  const color = pct >= 90 ? 'bg-danger' : pct >= 60 ? 'bg-caution' : 'bg-nominal'
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-none bg-panel-700">
      <div className={`h-full ${color} transition-[width]`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function RiskPanel({ status }) {
  const risk = status?.risk
  const isHalted = risk?.is_halted

  return (
    <section
      className={`border p-5 ${
        isHalted ? 'border-danger bg-danger/10' : 'border-panel-700 bg-panel-800'
      }`}
    >
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">Risk</h2>
        <span
          className={`font-mono text-xs tabular ${isHalted ? 'text-danger' : 'text-nominal'}`}
        >
          {isHalted ? 'halted' : 'nominal'}
        </span>
      </div>

      {isHalted && (
        <p className="mt-3 font-mono text-sm text-danger">
          {risk?.halt_reason || 'Trading halted.'}
        </p>
      )}

      <div className="mt-4 space-y-1.5">
        <div className="flex justify-between font-mono text-xs text-panel-400">
          <span>daily loss budget used</span>
          <span className="tabular">
            {risk?.risk_budget_used_fraction != null
              ? `${(risk.risk_budget_used_fraction * 100).toFixed(0)}%`
              : '—'}
          </span>
        </div>
        <BudgetBar fraction={risk?.risk_budget_used_fraction} />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-y-2 font-mono text-xs">
        <dt className="text-panel-500">daily loss limit</dt>
        <dd className="text-right text-panel-200 tabular">
          {risk?.max_daily_loss_pct != null ? `${(risk.max_daily_loss_pct * 100).toFixed(1)}%` : '—'}
        </dd>
        <dt className="text-panel-500">drawdown limit</dt>
        <dd className="text-right text-panel-200 tabular">
          {risk?.max_drawdown_halt != null ? `${(risk.max_drawdown_halt * 100).toFixed(1)}%` : '—'}
        </dd>
        <dt className="text-panel-500">peak equity</dt>
        <dd className="text-right text-panel-200 tabular">
          {risk?.peak_equity != null ? risk.peak_equity.toFixed(2) : '—'}
        </dd>
      </dl>
    </section>
  )
}
