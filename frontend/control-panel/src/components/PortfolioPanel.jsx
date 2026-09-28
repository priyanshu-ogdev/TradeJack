const ROLE_STYLES = {
  primary: { label: 'primary', text: 'text-nominal', bar: 'bg-nominal' },
  secondary: { label: 'secondary', text: 'text-caution', bar: 'bg-caution' },
}

function AllocationBar({ fraction, colorClass }) {
  const pct = Math.min(Math.max((fraction ?? 0) * 100, 0), 100)
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-none bg-panel-700">
      <div className={`h-full ${colorClass} transition-[width]`} style={{ width: `${pct}%` }} />
    </div>
  )
}

function RoleCard({ decision }) {
  const style = ROLE_STYLES[decision.role]
  return (
    <div className="border border-panel-700 p-3">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-sm text-panel-100">{decision.instrument}</span>
        <span className={`font-mono text-[11px] uppercase tracking-wide ${style.text}`}>
          {style.label}
        </span>
      </div>
      <div className="mt-2 flex items-baseline justify-between font-mono text-xs text-panel-400">
        <span>capital</span>
        <span className="tabular text-panel-200">
          {(decision.target_capital_fraction * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-1">
        <AllocationBar fraction={decision.target_capital_fraction} colorClass={style.bar} />
      </div>
      <div className="mt-2 flex items-baseline justify-between font-mono text-xs text-panel-500">
        <span>opportunity score</span>
        <span className="tabular">
          {decision.opportunity_score != null ? decision.opportunity_score.toExponential(2) : '—'}
        </span>
      </div>
      <p className="mt-2 font-mono text-[11px] leading-snug text-panel-500">{decision.reason}</p>
    </div>
  )
}

function timeAgo(generatedAt) {
  if (!generatedAt) return null
  const seconds = Date.now() / 1000 - generatedAt
  if (seconds < 90) return `${Math.round(seconds)}s ago`
  return `${Math.round(seconds / 60)}m ago`
}

/**
 * Shows execution/portfolio_allocator.py's Primary/Secondary allocation
 * (Kelly-sized, correlation-aware) plus whatever else is currently tracked
 * but not selected. Reads from GET /api/status's `portfolio` field, which
 * itself reads a plain persisted JSON file (see control_panel_api.py's
 * _collect_portfolio) written by PortfolioOrchestrator.run_cycle() -- NOT a
 * live orchestrator held by this API process. Nothing in this project runs
 * that loop on a schedule yet (see portfolio_orchestrator.py's own "what
 * this deliberately does not do"), so an empty state here is the honest,
 * expected state until that loop exists somewhere, not a bug in this panel.
 */
export default function PortfolioPanel({ status }) {
  const portfolio = status?.portfolio
  const decisions = portfolio?.decisions ?? []
  const active = decisions.filter((d) => d.role === 'primary' || d.role === 'secondary')
  const inactive = decisions.filter((d) => d.role === 'inactive')
  const age = timeAgo(portfolio?.generated_at)

  return (
    <section className="border border-panel-700 bg-panel-800 p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">
          Portfolio allocation
        </h2>
        <span className="font-mono text-[11px] text-panel-500 tabular">{age ?? '—'}</span>
      </div>

      {decisions.length === 0 ? (
        <p className="mt-4 font-mono text-xs text-panel-500">
          No allocation cycle has run yet -- the allocator and orchestrator exist
          (execution/portfolio_allocator.py, execution/portfolio_orchestrator.py) but
          nothing schedules run_cycle() in this deployment yet.
        </p>
      ) : (
        <>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {active.length === 0 ? (
              <p className="col-span-2 font-mono text-xs text-panel-500">
                No instrument currently clears the opportunity floor.
              </p>
            ) : (
              active
                .sort((a, b) => (a.role === 'primary' ? -1 : 1))
                .map((d) => <RoleCard key={d.instrument} decision={d} />)
            )}
          </div>

          {inactive.length > 0 && (
            <div className="mt-4 border-t border-panel-700 pt-3">
              <p className="font-mono text-[11px] uppercase tracking-wide text-panel-500">
                Not selected this cycle
              </p>
              <dl className="mt-2 space-y-1 font-mono text-xs">
                {inactive.map((d) => (
                  <div key={d.instrument} className="flex justify-between gap-3 text-panel-500">
                    <dt className="text-panel-400">{d.instrument}</dt>
                    <dd className="truncate text-right">{d.reason}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </>
      )}
    </section>
  )
}
