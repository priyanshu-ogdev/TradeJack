import { useState } from 'react'

export default function PendingPromotions({ pending, onApprove }) {
  const [approvingId, setApprovingId] = useState(null)
  const entries = Object.values(pending || {})

  async function handleApprove(agentId) {
    const record = pending[agentId]
    const confirmed = window.confirm(
      `Approve agent ${agentId} for promotion?\n\n` +
        `Sharpe: ${record?.avg_sharpe ?? '—'}  Max drawdown: ${record?.max_drawdown ?? '—'}\n\n` +
        `This will hot-swap it into live trading.`
    )
    if (!confirmed) return
    setApprovingId(agentId)
    try {
      await onApprove(agentId)
    } finally {
      setApprovingId(null)
    }
  }

  return (
    <section className="border border-panel-700 bg-panel-800 p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">
          Pending promotions
        </h2>
        <span className="font-mono text-xs text-panel-500 tabular">{entries.length}</span>
      </div>

      {entries.length === 0 ? (
        <p className="mt-4 font-mono text-xs text-panel-500">
          Nothing awaiting approval. A champion that clears the statistical gate
          will show up here.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-panel-700">
          {entries.map((entry) => (
            <li key={entry.agent_id} className="flex items-center justify-between py-3">
              <div className="font-mono text-xs">
                <div className="text-panel-100">
                  agent {entry.agent_id} · {entry.label}
                </div>
                <div className="mt-0.5 text-panel-500">
                  sharpe {entry.avg_sharpe?.toFixed?.(2) ?? '—'} · drawdown{' '}
                  {entry.max_drawdown != null ? `${(entry.max_drawdown * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
              <button
                onClick={() => handleApprove(entry.agent_id)}
                disabled={approvingId === entry.agent_id}
                className="border border-nominal px-3 py-1.5 font-mono text-xs text-nominal transition-colors hover:bg-nominal/10 disabled:opacity-40"
              >
                {approvingId === entry.agent_id ? 'approving…' : 'Approve'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
