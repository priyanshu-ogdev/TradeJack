function formatTime(ts) {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleTimeString('en-US', { hour12: false })
}

export default function TradeLog({ trades }) {
  const rows = trades || []

  return (
    <section className="border border-panel-700 bg-panel-800 p-5">
      <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">
        Recent trades
      </h2>

      {rows.length === 0 ? (
        <p className="mt-4 font-mono text-xs text-panel-500">No trade attempts yet.</p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full font-mono text-xs">
            <thead>
              <tr className="text-left text-panel-500">
                <th className="pb-2 font-normal">time</th>
                <th className="pb-2 font-normal">side</th>
                <th className="pb-2 font-normal text-right">qty</th>
                <th className="pb-2 font-normal text-right">price</th>
                <th className="pb-2 font-normal text-right">fee</th>
                <th className="pb-2 font-normal text-right">status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-panel-700">
              {rows.map((t, i) => (
                <tr key={i} className="text-panel-200">
                  <td className="py-1.5 text-panel-400 tabular">{formatTime(t.timestamp)}</td>
                  <td
                    className={`py-1.5 uppercase ${
                      t.side === 'buy' ? 'text-nominal' : 'text-danger'
                    }`}
                  >
                    {t.side}
                  </td>
                  <td className="py-1.5 text-right tabular">{t.filled_qty?.toFixed?.(6) ?? '—'}</td>
                  <td className="py-1.5 text-right tabular">{t.avg_price?.toFixed?.(2) ?? '—'}</td>
                  <td className="py-1.5 text-right tabular text-panel-400">
                    {t.fee_paid?.toFixed?.(4) ?? '—'}
                  </td>
                  <td className="py-1.5 text-right">
                    {t.rejected_reason ? (
                      <span className="text-caution">{t.rejected_reason}</span>
                    ) : t.fully_filled ? (
                      <span className="text-nominal">filled</span>
                    ) : (
                      <span className="text-panel-400">partial</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
