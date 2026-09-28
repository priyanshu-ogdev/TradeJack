function ConnectionDot({ state }) {
  const color = state === 'open' ? 'bg-nominal' : state === 'connecting' ? 'bg-caution' : 'bg-danger'
  return (
    <span className="relative flex h-2 w-2">
      {state === 'open' && (
        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${color} opacity-40`} />
      )}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${color}`} />
    </span>
  )
}

export default function StatusBar({ status, connectionState }) {
  const isHalted = status?.risk?.is_halted
  const tradingRunning = status?.trading_running

  return (
    <header className="flex items-center justify-between border-b border-panel-700 bg-panel-950 px-6 py-3">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isHalted ? 'bg-danger' : tradingRunning ? 'bg-nominal' : 'bg-panel-500'
            }`}
          />
          <span className="font-mono text-sm tracking-tight text-panel-100">
            {isHalted ? 'HALTED' : tradingRunning ? 'LIVE' : 'STOPPED'}
          </span>
        </div>
        <span className="text-panel-600">·</span>
        <span className="font-mono text-sm text-panel-300">
          {status?.symbol ?? '—'}
        </span>
        <span className="text-panel-600">·</span>
        <span className="font-mono text-xs text-panel-400 uppercase">
          {status?.exchange_mode ?? 'unknown'}
        </span>
      </div>

      <div className="flex items-center gap-5 text-xs font-mono text-panel-400">
        <div className="flex items-center gap-2">
          <ConnectionDot state={connectionState} />
          <span>{connectionState === 'open' ? 'connected' : connectionState}</span>
        </div>
        <span className="tabular">tick {status?.tick ?? 0}</span>
      </div>
    </header>
  )
}
