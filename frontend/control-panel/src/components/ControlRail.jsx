import { useState } from 'react'

function ActionButton({ label, onClick, tone = 'neutral', disabled, confirmMessage }) {
  const [pending, setPending] = useState(false)

  const toneClasses = {
    neutral: 'border-panel-500 text-panel-100 hover:bg-panel-700',
    go: 'border-nominal text-nominal hover:bg-nominal/10',
    stop: 'border-caution text-caution hover:bg-caution/10',
    danger: 'border-danger text-danger hover:bg-danger/10',
  }[tone]

  async function handleClick() {
    if (confirmMessage && !window.confirm(confirmMessage)) return
    setPending(true)
    try {
      await onClick()
    } finally {
      setPending(false)
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={disabled || pending}
      className={`border px-4 py-2 font-mono text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${toneClasses}`}
    >
      {pending ? 'working…' : label}
    </button>
  )
}

/**
 * Deliberately separated from the telemetry panels (its own bordered rail, its
 * own visual rhythm) -- these buttons start/stop real trading and reset a real
 * risk halt. Every consequential action goes through window.confirm() as a
 * simple, dependency-free friction step; this is a control surface, not a
 * dashboard, and the two should not feel visually interchangeable.
 */
export default function ControlRail({ status, onStartTrading, onStopTrading, onStartTraining, onStopTraining, onResetHalt }) {
  const tradingRunning = status?.trading_running
  const trainingRunning = status?.training_running
  const isHalted = status?.risk?.is_halted

  return (
    <section className="border border-panel-600 bg-panel-950 p-5">
      <h2 className="text-xs font-medium uppercase tracking-wide text-panel-400">Controls</h2>
      <div className="mt-4 flex flex-wrap gap-3">
        {!tradingRunning ? (
          <ActionButton
            label="Start trading"
            tone="go"
            onClick={onStartTrading}
            confirmMessage="Start live trading now? The system will begin placing real orders."
          />
        ) : (
          <ActionButton
            label="Stop trading"
            tone="stop"
            onClick={onStopTrading}
            confirmMessage="Stop trading? Any open position stays open until you close it manually."
          />
        )}

        {!trainingRunning ? (
          <ActionButton label="Start continuous training" onClick={onStartTraining} disabled={!tradingRunning} />
        ) : (
          <ActionButton label="Stop continuous training" onClick={onStopTraining} />
        )}

        <ActionButton
          label="Reset halt"
          tone="danger"
          onClick={onResetHalt}
          disabled={!isHalted}
          confirmMessage="Resume trading after a halt? Only do this once you've reviewed why it halted."
        />
      </div>
    </section>
  )
}
