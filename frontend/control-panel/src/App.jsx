import { useEffect, useState, useCallback } from 'react'
import { api, ApiError } from './lib/api.js'
import { useLiveSocket } from './lib/useLiveSocket.js'
import StatusBar from './components/StatusBar.jsx'
import EquityPanel from './components/EquityPanel.jsx'
import RiskPanel from './components/RiskPanel.jsx'
import ControlRail from './components/ControlRail.jsx'
import PendingPromotions from './components/PendingPromotions.jsx'
import TradeLog from './components/TradeLog.jsx'
import PortfolioPanel from './components/PortfolioPanel.jsx'

export default function App() {
  const { status: liveStatus, connectionState } = useLiveSocket()
  const [fallbackStatus, setFallbackStatus] = useState(null)
  const [pending, setPending] = useState({})
  const [trades, setTrades] = useState([])
  const [equityPoints, setEquityPoints] = useState([])
  const [error, setError] = useState(null)

  // The WebSocket carries the fast-moving numbers; REST covers the slower-moving
  // collections (trades, pending promotions) and gives the panel something to show
  // before the socket's first frame arrives.
  const refreshSlow = useCallback(async () => {
    try {
      const [statusRes, pendingRes, tradesRes, equityRes] = await Promise.all([
        api.getStatus(),
        api.getPendingPromotions(),
        api.getRecentTrades(50),
        api.getEquityHistory(200),
      ])
      setFallbackStatus(statusRes)
      setPending(pendingRes.pending)
      setTrades(tradesRes.trades)
      setEquityPoints(equityRes.points)
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not reach the control panel API.')
    }
  }, [])

  useEffect(() => {
    refreshSlow()
    const interval = setInterval(refreshSlow, 10000)
    return () => clearInterval(interval)
  }, [refreshSlow])

  const status = liveStatus ?? fallbackStatus

  async function withRefresh(action) {
    try {
      await action()
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Action failed.')
    } finally {
      refreshSlow()
    }
  }

  return (
    <div className="min-h-screen">
      <StatusBar status={status} connectionState={connectionState} />

      <main className="mx-auto max-w-5xl space-y-4 p-6">
        {error && (
          <div className="border border-danger bg-danger/10 px-4 py-2 font-mono text-xs text-danger">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <EquityPanel status={status} equityPoints={equityPoints} />
          <RiskPanel status={status} />
        </div>

        <PortfolioPanel status={status} />

        <ControlRail
          status={status}
          onStartTrading={() => withRefresh(api.startTrading)}
          onStopTrading={() => withRefresh(api.stopTrading)}
          onStartTraining={() => withRefresh(api.startTraining)}
          onStopTraining={() => withRefresh(api.stopTraining)}
          onResetHalt={() => withRefresh(api.resetHalt)}
        />

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <PendingPromotions
            pending={pending}
            onApprove={(agentId) => withRefresh(() => api.approvePromotion(agentId))}
          />
          <TradeLog trades={trades} />
        </div>
      </main>
    </div>
  )
}
