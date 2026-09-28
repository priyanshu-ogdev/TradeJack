const BASE_URL = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const TOKEN = import.meta.env.VITE_API_TOKEN || ''

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${TOKEN}`,
    },
    body: body ? JSON.stringify(body) : undefined,
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail || detail
    } catch {
      // response wasn't JSON -- fall back to statusText, already set above
    }
    throw new ApiError(`${method} ${path} failed: ${detail}`, res.status, detail)
  }

  if (res.status === 204) return null
  return res.json()
}

export const api = {
  getStatus: () => request('/api/status'),
  startTrading: () => request('/api/trading/start', { method: 'POST' }),
  stopTrading: () => request('/api/trading/stop', { method: 'POST' }),
  startTraining: () => request('/api/training/start', { method: 'POST' }),
  stopTraining: () => request('/api/training/stop', { method: 'POST' }),
  resetHalt: () => request('/api/risk/halt/reset', { method: 'POST' }),
  getPendingPromotions: () => request('/api/promotions/pending'),
  approvePromotion: (agentId) => request(`/api/promotions/${agentId}/approve`, { method: 'POST' }),
  getRecentTrades: (limit = 50) => request(`/api/trades/recent?limit=${limit}`),
  getEquityHistory: (limit = 200) => request(`/api/equity/history?limit=${limit}`),
}

export function wsUrl() {
  const base = BASE_URL.replace(/^http/, 'ws')
  return `${base}/ws/live?token=${encodeURIComponent(TOKEN)}`
}

export { ApiError }
