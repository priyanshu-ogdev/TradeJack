import type {
  StatusResponse,
  PricePoint,
  Trade,
  EquityPoint,
  Decision,
  RiskHaltResponse,
  PortfolioResponse,
} from "./types";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  status: () => getJSON<StatusResponse>("/api/status"),

  priceSeries: (accountId: string, limit = 2000) =>
    getJSON<{ points: PricePoint[] }>(`/api/price-series?account_id=${accountId}&limit=${limit}`),

  trades: (accountId: string, limit = 300) =>
    getJSON<{ trades: Trade[] }>(`/api/trades?account_id=${accountId}&limit=${limit}`),

  equityCurve: (accountId: string, limit = 2000) =>
    getJSON<{ points: EquityPoint[] }>(`/api/equity-curve?account_id=${accountId}&limit=${limit}`),

  decisions: (accountId: string, limit = 200) =>
    getJSON<{ decisions: Decision[] }>(`/api/decisions?account_id=${accountId}&limit=${limit}`),

  riskHalt: (accountId: string) => getJSON<RiskHaltResponse>(`/api/risk-halt?account_id=${accountId}`),

  portfolio: () => getJSON<PortfolioResponse>("/api/portfolio"),

  killSwitch: async (engage: boolean, token: string): Promise<{ kill_switch_engaged: boolean }> => {
    const res = await fetch("/api/kill-switch", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ engage }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.json();
  },
};

/** Polls `fn` every `intervalMs`, calling `onData` with each successful
 * result and `onError` (if given) on failure -- a failed poll is logged and
 * skipped, not fatal, since a dashboard should keep trying rather than
 * freeze on one dropped request. Returns a cleanup function for a React
 * effect's return value. */
export function poll<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  onData: (data: T) => void,
  onError?: (err: unknown) => void,
): () => void {
  let cancelled = false;

  const tick = () => {
    fn()
      .then((data) => {
        if (!cancelled) onData(data);
      })
      .catch((err) => {
        if (!cancelled) onError?.(err);
      });
  };

  tick();
  const handle = window.setInterval(tick, intervalMs);
  return () => {
    cancelled = true;
    window.clearInterval(handle);
  };
}
