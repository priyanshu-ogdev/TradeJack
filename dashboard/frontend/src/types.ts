/**
 * Types mirroring dashboard/telemetry_server.py's JSON responses exactly --
 * field names and nullability match the Python source, not an idealized
 * shape, so a backend change is a visible type error here rather than a
 * silent runtime mismatch.
 */

export interface StatusAccount {
  account_id: string;
  equity: number;
  cash: number;
  tick: number;
  market_timestamp: number;
}

export interface StatusResponse {
  state_dir: string;
  kill_switch_engaged: boolean;
  accounts: StatusAccount[];
  server_time: number;
}

export interface PricePoint {
  t: number;
  price: number;
}

export interface Trade {
  timestamp: number;
  side: string;
  filled_qty: number;
  avg_price: number;
  fee_paid: number;
  fully_filled: number;
  rejected_reason: string | null;
}

export interface EquityPoint {
  t: number;
  equity: number;
}

export interface Decision {
  tick: number;
  timestamp: number;
  target_frac: number;
  approved: number;
  reject_reason: string | null;
  equity: number;
}

export interface ActiveHalt {
  reason_code: string;
  message: string;
  halted_at: number | null;
  halted_at_iso: string | null;
  last_alert_at: number;
  last_alert_at_iso: string;
  last_alert_event: string;
}

export interface RiskHaltResponse {
  active: ActiveHalt | null;
}

export type AllocationRole = "primary" | "secondary" | "inactive";

export interface AllocationDecision {
  instrument: string;
  role: AllocationRole;
  target_capital_fraction: number;
  opportunity_score: number | null;
  reason: string;
}

export interface PortfolioResponse {
  generated_at: number | null;
  decisions: AllocationDecision[];
  correlation_matrix: Record<string, number>;
}
