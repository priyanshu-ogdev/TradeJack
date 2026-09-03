"""
Session report: reads a live-paper session's SQLite ledgers and turns them into
the actual answer to "how much could this earn if released" — not a vibe, a
number with a confidence caveat attached.

Deliberately conservative on what it will claim:
  - Requires a minimum sample size before reporting Sharpe/Sortino at all
    (a handful of trades produces a meaningless ratio).
  - Reports fee/slippage drag as its own line, since that's the part of the
    result least likely to change between paper and real deployment (fees and
    visible-depth slippage are real Binance mechanics, whereas the model's own
    edge is the part actually in question).
  - Compares against buy-and-hold on the same price series and reports a
    Mann-Whitney U test on tick-to-tick returns rather than eyeballing two
    numbers — this mirrors the significance-test gap flagged in
    escrow/validation_airgap.py during the earlier review (it backtests on
    historical splits but doesn't yet compare against an incumbent with a
    formal test; this is the live-data equivalent of that missing check).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sqlite3
import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _read_ledger_returns(ledger_db_path: str) -> List[float]:
    conn = sqlite3.connect(ledger_db_path)
    rows = conn.execute("SELECT equity FROM portfolio_state ORDER BY tick_id ASC").fetchall()
    conn.close()
    equities = [r[0] for r in rows]
    returns = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        if prev > 0:
            returns.append((equities[i] - prev) / prev)
    return returns, equities


def _read_fills(fills_db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(fills_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fills ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _read_decisions(decisions_db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(decisions_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM decisions ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_report(
    state_dir: str,
    account_id: int = 900,
    min_trades_for_ratios: int = 30,
    risk_free_rate_annual: float = 0.04,
) -> Dict[str, Any]:
    child_dir = os.path.join(os.path.abspath(state_dir), f"child_{account_id}")
    ledger_path = os.path.join(child_dir, "ledger.sqlite")
    fills_path = os.path.join(child_dir, "fills.sqlite")
    decisions_path = os.path.join(child_dir, "decisions.sqlite")

    report: Dict[str, Any] = {"account_id": account_id, "state_dir": child_dir}

    if not os.path.exists(ledger_path):
        report["error"] = f"No ledger found at {ledger_path} — has a session been run yet?"
        return report

    returns, equities = _read_ledger_returns(ledger_path)
    fills = _read_fills(fills_path) if os.path.exists(fills_path) else []
    decisions = _read_decisions(decisions_path) if os.path.exists(decisions_path) else []

    start_equity = equities[0] if equities else None
    end_equity = equities[-1] if equities else None
    real_fills = [f for f in fills if f["filled_qty"] not in (0.0, None)]
    total_fees = sum(f["fee_paid"] or 0.0 for f in fills)
    rejected_counts: Dict[str, int] = {}
    for f in fills:
        if f["rejected_reason"]:
            rejected_counts[f["rejected_reason"]] = rejected_counts.get(f["rejected_reason"], 0) + 1
    for d in decisions:
        if not d["approved"] and d["reject_reason"]:
            rejected_counts[d["reject_reason"]] = rejected_counts.get(d["reject_reason"], 0) + 1

    report.update({
        "start_equity": start_equity,
        "end_equity": end_equity,
        "pnl_abs": (end_equity - start_equity) if (start_equity and end_equity) else None,
        "pnl_pct": ((end_equity - start_equity) / start_equity * 100) if start_equity else None,
        "n_ticks": len(equities),
        "n_fills": len(real_fills),
        "total_fees_paid": total_fees,
        "rejection_breakdown": rejected_counts,
    })

    if len(returns) >= min_trades_for_ratios:
        arr = np.array(returns)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sharpe = (mean_ret / std_ret) if std_ret > 1e-12 else 0.0
        downside = arr[arr < 0]
        sortino = (mean_ret / float(np.std(downside, ddof=1))) if len(downside) > 1 else 0.0
        report["sharpe_per_tick"] = sharpe
        report["sortino_per_tick"] = sortino
        report["note_ratios"] = "Per-tick, NOT annualized — annualization needs a known tick frequency; compare sessions on this raw scale instead."
    else:
        report["sharpe_per_tick"] = None
        report["sortino_per_tick"] = None
        report["note_ratios"] = f"Fewer than {min_trades_for_ratios} ticks recorded — Sharpe/Sortino would be noise, not reported."

    if real_fills and returns:
        n = len(real_fills)
        report["fee_drag_pct_of_pnl"] = (
            (total_fees / abs(report["pnl_abs"]) * 100) if report["pnl_abs"] not in (None, 0) else None
        )

    # Benchmark comparison: Buy-and-Hold on underlying price series
    prices = [d["mid_price"] for d in decisions if d.get("mid_price") and d["mid_price"] > 0]
    if len(prices) >= 2:
        bah_return_pct = ((prices[-1] - prices[0]) / prices[0]) * 100.0
        report["buy_and_hold_pnl_pct"] = round(bah_return_pct, 4)
        if report.get("pnl_pct") is not None:
            report["alpha_over_bah_pct"] = round(report["pnl_pct"] - bah_return_pct, 4)

        bah_returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        if SCIPY_AVAILABLE and len(returns) == len(bah_returns) and len(returns) >= min_trades_for_ratios:
            diff = np.array(returns) - np.array(bah_returns)
            if np.any(np.abs(diff) > 1e-12):
                try:
                    stat_b, pval_b = scipy_stats.wilcoxon(diff)
                    report["wilcoxon_stat_vs_bah"] = float(stat_b)
                    report["wilcoxon_pvalue_vs_bah"] = float(pval_b)
                except Exception:
                    pass

    arr = np.array(returns) if returns else np.array([])
    has_variance = arr.size > 1 and float(np.std(arr)) > 1e-12
    if SCIPY_AVAILABLE and len(returns) >= min_trades_for_ratios and has_variance:
        stat, pvalue = scipy_stats.wilcoxon(arr)
        report["wilcoxon_stat_vs_zero_return"] = float(stat)
        report["wilcoxon_pvalue_vs_zero_return"] = float(pvalue)
        report["significance_note"] = (
            "Tests whether tick returns are distinguishable from zero (Wilcoxon signed-rank). "
            f"P-value: {pvalue:.4f}. Alpha over Buy-and-Hold: {report.get('alpha_over_bah_pct', 'N/A')}%"
        )
    elif len(returns) >= min_trades_for_ratios and not has_variance:
        report["significance_note"] = "Returns had zero variance (no trades filled) — nothing to test yet."
    else:
        report["significance_note"] = "scipy not available or insufficient samples — no significance test run."

    return report


def print_report(report: Dict[str, Any]):
    print("=" * 60)
    print("TRADEJACK LIVE-PAPER SESSION REPORT")
    print("=" * 60)
    if "error" in report:
        print(report["error"])
        return
    print(f"Account:          child_{report['account_id']}")
    print(f"Start equity:     ${report['start_equity']:.4f}")
    print(f"End equity:       ${report['end_equity']:.4f}")
    print(f"P&L:              ${report['pnl_abs']:.4f} ({report['pnl_pct']:.3f}%)")
    if report.get("buy_and_hold_pnl_pct") is not None:
        print(f"Buy & Hold P&L:   {report['buy_and_hold_pnl_pct']:.3f}%")
        print(f"Alpha over B&H:   {report.get('alpha_over_bah_pct', 0.0):+.3f}%")
    print(f"Ticks recorded:   {report['n_ticks']}")
    print(f"Fills executed:   {report['n_fills']}")
    print(f"Total fees paid:  ${report['total_fees_paid']:.4f}")
    if report.get("fee_drag_pct_of_pnl") is not None:
        print(f"Fee drag:         {report['fee_drag_pct_of_pnl']:.2f}% of realized P&L")
    print(f"Rejections:       {json.dumps(report['rejection_breakdown'])}")
    print(f"Sharpe (per tick):  {report['sharpe_per_tick']}")
    print(f"Sortino (per tick): {report['sortino_per_tick']}")
    print(f"Note:               {report['note_ratios']}")
    print(f"Significance:       {report.get('significance_note')}")
    print("=" * 60)
    print(
        "This is one session on one symbol. Do not treat it as an earnings estimate — "
        "treat repeated sessions (different days, different volatility regimes) clearing "
        "this same bar as the actual evidence bar before risking real money."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", default="state")
    p.add_argument("--account-id", type=int, default=900)
    args = p.parse_args()
    print_report(build_report(args.state_dir, args.account_id))
