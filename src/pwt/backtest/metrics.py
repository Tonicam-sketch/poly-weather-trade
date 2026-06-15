"""Performance metrics for a sequence of settled trades.

The headline numbers that decide whether the strategy is worth real capital:
returns, risk-adjusted return (Sharpe), and the drawdown that the risk controls
are supposed to bound (max drawdown, Calmar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(pnls: np.ndarray, initial_capital: float) -> np.ndarray:
    return initial_capital + np.cumsum(np.asarray(pnls, dtype="float64"))


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline as a positive fraction of the peak."""
    equity = np.asarray(equity, dtype="float64")
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (running_peak - equity) / running_peak
    return float(np.max(drawdowns))


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe of a per-period return series (risk-free = 0)."""
    returns = np.asarray(returns, dtype="float64")
    if returns.size < 2:
        return 0.0
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def summarize_trades(trades: pd.DataFrame, initial_capital: float) -> dict:
    """Aggregate a trades frame (needs columns: pnl, size_usdc, edge, won, day).

    `day` groups trades into return periods for the Sharpe calculation.
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "return_pct": 0.0,
            "win_rate": float("nan"),
            "avg_edge": float("nan"),
            "profit_factor": float("nan"),
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "final_equity": initial_capital,
        }

    pnls = trades["pnl"].to_numpy()
    equity = equity_curve(pnls, initial_capital)
    wins = trades["won"].astype(bool)
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()

    daily = trades.groupby("day")["pnl"].sum()
    daily_returns = daily.to_numpy() / initial_capital

    total_pnl = float(pnls.sum())
    return {
        "n_trades": int(len(trades)),
        "total_pnl": total_pnl,
        "return_pct": total_pnl / initial_capital * 100.0,
        "win_rate": float(wins.mean()),
        "avg_edge": float(trades["edge"].mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(daily_returns),
        "final_equity": float(equity[-1]),
    }
