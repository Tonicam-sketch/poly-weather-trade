"""Backtest engine.

`run_backtest` is a pure function over an evaluation frame — one row per
(market bin, as-of snapshot) with the raw model probability, the market price,
liquidity, and the realized outcome. It applies calibration, net-edge filtering,
shrinkage + fractional-Kelly sizing, and portfolio risk controls (per-day new
positions, daily-loss circuit breaker, hard drawdown kill-switch), then returns
the settled trades and a performance summary.

`build_evaluations_from_db` assembles that frame from collected data. It is the
data-alignment layer and runs only once real data has been collected; the engine
itself is fully unit-tested on synthetic frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pwt.model.calibration import IsotonicCalibrator
from pwt.model.edge import CostModel, EdgeFilter, net_edge
from pwt.model.ensemble import gaussian_bin_probabilities, normalize
from pwt.model.sizing import SizingPolicy
from pwt.model.units import to_native
from pwt.backtest.metrics import summarize_trades

REQUIRED_COLUMNS = {"market_id", "token_id", "as_of", "target_date", "p_model", "mid", "liquidity", "won"}


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    cost_model: CostModel = field(default_factory=CostModel)
    edge_filter: EdgeFilter = field(default_factory=EdgeFilter)
    sizing: SizingPolicy = field(default_factory=SizingPolicy)
    max_positions_per_day: int = 3
    daily_loss_limit_pct: float = 0.05      # halt new trades after this daily loss
    max_drawdown_stop: float = 0.25         # kill-switch: stop trading entirely
    calibrator: IsotonicCalibrator | None = None


def run_backtest(eval_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing columns: {sorted(missing)}")

    df = eval_df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["day"] = df["as_of"].dt.date
    df = df.sort_values(["as_of", "target_date", "market_id"]).reset_index(drop=True)
    if "ask" not in df.columns:
        df["ask"] = np.nan
    if "data_age_min" not in df.columns:
        df["data_age_min"] = 0.0
    if "liquidity_depth_usdc" not in df.columns:
        df["liquidity_depth_usdc"] = np.nan

    cfg = config
    capital = cfg.initial_capital
    peak_equity = cfg.initial_capital
    realized_by_day: dict = {}
    positions_by_day: dict = {}
    trades: list[dict] = []

    for _, row in df.iterrows():
        day = row["day"]
        drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0.0
        if drawdown >= cfg.max_drawdown_stop:
            break  # kill-switch: capital preservation overrides everything

        day_loss = realized_by_day.get(day, 0.0)
        if day_loss <= -cfg.daily_loss_limit_pct * cfg.initial_capital:
            continue  # daily circuit breaker tripped
        if positions_by_day.get(day, 0) >= cfg.max_positions_per_day:
            continue

        p_raw = float(row["p_model"])
        p = float(cfg.calibrator.transform([p_raw])[0]) if cfg.calibrator else p_raw

        mid = float(row["mid"])
        ask = None if pd.isna(row["ask"]) else float(row["ask"])
        ne = net_edge(p, mid, cfg.cost_model, ask)
        accepted, reason = cfg.edge_filter.accept(
            net_edge_value=ne,
            liquidity=float(row["liquidity"]),
            data_age_min=float(row["data_age_min"]),
        )
        if not accepted:
            continue

        price = cfg.cost_model.executable_ask(mid, ask)
        depth = None if pd.isna(row["liquidity_depth_usdc"]) else float(row["liquidity_depth_usdc"])
        size = cfg.sizing.position_size_usdc(
            p_model=p, price=price, capital=capital, liquidity_depth_usdc=depth
        )
        if size <= 0:
            continue

        won = bool(row["won"])
        pnl = size * ((1.0 - price) / price) if won else -size

        capital += pnl
        peak_equity = max(peak_equity, capital)
        realized_by_day[day] = realized_by_day.get(day, 0.0) + pnl
        positions_by_day[day] = positions_by_day.get(day, 0) + 1

        trades.append(
            {
                "as_of": row["as_of"],
                "day": day,
                "market_id": row["market_id"],
                "token_id": row["token_id"],
                "city": row.get("city"),
                "target_date": row["target_date"],
                "p_model": p,
                "price": price,
                "edge": ne,
                "size_usdc": size,
                "won": won,
                "pnl": pnl,
                "equity_after": capital,
            }
        )

    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df, cfg.initial_capital)
    return trades_df, summary


def build_evaluations_from_db(
    db,
    *,
    lead_days: int = 1,
    variance_inflation: float = 1.0,
    station_precision: int = 1,
) -> pd.DataFrame:
    """Assemble an evaluation frame from collected data (needs a populated DB).

    For each resolved market bin it computes the model probability from the
    ensemble issued `lead_days` before the target date, prices it at the snapshot
    nearest that issue time, and labels it with the realized outcome.
    """
    conn = db.conn
    resolved = pd.read_sql_query(
        """
        SELECT m.market_id, m.city, m.target_date, m.units, m.volume_num,
               r.actual_value_c
        FROM markets m JOIN resolutions r ON r.market_id = m.market_id
        WHERE m.city IS NOT NULL AND m.target_date IS NOT NULL
              AND r.actual_value_c IS NOT NULL
        """,
        conn,
    )
    rows: list[dict] = []
    for _, mk in resolved.iterrows():
        units = mk["units"] or "C"
        actual_native = round(to_native(float(mk["actual_value_c"]), units), station_precision)

        bins = pd.read_sql_query(
            "SELECT token_id, outcome, bin_low, bin_high FROM market_bins WHERE market_id = ?",
            conn,
            params=(mk["market_id"],),
        )
        if bins.empty:
            continue

        members = pd.read_sql_query(
            """
            SELECT tmax_c FROM forecast_tmax
            WHERE city = ? AND target_date = ? AND lead_days = ?
            """,
            conn,
            params=(mk["city"], mk["target_date"], lead_days),
        )
        if members.empty:
            continue
        native_members = to_native(members["tmax_c"].to_numpy(), units)
        mu = float(np.mean(native_members))
        sigma = float(np.std(native_members, ddof=0)) * variance_inflation

        # SQLite NULLs arrive via pandas as NaN; an open-ended bound must be None.
        def _bound(v):
            return None if v is None or pd.isna(v) else float(v)

        bin_bounds = [(_bound(r.bin_low), _bound(r.bin_high)) for r in bins.itertuples()]
        probs = normalize(gaussian_bin_probabilities(mu, sigma, bin_bounds))

        issue_date = (pd.Timestamp(mk["target_date"]) - pd.Timedelta(days=lead_days)).date()
        for ((_, b), (lo, hi)), p_model in zip(zip(bins.iterrows(), bin_bounds), probs):
            price_row = pd.read_sql_query(
                """
                SELECT mid, ask, ts FROM market_prices
                WHERE token_id = ? AND date(ts) <= ?
                ORDER BY ts DESC LIMIT 1
                """,
                conn,
                params=(b["token_id"], str(issue_date)),
            )
            if price_row.empty:
                continue
            won = ((lo is None or actual_native >= lo) and (hi is None or actual_native <= hi))
            rows.append(
                {
                    "market_id": mk["market_id"],
                    "token_id": b["token_id"],
                    "city": mk["city"],
                    "target_date": mk["target_date"],
                    "as_of": f"{issue_date}T12:00:00",
                    "p_model": float(p_model),
                    "mid": float(price_row.iloc[0]["mid"]),
                    "ask": price_row.iloc[0]["ask"],
                    "liquidity": float(mk["volume_num"] or 0.0),
                    "won": 1 if won else 0,
                }
            )
    return pd.DataFrame(rows)
