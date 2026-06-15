import numpy as np
import pandas as pd
import pytest

from pwt.backtest.engine import BacktestConfig, build_evaluations_from_db, run_backtest
from pwt.backtest.metrics import equity_curve, max_drawdown, sharpe_ratio, summarize_trades
from pwt.model.edge import EdgeFilter
from pwt.model.sizing import SizingPolicy
from pwt.storage.db import connect


def _row(**kw):
    base = dict(
        market_id="m",
        token_id="t",
        city="nyc",
        target_date="2025-06-15",
        as_of="2025-06-14T12:00:00",
        p_model=0.6,
        mid=0.5,
        liquidity=10000.0,
        won=1,
    )
    base.update(kw)
    return base


# ----- metrics -----
def test_max_drawdown():
    eq = np.array([100, 120, 90, 110, 80])
    # peak 120 -> trough 80 => (120-80)/120
    assert np.isclose(max_drawdown(eq), 40 / 120)


def test_sharpe_zero_variance():
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0


def test_equity_curve():
    assert list(equity_curve([10, -5, 20], 100)) == [110, 105, 125]


def test_summarize_empty():
    s = summarize_trades(pd.DataFrame(), 1000)
    assert s["n_trades"] == 0 and s["final_equity"] == 1000


# ----- engine -----
def test_missing_columns_raises():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame({"market_id": ["m"]}), BacktestConfig())


def test_winning_trades_make_money():
    df = pd.DataFrame([_row(token_id=f"t{i}", market_id=f"m{i}", as_of=f"2025-06-{10+i}T12:00:00", won=1) for i in range(5)])
    trades, summary = run_backtest(df, BacktestConfig(initial_capital=1000))
    assert summary["n_trades"] == 5
    assert summary["total_pnl"] > 0
    assert summary["final_equity"] > 1000
    assert summary["win_rate"] == 1.0


def test_no_edge_no_trade():
    # p_model == mid, after costs the net edge is negative -> filtered out
    df = pd.DataFrame([_row(p_model=0.5, mid=0.5)])
    trades, summary = run_backtest(df, BacktestConfig())
    assert summary["n_trades"] == 0


def test_alarm_edge_rejected():
    # huge paper edge -> treated as data bug, not a trade
    df = pd.DataFrame([_row(p_model=0.95, mid=0.30)])
    trades, summary = run_backtest(df, BacktestConfig())
    assert summary["n_trades"] == 0


def test_daily_loss_circuit_breaker():
    # 10 losers on the SAME day; breaker should halt before all execute
    df = pd.DataFrame([_row(token_id=f"t{i}", market_id=f"m{i}", won=0) for i in range(10)])
    cfg = BacktestConfig(
        initial_capital=1000,
        daily_loss_limit_pct=0.05,
        max_positions_per_day=100,  # isolate the loss breaker
    )
    trades, summary = run_backtest(df, cfg)
    assert 0 < summary["n_trades"] < 10
    assert summary["total_pnl"] < 0


def test_max_positions_per_day_cap():
    df = pd.DataFrame([_row(token_id=f"t{i}", market_id=f"m{i}", won=1) for i in range(10)])
    cfg = BacktestConfig(max_positions_per_day=2, daily_loss_limit_pct=1.0)
    trades, summary = run_backtest(df, cfg)
    assert summary["n_trades"] == 2


def test_drawdown_kill_switch():
    # losers spread across distinct days; kill-switch stops once dd >= threshold
    rows = [_row(token_id=f"t{i}", market_id=f"m{i}", as_of=f"2025-{6:02d}-{(i % 27) + 1:02d}T12:00:00", won=0) for i in range(60)]
    df = pd.DataFrame(rows)
    cfg = BacktestConfig(
        initial_capital=1000,
        max_drawdown_stop=0.10,
        daily_loss_limit_pct=1.0,
        max_positions_per_day=100,
        sizing=SizingPolicy(kelly_fraction=1.0, shrink_weight=1.0, max_position_pct=0.10),
        edge_filter=EdgeFilter(min_edge=0.03, alarm_edge=0.20),
    )
    trades, summary = run_backtest(df, cfg)
    assert summary["n_trades"] < 60  # stopped early
    # never lost dramatically more than the stop threshold
    assert summary["final_equity"] > 1000 * (1 - 0.10) - 100


# ----- DB integration: build evaluation frame end to end -----
def test_build_evaluations_from_db():
    with connect(":memory:") as db:
        fetched = "2025-06-14T00:00:00Z"
        db.upsert_market(
            {
                "market_id": "m1",
                "city": "nyc",
                "target_date": "2025-06-15",
                "question": "High temp NYC June 15",
                "units": "F",
                "end_date": "2025-06-15T23:59:00Z",
                "volume_num": 10000.0,
                "raw_json": "{}",
                "fetched_at": fetched,
            }
        )
        db.upsert_bins(
            [
                {"token_id": "t1", "market_id": "m1", "outcome": "84-85F", "bin_low": 84.0, "bin_high": 85.0},
                {"token_id": "t2", "market_id": "m1", "outcome": "86-87F", "bin_low": 86.0, "bin_high": 87.0},
                {"token_id": "t3", "market_id": "m1", "outcome": "88F+", "bin_low": 88.0, "bin_high": None},
            ]
        )
        # ensemble members centered ~30C == ~86F, issued one day before target
        members = np.linspace(28.0, 32.0, 21)
        db.upsert_forecast_members(
            [
                {
                    "city": "nyc", "model": "icon", "issue_date": "2025-06-14",
                    "target_date": "2025-06-15", "lead_days": 1, "member": i,
                    "tmax_c": float(v), "fetched_at": fetched,
                }
                for i, v in enumerate(members)
            ]
        )
        db.upsert_prices(
            [
                {"token_id": tid, "ts": "2025-06-14T12:00:00+00:00", "mid": mid, "bid": None, "ask": None, "fetched_at": fetched}
                for tid, mid in [("t1", 0.30), ("t2", 0.30), ("t3", 0.20)]
            ]
        )
        # actual high = 30C -> 86F -> bin t2 wins
        db.upsert_resolution(
            {"market_id": "m1", "winning_token_id": None, "actual_value_c": 30.0, "resolved_at": "2025-06-15", "fetched_at": fetched}
        )

        eval_df = build_evaluations_from_db(db, lead_days=1, variance_inflation=1.0)

    assert not eval_df.empty
    assert set(eval_df["token_id"]) == {"t1", "t2", "t3"}
    won = dict(zip(eval_df["token_id"], eval_df["won"]))
    assert won["t2"] == 1 and won["t1"] == 0 and won["t3"] == 0
    # probabilities are valid, and the bin straddling the forecast mean (t2,
    # 86-87F vs members centered on 86F) carries real mass
    p = dict(zip(eval_df["token_id"], eval_df["p_model"]))
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert p["t2"] > 0.1
