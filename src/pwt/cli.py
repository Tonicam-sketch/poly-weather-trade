"""Command-line entrypoints.

    pwt init-db
    pwt status
    pwt collect-weather  --issue-date 2025-06-01 --horizon 5 [--cities nyc,london]
    pwt collect-actuals  --start 2025-06-01 --end 2025-06-30 [--cities ...]
    pwt collect-markets  [--open] [--limit 500] [--no-prices]
    pwt backtest         [--lead-days 1] [--inflation 1.0] [--capital 1000]

Collection commands require network access to Open-Meteo and Polymarket. If a
host is blocked, add it to the environment's egress allowlist:
    ensemble-api.open-meteo.com, historical-forecast-api.open-meteo.com,
    archive-api.open-meteo.com, gamma-api.polymarket.com, clob.polymarket.com
"""

from __future__ import annotations

import argparse
import json

from pwt.storage.db import connect


def _cities(arg: str | None) -> list[str] | None:
    return [c.strip() for c in arg.split(",")] if arg else None


def cmd_init_db(args):
    with connect() as db:
        print(f"DB initialized at {db.conn.execute('PRAGMA database_list').fetchall()[0][2]}")


def cmd_status(args):
    tables = ["forecast_tmax", "actuals", "markets", "market_bins", "market_prices", "resolutions"]
    with connect() as db:
        print("Row counts:")
        for t in tables:
            n = db.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
            print(f"  {t:<16} {n}")


def cmd_collect_weather(args):
    from pwt.collect.weather import collect_forecasts

    with connect() as db:
        n = collect_forecasts(
            db, city_keys=_cities(args.cities), issue_date=args.issue_date, horizon_days=args.horizon
        )
    print(f"upserted {n} forecast member-rows")


def cmd_collect_actuals(args):
    from pwt.collect.weather import collect_actuals

    with connect() as db:
        n = collect_actuals(db, start_date=args.start, end_date=args.end, city_keys=_cities(args.cities))
    print(f"upserted {n} actuals rows")


def cmd_collect_markets(args):
    from pwt.collect.markets import collect_markets

    with connect() as db:
        result = collect_markets(
            db, closed=(None if args.open else True), limit=args.limit, with_prices=not args.no_prices
        )
    print(json.dumps(result, indent=2))


def cmd_backtest(args):
    from pwt.backtest.engine import BacktestConfig, build_evaluations_from_db, run_backtest

    with connect() as db:
        eval_df = build_evaluations_from_db(
            db, lead_days=args.lead_days, variance_inflation=args.inflation
        )
    if eval_df.empty:
        print("No evaluations could be built. Collect weather + markets + actuals first.")
        return
    config = BacktestConfig(initial_capital=args.capital)
    trades, summary = run_backtest(eval_df, config)
    print(f"Evaluations: {len(eval_df)} | Trades: {len(trades)}")
    print(json.dumps(summary, indent=2, default=float))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pwt", description="Polymarket weather backtester")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("status").set_defaults(func=cmd_status)

    cw = sub.add_parser("collect-weather")
    cw.add_argument("--issue-date", default=None, help="forecast run date YYYY-MM-DD (default today)")
    cw.add_argument("--horizon", type=int, default=5, help="forecast horizon in days")
    cw.add_argument("--cities", default=None, help="comma-separated city keys")
    cw.set_defaults(func=cmd_collect_weather)

    ca = sub.add_parser("collect-actuals")
    ca.add_argument("--start", required=True, help="YYYY-MM-DD")
    ca.add_argument("--end", required=True, help="YYYY-MM-DD")
    ca.add_argument("--cities", default=None)
    ca.set_defaults(func=cmd_collect_actuals)

    cm = sub.add_parser("collect-markets")
    cm.add_argument("--open", action="store_true", help="include open markets (default closed only)")
    cm.add_argument("--limit", type=int, default=500)
    cm.add_argument("--no-prices", action="store_true", help="skip CLOB price history")
    cm.set_defaults(func=cmd_collect_markets)

    bt = sub.add_parser("backtest")
    bt.add_argument("--lead-days", type=int, default=1)
    bt.add_argument("--inflation", type=float, default=1.0, help="ensemble variance inflation factor")
    bt.add_argument("--capital", type=float, default=1000.0)
    bt.set_defaults(func=cmd_backtest)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
