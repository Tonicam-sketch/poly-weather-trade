"""poly-weather-trade: data collection + backtesting for Polymarket weather markets.

Layout:
    datasources/  raw API clients (Open-Meteo, Polymarket)
    collect/      orchestration: pull data into storage
    storage/      SQLite persistence layer
    model/        deterministic quant core (ensemble -> prob, calibration, edge, sizing)
    backtest/     align data, simulate trades, compute metrics
    cli.py        command-line entrypoints
"""

__version__ = "0.1.0"
