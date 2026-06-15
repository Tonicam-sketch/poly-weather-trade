"""SQLite storage layer.

A single relational store keeps the backtest reproducible: every forecast,
actual, market, price tick and resolution is timestamped and queryable. This
replaces the original strategy's pile of concurrently-written JSON files, which
is prone to races and silent account drift.

Schema (all temperatures stored in Celsius internally; convert at the market
boundary using the city's `units`):

  forecast_tmax   per-ensemble-member predicted daily Tmax, keyed by the run
                  that produced it (issue_date) and the day it predicts
                  (target_date). lead_days = target_date - issue_date.
  actuals         observed daily Tmax per city/date (resolution truth).
  markets         Polymarket weather markets (one row per market/condition).
  market_bins     outcome tokens of a market, with parsed numeric bin bounds.
  market_prices   price ticks per outcome token over time (mid/bid/ask).
  resolutions     final settled outcome per market.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pwt.config import DB_PATH, ensure_data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_tmax (
    city        TEXT NOT NULL,
    model       TEXT NOT NULL,
    issue_date  TEXT NOT NULL,   -- YYYY-MM-DD, the forecast run date (as-of)
    target_date TEXT NOT NULL,   -- YYYY-MM-DD, the day predicted
    lead_days   INTEGER NOT NULL,
    member      INTEGER NOT NULL,
    tmax_c      REAL NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (city, model, issue_date, target_date, member)
);

CREATE TABLE IF NOT EXISTS actuals (
    city       TEXT NOT NULL,
    date       TEXT NOT NULL,    -- YYYY-MM-DD
    tmax_c     REAL NOT NULL,
    source     TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (city, date, source)
);

CREATE TABLE IF NOT EXISTS markets (
    market_id   TEXT PRIMARY KEY,  -- condition_id
    city        TEXT,
    target_date TEXT,
    question    TEXT,
    units       TEXT,
    end_date    TEXT,
    volume_num  REAL,
    raw_json    TEXT,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_bins (
    token_id     TEXT PRIMARY KEY,
    market_id    TEXT NOT NULL,
    outcome      TEXT,
    -- Bounds are in the market's NATIVE unit (see markets.units), inclusive,
    -- because rounding/resolution happens in native degrees. NULL = open-ended.
    bin_low      REAL,
    bin_high     REAL,
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

CREATE TABLE IF NOT EXISTS market_prices (
    token_id   TEXT NOT NULL,
    ts         TEXT NOT NULL,     -- ISO8601 UTC
    mid        REAL,
    bid        REAL,
    ask        REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (token_id, ts)
);

CREATE TABLE IF NOT EXISTS resolutions (
    market_id        TEXT PRIMARY KEY,
    winning_token_id TEXT,
    actual_value_c   REAL,
    resolved_at      TEXT,
    fetched_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forecast_target ON forecast_tmax(city, target_date);
CREATE INDEX IF NOT EXISTS idx_bins_market ON market_bins(market_id);
CREATE INDEX IF NOT EXISTS idx_prices_ts ON market_prices(token_id, ts);
"""


class Database:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_forecast_members(self, rows: list[dict]) -> int:
        sql = """
        INSERT OR REPLACE INTO forecast_tmax
            (city, model, issue_date, target_date, lead_days, member, tmax_c, fetched_at)
        VALUES (:city, :model, :issue_date, :target_date, :lead_days, :member, :tmax_c, :fetched_at)
        """
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur.rowcount

    def upsert_actuals(self, rows: list[dict]) -> int:
        sql = """
        INSERT OR REPLACE INTO actuals (city, date, tmax_c, source, fetched_at)
        VALUES (:city, :date, :tmax_c, :source, :fetched_at)
        """
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur.rowcount

    def upsert_market(self, row: dict) -> None:
        sql = """
        INSERT OR REPLACE INTO markets
            (market_id, city, target_date, question, units, end_date, volume_num, raw_json, fetched_at)
        VALUES (:market_id, :city, :target_date, :question, :units, :end_date, :volume_num, :raw_json, :fetched_at)
        """
        self.conn.execute(sql, row)
        self.conn.commit()

    def upsert_bins(self, rows: list[dict]) -> int:
        sql = """
        INSERT OR REPLACE INTO market_bins
            (token_id, market_id, outcome, bin_low, bin_high)
        VALUES (:token_id, :market_id, :outcome, :bin_low, :bin_high)
        """
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur.rowcount

    def upsert_prices(self, rows: list[dict]) -> int:
        sql = """
        INSERT OR REPLACE INTO market_prices (token_id, ts, mid, bid, ask, fetched_at)
        VALUES (:token_id, :ts, :mid, :bid, :ask, :fetched_at)
        """
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur.rowcount

    def upsert_resolution(self, row: dict) -> None:
        sql = """
        INSERT OR REPLACE INTO resolutions
            (market_id, winning_token_id, actual_value_c, resolved_at, fetched_at)
        VALUES (:market_id, :winning_token_id, :actual_value_c, :resolved_at, :fetched_at)
        """
        self.conn.execute(sql, row)
        self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()


@contextmanager
def connect(path: Path | str | None = None) -> Iterator[Database]:
    if path is None:
        ensure_data_dir()
        path = DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        db = Database(conn)
        db.init_schema()
        yield db
    finally:
        conn.close()
