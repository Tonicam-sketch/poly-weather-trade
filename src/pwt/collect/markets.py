"""Collect Polymarket weather markets, price history and resolutions.

City and target-date are inferred from the market question (best-effort) so the
backtest can join a market to its forecast and actual. Resolution truth is taken
from the `actuals` table (collect weather actuals first).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from pwt.config import load_cities
from pwt.datasources.polymarket import ClobClient, GammaClient, parse_market
from pwt.storage.db import Database

_EXTRA_ALIASES = {
    "nyc": ["new york", "nyc", "manhattan"],
    "toronto": ["toronto"],
    "london": ["london"],
    "shanghai": ["shanghai"],
    "seoul": ["seoul"],
    "tokyo": ["tokyo"],
    "singapore": ["singapore"],
    "shenzhen": ["shenzhen"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def match_city(question: str) -> str | None:
    q = (question or "").lower()
    cities = load_cities()
    for key in cities:
        aliases = _EXTRA_ALIASES.get(key, []) + [key, cities[key].name.lower()]
        if any(a in q for a in aliases):
            return key
    return None


def parse_target_date(question: str, end_date: str | None) -> str | None:
    """Best-effort target date: a date in the question, else the market end date."""
    q = question or ""
    m = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})",
        q,
        re.IGNORECASE,
    )
    if m:
        try:
            year = datetime.now(timezone.utc).year
            dt = pd.to_datetime(f"{m.group(1)} {m.group(2)} {year}")
            return dt.date().isoformat()
        except (ValueError, TypeError):
            pass
    if end_date:
        try:
            return pd.to_datetime(end_date).date().isoformat()
        except (ValueError, TypeError):
            return None
    return None


def collect_markets(
    db: Database,
    *,
    closed: bool | None = True,
    limit: int = 500,
    with_prices: bool = True,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
) -> dict:
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()
    fetched = _now_iso()
    cities = load_cities()

    raw_markets = gamma.weather_markets(closed=closed, limit=limit)
    n_markets = n_bins = n_prices = n_resolutions = 0

    for raw in raw_markets:
        market_row, bins = parse_market(raw)
        city = match_city(market_row["question"])
        market_row["city"] = city
        market_row["units"] = cities[city].units if city else "F"
        market_row["target_date"] = parse_target_date(market_row["question"], market_row["end_date"])
        market_row["fetched_at"] = fetched
        db.upsert_market(market_row)
        n_markets += 1
        n_bins += db.upsert_bins(bins)

        if with_prices:
            for b in bins:
                ticks = clob.price_history(b["token_id"])
                price_rows = [
                    {
                        "token_id": b["token_id"],
                        "ts": datetime.fromtimestamp(t["ts"], tz=timezone.utc).isoformat(),
                        "mid": t["mid"],
                        "bid": None,
                        "ask": None,
                        "fetched_at": fetched,
                    }
                    for t in ticks
                ]
                n_prices += db.upsert_prices(price_rows)

        # Resolution truth from the actuals table (collect actuals first).
        if city and market_row["target_date"]:
            actual = db.query(
                "SELECT tmax_c FROM actuals WHERE city = ? AND date = ? ORDER BY source LIMIT 1",
                (city, market_row["target_date"]),
            )
            if actual:
                db.upsert_resolution(
                    {
                        "market_id": market_row["market_id"],
                        "winning_token_id": None,
                        "actual_value_c": float(actual[0]["tmax_c"]),
                        "resolved_at": market_row["end_date"],
                        "fetched_at": fetched,
                    }
                )
                n_resolutions += 1

    return {
        "markets": n_markets,
        "bins": n_bins,
        "prices": n_prices,
        "resolutions": n_resolutions,
    }
