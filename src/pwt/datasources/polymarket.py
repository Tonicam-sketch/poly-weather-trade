"""Polymarket clients: Gamma (market metadata) and CLOB (price history / book).

The Gamma market object encodes outcomes as parallel JSON-encoded string lists
(`outcomes`, `clobTokenIds`, `outcomePrices`). For temperature-range weather
markets the numeric bin lives in the outcome label, so `parse_temperature_bin`
turns labels like "85-86°F", "90°F or above", "Below 32°" into numeric bounds
in the market's native unit. Getting this parse right is load-bearing: a
mis-mapped bin makes the model think it has edge on the wrong token.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pwt.datasources.http import HttpClient

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com"

# A degree number, optionally signed/decimal.
_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_temperature_bin(label: str) -> tuple[Optional[float], Optional[float]]:
    """Parse an outcome label into (low, high) inclusive bounds in native units.

    Returns (None, None) if no numbers are found (e.g. a plain "Yes"/"No" — the
    bin bounds for binary markets are derived from the question instead).
    Open-ended bins return one None: "90 or above" -> (90, None),
    "below 32" -> (None, 32).
    """
    if label is None:
        return (None, None)
    text = label.strip().lower().replace("–", "-").replace("—", "-")
    text = text.replace("°f", "").replace("°c", "").replace("°", "").replace("℉", "").replace("℃", "")

    # Range: "85-86", "85 to 86"
    m = re.search(rf"{_NUM}\s*(?:-|to)\s*{_NUM}", text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))

    nums = re.findall(_NUM, text)
    if not nums:
        return (None, None)
    val = float(nums[0])

    if any(w in text for w in ("above", "over", "greater", "at least", "or more", "or higher", ">=", ">", "≥")):
        return (val, None)
    if any(w in text for w in ("below", "under", "less", "at most", "or fewer", "or lower", "<=", "<", "≤")):
        return (None, val)
    # Single number with no qualifier: treat as the exact-degree bin [val, val].
    return (val, val)


def _json_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


def parse_market(raw: dict) -> tuple[dict, list[dict]]:
    """Gamma market dict -> (market_row, [bin_rows]) for storage (sans fetched_at)."""
    market_id = raw.get("conditionId") or raw.get("id")
    outcomes = _json_list(raw.get("outcomes"))
    token_ids = _json_list(raw.get("clobTokenIds"))

    market_row = {
        "market_id": market_id,
        "city": None,  # filled by the collector via question matching
        "target_date": None,
        "question": raw.get("question"),
        "units": "F",  # refined by collector from question / city
        "end_date": raw.get("endDate"),
        "volume_num": float(raw.get("volumeNum") or 0.0),
        "raw_json": json.dumps(raw, separators=(",", ":")),
    }

    bins: list[dict] = []
    for outcome, token_id in zip(outcomes, token_ids):
        lo, hi = parse_temperature_bin(str(outcome))
        bins.append(
            {
                "token_id": str(token_id),
                "market_id": market_id,
                "outcome": str(outcome),
                "bin_low": lo,
                "bin_high": hi,
            }
        )
    return market_row, bins


class GammaClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient()

    def weather_markets(self, *, closed: bool | None = None, limit: int = 500) -> list[dict]:
        """Page through weather-tagged markets. closed=True for backtest history."""
        out: list[dict] = []
        offset = 0
        page = 100
        while len(out) < limit:
            params = {"tag": "weather", "limit": page, "offset": offset}
            if closed is not None:
                params["closed"] = str(closed).lower()
            batch = self.http.get_json(GAMMA_URL, params=params)
            if not isinstance(batch, list) or not batch:
                break
            out.extend(batch)
            if len(batch) < page:
                break
            offset += page
        return out[:limit]


class ClobClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient(base_url=CLOB_URL)

    def price_history(self, token_id: str, *, interval: str = "max", fidelity: int = 60) -> list[dict]:
        """Midpoint price ticks: [{ts: unix_seconds, mid: price}]."""
        payload = self.http.get_json(
            "/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": fidelity},
        )
        history = payload.get("history", []) if isinstance(payload, dict) else []
        return [{"ts": int(h["t"]), "mid": float(h["p"])} for h in history if "t" in h and "p" in h]

    def book(self, token_id: str) -> dict:
        """Live order book: {bids: [...], asks: [...]} — for live trading/depth."""
        return self.http.get_json("/book", params={"token_id": token_id})
