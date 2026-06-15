"""Station observations — the actual resolution truth for weather markets.

Polymarket weather markets settle on a SPECIFIC station's official daily
max/min (e.g. KNYC Central Park), not on a model grid point or reanalysis.
Aligning to that station is the single highest-leverage correctness fix: you can
be right about the weather and wrong about the contract if you settle against the
wrong source.

This client pulls daily station data from Meteostat's bulk endpoint, a uniform
global source of station TMAX. A station is resolved either by an explicit
Meteostat id (preferred — pin the verified resolution station in config) or by
nearest station to the city's configured coordinates.

All parsing is pure and unit-tested; only fetching needs network.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass

from pwt.datasources.http import HttpClient

BULK_URL = "https://bulk.meteostat.net"

# Meteostat daily CSV is headerless; these are the column positions we use.
_COL_DATE = 0
_COL_TMAX = 3


@dataclass(frozen=True)
class Station:
    id: str
    lat: float
    lon: float
    name: str = ""
    icao: str | None = None
    wmo: str | None = None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_stations_meta(raw_json: bytes | str) -> list[Station]:
    """Parse Meteostat stations metadata (lite.json) into Station records."""
    data = json.loads(raw_json)
    out: list[Station] = []
    for s in data:
        loc = s.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue
        ident = s.get("identifiers") or {}
        name = s.get("name") or {}
        out.append(
            Station(
                id=str(s.get("id")),
                lat=float(lat),
                lon=float(lon),
                name=name.get("en", "") if isinstance(name, dict) else str(name),
                icao=ident.get("icao"),
                wmo=ident.get("wmo"),
            )
        )
    return out


def find_nearest_station(
    stations: list[Station], lat: float, lon: float, *, icao: str | None = None
) -> Station | None:
    """Pick the station matching `icao` if given, else the geographically nearest."""
    if not stations:
        return None
    if icao:
        for s in stations:
            if s.icao and s.icao.upper() == icao.upper():
                return s
    return min(stations, key=lambda s: haversine_km(lat, lon, s.lat, s.lon))


def parse_meteostat_daily_csv(text: str) -> list[dict]:
    """Headerless Meteostat daily CSV -> [{date, tmax_c}] (skips missing TMAX)."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        if len(cols) <= _COL_TMAX:
            continue
        tmax = cols[_COL_TMAX].strip()
        if not tmax:
            continue
        try:
            out.append({"date": cols[_COL_DATE].strip(), "tmax_c": float(tmax)})
        except ValueError:
            continue
    return out


class MeteostatClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient(base_url=BULK_URL)

    def stations_meta(self) -> list[Station]:
        raw = self.http.get_bytes("/v2/stations/lite.json.gz")
        return parse_stations_meta(gzip.decompress(raw))

    def daily_tmax(
        self, station_id: str, *, start_date: str | None = None, end_date: str | None = None
    ) -> list[dict]:
        raw = self.http.get_bytes(f"/v2/daily/{station_id}.csv.gz")
        rows = parse_meteostat_daily_csv(gzip.decompress(raw).decode("utf-8"))
        if start_date:
            rows = [r for r in rows if r["date"] >= start_date]
        if end_date:
            rows = [r for r in rows if r["date"] <= end_date]
        return rows
