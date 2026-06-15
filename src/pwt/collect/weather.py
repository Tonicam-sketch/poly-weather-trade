"""Collect weather data into storage.

Forecasts: per-member ensemble Tmax for a horizon of target dates, tagged with the
issue date so the backtest can pick any forecast lead time. Actuals: ERA5 Tmax as
the resolution-truth proxy (swap in station data for production).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from pwt.config import City, get_city, ensemble_models, load_cities
from pwt.datasources.open_meteo import OpenMeteoClient
from pwt.datasources.station import MeteostatClient, Station, find_nearest_station
from pwt.storage.db import Database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_forecasts(
    db: Database,
    *,
    city_keys: list[str] | None = None,
    issue_date: str | None = None,
    horizon_days: int = 5,
    client: OpenMeteoClient | None = None,
) -> int:
    """Pull ensemble forecasts issued on `issue_date` for the next `horizon_days`."""
    client = client or OpenMeteoClient()
    city_keys = city_keys or list(load_cities().keys())
    issue = date.fromisoformat(issue_date) if issue_date else date.today()
    start = issue.isoformat()
    end = (issue + timedelta(days=horizon_days)).isoformat()
    fetched = _now_iso()
    models = ensemble_models()

    total = 0
    for key in city_keys:
        city = get_city(key)
        for model in models:
            parsed = client.ensemble_tmax(
                city.lat, city.lon, start_date=start, end_date=end, model=model, tz=city.tz
            )
            rows = []
            for r in parsed:
                target = date.fromisoformat(r["target_date"])
                rows.append(
                    {
                        "city": key,
                        "model": model,
                        "issue_date": issue.isoformat(),
                        "target_date": r["target_date"],
                        "lead_days": (target - issue).days,
                        "member": r["member"],
                        "tmax_c": r["tmax_c"],
                        "fetched_at": fetched,
                    }
                )
            total += db.upsert_forecast_members(rows)
    return total


def collect_actuals(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    city_keys: list[str] | None = None,
    client: OpenMeteoClient | None = None,
) -> int:
    """Pull ERA5 daily Tmax actuals for the date range (resolution-truth proxy)."""
    client = client or OpenMeteoClient()
    city_keys = city_keys or list(load_cities().keys())
    fetched = _now_iso()

    total = 0
    for key in city_keys:
        city = get_city(key)
        parsed = client.archive_tmax(
            city.lat, city.lon, start_date=start_date, end_date=end_date, tz=city.tz
        )
        rows = [
            {
                "city": key,
                "date": r["date"],
                "tmax_c": r["tmax_c"],
                "source": "era5",
                "fetched_at": fetched,
            }
            for r in parsed
        ]
        total += db.upsert_actuals(rows)
    return total


def resolve_station_id(city: City, stations: list[Station] | None) -> str | None:
    """Verified id if pinned, else the station matching ICAO / nearest to coords."""
    if city.meteostat_id:
        return city.meteostat_id
    if not stations:
        return None
    match = find_nearest_station(stations, city.lat, city.lon, icao=city.icao)
    return match.id if match else None


def collect_station_actuals(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    city_keys: list[str] | None = None,
    client: MeteostatClient | None = None,
) -> dict:
    """Pull official station daily Tmax (resolution truth) via Meteostat.

    Stored under source="station", which the resolver prefers over ERA5.
    """
    client = client or MeteostatClient()
    city_keys = city_keys or list(load_cities().keys())
    fetched = _now_iso()

    # Station metadata is only needed for cities without a pinned meteostat_id.
    need_meta = any(get_city(k).meteostat_id is None for k in city_keys)
    stations = client.stations_meta() if need_meta else None

    total = 0
    resolved: dict[str, str | None] = {}
    for key in city_keys:
        city = get_city(key)
        station_id = resolve_station_id(city, stations)
        resolved[key] = station_id
        if not station_id:
            continue
        parsed = client.daily_tmax(station_id, start_date=start_date, end_date=end_date)
        rows = [
            {
                "city": key,
                "date": r["date"],
                "tmax_c": r["tmax_c"],
                "source": "station",
                "fetched_at": fetched,
            }
            for r in parsed
        ]
        total += db.upsert_actuals(rows)
    return {"rows": total, "stations": resolved}
