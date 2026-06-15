"""Open-Meteo clients: ensemble forecasts and historical actuals.

Three endpoints matter for a weather backtest:

  ensemble-api            current/recent ensemble runs -> per-member forecast.
                          Used live and for the recent forward-test window.
  historical-forecast-api archived past forecasts (what the model said on a
                          past date). Used to backtest the model's skill.
  archive-api (ERA5)      reanalysis actuals. A *proxy* for the resolution
                          truth. NOTE: Polymarket settles on a specific station,
                          not ERA5 — wire a station source (Meteostat/GHCN) into
                          `actuals` for production; ERA5 is the bootstrap.

Parsing is split into pure functions so it is unit-testable without network.
"""

from __future__ import annotations

import re

import pandas as pd

from pwt.datasources.http import HttpClient

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_MEMBER_RE = re.compile(r"member(\d+)")


def parse_ensemble_response(payload: dict, *, model: str) -> list[dict]:
    """Hourly per-member temperatures -> daily Tmax per (member, target_date).

    The API applies the requested timezone, so `time` is already local; we group
    by local calendar date to get each member's daily maximum.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return []
    times = pd.to_datetime(pd.Series(hourly["time"]))
    out: list[dict] = []
    for key, values in hourly.items():
        if not key.startswith("temperature_2m"):
            continue
        if key == "temperature_2m":
            member = 0
        else:
            m = _MEMBER_RE.search(key)
            if not m:
                continue
            member = int(m.group(1))
        series = pd.Series(values, index=times, dtype="float64").dropna()
        if series.empty:
            continue
        daily_max = series.groupby(series.index.date).max()
        for day, tmax in daily_max.items():
            out.append(
                {
                    "model": model,
                    "member": member,
                    "target_date": day.isoformat(),
                    "tmax_c": float(tmax),
                }
            )
    return out


def parse_archive_response(payload: dict) -> list[dict]:
    """archive-api daily block -> [{date, tmax_c}]."""
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return []
    dates = daily["time"]
    tmax = daily.get("temperature_2m_max", [])
    out = []
    for d, t in zip(dates, tmax):
        if t is None:
            continue
        out.append({"date": d, "tmax_c": float(t)})
    return out


class OpenMeteoClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient()

    def ensemble_tmax(
        self, lat: float, lon: float, *, start_date: str, end_date: str, model: str, tz: str
    ) -> list[dict]:
        payload = self.http.get_json(
            ENSEMBLE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "models": model,
                "start_date": start_date,
                "end_date": end_date,
                "timezone": tz,
                "temperature_unit": "celsius",
            },
        )
        return parse_ensemble_response(payload, model=model)

    def historical_forecast_tmax(
        self, lat: float, lon: float, *, start_date: str, end_date: str, tz: str
    ) -> list[dict]:
        """Archived deterministic forecast Tmax (single trajectory)."""
        payload = self.http.get_json(
            HIST_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": tz,
                "temperature_unit": "celsius",
            },
        )
        return parse_archive_response(payload)

    def archive_tmax(
        self, lat: float, lon: float, *, start_date: str, end_date: str, tz: str
    ) -> list[dict]:
        """ERA5 reanalysis Tmax — actuals proxy for resolution/calibration."""
        payload = self.http.get_json(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": tz,
                "temperature_unit": "celsius",
            },
        )
        return parse_archive_response(payload)
