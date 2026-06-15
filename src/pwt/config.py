"""Configuration loading: city registry and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = Path(os.environ.get("PWT_DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.environ.get("PWT_DB_PATH", DATA_DIR / "pwt.sqlite"))


@dataclass(frozen=True)
class City:
    key: str
    name: str
    lat: float
    lon: float
    tz: str
    units: str  # "F" or "C" — the unit the market resolves in
    station: str
    icao: str | None = None          # resolution station ICAO (hint for matching)
    meteostat_id: str | None = None  # verified Meteostat station id (override)

    @property
    def is_fahrenheit(self) -> bool:
        return self.units.upper() == "F"


@lru_cache(maxsize=1)
def _raw_config() -> dict:
    with open(CONFIG_DIR / "cities.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_cities() -> dict[str, City]:
    raw = _raw_config()["cities"]
    return {
        key: City(
            key=key,
            name=c["name"],
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            tz=c["tz"],
            units=c.get("units", "C"),
            station=c.get("station", ""),
            icao=c.get("icao"),
            meteostat_id=c.get("meteostat_id"),
        )
        for key, c in raw.items()
    }


def get_city(key: str) -> City:
    cities = load_cities()
    if key not in cities:
        raise KeyError(f"unknown city '{key}'. known: {sorted(cities)}")
    return cities[key]


def ensemble_models() -> list[str]:
    return list(_raw_config().get("ensemble_models", ["icon_seamless"]))


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
