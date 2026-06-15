from pwt.collect.weather import collect_station_actuals, resolve_station_id
from pwt.config import get_city
from pwt.datasources.station import (
    Station,
    find_nearest_station,
    haversine_km,
    parse_meteostat_daily_csv,
    parse_stations_meta,
)
from pwt.storage.db import connect


def test_parse_meteostat_daily_csv():
    text = "\n".join(
        [
            "2025-06-01,18.0,12.0,25.5,0.0,,,,,,",
            "2025-06-02,,,,,,,,,,",          # missing tmax -> skipped
            "2025-06-03,19.0,13.0,27.1,0.0,,,,,,",
            "",                               # blank -> skipped
        ]
    )
    rows = parse_meteostat_daily_csv(text)
    assert rows == [
        {"date": "2025-06-01", "tmax_c": 25.5},
        {"date": "2025-06-03", "tmax_c": 27.1},
    ]


def test_parse_stations_meta():
    raw = (
        '[{"id":"72503","name":{"en":"New York"},'
        '"location":{"latitude":40.78,"longitude":-73.97},'
        '"identifiers":{"icao":"KNYC","wmo":"72506"}},'
        '{"id":"00000","name":{"en":"NoLoc"},"location":{},"identifiers":{}}]'
    )
    stations = parse_stations_meta(raw)
    assert len(stations) == 1  # second has no location -> dropped
    assert stations[0].id == "72503"
    assert stations[0].icao == "KNYC"


def test_haversine_known_distance():
    # NYC to LA ~ 3936 km
    d = haversine_km(40.71, -74.01, 34.05, -118.24)
    assert 3900 < d < 4000


def test_find_nearest_station_prefers_icao():
    stations = [
        Station(id="far", lat=0.0, lon=0.0, icao="ZZZZ"),
        Station(id="knyc", lat=10.0, lon=10.0, icao="KNYC"),
    ]
    # even though "far" is closer to (0,0), the ICAO match wins
    s = find_nearest_station(stations, 0.0, 0.0, icao="KNYC")
    assert s.id == "knyc"


def test_find_nearest_station_falls_back_to_distance():
    stations = [
        Station(id="a", lat=40.0, lon=-74.0),
        Station(id="b", lat=51.0, lon=0.0),
    ]
    s = find_nearest_station(stations, 40.7, -73.9)
    assert s.id == "a"


def test_resolve_station_id_uses_pinned_id():
    city = get_city("nyc")
    pinned = city.__class__(**{**city.__dict__, "meteostat_id": "99999"})
    assert resolve_station_id(pinned, stations=None) == "99999"


class _FakeMeteostat:
    """Offline stand-in for MeteostatClient."""

    def __init__(self):
        self.requested_ids = []

    def stations_meta(self):
        return [
            Station(id="nyc1", lat=40.78, lon=-73.97, icao="KNYC"),
            Station(id="lon1", lat=51.48, lon=-0.45, icao="EGLL"),
        ]

    def daily_tmax(self, station_id, *, start_date=None, end_date=None):
        self.requested_ids.append(station_id)
        return [{"date": "2025-06-01", "tmax_c": 26.0}, {"date": "2025-06-02", "tmax_c": 27.5}]


def test_collect_station_actuals_end_to_end():
    fake = _FakeMeteostat()
    with connect(":memory:") as db:
        result = collect_station_actuals(
            db, start_date="2025-06-01", end_date="2025-06-02", city_keys=["nyc"], client=fake
        )
        assert result["rows"] == 2
        assert result["stations"]["nyc"] == "nyc1"  # matched by ICAO KNYC
        stored = db.query("SELECT source, tmax_c FROM actuals WHERE city='nyc' ORDER BY date")
    assert [r["source"] for r in stored] == ["station", "station"]
    assert stored[0]["tmax_c"] == 26.0
