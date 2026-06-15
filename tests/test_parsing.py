from pwt.datasources.open_meteo import parse_archive_response, parse_ensemble_response
from pwt.datasources.polymarket import parse_market, parse_temperature_bin


def test_parse_temperature_bin_range():
    assert parse_temperature_bin("85-86°F") == (85.0, 86.0)
    assert parse_temperature_bin("85 to 86") == (85.0, 86.0)
    assert parse_temperature_bin("85°–86°") == (85.0, 86.0)


def test_parse_temperature_bin_open_ended():
    assert parse_temperature_bin("90°F or above") == (90.0, None)
    assert parse_temperature_bin("Below 32") == (None, 32.0)
    assert parse_temperature_bin("at least 100") == (100.0, None)


def test_parse_temperature_bin_exact_and_empty():
    assert parse_temperature_bin("88°F") == (88.0, 88.0)
    assert parse_temperature_bin("Yes") == (None, None)
    assert parse_temperature_bin(None) == (None, None)


def test_parse_market_pairs_outcomes_and_tokens():
    raw = {
        "conditionId": "0xabc",
        "question": "Highest temperature in NYC on June 15?",
        "outcomes": '["84-85\\u00b0F", "86-87\\u00b0F", "88\\u00b0F or above"]',
        "clobTokenIds": '["t1", "t2", "t3"]',
        "volumeNum": 12345.6,
        "endDate": "2025-06-15T23:59:00Z",
    }
    market, bins = parse_market(raw)
    assert market["market_id"] == "0xabc"
    assert market["volume_num"] == 12345.6
    assert len(bins) == 3
    assert bins[0]["token_id"] == "t1"
    assert bins[0]["bin_low"] == 84.0 and bins[0]["bin_high"] == 85.0
    assert bins[2]["bin_low"] == 88.0 and bins[2]["bin_high"] is None


def test_parse_ensemble_response():
    payload = {
        "hourly": {
            "time": ["2025-06-01T12:00", "2025-06-01T14:00", "2025-06-02T13:00"],
            "temperature_2m": [20.0, 22.0, 19.0],            # control -> member 0
            "temperature_2m_member01": [21.0, 23.0, 18.0],   # member 1
        }
    }
    rows = parse_ensemble_response(payload, model="icon_seamless")
    # 2 members x 2 days = 4 rows
    assert len(rows) == 4
    by = {(r["member"], r["target_date"]): r["tmax_c"] for r in rows}
    assert by[(0, "2025-06-01")] == 22.0  # daily max of control on day 1
    assert by[(1, "2025-06-01")] == 23.0
    assert by[(0, "2025-06-02")] == 19.0
    assert all(r["model"] == "icon_seamless" for r in rows)


def test_parse_archive_response_skips_nulls():
    payload = {"daily": {"time": ["2025-06-01", "2025-06-02"], "temperature_2m_max": [25.0, None]}}
    rows = parse_archive_response(payload)
    assert rows == [{"date": "2025-06-01", "tmax_c": 25.0}]
