"""Temperature unit helpers.

Forecasts/actuals are stored in Celsius; market bins are in native units. All
unit crossing goes through here so it is consistent and testable.
"""

from __future__ import annotations


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def to_native(temp_c: float, units: str) -> float:
    return c_to_f(temp_c) if units.upper() == "F" else temp_c


def from_native(temp_native: float, units: str) -> float:
    return f_to_c(temp_native) if units.upper() == "F" else temp_native
