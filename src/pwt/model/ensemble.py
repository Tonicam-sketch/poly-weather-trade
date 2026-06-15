"""Turn ensemble members into a probability distribution over market bins.

Two estimators are provided:

  member_bin_probabilities  raw histogram frequency. Simple, but inherits the
                            ensemble's under-dispersion: raw members are too
                            tightly packed, so the modal bin gets an inflated
                            probability. Do NOT trade off this directly.

  gaussian_bin_probabilities fit a Normal(mu, sigma) to the members and integrate
                            over each bin. Combined with a calibrated variance
                            inflation factor (see model.calibration), this is the
                            EMOS-lite estimator that fixes the dispersion bias.

A bin is (low, high) inclusive in the SAME unit as the member values; None means
open-ended on that side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

Bin = tuple[Optional[float], Optional[float]]


def ensemble_mean(members: Sequence[float]) -> float:
    return float(np.mean(members))


def ensemble_spread(members: Sequence[float]) -> float:
    """Population std of the members (the raw, uncalibrated spread)."""
    return float(np.std(members, ddof=0))


def member_bin_probabilities(members: Sequence[float], bins: Sequence[Bin]) -> np.ndarray:
    m = np.asarray(members, dtype="float64")
    n = m.size
    if n == 0:
        return np.zeros(len(bins))
    out = np.empty(len(bins))
    for i, (lo, hi) in enumerate(bins):
        mask = np.ones(n, dtype=bool)
        if lo is not None:
            mask &= m >= lo
        if hi is not None:
            mask &= m <= hi
        out[i] = mask.sum() / n
    return out


def gaussian_bin_probabilities(mu: float, sigma: float, bins: Sequence[Bin]) -> np.ndarray:
    """P(bin) under Normal(mu, sigma). sigma should already be calibration-inflated.

    Bins are treated as half-open [lo, hi) for integration so adjacent bins do
    not double-count the shared edge; open ends use the tail.
    """
    sigma = max(float(sigma), 1e-6)
    out = np.empty(len(bins))
    for i, (lo, hi) in enumerate(bins):
        p_lo = 0.0 if lo is None else norm.cdf(lo, mu, sigma)
        p_hi = 1.0 if hi is None else norm.cdf(hi, mu, sigma)
        out[i] = max(p_hi - p_lo, 0.0)
    return out


def normalize(probs: np.ndarray) -> np.ndarray:
    """Renormalize a probability vector (bins may not perfectly tile the line)."""
    total = float(np.sum(probs))
    if total <= 0:
        return probs
    return probs / total
