"""Probability calibration — the step that decides whether the edge is real.

Raw ensemble frequencies are not calibrated probabilities. Two corrections:

1. Variance inflation (dispersion fix). Standardized errors
   z = (actual - mu) / sigma should be ~N(0,1). Under-dispersed ensembles give
   std(z) > 1, so we inflate sigma by `fit_variance_inflation` before turning the
   forecast into bin probabilities.

2. Probability calibration (reliability fix). Even after dispersion correction,
   predicted probabilities may not match observed frequencies. Isotonic
   regression learns a monotone map p_raw -> p_calibrated from history.

`brier_score` and `reliability_curve` quantify how well-calibrated we are and
feed the live drift kill-switch.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_variance_inflation(means: np.ndarray, sigmas: np.ndarray, actuals: np.ndarray) -> float:
    """Return factor c such that sigma*c makes standardized errors unit-variance.

    c = std((actual - mu) / sigma). c > 1 means the raw ensemble was too tight.
    """
    means = np.asarray(means, dtype="float64")
    sigmas = np.asarray(sigmas, dtype="float64")
    actuals = np.asarray(actuals, dtype="float64")
    valid = sigmas > 1e-9
    if valid.sum() < 2:
        return 1.0
    z = (actuals[valid] - means[valid]) / sigmas[valid]
    c = float(np.sqrt(np.mean(z**2)))
    return max(c, 1e-6)


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted prob and {0,1} outcome. Lower better."""
    probs = np.asarray(probs, dtype="float64")
    outcomes = np.asarray(outcomes, dtype="float64")
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - outcomes) ** 2))


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Bucket predictions and compare predicted vs observed frequency.

    Returns (bin_centers, observed_freq, counts). A well-calibrated model has
    observed_freq ~= bin_centers.
    """
    probs = np.asarray(probs, dtype="float64")
    outcomes = np.asarray(outcomes, dtype="float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, obs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        c = int(mask.sum())
        centers.append((lo + hi) / 2.0)
        counts.append(c)
        obs.append(float(np.mean(outcomes[mask])) if c else float("nan"))
    return np.array(centers), np.array(obs), np.array(counts)


class IsotonicCalibrator:
    """Monotone probability calibration learned from (predicted, outcome) pairs."""

    def __init__(self):
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(np.asarray(probs, dtype="float64"), np.asarray(outcomes, dtype="float64"))
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.asarray(probs, dtype="float64")
        return self._iso.predict(np.asarray(probs, dtype="float64"))
