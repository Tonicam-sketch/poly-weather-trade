import numpy as np

from pwt.model.calibration import (
    IsotonicCalibrator,
    brier_score,
    fit_variance_inflation,
    reliability_curve,
)
from pwt.model.edge import CostModel, EdgeFilter, net_edge
from pwt.model.ensemble import (
    ensemble_mean,
    ensemble_spread,
    gaussian_bin_probabilities,
    member_bin_probabilities,
    normalize,
)
from pwt.model.sizing import SizingPolicy, full_kelly_fraction, shrink_probability
from pwt.model.units import c_to_f, f_to_c, from_native, to_native


# ----- units -----
def test_unit_roundtrip():
    assert c_to_f(0) == 32
    assert c_to_f(100) == 212
    assert abs(f_to_c(c_to_f(21.7)) - 21.7) < 1e-9
    assert to_native(0, "F") == 32
    assert to_native(10, "C") == 10
    assert abs(from_native(50, "F") - 10.0) < 1e-9


# ----- ensemble -----
def test_member_bin_probabilities_basic():
    members = [10, 11, 12, 12, 13]  # 5 members
    bins = [(None, 11), (11, 12), (12, None)]  # <=11, 11-12, >=12
    p = member_bin_probabilities(members, bins)
    # <=11: 10,11 -> 2/5 ; 11-12: 11,12,12 -> 3/5 ; >=12: 12,12,13 -> 3/5 (overlap by design)
    assert np.isclose(p[0], 2 / 5)
    assert np.isclose(p[2], 3 / 5)


def test_member_bin_probabilities_empty():
    assert member_bin_probabilities([], [(0, 1)]).sum() == 0.0


def test_gaussian_bin_probabilities_sums_to_one_over_full_partition():
    bins = [(None, 20), (20, 25), (25, 30), (30, None)]
    p = gaussian_bin_probabilities(mu=25, sigma=3, bins=bins)
    assert np.isclose(p.sum(), 1.0, atol=1e-6)
    # symmetric around mu=25 with this partition's middle split at 25
    assert np.isclose(p[0] + p[1], p[2] + p[3], atol=1e-6)


def test_gaussian_handles_zero_sigma():
    p = gaussian_bin_probabilities(mu=25, sigma=0.0, bins=[(24, 26), (26, None)])
    assert p[0] > 0.99


def test_spread_and_mean():
    assert ensemble_mean([1, 2, 3]) == 2.0
    assert ensemble_spread([1, 1, 1]) == 0.0


def test_normalize():
    assert np.isclose(normalize(np.array([1.0, 1.0, 2.0])).sum(), 1.0)
    assert normalize(np.array([0.0, 0.0])).sum() == 0.0


# ----- calibration -----
def test_variance_inflation_detects_underdispersion():
    rng = np.random.default_rng(0)
    n = 5000
    means = np.zeros(n)
    sigmas = np.ones(n)  # forecast claims sigma=1
    actuals = rng.normal(0, 2, n)  # truth has sigma=2 -> under-dispersed by 2x
    c = fit_variance_inflation(means, sigmas, actuals)
    assert 1.8 < c < 2.2


def test_variance_inflation_degenerate():
    assert fit_variance_inflation(np.array([0.0]), np.array([0.0]), np.array([0.0])) == 1.0


def test_brier_score():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert np.isclose(brier_score([0.5, 0.5], [1, 0]), 0.25)


def test_reliability_curve_shapes():
    p = np.array([0.05, 0.15, 0.95, 0.85])
    y = np.array([0, 0, 1, 1])
    centers, obs, counts = reliability_curve(p, y, n_bins=10)
    assert len(centers) == 10
    assert counts.sum() == 4


def test_isotonic_calibrator_monotone():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p**2).astype(int)  # miscalibrated (overconfident)
    cal = IsotonicCalibrator().fit(p, y)
    out = cal.transform(np.array([0.2, 0.5, 0.8]))
    assert out[0] <= out[1] <= out[2]  # monotone


def test_isotonic_passthrough_when_unfitted():
    cal = IsotonicCalibrator()
    assert np.allclose(cal.transform([0.3, 0.7]), [0.3, 0.7])


# ----- edge -----
def test_executable_ask_uses_real_ask_when_present():
    c = CostModel(half_spread=0.02)
    assert c.executable_ask(0.50, ask=0.53) == 0.53
    assert np.isclose(c.executable_ask(0.50), 0.52)


def test_net_edge_subtracts_costs():
    c = CostModel(half_spread=0.02, taker_fee=0.01)
    # buy cost = 0.50 + 0.02 + 0.01 = 0.53
    assert np.isclose(net_edge(0.60, 0.50, c), 0.07)


def test_edge_filter_rejects_alarm_and_stale_and_thin():
    f = EdgeFilter(min_edge=0.03, alarm_edge=0.10, min_liquidity=5000, max_data_age_min=30)
    assert f.accept(net_edge_value=0.05, liquidity=10000, data_age_min=5)[0] is True
    assert f.accept(net_edge_value=0.20, liquidity=10000, data_age_min=5)[0] is False  # alarm
    assert f.accept(net_edge_value=0.05, liquidity=100, data_age_min=5)[0] is False  # thin
    assert f.accept(net_edge_value=0.05, liquidity=10000, data_age_min=99)[0] is False  # stale
    assert f.accept(net_edge_value=0.01, liquidity=10000, data_age_min=5)[0] is False  # below min


# ----- sizing -----
def test_full_kelly_formula():
    # p=0.6, price=0.5 -> (0.6-0.5)/(1-0.5) = 0.2
    assert np.isclose(full_kelly_fraction(0.6, 0.5), 0.2)
    assert full_kelly_fraction(0.4, 0.5) == 0.0  # no edge -> no bet
    assert full_kelly_fraction(0.6, 0.0) == 0.0  # degenerate price


def test_shrinkage_blends():
    assert shrink_probability(0.8, 0.5, w=0.5) == 0.65
    assert shrink_probability(0.8, 0.5, w=1.0) == 0.8
    assert shrink_probability(0.8, 0.5, w=0.0) == 0.5


def test_sizing_caps_and_minimum():
    pol = SizingPolicy(kelly_fraction=0.25, shrink_weight=1.0, max_position_pct=0.10, min_order_usdc=5.0)
    # strong edge but capped at 10% of capital
    size = pol.position_size_usdc(p_model=0.9, price=0.5, capital=1000)
    assert size <= 100.0 + 1e-9
    # below minimum -> 0
    tiny = pol.position_size_usdc(p_model=0.501, price=0.5, capital=10)
    assert tiny == 0.0


def test_sizing_respects_liquidity_depth():
    pol = SizingPolicy(kelly_fraction=1.0, shrink_weight=1.0, max_position_pct=1.0, max_liquidity_frac=0.10)
    size = pol.position_size_usdc(p_model=0.9, price=0.5, capital=100000, liquidity_depth_usdc=1000)
    assert size <= 100.0 + 1e-9  # 10% of 1000 depth
