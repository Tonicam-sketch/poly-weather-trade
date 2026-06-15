"""Position sizing — fractional Kelly with shrinkage and hard caps.

Kelly is extremely sensitive to errors in the probability estimate, so we:
  1. Shrink the model probability toward the market price (estimation humility).
  2. Use a small Kelly fraction (default 1/4), not half.
  3. Cap each position at a fraction of capital and at available liquidity depth.

For a binary outcome token bought at price q (pays 1 if it resolves YES):
  full Kelly fraction f* = (p - q) / (1 - q)
"""

from __future__ import annotations

from dataclasses import dataclass


def shrink_probability(p_model: float, p_market: float, w: float) -> float:
    """Blend model and market: w=1 trusts model fully, w=0 trusts market fully."""
    w = min(max(w, 0.0), 1.0)
    return w * p_model + (1.0 - w) * p_market


def full_kelly_fraction(p: float, price: float) -> float:
    """Kelly fraction of bankroll for a token bought at `price` paying 1 on YES."""
    if price <= 0.0 or price >= 1.0:
        return 0.0
    f = (p - price) / (1.0 - price)
    return max(f, 0.0)


@dataclass(frozen=True)
class SizingPolicy:
    kelly_fraction: float = 0.25      # fraction of full Kelly to actually bet
    shrink_weight: float = 0.5        # weight on model vs market in shrinkage
    max_position_pct: float = 0.10    # cap per position as fraction of capital
    max_liquidity_frac: float = 0.10  # never take more than this share of book depth
    min_order_usdc: float = 5.0       # CLOB minimum

    def position_size_usdc(
        self,
        *,
        p_model: float,
        price: float,
        capital: float,
        liquidity_depth_usdc: float | None = None,
    ) -> float:
        """Dollar size after shrinkage, fractional Kelly and all caps.

        Returns 0.0 if the capped size falls below the exchange minimum.
        """
        p = shrink_probability(p_model, price, self.shrink_weight)
        kelly = full_kelly_fraction(p, price) * self.kelly_fraction
        size = kelly * capital
        size = min(size, self.max_position_pct * capital)
        if liquidity_depth_usdc is not None:
            size = min(size, self.max_liquidity_frac * liquidity_depth_usdc)
        size = min(size, capital)
        if size < self.min_order_usdc:
            return 0.0
        return float(size)
