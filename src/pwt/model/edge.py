"""Edge net of trading costs.

The original strategy compares P_model to a single market probability and trades
on a 5% paper edge. That ignores that you pay the ASK when buying, and the bid-ask
spread on thin weather books (2-5c) routinely exceeds the paper edge. Here edge is
computed against the executable price, after fees, with an explicit safety margin.
A large positive edge (> `alarm_edge`) is treated as a data bug, not a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CostModel:
    half_spread: float = 0.02   # assumed half bid-ask spread when only mid is known
    taker_fee: float = 0.0      # Polymarket CLOB taker fee (fraction of notional)
    slippage: float = 0.0       # extra adverse fill beyond half-spread

    def executable_ask(self, mid: float, ask: Optional[float] = None) -> float:
        """Price actually paid to buy: use real ask if known, else mid+half_spread."""
        base = ask if ask is not None else mid + self.half_spread
        return min(1.0, base + self.slippage)

    def buy_cost(self, mid: float, ask: Optional[float] = None) -> float:
        return self.executable_ask(mid, ask) + self.taker_fee


def net_edge(p_model: float, mid: float, cost: CostModel, ask: Optional[float] = None) -> float:
    """Calibrated probability minus the all-in cost of acquiring the token."""
    return p_model - cost.buy_cost(mid, ask)


@dataclass(frozen=True)
class EdgeFilter:
    min_edge: float = 0.03          # minimum NET edge after costs
    alarm_edge: float = 0.10        # edge above this => likely data error, reject
    min_liquidity: float = 5000.0   # market volume floor
    max_data_age_min: float = 30.0  # forecast/price staleness cap (minutes)

    def accept(
        self,
        *,
        net_edge_value: float,
        liquidity: float,
        data_age_min: float,
    ) -> tuple[bool, str]:
        """Return (accepted, reason). Reason is the first failing check."""
        if data_age_min > self.max_data_age_min:
            return False, f"stale_data({data_age_min:.0f}min)"
        if liquidity < self.min_liquidity:
            return False, f"low_liquidity({liquidity:.0f})"
        if net_edge_value < self.min_edge:
            return False, f"edge_below_min({net_edge_value:.3f})"
        if net_edge_value > self.alarm_edge:
            return False, f"edge_alarm({net_edge_value:.3f})_check_data"
        return True, "ok"
