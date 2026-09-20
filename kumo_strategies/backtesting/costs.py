"""Execution cost model built from MEASURED spreads, not an assumed rate.

Every result in this repo charged cost as arithmetic on notional at a rate someone picked. That
rate decided outcomes: one intraday variant returned +139% at 10bps and +65% — exactly the
do-nothing baseline — at 25bps. The rate has to come from the tape.

`research/residual-gate/half_spreads.json` holds the median half-spread in bps per symbol, sampled
from Alpaca SIP NBBO quotes across the session (217 names, ~700k quotes). Two facts from it shape
this model:

  dispersion  NVDA/AAPL/KO cost ~0.6bps a side; APEI/MTSI/VICR cost 25-34. A flat rate is wrong in
              both directions, and this book trades the mid-caps where it is wrong by the most.
  time of day widest at the open (3.59bps a side at 09:35) and tightest late (1.36 by 15:55) — so
              the daily strategy, which trades at the open, pays the worst spread of the session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Half-spread multiplier by time of day, as ratios to the all-day median, taken from the measured
# profile (SIP NBBO, ~200k quotes): 09:35 5.53bps, 10:30 2.87, 11:30 2.12, 12:30 1.90, 13:30 1.90,
# 15:00 2.00, 15:55 1.81, against an all-day median of 2.46. Note it does NOT fall monotonically —
# it drops steeply off the open, bottoms around midday, and ticks up slightly into the close.
_TOD = ((10, 0, 2.25), (11, 0, 1.17), (12, 0, 0.86), (14, 0, 0.77), (15, 30, 0.81))
_TOD_LATE = 0.74


@dataclass(frozen=True)
class CostModel:
    """Cost of crossing the spread, per side, in basis points."""

    half_spread_bps: dict[str, float]
    default_bps: float
    commission_bps: float = 0.0        # Alpaca US equities are commission-free
    slippage_mult: float = 1.0         # >1 to stress impact beyond the quoted spread
    time_of_day: bool = True

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> CostModel:
        d = json.loads(Path(path).read_text())
        med = sorted(d.values())[len(d) // 2] if d else 5.0
        return cls(half_spread_bps=d, default_bps=med, **kw)

    def _tod_mult(self, hour: int, minute: int) -> float:
        if not self.time_of_day:
            return 1.0
        for h, m, mult in _TOD:
            if (hour, minute) < (h, m):
                return mult
        return _TOD_LATE

    def bps(self, symbol: str, hour: int = 12, minute: int = 0) -> float:
        base = self.half_spread_bps.get(symbol, self.default_bps)
        return base * self.slippage_mult * self._tod_mult(hour, minute) + self.commission_bps

    def charge(self, symbol: str, notional: float, hour: int = 12, minute: int = 0) -> float:
        """Dollar cost of one side of a trade in `symbol`."""
        return abs(notional) * self.bps(symbol, hour, minute) / 1e4

    def coverage(self, symbols: list[str]) -> float:
        if not symbols:
            return 0.0
        return sum(1 for s in symbols if s in self.half_spread_bps) / len(symbols)
