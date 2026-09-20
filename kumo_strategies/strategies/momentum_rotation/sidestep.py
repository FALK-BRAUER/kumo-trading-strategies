"""Sell and rebuy WITHOUT giving up the position — in either direction (#32, 2026-08-17).

    "Hold, sell when it looses, rebuy when it bounces back, latest on session close."
    "It can also sell at strength and rebuy the dip."

WHY THIS IS NOT AN EXIT RULE
----------------------------
Every rule in `exits.py` is terminal: the position closes and the name is gone until the strategy
next selects it. That is right for a strategy rotating weekly, where the next decision is days away.

QC345 holds five names for a MONTH. A name that swings 15% and comes back is not a position the
strategy wants to abandon — the monthly thesis is intact — but sitting through the round trip earns
nothing. A terminal exit gets the first half right and the second half wrong, because nothing rebuys
until the next rebalance.

So this is a SIDESTEP: the name is sold, the slot stays reserved, and the position is restored on
the reversal. Selection never sees it. If the reversal never comes, the sidestep has behaved like an
exit and the slot frees at the rebalance — the good case, not a failure.

TWO DIRECTIONS, AND THEY ARE NOT THE SAME TRADE
-----------------------------------------------
  WEAKNESS   sell `drop_pct` below entry, rebuy `rebuy_pct` above the low made while out.
  STRENGTH   sell `pop_pct` above entry, rebuy `dip_pct` below the high made while out.

They look symmetric and are not, because of which side of the spread each one crosses.

The weakness version SELLS INTO A FALLING BID and REBUYS INTO A RISING OFFER — the wrong side both
times, on top of two commissions. On a five-name book one whipsaw is 20% of the portfolio paying a
full round trip to end where it started. It has to overcome that before it has predicted anything.

The strength version does the opposite: it SELLS INTO STRENGTH, where the bid is being lifted, and
REBUYS INTO A DIP, where the market is offering down. Same two fills, favourable side both times.

That asymmetry is a property of the mechanism rather than a forecast, so it survives whatever the
return numbers say — which matters here, because on this repo's evidence return numbers from short
samples are the least durable thing measured.

WHAT IS MEASURED AGAINST WHAT, DELIBERATELY
-------------------------------------------
Both SELL triggers measure from ENTRY. "When it looses" means losing money on the position, not
fading from a high it made; measuring the weakness sell from a peak would fire on a name up 30% that
gave back 12% while still deeply profitable, which is give-back's job and a different question.

Both REBUY triggers measure from the EXTREME REACHED WHILE OUT, not from the sell price. A turn is
visible from the extreme. Waiting to get back to the sell price means a name that drops 20% and
recovers 15% is never rebought — the sidestep captures the whole decline and none of the recovery,
which is the worst of both.

Neither trigger uses a peak-since-entry, so none of `TrailState`'s trustworthiness machinery is
needed: entry price is known even for an adopted position.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

WEAK = "weak"
STRONG = "strong"


@dataclass(frozen=True)
class SidestepConfig:
    drop_pct: float | None = None
    """Sell when the close is this far BELOW entry. None disables the weakness direction."""

    rebuy_pct: float = 0.02
    """Rebuy a weakness sidestep when the close is this far above the low made while out.

    Not zero: a name resting on its low would rebuy on any tick up, which is not a bounce and pays
    the round trip for nothing. Not large either — waiting for a big recovery rebuys near where it
    sold, capturing the decline and none of the turn.
    """

    pop_pct: float | None = None
    """Sell when the close is this far ABOVE entry. None disables the strength direction."""

    dip_pct: float = 0.02
    """Rebuy a strength sidestep when the close is this far below the high made while out."""

    max_sidesteps: int | None = None
    """Round trips one position may make before the sidestep stops arming for it.

    A whipsawing name is the failure mode: each cycle pays two spreads, and on a five-name book that
    is 20% of the portfolio churning. None means uncapped, which is the honest default for
    measurement — the cap is a parameter to test, not a safety blanket to assume.
    """


@dataclass(frozen=True)
class SidestepState:
    entry_px: float
    out: str | None = None
    """WEAK, STRONG, or None. Which direction we sidestepped decides which rebuy rule applies —
    a strength sidestep waiting for a bounce off the low would never rebuy."""

    extreme_px: float | None = None
    """Low (weakness) or high (strength) reached while out. The rebuy trigger measures from here."""

    cycles: int = 0
    """Completed round trips. Reset when the position itself is replaced at rebalance."""


@dataclass(frozen=True)
class SidestepPlan:
    sells: dict[str, str]
    buys: dict[str, str]
    state: dict[str, SidestepState]


def _armed(cfg: SidestepConfig, st: SidestepState) -> bool:
    return cfg.max_sidesteps is None or st.cycles < cfg.max_sidesteps


def evaluate_sidestep(cfg: SidestepConfig, prices: dict[str, float],
                      state: dict[str, SidestepState]) -> SidestepPlan:
    """One session's sidestep decisions, at the close.

    A symbol with no usable price is left entirely alone and its state does not advance. Acting on a
    missing price is how a data outage becomes a trade.

    When both directions are configured the WEAKNESS trigger is checked first. They cannot both fire
    on one bar — a close cannot be below entry and above it — so the order is not a tie-break, only
    a reading order.
    """
    sells: dict[str, str] = {}
    buys: dict[str, str] = {}
    out: dict[str, SidestepState] = {}

    for sym, st in state.items():
        px = prices.get(sym)
        if px is None or px != px or px <= 0:
            out[sym] = st
            continue

        if st.out is None:
            if _armed(cfg, st) and cfg.drop_pct is not None and px <= st.entry_px * (1 - cfg.drop_pct):
                sells[sym] = (f"sidestepped weakness {100 * (1 - px / st.entry_px):.1f}% below entry "
                              f"(trigger {100 * cfg.drop_pct:.0f}%)")
                out[sym] = replace(st, out=WEAK, extreme_px=px)
            elif _armed(cfg, st) and cfg.pop_pct is not None and px >= st.entry_px * (1 + cfg.pop_pct):
                sells[sym] = (f"sold strength {100 * (px / st.entry_px - 1):.1f}% above entry "
                              f"(trigger {100 * cfg.pop_pct:.0f}%)")
                out[sym] = replace(st, out=STRONG, extreme_px=px)
            else:
                out[sym] = st
            continue

        # Out. The extreme updates FIRST, so a new extreme in the same session cannot also trigger a
        # rebuy against the old one — that would rebuy into a name still running away from us.
        prev = st.extreme_px if st.extreme_px is not None else px
        if st.out == WEAK:
            extreme = min(prev, px)
            if px >= extreme * (1 + cfg.rebuy_pct):
                buys[sym] = f"bounced {100 * (px / extreme - 1):.1f}% off {extreme:.2f}"
                out[sym] = replace(st, out=None, extreme_px=None, cycles=st.cycles + 1)
            else:
                out[sym] = replace(st, extreme_px=extreme)
        else:
            extreme = max(prev, px)
            if px <= extreme * (1 - cfg.dip_pct):
                buys[sym] = f"dipped {100 * (1 - px / extreme):.1f}% off {extreme:.2f}"
                out[sym] = replace(st, out=None, extreme_px=None, cycles=st.cycles + 1)
            else:
                out[sym] = replace(st, extreme_px=extreme)

    return SidestepPlan(sells=sells, buys=buys, state=out)
