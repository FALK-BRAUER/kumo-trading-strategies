"""CRSISHORT's covers, on the SHORT side. Pure: no I/O, no clock, no broker.

    FLAT      once the position has been in profit, cover on a return to the entry price
    REVERSAL  cover after a close ABOVE the prior session's high
    WEAKNESS  cover after a close BELOW the prior session's low   (off in the frozen spec)

Both structural rules are FLAGGED AT A CLOSE AND ACTED AT THE NEXT OPEN, which is the lab engine's
model and the only honest one for a daily lane: the signal is the close, so the earliest tradable
moment is the following open. Acting on the close you have just observed needs a market-on-close
order this book does not send, and it is worth several points of return.

The FLAT cover is different in kind and fires INTRADAY: it is a resting buy at the trail price, so
it fills the moment the tape reaches that level, and on a session that gaps through it, at the open.

Rule order is the lab's and it is the semantics: flat, then weakness, then reversal. A session can
satisfy more than one, and the flat cover happened first.

STATE IS RETURNED, NOT MUTATED, and it is evaluated AS OF THE PRIOR CLOSE — this session's low
cannot arm this session's flat exit. The backtest holds state in a dict for one process and the live
runner reloads it from Postgres every session; both are ways to SUPPLY it, neither is a reason for a
second copy of the arithmetic. The give-back arithmetic is not even in this file: it is
`strategies/give_back.py`, shared with the long side (#110).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.give_back import (
    SHORT, favourable_extreme, favourable_gain, give_back_trigger_px)

FLAT = "flat"
REVERSAL = "reversal"
WEAKNESS = "weakness"


@dataclass(frozen=True)
class SessionBar:
    """One session's OHLC for one symbol. The rules need all four."""

    open: float
    high: float
    low: float
    close: float

    @property
    def usable(self) -> bool:
        return all(_finite(x) and x > 0 for x in (self.open, self.high, self.low, self.close))


@dataclass(frozen=True)
class ShortTrailState:
    """What the covers need to know about one open SHORT, as of the PRIOR close."""

    entry_px: float
    best_px: float
    """The LOWEST price seen since entry — the favourable extreme for a short.

    Named `best`, not `peak`: `peak` is the long-side word and it is what made the shared trail
    unusable on this side. A field called `peak` invites `max()`."""

    prev_low: float
    prev_high: float
    flag_reversal: bool = False
    """Set by the PRIOR close breaking above the prior-prior high. Acted at this session's open."""

    flag_weakness: bool = False
    opened_session: date | None = None
    sessions_held: int = 0

    @property
    def ever_in_profit(self) -> bool:
        """DERIVED, never stored beside `best_px`. Two fields for one fact disagree eventually, and
        the disagreement is invisible: the flat exit simply stops arming."""
        return self.best_px < self.entry_px


@dataclass(frozen=True)
class ShortExitPlan:
    """`exits` is symbol -> reason; `state` is the advanced trail to persist."""

    exits: dict[str, str] = field(default_factory=dict)
    state: dict[str, ShortTrailState] = field(default_factory=dict)
    exit_px: dict[str, float] = field(default_factory=dict)
    """The price each cover is DEFINED at — the trail level (or the open, if the session gapped
    through it) for a flat cover, the open for a flagged one.

    Returned rather than left to the caller because the price and the condition are one fact. A
    driver that detected the flat cover at the session high and booked the fill there would report
    a loss the rule never took. The driver still owns slippage and fees; it must not own WHICH
    price the rule meant."""

    kind: dict[str, str] = field(default_factory=dict)
    """symbol -> FLAT / REVERSAL / WEAKNESS. The exit MIX is an acceptance number (#123: flat 48%,
    reversal 48%, no-data 4%), so it is returned as a machine-readable label and not only inside
    the human reason."""


def evaluate_short_exits(cfg: CrsiShortConfig, bars: dict[str, SessionBar],
                         state: dict[str, ShortTrailState]) -> ShortExitPlan:
    """Apply the covers to every position with a usable bar. First match wins.

    A symbol missing from `bars`, or carrying a non-finite one, is left alone entirely and its state
    is NOT advanced — no exit, no ageing. Whether a position with no data is CLOSED is the driver's
    call (the lab closes it at the last mark and books it as `nodata`, 4% of trades); what must not
    happen here is a data outage quietly ageing a position through a holding rule.
    """
    exits: dict[str, str] = {}
    out: dict[str, ShortTrailState] = {}
    exit_px: dict[str, float] = {}
    kind: dict[str, str] = {}

    for sym, st in state.items():
        bar = bars.get(sym)
        if bar is None or not bar.usable:
            out[sym] = st
            continue

        # --- act on what was true AT THE PRIOR CLOSE ------------------------------------------
        if cfg.give_back_frac is not None and st.ever_in_profit:
            trigger = give_back_trigger_px(entry_px=st.entry_px, extreme_px=st.best_px,
                                           frac=cfg.give_back_frac, side=SHORT)
            if bar.high >= trigger:
                # A resting buy at `trigger`. A session that OPENS above it fills at the open, and
                # that gap is the risk #123 names explicitly: "latest at flat" cannot protect
                # against an overnight squeeze, and the worst observed trade is -53%.
                px = max(trigger, bar.open)
                # The excursion comes from the shared module too, not from an inline ratio: a
                # local `1 - best/entry` next to the call is how a shared function ends up
                # decorative while a stale copy decides what gets reported.
                best_gain = favourable_gain(st.entry_px, st.best_px, side=SHORT)
                exits[sym], exit_px[sym], kind[sym] = (
                    f"back to {trigger:.2f} after {100 * best_gain:.1f}% in profit"
                    + (f", gapped to {bar.open:.2f}" if bar.open > trigger else ""),
                    px, FLAT)
                out[sym] = st
                continue
        # THE CONFIG IS CONSULTED AT THE ACT, not only at the flag. A flag persisted by an earlier
        # session — a live runner reloading its trail from Postgres after the rule was switched off
        # in settings — would otherwise still cover the position, on a rule nobody has armed.
        if cfg.weakness_exit and st.flag_weakness:
            exits[sym], exit_px[sym], kind[sym] = (
                f"prior close broke below the session low before it ({st.prev_low:.2f})",
                bar.open, WEAKNESS)
            out[sym] = st
            continue
        if cfg.reversal_exit and st.flag_reversal:
            exits[sym], exit_px[sym], kind[sym] = (
                f"prior close broke above the session high before it ({st.prev_high:.2f})",
                bar.open, REVERSAL)
            out[sym] = st
            continue

        # --- otherwise advance the trail on THIS session ---------------------------------------
        out[sym] = replace(
            st,
            best_px=favourable_extreme(st.best_px, bar.low, side=SHORT),
            flag_weakness=bool(cfg.weakness_exit and bar.close < st.prev_low),
            flag_reversal=bool(cfg.reversal_exit and bar.close > st.prev_high),
            prev_low=bar.low,
            prev_high=bar.high,
            sessions_held=st.sessions_held + 1)

    return ShortExitPlan(exits=exits, state=out, exit_px=exit_px, kind=kind)


def open_state(entry_px: float, bar: SessionBar, session: date | None = None) -> ShortTrailState:
    """The trail for a position filled during `bar`. ONE constructor, so no driver invents its own.

    `best_px = entry_px`, i.e. NOT in profit yet, even if the session traded below entry after the
    fill. `prev_low`/`prev_high` are this session's, so the first flag can only be set at the NEXT
    close — a position cannot be flagged out on the session it was opened in.
    """
    return ShortTrailState(entry_px=entry_px, best_px=entry_px,
                           prev_low=bar.low, prev_high=bar.high, opened_session=session)


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
