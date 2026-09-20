"""The give-back trail, ONE implementation, side-aware. (#110, #123)

`momentum_rotation/exits.py` has owned this arithmetic since #197 P2 collapsed two copies into one.
It is written LONG: `peak_px` ratchets up, `peak_gain = peak/entry - 1`, and a short position run
through it reads a rising price as profit. #123's **flat** exit is the same rule on the other side —
"once the position has been in profit, never return to entry" is `give_back_frac = 1.0` with the
favourable extreme measured DOWNWARD.

Two arrivals at one mechanism is the reason to share the arithmetic rather than copy it. #110 records
the long-side result from the sell-in-strength programme; #123 records the short-side result from
someone else's entries. If those are two implementations, the claim that they are the same mechanism
is unverifiable, and a fix to one silently leaves the other wrong — which is the shape #197 B12 was.

WHAT MOVES WHEN THIS IS SHARED: the breach test is `<=`, not `<`. At `frac = 1.0` a strict `<`
demands the position give back MORE than all of its peak, so the flat exit could only fire once the
trade was already a LOSS — the rule inverted by one character. For `frac < 1.0` the two differ only
at exact float equality, so the long side is unchanged in every case but that boundary.

PURE. No config type, no state type, no I/O: the two callers hold different state (`TrailState` here,
`ShortTrailState` there) and the disagreement this module prevents is about the ARITHMETIC.
"""

from __future__ import annotations

from typing import Literal

Side = Literal["long", "short"]
LONG: Side = "long"
SHORT: Side = "short"


def favourable_gain(entry_px: float, px: float, *, side: Side) -> float:
    """Return in the position's OWN direction: positive means profit, both sides.

    A short at 100 now trading 90 is +10%, not -10%. Every rule downstream compares gains to zero or
    to each other, so the sign convention has to be the position's and not the tape's.
    """
    if entry_px <= 0:
        raise ValueError(f"entry_px must be positive, got {entry_px!r}")
    if side == LONG:
        return px / entry_px - 1.0
    if side == SHORT:
        return 1.0 - px / entry_px
    raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")


def favourable_extreme(a: float, b: float, *, side: Side) -> float:
    """The better of two prices FOR THIS POSITION — the max for a long, the min for a short.

    Named for what it means rather than `max`, because the long-side name (`peak`) is precisely what
    made this arithmetic unusable on a short: the extreme a short cares about is the LOW.
    """
    if side == LONG:
        return max(a, b)
    if side == SHORT:
        return min(a, b)
    raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")


def gave_back(*, entry_px: float, extreme_px: float, price: float, frac: float,
              side: Side) -> tuple[bool, float, float]:
    """`(breached, peak_gain, now_gain)` — gains signed in the position's favour.

    `peak_gain > 0` is the arming condition: a position that has never been in profit has given
    nothing back, and without it a trade that went straight against us reports a give-back of a peak
    that never existed (the SU journal line in #197: "gave back 30825% of a 0.1% peak").

    The caller decides what to do about confirmation counters, minimum peaks and reason strings.
    Those differ per lane; this does not.
    """
    peak_gain = favourable_gain(entry_px, extreme_px, side=side)
    now_gain = favourable_gain(entry_px, price, side=side)
    breached = peak_gain > 0 and now_gain <= peak_gain * (1.0 - frac)
    return breached, peak_gain, now_gain


def give_back_trigger_px(*, entry_px: float, extreme_px: float, frac: float, side: Side) -> float:
    """The price at which the trail is breached — where a resting exit order would sit.

    At `frac = 1.0` this is the entry price, which is what makes the flat exit a plain resting
    order rather than something that has to be re-evaluated each session. Below 1.0 it is the level
    that gives back `frac` of the favourable excursion so far, and it MOVES as the extreme moves.

    Returned rather than left to the driver because the exit PRICE and the exit CONDITION are the
    same fact: a runner that detected the breach at the session high and then booked the fill there
    would report a loss the rule never took.
    """
    peak_gain = favourable_gain(entry_px, extreme_px, side=side)
    kept = peak_gain * (1.0 - frac)
    if side == LONG:
        return entry_px * (1.0 + kept)
    if side == SHORT:
        return entry_px * (1.0 - kept)
    raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")
