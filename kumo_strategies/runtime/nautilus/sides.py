"""WHICH SIDE A LANE HOLDS, stated and asserted, instead of assumed everywhere (#88, #123).

Every rotation lane in this package opens with BUY and closes with SELL — `momentum_rotation.py`,
`qc27_rotation.py`, `qc345_rotation.py`. That was never written down, so the whole
exit path quietly depended on it and no single place said so.

What the dependence costs, found 2026-08-28 against four UNCLAIMED positions the UI labelled SHORT:

    qc345_rotation.py:679   Quantity.from_str(str(abs(position.quantity)))  + OrderSide.SELL
                            `abs()` on a short, then SELL — which DOUBLES the short rather than
                            closing it, and reports success.

    momentum_rotation.py:840  Quantity.from_int(int(pos.quantity))
                            a negative raises inside Nautilus instead, from a line that reads as an
                            ordinary exit. Different failure, equally unhandled.

Neither considered the sign. `abs()` is the dangerous one because it SUCCEEDS.

CRSISHORT (#123) IS THE FIRST LANE THAT HOLDS THE OTHER SIDE, and that does not weaken the rule —
it is the reason the rule has to be about a DECLARED side rather than about the word "long". A long
position in a short lane is exactly as unmanageable as a short in a long lane, and covering it with
`abs()` would be the same defect mirrored. So the side is a required, keyword-only argument at every
call site: there is no default to inherit and no lane that fails to state what it holds.

A position on the wrong side cannot originate from a lane that declares its side, so if one exists
it came from reconciliation, a manual order, or a phantom — and none of those make it ours to
double. REFUSING remains the only correct answer.

The vocabulary is `strategies/give_back.py`'s, imported rather than restated: one set of side names
for the decision layer, the reporting layer and the order path.
"""

from __future__ import annotations

from kumo_strategies.strategies.give_back import LONG, SHORT, Side

__all__ = ["LONG", "SHORT", "Side", "closing_quantity", "closing_order_side", "refusal_reason"]

#: The order side that FLATTENS a position held on each side. A long is closed by selling it.
_CLOSES_WITH = {LONG: "SELL", SHORT: "BUY"}


def closing_order_side(side: Side) -> str:
    """"BUY" or "SELL" — the order that reduces a position held on `side`.

    A lookup rather than a conditional at four call sites. The mapping is trivial and that is the
    point: the defect it replaces was not a hard sum, it was four places each deciding for
    themselves and one of them wrapping the quantity in `abs()` first.
    """
    try:
        return _CLOSES_WITH[side]
    except KeyError:
        raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}") from None


def closing_quantity(quantity, *, side: Side) -> int | None:
    """How many units to trade to flatten a position held on `side`, or None if that is not what
    this is.

    Returns None for a position on the OTHER side and for a flat one — the caller must not trade
    either, and both are the same instruction ("do nothing here") even though they are different
    facts. The caller tells them apart with `refusal_reason`, because they need different words: a
    flat position is ordinary, and a wrong-sided one is a book we cannot manage and must say so
    about.

    Always POSITIVE, and deliberately not `abs()`. The magnitude is derived from a quantity whose
    sign has already been checked against the declared side; `abs()` skips that check, which is
    exactly the defect this module exists for — it turns "I do not understand this position" into a
    confident order in the wrong direction.
    """
    if side not in _CLOSES_WITH:
        raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        return None
    if side == LONG:
        return qty if qty > 0 else None
    return -qty if qty < 0 else None


def refusal_reason(quantity, *, side: Side) -> str | None:
    """Why `closing_quantity` said no, phrased for an operator. None when there is nothing to say.

    A flat position gets no message: it is the ordinary end of a holding, and a log line per flat
    symbol per session would bury the one that matters.
    """
    if side not in _CLOSES_WITH:
        raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        return (f"position quantity {quantity!r} is not a whole number of units, so no exit size "
                f"can be derived — refusing to guess one")
    wrong = qty < 0 if side == LONG else qty > 0
    if not wrong:
        return None
    held, order = ("SHORT", "Selling") if side == LONG else ("LONG", "Buying")
    return (f"position is {held} {abs(qty)} and this strategy is {side}-only — refusing to exit. "
            f"{order} would DOUBLE it, not close it. Nothing in this lane opens a {held.lower()}, "
            f"so this came from reconciliation, a manual order, or a phantom, and it must be "
            f"resolved outside the strategy (#88)")
