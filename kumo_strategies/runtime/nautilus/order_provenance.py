"""Is this order event ours? (kumo-trading-platform issue 748.)

Nautilus routes order events by the ORDER's strategy_id — `Strategy.submit_order` publishes to
`events.order.{order.strategy_id}` — so a strategy receives events for every order stamped with its
id, whoever built it.

That was harmless while cockpit stamped its protective stops with its own display strategy. It is
about to stop being harmless: under NETTING a fill's position is derived as
`{instrument}-{fill.strategy_id}`, so a stop stamped with the display strategy resolves to a position
that never existed, its reduce-only fill lands on nothing, and the position poll fabricates the
difference — a phantom short. The fix over there is to stamp each protective stop with the LANE whose
shares it covers, and the consequence over here is that `PROT-*` fills, cancels and rejections start
arriving at handlers that assume every event is their own.

THEY DO ASSUME THAT. Every rotation strategy here pops `_pending` by SYMBOL alone, so a protective
SELL landing on a pending "enter" is read as that entry completing: the strategy believes it holds a
position A STOP JUST SOLD, sizes its next decision off that belief, and skips re-entering a name it
thinks it owns. `template_rotation` is worse — its `else` branch adds to `_held` for ANY non-exit
fill, so a foreign sell with no pending intent marks the symbol held out of nothing.

TWO CHECKS — AND THE REDUNDANCY IS HALF A BACKSTOP, NOT A SYMMETRIC ONE. Read this before trusting
"either can drift": `completes` only catches foreign fills whose SIDE is wrong for the pending intent
(a SELL on an enter, a BUY on an exit). A foreign fill with the RIGHT side passes it and is stopped by
the PREFIX ALONE — and two such shapes are live rather than exotic: a book-repair BUY leg landing on
a pending enter, and a BUY-side protective stop, which exists because the current book carries
mirrored SHORT positions whose protection reduces by BUYING. So prefix completeness is load-bearing
on its own for those, which is why the list below is pinned by a conformance test rather than
maintained by hand.

  - `is_foreign` excludes orders another component minted, by client order id. EXCLUSION, never
    attribution: it answers "did someone else place this", not "who owns it", and the failure
    direction is safe — a prefix that stops matching returns today's behaviour, and no strategy here
    mints such an id, so it cannot exclude its own order.
  - `fill_side` lets a handler refuse a fill that cannot be the thing it is waiting for. A SELL
    cannot complete an entry, whoever sent it, and this needs no shared vocabulary with the other
    repo at all — it is what survives the prefix convention changing.

WHY THE CLIENT ORDER ID AND NOT THE TAGS. Cockpit's own `engine_node.py` records that reconciled
orders are rebuilt with NO TAGS AT ALL, so a tag-based check goes blind on exactly the orders that
survive a restart. The client order id is the order's identity and survives reconciliation.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.sides import LONG, SHORT, Side, closing_order_side

#: Client-order-id prefixes minted by KUMO-TRADING-PLATFORM for orders IT placed. Mirrors that repo's
#: `cancel_attribution.OUR_STOP_PREFIXES`; it cannot be imported across the seam, so the coupling is
#: named here rather than left implicit. If cockpit adds a prefix and this is not updated, the SIDE
#: check above is what still holds — which is why there are two.
#: `TR-` IS NOT A STOP AND IT REACHES THESE HANDLERS TODAY, WITHOUT ANY OF #748. Cockpit's transfer
#: legs (`transfers.py`) are built by `_apply_leg` STAMPED WITH THE LANE and emitted as synthetic
#: `OrderFilled` events straight onto `events.order.{lane}`. A `TR-` SELL landing on a pending
#: "enter" is the defect in this file, live, now; a `TR-` BUY would sail through BOTH guards. It was
#: missing because the list was copied from cockpit's `OUR_STOP_PREFIXES` — which is correct for the
#: question IT asks ("is this one of our stops") and incomplete for the question asked here ("did
#: someone else place this"). Copying an answer to a different question is how the gap opened.
FOREIGN_ORDER_PREFIXES = ("PROT-", "PKW-", "PK-", "FL-", "TR-")


def is_foreign(event) -> bool:
    """Did some other component place the order behind this event?"""
    coid = str(getattr(event, "client_order_id", "") or "")
    return coid.startswith(FOREIGN_ORDER_PREFIXES)


def fill_side(event) -> str | None:
    """"BUY", "SELL", or None when the event cannot say.

    None is a THIRD STATE, not a default. A shape this handler cannot read is not a confirmation of
    anything, and moving book state on it is how a strategy comes to believe it holds something
    nobody bought.
    """
    side = getattr(event, "order_side", None)
    name = getattr(side, "name", None)
    if name is None and isinstance(side, str):
        name = side
    return str(name).upper() if name else None


def completes(intent: str | None, event, *, side: Side) -> bool:
    """Can this fill be the thing `intent` is waiting for, in a lane that holds `side`?

    An unreadable side answers NO for every intent — including `None`, which is what stops a handler
    falling through to write a terminal journal row asserting an outcome nobody established.

    THE LANE'S SIDE IS REQUIRED AND HAS NO DEFAULT. This used to map `enter -> BUY` unconditionally,
    with a comment saying that is a property of this long-only shop rather than of trading, and that
    a future short-entry strategy would make it wrong. CRSISHORT (#123) is that strategy.

    The failure it would have had is not a refused good fill. For a short lane the mapping is
    INVERTED: the entry SELL that actually filled is refused as "cannot complete an enter", and the
    covering BUY is accepted as though it were the entry — so the lane believes it is short a name it
    has just closed, and its next decision sizes off that belief. Defaulting to LONG would have been
    the polite change and the worse one, because every short lane written afterwards would inherit
    the wrong mapping and read correctly at the call site while doing it.
    """
    lane_side = fill_side(event)
    if side not in (LONG, SHORT):
        raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")
    if lane_side is None:
        return False
    #: A lane OPENS with the side that is not its closing side, and closes with the closing one.
    if intent == "enter":
        return lane_side != closing_order_side(side)
    if intent == "exit":
        return lane_side == closing_order_side(side)
    return True
