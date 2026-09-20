"""The order path must decide OPEN vs EXIT from the lane's declared side, not from the word "SELL".

`broker.py` read `is_exit = req.side == "SELL"` and passed `reduce_only=is_exit`. For every lane
written before #123 that is correct and untestably so: they are all LONG, so a SELL really is always
an exit. CRSISHORT (#123) holds the other side, and there the same line inverts the meaning of every
order it sends — a short ENTRY is a SELL, so it goes out `reduce_only=True` against a flat book,
which is not a weaker order but a DIFFERENT one. The venue has nothing to reduce and rejects it.

WHY THE DOUBLE REJECTS. Asserting on the `reduce_only` kwarg alone would pass the moment someone
flipped the flag for the wrong reason, and would say nothing about what a venue does with it. So the
factory here refuses a reduce-only order that would have to OPEN a position, the way Alpaca and IB
both do ("reduce only order would increase position size"). The defect then presents the way it
would live: an entry that is rejected, in a lane that reports it submitted nothing.

The long cases are here for the same reason the short ones are — this fix must not become a lane
flag. A LONG lane's BUY still opens and its SELL still exits, and if that changes the four deployed
lanes stop being able to close a position.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide, OrderStatus
from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.executor.broker import OrderRequest
from kumo_strategies.runtime.nautilus.broker import NautilusBroker
from kumo_strategies.runtime.nautilus.sides import LONG, SHORT

AAPL = InstrumentId.from_str("AAPL.XNAS")


def _position(symbol: str, signed_qty: int):
    """Nautilus's own shape: `quantity` is the MAGNITUDE, `signed_qty` carries the direction."""
    return SimpleNamespace(instrument_id=InstrumentId.from_str(f"{symbol}.XNAS"),
                           quantity=abs(signed_qty), signed_qty=signed_qty)


def _strategy(*, declared_side=None, positions=()):
    """A Nautilus double that REJECTS what the venue rejects.

    `declared_side=None` is a lane that declares nothing, which must keep reading as LONG — every
    adapter written before #123 is in that state and none of them may change behaviour here.
    """
    made, submitted = [], []
    held = {p.instrument_id: p.signed_qty for p in positions}

    def market(**kw):
        made.append(kw)
        iid, side = kw["instrument_id"], kw["order_side"]
        qty = int(kw["quantity"])
        signed = held.get(iid, 0)
        # What the venue actually checks: a reduce-only order must move the position TOWARDS flat,
        # and it cannot do that if there is nothing there or if it points the wrong way.
        reducing = (signed > 0 and side == OrderSide.SELL) or (signed < 0 and side == OrderSide.BUY)
        if kw.get("reduce_only") and not reducing:
            return SimpleNamespace(
                status=OrderStatus.REJECTED,
                last_event=SimpleNamespace(
                    reason=f"reduce only order would increase position size (held {signed}, "
                           f"{side} {qty})"))
        return SimpleNamespace(status=OrderStatus.SUBMITTED, last_event=None)

    strat = SimpleNamespace(
        cache=SimpleNamespace(
            positions_open=lambda **kw: list(positions),
            instrument=lambda iid: object(),
            order=lambda coid: made and made[-1]["_order"] or None,
            accounts=lambda: [],
        ),
        order_factory=SimpleNamespace(market=market),
        submit_order=submitted.append,
    )
    if declared_side is not None:
        strat.POSITION_SIDE = declared_side

    # `submit()` reads the order back out of the cache by id to learn the venue's answer, so the
    # cache has to return the order the factory just built rather than a fresh accepted one.
    def market_and_remember(**kw):
        order = market(**kw)
        made[-1]["_order"] = order
        return order

    strat.order_factory.market = market_and_remember
    strat.cache.order = lambda coid: (made[-1]["_order"] if made else None)
    return strat, submitted, made


def _submit(strat, side, *, strategy_id):
    return NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", side, 10, "2026-09-10", strategy_id=strategy_id))


def test_a_short_lane_opens_with_a_sell_that_is_not_reduce_only():
    """THE DEFECT. CRSISHORT's entry is a sell-short on a flat book.

    Under `is_exit = req.side == "SELL"` this order carries `reduce_only=True`, the venue has no
    position to reduce, and the lane's first trade is rejected — while every log line and every
    call site reads correctly.
    """
    strat, submitted, made = _strategy(declared_side=SHORT)
    r = _submit(strat, "SELL", strategy_id="CRSISHORT-001")
    assert made[0]["order_side"] == OrderSide.SELL
    assert made[0]["reduce_only"] is False, (
        "a short ENTRY was submitted reduce_only — the venue has nothing to reduce on a flat book")
    assert r.ok, f"the short entry did not reach the venue: {r.detail}"
    assert len(submitted) == 1


def test_a_short_lane_covers_with_a_buy_that_is_reduce_only():
    """The cover is the exit, so it keeps the guard the exit path exists for: an order that outlives
    the position it was closing must not be able to open the opposite one."""
    strat, submitted, made = _strategy(declared_side=SHORT, positions=[_position("AAPL", -10)])
    r = _submit(strat, "BUY", strategy_id="CRSISHORT-001")
    assert made[0]["order_side"] == OrderSide.BUY
    assert made[0]["reduce_only"] is True, "a short COVER was submitted without reduce_only"
    assert r.ok, f"the cover did not reach the venue: {r.detail}"
    assert len(submitted) == 1


@pytest.mark.parametrize("declared", [None, LONG])
def test_a_long_lane_is_unchanged(declared):
    """Four deployed lanes depend on this exactly as it is, and one of them declares nothing."""
    strat, _, made = _strategy(declared_side=declared)
    assert _submit(strat, "BUY", strategy_id="MOMENTUM-002").ok
    assert made[0]["order_side"] == OrderSide.BUY and made[0]["reduce_only"] is False

    strat, _, made = _strategy(declared_side=declared, positions=[_position("AAPL", 10)])
    assert _submit(strat, "SELL", strategy_id="MOMENTUM-002").ok
    assert made[0]["order_side"] == OrderSide.SELL and made[0]["reduce_only"] is True


def test_the_order_side_is_the_requested_one_not_a_re_derived_one():
    """`OrderSide.BUY if not is_exit else OrderSide.SELL` derives the side a SECOND time, from the
    same field, through a variable that means something else. The two derivations agree for a long
    lane and disagree for a short one; there should only ever have been one."""
    for declared, side, expected in [(SHORT, "SELL", OrderSide.SELL), (SHORT, "BUY", OrderSide.BUY),
                                     (LONG, "BUY", OrderSide.BUY), (LONG, "SELL", OrderSide.SELL)]:
        strat, _, made = _strategy(declared_side=declared,
                                   positions=[_position("AAPL", -10 if declared == SHORT else 10)])
        _submit(strat, side, strategy_id="X-001")
        assert made[0]["order_side"] == expected, f"{declared} lane sent {side} as {made[0]}"


def test_a_side_that_is_neither_buy_nor_sell_is_refused_not_bought():
    """`req.side` is a plain string, and the old line asked only whether it equalled "SELL". Every
    other value — a typo, a lowercase "sell", a None that stringified — fell through to BUY and was
    submitted as a real order in the wrong direction, reporting success."""
    for bad in ["sell", "Sell", "SHORT", "", "BUYY"]:
        strat, submitted, made = _strategy(declared_side=SHORT)
        r = _submit(strat, bad, strategy_id="CRSISHORT-001")
        assert not r.ok, f"side {bad!r} was accepted"
        assert submitted == [] and made == [], f"side {bad!r} reached the order factory"
