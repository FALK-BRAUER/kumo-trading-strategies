"""`reduce_only` IS NOT A GUARANTEE ON EVERY VENUE, so nothing may depend on it for safety.

MEASURED against the installed Nautilus, not assumed: the Interactive Brokers adapter contains the
string `reduce_only` ZERO times, while nineteen other venue adapters implement it. IB's own API has
no such field, so the flag is accepted by `order_factory.market(...)`, carried through Nautilus, and
then SILENTLY DROPPED at the adapter boundary. Nothing raises and nothing logs.

This matters here and now: CRSISHORT goes to ibkr-paper first, and ibkr-paper is the IBKR instance.

`broker.py` decides open-vs-exit from the lane's declared side and deliberately does not consult the
book (#88). The first version of that reasoning said `reduce_only` was the backstop for a lane that
tries to close what it does not hold. On IB there is no backstop. The real guard is one level up and
always was — `sides.closing_quantity`, which refuses a flat or wrong-sided position at the adapter
before an order is ever built, and which `test_position_side_is_declared` already requires every
`_close` path to call.

So this file pins the property that actually has to hold: the EXIT PATH IS CORRECT WITH THE FLAG
REMOVED ENTIRELY. If a future change makes safety depend on the venue honouring it, these fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide, OrderStatus
from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.executor.broker import OrderRequest
from kumo_strategies.runtime.nautilus.broker import NautilusBroker
from kumo_strategies.runtime.nautilus.sides import LONG, SHORT, closing_quantity, refusal_reason

AAPL = InstrumentId.from_str("AAPL.XNAS")


def _ib_adapter_dir() -> Path | None:
    try:
        import nautilus_trader.adapters.interactive_brokers as ib
    except ImportError:                                                # pragma: no cover
        return None
    return Path(os.path.dirname(ib.__file__))


def test_the_IB_adapter_really_does_ignore_reduce_only():
    """The observation this whole file rests on, re-taken on every run rather than quoted.

    If a future Nautilus release implements it, this fails and the comments above become wrong —
    which is the point: the claim is pinned to the installed package, not to a memory of it.
    """
    d = _ib_adapter_dir()
    if d is None:                                                      # pragma: no cover
        pytest.skip("interactive_brokers adapter not installed")
    hits = [p for p in d.rglob("*.py") if "reduce_only" in p.read_text(errors="ignore")]
    assert not hits, (
        f"the IB adapter now mentions reduce_only in {[p.name for p in hits]} — re-read whether it "
        f"is honoured, and update the reasoning in broker.py and this file")


def _strategy(*, declared_side, honours_reduce_only: bool, positions=()):
    """A venue that either honours `reduce_only` or drops it, so the same exit runs both ways.

    `honours_reduce_only=False` is IB. The order goes to the venue as a plain side with a quantity
    and NOTHING protecting it from opening the opposite position — which is exactly why the quantity
    has to have been checked before it was built.
    """
    made, submitted = [], []
    held = {p["iid"]: p["qty"] for p in positions}

    def market(**kw):
        made.append(kw)
        signed = held.get(kw["instrument_id"], 0)
        qty = int(kw["quantity"])
        reducing = ((signed > 0 and kw["order_side"] == OrderSide.SELL)
                    or (signed < 0 and kw["order_side"] == OrderSide.BUY))
        if honours_reduce_only and kw.get("reduce_only") and not reducing:
            order = SimpleNamespace(status=OrderStatus.REJECTED,
                                    last_event=SimpleNamespace(reason="reduce only would increase"))
        else:
            # The venue fills it. On a book of `signed`, an order of `qty` on this side lands the
            # position here — which is what tells us whether the flag was load-bearing.
            delta = qty if kw["order_side"] == OrderSide.BUY else -qty
            order = SimpleNamespace(status=OrderStatus.SUBMITTED, last_event=None,
                                    resulting_position=signed + delta)
        made[-1]["_order"] = order
        return order

    strat = SimpleNamespace(
        cache=SimpleNamespace(positions_open=lambda **kw: [], instrument=lambda iid: object(),
                              order=lambda coid: (made[-1]["_order"] if made else None),
                              accounts=lambda: []),
        order_factory=SimpleNamespace(market=market),
        submit_order=submitted.append,
        POSITION_SIDE=declared_side,
    )
    return strat, submitted, made


@pytest.mark.parametrize("honours", [True, False], ids=["venue-honours-it", "IB-drops-it"])
@pytest.mark.parametrize("side,held_qty,exit_side", [(LONG, 10, "SELL"), (SHORT, -10, "BUY")])
def test_an_exit_sized_by_closing_quantity_flattens_the_book_either_way(honours, side, held_qty,
                                                                       exit_side):
    """The quantity comes from `sides.closing_quantity`, which has already checked the sign against
    the declared side. Given that, the exit lands the position at exactly flat whether or not the
    venue honours the flag."""
    qty = closing_quantity(held_qty, side=side)
    assert qty == abs(held_qty)
    strat, submitted, made = _strategy(declared_side=side, honours_reduce_only=honours,
                                       positions=[{"iid": AAPL, "qty": held_qty}])
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", exit_side, qty, "2026-09-10", strategy_id="X-001"))
    assert r.ok, r.detail
    assert made[0]["_order"].resulting_position == 0, "the exit did not flatten the book"
    assert len(submitted) == 1


@pytest.mark.parametrize("side,book", [(LONG, -10), (SHORT, 10)])
def test_a_wrong_SIDED_book_is_refused_BEFORE_an_order_exists(side, book):
    """The case `reduce_only` was supposed to catch, caught where it actually is caught.

    `closing_quantity` returns None for a position on the other side, so no quantity exists to build
    an order from and `refusal_reason` says what an operator must do about it. On IB this is the ONLY
    thing standing between a lane and an order that DOUBLES a position it did not open.
    """
    assert closing_quantity(book, side=side) is None
    why = refusal_reason(book, side=side)
    assert why and "DOUBLE" in why


@pytest.mark.parametrize("side", [LONG, SHORT])
def test_a_flat_book_yields_no_exit_quantity_either(side):
    assert closing_quantity(0, side=side) is None
    # Flat is ordinary and gets no operator message; only a wrong-sided book does.
    assert refusal_reason(0, side=side) is None
