"""The order path has to be able to REST A LIMIT and CANCEL it (#131).

`broker.py` could send exactly one thing: a `TimeInForce.DAY` MARKET order. Every lane written
before CRSISHORT enters at market, so nothing needed anything else and the limitation was invisible.

CRSISHORT's entry is a sell-short LIMIT 3% above the signal close, resting until the session it was
computed for, cancelled if the name leaves the universe overnight. That is not a decoration on the
rule — #123 measures the limit arm at +79.7% on 228 trades and the market arm at +99.4% on 384 with
a worse tail, and the market version LOSES above 100bps of cost. A runner that quietly sent a market
order would measure one strategy and deploy another.

WHY THE PRICE IS REFUSED RATHER THAN ROUNDED. `engine.apply_gates` already rounds `limit_px` to the
penny, deliberately, because "a limit at 12.3449 rests at 12.34, not 12.35" and that changes fills.
If this layer rounded too, a caller that computed the price differently would be silently corrected
here and the live fill would differ from the measured one with nothing to show for it. Two roundings
of one number is two derivations; the second is where they disagree. So a price that is not a whole
number of cents is REFUSED, loudly, and the caller is told which one it sent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide, OrderStatus, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.executor.broker import Broker, OrderRequest
from kumo_strategies.runtime.nautilus.broker import NautilusBroker
from kumo_strategies.runtime.nautilus.sides import SHORT

AAPL = InstrumentId.from_str("AAPL.XNAS")


def _strategy(*, declared_side=None, status=OrderStatus.SUBMITTED):
    made, submitted, cancelled = [], [], []

    def _order(**kw):
        made.append(kw)
        o = SimpleNamespace(status=status, last_event=None,
                            client_order_id=kw.get("client_order_id"))
        made[-1]["_order"] = o
        return o

    strat = SimpleNamespace(
        cache=SimpleNamespace(instrument=lambda iid: object(),
                              order=lambda coid: (made[-1]["_order"] if made else None),
                              positions_open=lambda **kw: [], accounts=lambda: []),
        order_factory=SimpleNamespace(market=_order, limit=_order),
        submit_order=submitted.append,
        cancel_order=cancelled.append,
    )
    if declared_side is not None:
        strat.POSITION_SIDE = declared_side
    return strat, submitted, made, cancelled


def _req(**kw):
    base = dict(symbol="AAPL", side="SELL", qty=10, session="2026-09-10",
                strategy_id="CRSISHORT-001")
    base.update(kw)
    return OrderRequest(**base)


# -- the limit path -------------------------------------------------------------------------------

def test_a_request_with_a_limit_price_builds_a_LIMIT_order():
    strat, submitted, made, _ = _strategy(declared_side=SHORT)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(_req(limit_px=103.5))
    assert r.ok, r.detail
    assert len(submitted) == 1
    assert "price" in made[0], f"no price on the order — a market order was sent: {made[0]}"
    assert float(str(made[0]["price"])) == 103.5


def test_a_request_WITHOUT_a_limit_price_is_still_a_MARKET_order():
    """Four deployed lanes enter at market. This is the assertion that keeps them working."""
    strat, submitted, made, _ = _strategy()
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(_req(side="BUY"))
    assert r.ok and len(submitted) == 1
    assert "price" not in made[0], "a market entry acquired a limit price"


def test_a_short_ENTRY_at_a_limit_is_not_reduce_only():
    """#88 and #131 meet here: the entry is a SELL, and it is an OPEN. Sending it reduce-only against
    a flat book is a rejected order, whether it is a market or a limit."""
    strat, _, made, _ = _strategy(declared_side=SHORT)
    NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(_req(limit_px=103.5))
    assert made[0]["order_side"] == OrderSide.SELL
    assert made[0]["reduce_only"] is False


@pytest.mark.parametrize("bad", [103.456, 103.4449, 0.005])
def test_a_sub_penny_limit_is_REFUSED_not_rounded(bad):
    """Rounding here would be a SECOND derivation of a number `engine.apply_gates` already rounds,
    and the two would disagree exactly where it matters — a limit at 12.3449 rests at 12.34, not
    12.35, and that changes which fills happen at all."""
    strat, submitted, made, _ = _strategy(declared_side=SHORT)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(_req(limit_px=bad))
    assert not r.ok, f"accepted a sub-penny limit {bad}"
    assert submitted == [] and made == []
    # The OFFENDING PRICE has to be in the message. "invalid limit price" tells an operator
    # nothing; the number they sent tells them where to look. Asserting on prose would be a
    # source-substring test — satisfied by a comment, broken by a rewording.
    assert str(bad) in r.detail, r.detail


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_UNUSABLE_limit_price_is_refused(bad):
    """A NaN passes `is not None` and `float()` happily and then disarms every comparison
    downstream. Zero and negative are prices no venue accepts."""
    strat, submitted, _, _ = _strategy(declared_side=SHORT)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(_req(limit_px=bad))
    assert not r.ok and submitted == []


def test_the_time_in_force_is_carried_and_defaults_to_DAY():
    """DAY is what every lane sends today and it must not change by accident. CRSISHORT needs a
    different one — its order rests for a session it has not reached yet — so the field exists, with
    today's value as the default."""
    strat, _, made, _ = _strategy(declared_side=SHORT)
    b = NautilusBroker(strategy=strat, instrument_ids=[AAPL])
    b.submit(_req(limit_px=103.5))
    assert made[0]["time_in_force"] == TimeInForce.DAY

    b.submit(_req(limit_px=103.5, time_in_force="GTC"))
    assert made[1]["time_in_force"] == TimeInForce.GTC


def test_AT_THE_OPEN_is_expressible_because_it_is_what_this_lane_ACTUALLY_needs():
    """The entry has to fill at the OPENING PRINT, not merely rest during the session.

    Measured: 108 of 231 accepted trades (46.8%) gapped through the limit and filled at the open,
    and they carry 77.6% of the total return at +9.99%/trade against +2.53% for a limit fill. The
    backtest's `fill = max(limit, open)` IS auction participation.

    Verified against the installed Nautilus 1.229 rather than assumed: `TimeInForce.AT_THE_OPEN`
    exists, and the IB adapter maps it to IB's `"OPG"` at
    `adapters/interactive_brokers/parsing/execution.py:42` — a limit-on-open, which IB routes to the
    opening cross.
    """
    strat, _, made, _ = _strategy(declared_side=SHORT)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        _req(limit_px=103.5, time_in_force="AT_THE_OPEN"))
    assert r.ok, r.detail
    assert made[0]["time_in_force"] == TimeInForce.AT_THE_OPEN
    assert "price" in made[0], "a limit-on-open must carry its limit"


def test_an_UNKNOWN_time_in_force_is_refused_rather_than_defaulted():
    """Silently falling back to DAY would turn a resting overnight order into one that expires at
    the close, and the lane would never know its entry was gone."""
    strat, submitted, _, _ = _strategy(declared_side=SHORT)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        _req(limit_px=103.5, time_in_force="FOREVER"))
    assert not r.ok and submitted == []


# -- the cancel path ------------------------------------------------------------------------------

def test_a_resting_order_can_be_CANCELLED_by_its_client_order_id():
    """The overnight cancel: a name that has dropped out of the universe must have its resting entry
    pulled before the session it was computed for. The lab does this every session and dropping it
    changes the trade set."""
    strat, _, made, cancelled = _strategy(declared_side=SHORT)
    b = NautilusBroker(strategy=strat, instrument_ids=[AAPL])
    req = _req(limit_px=103.5)
    b.submit(req)
    r = b.cancel(req.client_order_id)
    assert r.ok, r.detail
    assert len(cancelled) == 1


def test_cancelling_an_order_the_cache_does_not_have_is_reported_not_raised():
    """It runs on the session path beside other symbols. An order already filled, already cancelled
    or never placed is an ordinary answer, not an exception — and it must not stop the rest."""
    strat, _, _, cancelled = _strategy()
    strat.cache.order = lambda coid: None
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).cancel("CRSISHORT-001-NOPE")
    assert not r.ok and cancelled == []
    assert "not" in r.detail.lower()


# -- the contract ---------------------------------------------------------------------------------

def test_the_Broker_PROTOCOL_declares_cancel():
    """"A protocol narrower than its callers is not a contract, it is a suggestion" — this file's
    own lesson, from the time `broker.price()` existed nowhere and TECHIVOL sized every entry to
    zero shares."""
    assert hasattr(Broker, "cancel"), "the runner will call cancel(); the protocol must declare it"


def test_OrderRequest_carries_the_limit_and_the_time_in_force():
    req = _req(limit_px=103.5, time_in_force="GTC")
    assert req.limit_px == 103.5 and req.time_in_force == "GTC"
    plain = _req()
    assert plain.limit_px is None and plain.time_in_force == "DAY"


def test_the_client_order_id_DIFFERS_for_a_limit_and_a_market_order():
    """The id is the replay guard. A market retry of an entry that rested as a limit is a DIFFERENT
    order, and if both hash the same the second is denied locally as a duplicate — which is how a
    seven-hour-old rejection was once reported as this morning's answer."""
    assert _req(limit_px=103.5).client_order_id != _req().client_order_id
