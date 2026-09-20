"""Tests for the Nautilus `Broker` adapter.

Each of these pins a property the session runner silently depends on. The runner cannot tell a
correct broker from a subtly wrong one — it just trades on what it is told.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.executor.broker import OrderRequest
from kumo_strategies.runtime.nautilus.broker import NautilusBroker

AAPL = InstrumentId.from_str("AAPL.XNAS")
MSFT = InstrumentId.from_str("MSFT.XNAS")


def _async(result, *, calls=None):
    async def f(*a, **kw):
        if calls is not None:
            calls.append((a, kw))
        return result
    return f


#: The signature of the DEPLOYED `UiFeedStrategy.release_for_exit`, read out of the running
#: container on 2026-08-23 rather than out of cockpit's repo:
#:
#:     docker exec kumo-paper-api-1 python -c "inspect.signature(UiFeedStrategy.release_for_exit)"
#:     -> ['self', 'instrument_id', 'qty', 'side', 'strategy_id']
#:
#: An image tag is a claim; a signature read inside the container is an observation. Pinned here so
#: this repo cannot start passing an argument the running acceptor does not take — which is the
#: 2026-08-17 failure exactly, and on the exit path it happens with protection already suppressed.
DEPLOYED_RELEASE_SIGNATURE = ("self", "instrument_id", "qty", "side", "strategy_id")


class _Feed:
    """Binds cockpit's REAL `release_for_exit` signature, not `**kwargs`.

    A permissive fake would swallow an argument the deployed acceptor does not take, and the failure
    would surface live, on an exit, after protection was released. This raises `TypeError` here
    instead — the same way the actual `UiFeedStrategy` would.

    `strategy_id` was added back on 2026-08-23 for kumo-trading-platform issue 462, and it is a DIFFERENT ARGUMENT
    with the same name as the one deleted in #245/#252. That earlier one was the FILTER — which
    resting order to cancel and wait on — and passing the lane made it match nothing, so the
    cancel-confirm step returned True having waited for nothing: a naked position presenting as a
    successful exit. The new one is the CANCELLER, the authorisation identity. Cockpit keeps the
    filter pinned to MANUAL-001, who actually holds the stops, and verified that separation on their
    side before merging. The two must never be collapsed again."""

    def __init__(self, result=True, *, calls=None):
        self._result = result
        self._calls = calls

    async def release_for_exit(self, instrument_id: str, qty, side: str = "SELL",
                               strategy_id: str | None = None) -> bool:
        if self._calls is not None:
            self._calls.append((instrument_id, qty, side, strategy_id))
        return self._result


def _position(symbol: str, qty: int):
    return SimpleNamespace(instrument_id=InstrumentId.from_str(f"{symbol}.XNAS"), quantity=qty)


def _account(equity=100_000.0, ccy_total=True):
    return SimpleNamespace(
        id="ALPACA-master",
        balance_total=lambda ccy: (SimpleNamespace(as_double=lambda: equity)
                                   if ccy_total else None))


def _strategy(positions=(), equity=100_000.0, instrument=object(),
              status=OrderStatus.SUBMITTED, accounts=None, cached_order=True):
    def broker_equity():
        if equity is None:
            raise RuntimeError("no broker account snapshot yet")
        return equity

    submitted, made = [], []

    def market(**kw):
        made.append(kw)
        return SimpleNamespace(**kw)

    order = SimpleNamespace(status=status, last_event=SimpleNamespace(reason="denied by risk engine"))
    return SimpleNamespace(
        cache=SimpleNamespace(
            positions_open=lambda **kw: list(positions),
            instrument=lambda iid: instrument,
            order=lambda coid: (order if cached_order else None),
            accounts=lambda: (list(accounts) if accounts is not None else [_account(equity)]),
        ),
        order_factory=SimpleNamespace(market=market),
        submit_order=submitted.append,
        broker_equity=broker_equity,
    ), submitted, made


def test_positions_reports_the_whole_account_not_just_this_strategy():
    """The runner subtracts what it OWNS from this to find foreign holdings and leave them alone.
    Filtering by strategy_id here would make `foreign` permanently empty and delete that check
    without any test failing — the runner would then read a manual position as one of ours that
    'fell out of the ranking', and sell it."""
    seen = {}
    strat, _, _ = _strategy(positions=[_position("AAPL", 10), _position("MSFT", 5)])
    strat.cache.positions_open = lambda **kw: seen.update(kw) or [_position("AAPL", 10)]
    NautilusBroker(strategy=strat, instrument_ids=[AAPL]).positions()
    assert "strategy_id" not in seen, "filtered the account down to this strategy's own positions"


def test_positions_sums_by_symbol():
    strat, _, _ = _strategy(positions=[_position("AAPL", 10), _position("AAPL", 5),
                                       _position("MSFT", 3)])
    assert NautilusBroker(strategy=strat, instrument_ids=[AAPL, MSFT]).positions() == {
        "AAPL": 15, "MSFT": 3}


def test_submit_carries_the_deterministic_client_order_id():
    """`AlpacaBroker` relied on Alpaca rejecting a duplicate id so a replayed session could not
    double-fill. Nautilus mints its own ids, so letting it do that here would drop that guard."""
    strat, submitted, made = _strategy()
    req = OrderRequest("AAPL", "BUY", 10, "2026-03-02", strategy_id="MOMENTUM-002")
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(req)
    assert r.ok and len(submitted) == 1
    assert str(made[0]["client_order_id"]) == req.client_order_id


def test_submit_refuses_a_symbol_that_is_not_subscribed():
    strat, submitted, _ = _strategy()
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("TSLA", "BUY", 1, "2026-03-02", strategy_id="MOMENTUM-002"))
    assert not r.ok and submitted == []


def test_submit_refuses_when_the_instrument_is_not_in_the_cache():
    """Nautilus needs the definition to size and price the order; without it the submit raises deep
    in the exec engine, well after the journal has already recorded the intent."""
    strat, submitted, _ = _strategy(instrument=None)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", "BUY", 1, "2026-03-02", strategy_id="MOMENTUM-002"))
    assert not r.ok and submitted == []


def test_a_denied_order_is_reported_as_failure_not_success():
    """`submit_order()` returning means Nautilus ACCEPTED THE COMMAND, not that the order lives. A
    local denial -- risk engine, or a duplicate ClientOrderId -- lands synchronously as an order
    STATE. Reading the return as success let the runner write durable position state for an order
    Nautilus had already denied: phantom ownership on a denied buy, and on a denied sell the
    strategy forgetting it owns a live position that the next session then treats as foreign."""
    strat, submitted, _ = _strategy(status=OrderStatus.DENIED)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", "BUY", 1, "2026-03-02", strategy_id="MOMENTUM-002"))
    assert submitted, "the order was still handed to nautilus"
    assert not r.ok
    assert "denied" in r.detail and "risk engine" in r.detail


def test_a_rejected_order_is_reported_as_failure():
    strat, _, _ = _strategy(status=OrderStatus.REJECTED)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", "SELL", 1, "2026-03-02", strategy_id="MOMENTUM-002"))
    assert not r.ok


def test_an_order_missing_from_the_cache_is_not_success():
    """If it is not in the cache, nothing can be claimed about it."""
    strat, _, _ = _strategy(cached_order=False)
    r = NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", "BUY", 1, "2026-03-02", strategy_id="MOMENTUM-002"))
    assert not r.ok


def test_exit_sends_nothing_when_release_is_not_confirmed():
    """FSM and VCTR were both refused `available: 0` while their own resting protective stop held
    the full quantity — the position was real, just fully reserved. `release_for_exit` returning
    False means the reservation did not clear in time; the position KEEPS its protection, which is
    the safe end state, so nothing may reach the venue. Sending anyway on an unconfirmed release is
    how a naked position happens."""
    strat, submitted, _ = _strategy()
    r = asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(False),
                                   instrument_ids=[AAPL]).exit(
        OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="MOMENTUM-002")))
    assert not r.ok and submitted == [], "sent an order the release never confirmed"


def test_exit_submits_once_release_is_confirmed():
    strat, submitted, made = _strategy()
    req = OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="MOMENTUM-002")
    r = asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(True),
                                   instrument_ids=[AAPL]).exit(req))
    assert r.ok and len(submitted) == 1
    assert str(made[0]["client_order_id"]) == req.client_order_id


def test_exit_calls_feed_not_the_platform_strategy():
    """`feed` (cockpit's `UiFeedStrategy`) and `strategy` (the Nautilus rotation strategy `submit()`
    uses) share no base beyond Nautilus's own `Strategy` and no `__getattr__` delegation between
    them. `release_for_exit` exists ONLY on `feed` — calling it on `strategy` raises AttributeError
    and, with nothing wrapping `_submit`, aborts the whole session after the first symbol's intent
    row is already journalled (found and reverted same night, before it shipped)."""
    strat, _, _ = _strategy()
    calls = []
    r = asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(True, calls=calls),
                                   instrument_ids=[AAPL]).exit(
        OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="MOMENTUM-002")))
    assert r.ok and calls, "release_for_exit on feed was never called"


def test_exit_NAMES_the_lane_as_the_canceller():
    """kumo-trading-platform issue 462: a lane's exit could cancel ANOTHER lane's protective stop.

    `release_for_exit` carried no owner, so cockpit resolved it internally as `self.id` on the feed —
    MANUAL-001. Every exit from every lane arrived as MANUAL-001, and "MOMENTUM exiting its own stop"
    and "MOMENTUM exiting BCTROT's" were literally the same call. That is the #437 incident,
    mechanically.

    Passing the lane makes cockpit's `proxy` concession flip to False and the strict ownership rule
    engage on the exit path for the first time: MOMENTUM exiting BCTROT's WHD is now REFUSED rather
    than silently cancelling BCTROT's protection.

    THIS IS NOT THE ARGUMENT THAT WAS DELETED IN #245/#252, despite the identical name. That one was
    the FILTER — which resting order to cancel and wait on — and the lane matched nothing there, so
    the wait cleared instantly against a stop still resting and returned True. This one is the
    CANCELLER. Same name, opposite consequence; see `_Feed`.
    """
    calls: list = []
    strat, _, _ = _strategy()
    r = asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(True, calls=calls),
                                   instrument_ids=[AAPL]).exit(
        OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="BCTROT-004")))
    assert r.ok
    assert calls, "release_for_exit was never called"
    assert calls[0][3] == "BCTROT-004", (
        f"the release named {calls[0][3]!r}; cockpit authorises the cancel against this value, so a "
        f"wrong or absent lane either refuses a lane's own stop or permits it to cancel another's")


def test_the_feed_double_matches_the_signature_RUNNING_IN_THE_CONTAINER():
    """Pinned against an observation, not against cockpit's repo.

    The stale-build incident of 2026-08-22 is why: the build log was byte-identical to a correct one
    and only the artifact stamp knew the wrong commit had shipped. So the acceptor's shape is
    recorded as what `inspect.signature` returned INSIDE the running api container.

    If this repo passes an argument the deployed acceptor does not take, every exit dies with
    `TypeError` — the 2026-08-17 failure, on the exit path, with protection already suppressed.
    """
    import inspect

    actual = tuple(inspect.signature(_Feed.release_for_exit).parameters)
    assert actual == DEPLOYED_RELEASE_SIGNATURE, (
        f"the double binds {actual}, the deployed acceptor takes {DEPLOYED_RELEASE_SIGNATURE}; a "
        f"double that disagrees with the running acceptor cannot test the call")


def test_the_release_is_told_the_LANE_not_the_feed():
    """Passing the feed's own id would be indistinguishable from passing nothing.

    MANUAL-001 is exactly what cockpit already defaults to, so a bug that sent the feed id would
    restore the old behaviour while looking like the fix — `proxy` would be False and the
    authorisation check would pass for every lane, on every instrument.
    """
    calls: list = []
    strat, _, _ = _strategy()
    asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(True, calls=calls),
                               instrument_ids=[AAPL]).exit(
        OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="QC345-003")))
    assert calls[0][3] not in ("MANUAL-001", None, ""), (
        "the release was told the feed's identity or nothing at all, which authorises every lane "
        "against every instrument while appearing to implement the ownership rule")


def test_exit_fails_loudly_when_feed_is_not_wired():
    """`feed=None` must be a named, diagnosable failure — not a fall-through to a plain `submit()`,
    which would silently restore the exact reservation bug this exists to fix."""
    strat, submitted, _ = _strategy()
    r = asyncio.run(NautilusBroker(strategy=strat, instrument_ids=[AAPL]).exit(
        OrderRequest("AAPL", "SELL", 10, "2026-08-19", strategy_id="MOMENTUM-002")))
    assert not r.ok and submitted == []
    assert "feed" in r.detail.lower() and "not wired" in r.detail.lower()


def test_exit_refuses_a_symbol_that_is_not_subscribed_before_asking_for_release():
    strat, submitted, _ = _strategy()
    called = []
    r = asyncio.run(NautilusBroker(strategy=strat, feed=_Feed(True, calls=called),
                                   instrument_ids=[AAPL]).exit(
        OrderRequest("TSLA", "SELL", 1, "2026-08-19", strategy_id="MOMENTUM-002")))
    assert not r.ok and submitted == [] and not called


def test_equity_raises_rather_than_returning_a_plausible_zero():
    """The daily-loss halt anchors on equity. A zero would not look like an error — it would make
    the limit unreachable and quietly disarm the only automatic stop this strategy has."""
    strat, _, _ = _strategy(equity=None)
    with pytest.raises(RuntimeError):
        NautilusBroker(strategy=strat, instrument_ids=[AAPL]).equity()


def test_equity_is_net_liquidation_not_the_nautilus_cash_ledger():
    """Nautilus `AccountState` models CASH ONLY, and cockpit's Alpaca client puts the cash figure in
    AccountBalance.total. Reading it would compare $20k cash against a $100k session-start equity
    and halt for a drawdown that never happened — while missing a real mark-to-market drawdown,
    because cash does not move when positions do. It comes off the broker's own snapshot instead."""
    strat, _, _ = _strategy(equity=123_456.0)
    b = NautilusBroker(strategy=strat, instrument_ids=[AAPL, InstrumentId.from_str("BAC.XNYS")])
    assert b.equity() == 123_456.0
    assert not hasattr(strat, "portfolio"), "equity must not be reachable via portfolio/venue again"


def test_status_is_compared_as_an_enum_not_a_string():
    """`str(OrderStatus.DENIED)` is "2" — the enum stringifies to its integer value. A substring
    test for "DENIED" therefore never matches, and every denied order reads as accepted. This
    guards the fix against being "simplified" back into a string compare."""
    assert "DENIED" not in str(OrderStatus.DENIED), "premise: the enum does not stringify by name"
    strat, _, _ = _strategy(status=OrderStatus.DENIED)
    assert not NautilusBroker(strategy=strat, instrument_ids=[AAPL]).submit(
        OrderRequest("AAPL", "BUY", 1, "2026-03-02", strategy_id="MOMENTUM-002")).ok


def test_last_price_falls_back_to_the_freshest_bar():
    """The strategy subscribes to BARS ONLY — no quotes, no trades — so the tick caches are
    permanently empty. Without a bar fallback, last_price() always returned None, sizing refused
    every entry with "no live price to size against", and a decided session submitted nothing.
    That is exactly what happened on the first live run: 8 names ranked, 0 orders."""
    strat, _, _ = _strategy()
    strat.cache.price = lambda iid: None
    strat.cache.quote_tick = lambda iid: None
    strat.cache.trade_tick = lambda iid: None
    strat.cache.bar = lambda bt: SimpleNamespace(close=SimpleNamespace(as_double=lambda: 42.5))
    strat._bar_type = lambda iid: "AAPL.XNAS-1-DAY-LAST-EXTERNAL"
    assert NautilusBroker(strategy=strat, instrument_ids=[AAPL]).last_price("AAPL") == 42.5


def test_last_price_is_none_when_there_is_no_bar_either():
    """Still no fabrication — an instrument with nothing cached must size nothing."""
    strat, _, _ = _strategy()
    for a in ("price", "quote_tick", "trade_tick"):
        setattr(strat.cache, a, lambda iid: None)
    strat.cache.bar = lambda bt: None
    strat._bar_type = lambda iid: "AAPL.XNAS-1-DAY-LAST-EXTERNAL"
    assert NautilusBroker(strategy=strat, instrument_ids=[AAPL]).last_price("AAPL") is None


def test_last_price_without_a_bound_is_unchanged():
    """Opt-in, matching kumo-trading-platform's item 4 on the same problem: no `max_age_ns` must behave
    exactly as before, existence-only, even against a bar timestamped days ago."""
    strat, _, _ = _strategy()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 10 * 24 * 3600 * 1_000_000_000)
    for a in ("price", "quote_tick", "trade_tick"):
        setattr(strat.cache, a, lambda iid: None)
    strat.cache.bar = lambda bt: SimpleNamespace(
        close=SimpleNamespace(as_double=lambda: 42.5), ts_event=0)   # 10 days old
    strat._bar_type = lambda iid: "AAPL.XNAS-1-DAY-LAST-EXTERNAL"
    assert NautilusBroker(strategy=strat, instrument_ids=[AAPL]).last_price("AAPL") == 42.5


def test_last_price_with_a_bound_rejects_a_stale_bar_instead_of_fabricating():
    """Both this repo's staleness guards were existence checks — `last_price` returned SOMETHING
    the instant any bar existed, however old, so the guards downstream had nothing to reject.
    A bound must actually refuse an observation older than it, the same way the no-bar-at-all case
    already refuses (#54's sibling on the entry/exit sizing path)."""
    strat, _, _ = _strategy()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 10 * 24 * 3600 * 1_000_000_000)   # day 10
    for a in ("price", "quote_tick", "trade_tick"):
        setattr(strat.cache, a, lambda iid: None)
    strat.cache.bar = lambda bt: SimpleNamespace(
        close=SimpleNamespace(as_double=lambda: 42.5), ts_event=0)   # day 0 -- 10 days stale
    strat._bar_type = lambda iid: "AAPL.XNAS-1-DAY-LAST-EXTERNAL"
    broker = NautilusBroker(strategy=strat, instrument_ids=[AAPL])
    one_day_ns = 24 * 3600 * 1_000_000_000
    assert broker.last_price("AAPL", max_age_ns=one_day_ns) is None, (
        "a bar ten days old must be rejected under a one-day bound, not returned as fresh")


def test_last_price_with_a_bound_accepts_an_observation_inside_it():
    strat, _, _ = _strategy()
    now = 10 * 24 * 3600 * 1_000_000_000
    strat.clock = SimpleNamespace(timestamp_ns=lambda: now)
    for a in ("price", "quote_tick", "trade_tick"):
        setattr(strat.cache, a, lambda iid: None)
    strat.cache.bar = lambda bt: SimpleNamespace(
        close=SimpleNamespace(as_double=lambda: 42.5), ts_event=now - 60_000_000_000)  # 60s old
    strat._bar_type = lambda iid: "AAPL.XNAS-1-DAY-LAST-EXTERNAL"
    broker = NautilusBroker(strategy=strat, instrument_ids=[AAPL])
    one_day_ns = 24 * 3600 * 1_000_000_000
    assert broker.last_price("AAPL", max_age_ns=one_day_ns) == 42.5


def test_last_price_with_a_bound_skips_a_bare_price_with_no_timestamp():
    """`cache.price()` returns a bare `Price` with no `ts_event` at all. A bound cannot honour an
    unaged value, so it must skip straight to a source that carries one rather than either trusting
    it blindly or refusing everything the moment any bound is set."""
    strat, _, _ = _strategy()
    now = 10 * 24 * 3600 * 1_000_000_000
    strat.clock = SimpleNamespace(timestamp_ns=lambda: now)
    strat.cache.price = lambda iid: SimpleNamespace(as_double=lambda: 999.0)   # no ts_event
    strat.cache.quote_tick = lambda iid: None
    strat.cache.trade_tick = lambda iid: SimpleNamespace(
        last_px=SimpleNamespace(as_double=lambda: 42.5), ts_event=now - 60_000_000_000)
    broker = NautilusBroker(strategy=strat, instrument_ids=[AAPL])
    one_day_ns = 24 * 3600 * 1_000_000_000
    assert broker.last_price("AAPL", max_age_ns=one_day_ns) == 42.5, (
        "the unaged bare price must not be trusted; the aged trade tick should win instead")
