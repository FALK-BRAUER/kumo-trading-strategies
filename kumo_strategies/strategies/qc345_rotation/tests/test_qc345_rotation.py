"""Tests for the QC345 Nautilus binding.

Same standard as `test_momentum_rotation.py`: no `BacktestEngine` (a second engine per interpreter
aborts natively), and the clock is a REAL Nautilus `TestClock` rather than a stub — the whole reason
the trigger lives on Nautilus's clock is that a hand-rolled one could not be exercised by a backtest,
and testing against a hand-rolled stub would give that back.

What is pinned: identity, warmup, the rule that book state moves only on order EVENTS, and — new for
this strategy — that the MONTHLY cadence is the backtest's own rule rather than a second copy.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.strategies.qc345_rotation.nautilus import (
    STRATEGY_NAME,
    STRATEGY_TAG,
    QC345RotationStrategy,
    warmup_bars_needed,
)
from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
from kumo_strategies.strategies.qc345_rotation import ExitConfig, QC345RotationConfig, rebalance_dates


def _live_cfg(**kw) -> QC345RotationConfig:
    base = {
        "momentum_price_field": "close",
        "corporate_action_window": 252,
        "exits": ExitConfig(stall_days=12),
    }
    base.update(kw)
    return QC345RotationConfig(**base)


def _strategy(**kw) -> QC345RotationStrategy:
    return QC345RotationStrategy(
        cfg=kw.pop("cfg", _live_cfg()),
        source=kw.pop("source", StaticList(["AAA", "BBB", "CCC"])),
        instrument_ids=kw.pop("instrument_ids", []),
        **kw,
    )


#: A deliberately short lookback for the CADENCE tests. Warmup and cadence are independent rules, and
#: proving the monthly rule should not require generating fourteen months of bars per symbol. The
#: warmup arithmetic itself is pinned separately, against the real default config.
_FAST = _live_cfg(
    lookback_sessions=5,
    corporate_action_window=5,
    liquidity_window=3,
    min_liquidity_history=3,
    realized_vol_window=3,
    min_realized_vol_history=3,
    portfolio_size=3,
    exits=ExitConfig(),
)


def _registered(now: str = "2026-03-02 12:00", **kw):
    """A strategy wired to a real TestClock. Returns (strategy, clock)."""
    from nautilus_trader.common.component import TestClock
    from nautilus_trader.test_kit.stubs.component import TestComponentStubs
    from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs

    from kumo_strategies.runtime.calendar import WeekdayCalendar

    clock = TestClock()
    clock.set_time(pd.Timestamp(now, tz="UTC").value)
    kw.setdefault("cfg", _FAST)
    s = _strategy(calendar=kw.pop("calendar", WeekdayCalendar()), **kw)
    s.register(TestIdStubs.trader_id(), TestComponentStubs.portfolio(),
               TestComponentStubs.msgbus(), TestComponentStubs.cache(), clock)
    s._arm(s.clock.utc_now())
    return s, clock


def _fire(clock, when: str) -> None:
    for handler in clock.advance_time(pd.Timestamp(when, tz="UTC").value):
        handler.handle()


def _row(day: str, sym: str, close: float = 100.0) -> dict:
    return {"ticker": sym, "date": pd.Timestamp(day).normalize(), "open": close, "high": close,
            "low": close, "close": close, "volume": 1_000_000.0}


#: Sessions that FIT the rolling window. `_bars` is a deque capped at `need + 5`, so a longer span is
#: silently truncated to its tail — the first attempt fed 65 sessions and the panel began mid-March,
#: which made 2026-03-02 correctly NOT the first session present and the cadence test fail for a
#: reason production would never hit. 2026-03-02 is the first March business day (the 1st is a
#: Sunday); 2026-03-10 is an ordinary Tuesday.
_SESSIONS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-02-23", "2026-03-10", freq="B")]
_LIVE_SESSIONS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-02-20", "2026-03-03", freq="B")]


def _fill_panel(s, sessions: list[str] | None = None, symbols: str = "ABCDEFGH") -> None:
    """Give every symbol a real bar on every session — the shape `_panel()` actually consumes.

    Deliberately NOT `range(need)` ints, which is enough for MOMENTUM's warmth check but would make
    `_panel()` raise here. A double that cannot represent production is the bug.
    """
    sessions = sessions if sessions is not None else _SESSIONS
    for sym in symbols:
        for day in sessions:
            s._bars[sym].append(_row(day, sym))


# -- identity ------------------------------------------------------------------------------------


def test_identity_is_qc345_003():
    """`QC345-003` is the cycle-attribution key — the NETTING position id is
    {instrument}-{strategy_id}. It must be set via the CONFIG so the order-id tag binds at
    registration; `change_id()` would leave the tag at its default and desync client_order_ids.

    Tag 003, not 001: Nautilus requires `order_id_tag` unique across every strategy in one trader.
    MANUAL holds 001 with live positions keyed to it and MOMENTUM holds 002, so registering this one
    as 001 does not degrade — it stops the node booting at `add_strategy`.
    """
    s = _strategy()
    assert (STRATEGY_NAME, STRATEGY_TAG) == ("QC345", "003")
    assert s.config.strategy_id == "QC345"
    assert s.config.order_id_tag == "003"


def test_default_helper_uses_the_promoted_live_candidate():
    cfg = _strategy()._cfg
    assert cfg.momentum_price_field == "close"
    assert cfg.corporate_action_window == 252
    assert cfg.exits.stall_days == 12


def test_the_three_registered_tags_are_distinct():
    """The collision this file exists to prevent, stated as an invariant rather than a comment.

    MANUAL-001 and MOMENTUM-002 are live. A fourth strategy added with any of these three fails the
    same way, which is why the cockpit registry must ALLOCATE tags rather than trust the next author.
    """
    assert len({"001", "002", STRATEGY_TAG}) == 3


# -- warmup --------------------------------------------------------------------------------------


def test_warmup_is_the_LONGEST_window_not_the_sum_of_them():
    """MAX, not sum — and this is a corrected number, not an assumed one.

    Every window is a TRAILING window ending at the same bar, so they overlap: the 41 sessions the
    split guard inspects are the most recent 41 of the 254 momentum already needs. A first version
    summed them and reported 295, overstating by two months.

    254 is MEASURED against `build_feature_panel`: binary-searching a synthetic panel, the first bar
    count at which any name becomes `eligible` is exactly 254. Pinned because the failure is SILENT —
    decide while short and `eligible` merely goes empty, so the strategy appears to choose to hold
    nothing rather than to be unready.
    """
    cfg = QC345RotationConfig()
    assert warmup_bars_needed(cfg) == 254 == cfg.lookback_sessions + 2
    assert warmup_bars_needed(cfg) < cfg.lookback_sessions + 2 + cfg.corporate_action_window


def test_a_long_split_window_still_dominates_when_it_is_the_longest():
    """The counter-case, so `max` is not silently equivalent to "just use the momentum window".

    The raw-close guard requires `corporate_action_window >= lookback_sessions`, which is a config an
    operator can legitimately reach (#319). There the split window IS the binding constraint.
    """
    cfg = _live_cfg(lookback_sessions=20, corporate_action_window=100)
    assert warmup_bars_needed(cfg) == 101


def test_a_shorter_lookback_shortens_warmup():
    """The fixture's own property: warmup must TRACK config, not be a constant that happens to match.
    Without this, `warmup_bars_needed` could return a hardcoded 295 and the test above would pass.
    """
    assert warmup_bars_needed(QC345RotationConfig(lookback_sessions=60)) < warmup_bars_needed(
        QC345RotationConfig(lookback_sessions=252)
    )


def test_refuses_to_trade_before_warmup():
    s = _strategy()
    assert s.warm is False


def test_warms_as_a_universe_not_per_symbol():
    """A cross-sectional strategy ranks names AGAINST each other. Three warm names out of fifty is
    not an early answer, it is a different and wrong strategy — so warmth is all-or-nothing on the
    count of symbols that have cleared the bar requirement.
    """
    # `momentum_price_field="close"` because the adapter now REFUSES a field no live bar can supply,
    # and the config's own default (`close_split_dividend`) is one of those — it works in production
    # only because cockpit overrides it. A test constructing the bare default was constructing a
    # configuration that cannot run live, which is exactly what the guard exists to make impossible.
    cfg = QC345RotationConfig(portfolio_size=5, momentum_price_field="close")
    s = _strategy(cfg=cfg)
    for sym in "ABCD":                       # one short of portfolio_size
        s._bars[sym].extend(range(s._need))
    assert s.warm is False, "ranked a universe smaller than the portfolio it must fill"
    s._bars["E"].extend(range(s._need))
    assert s.warm is True


# -- book state ----------------------------------------------------------------------------------


def _event(sym: str, side: str = "BUY", coid: str = "O-20260831-000001"):
    """A REAL `InstrumentId`, not a SimpleNamespace.

    Production reads `str(event.instrument_id.symbol)`, and a Nautilus `Symbol` stringifies to its
    value. A namespace double stringifies to "namespace(value='AAA')" instead, so every lookup missed
    and the tests failed for a reason production would never hit — the drifted double this repo keeps
    paying for. Use the real type and it rejects what production rejects.
    """
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import InstrumentId

    # A REAL `OrderSide` too, for the same reason (kumo-trading-platform issue 748). Production's `OrderFilled`
    # carries a side and a client order id, and the handlers now read both: a SELL cannot complete an
    # entry, and an order another component minted is ignored. A double without them was testing
    # handlers that could not tell their own fills from a protective stop's.
    return SimpleNamespace(
        instrument_id=InstrumentId.from_str(f"{sym}.XNAS"),
        order_side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
        client_order_id=coid,
    )


# -- terminal recording (kumo-trading-platform issue 383) -------------------------------------------------------
# These must not be satisfiable by the `getattr(runner, "record_terminal", None)` no-op. That is not
# hypothetical: MOMENTUM has called `_record_terminal` from all four handlers for weeks, and the
# 2026-08-20 live session produced 22 OrderFilled events and ZERO terminal rows, because cockpit's
# SessionGateway never defined the method. A test that only asserts "the handler calls it" passes
# in exactly that broken state, which is why the runner below genuinely implements it and the
# assertions are on what it RECEIVED.


class _RecordingRunner:
    """A runner that actually implements `record_terminal`, so a no-op cannot satisfy the test."""

    def __init__(self):
        self.terminal = []

    async def record_terminal(self, session, symbol, ok, detail):
        self.terminal.append({"session": session, "symbol": symbol, "ok": ok, "detail": detail})


def _tagged_event(s, sym: str, session: str = "2026-08-20"):
    """An order event whose order carries the `session:` tag production reads the session from.

    `_record_terminal` recovers the session from `order.tags`, not from a parallel map, because that
    is what survives a restart between submit and fill. So the double has to put a real tagged order
    in the cache or the lookup returns None and the test would pass for the wrong reason.
    """
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Quantity

    iid = InstrumentId.from_str(f"{sym}.XNAS")
    order = s.order_factory.market(
        instrument_id=iid, order_side=OrderSide.BUY, quantity=Quantity.from_int(1),
        time_in_force=TimeInForce.DAY, tags=[f"session:{session}"])
    s.cache.add_order(order)
    # THE SIDE COMES FROM THE REAL ORDER, not a literal. Production's `OrderFilled` carries the side
    # of the order that filled, and the handlers now read it (kumo-trading-platform issue 748: a SELL cannot
    # complete an entry). Taking it from the order this double actually built keeps the two from
    # drifting the way the missing field just did.
    return SimpleNamespace(instrument_id=iid, client_order_id=order.client_order_id,
                           order_side=order.side,
                           reason="insufficient qty available")


def test_a_fill_is_recorded_as_a_terminal_outcome():
    async def go():
        runner = _RecordingRunner()
        s, _clock = _registered(session_runner=runner)
        s._loop = asyncio.get_running_loop()
        s._pending["AAA"] = "enter"
        s.on_order_filled(_tagged_event(s, "AAA"))
        await asyncio.sleep(0.01)
        return runner

    runner = asyncio.run(go())
    assert len(runner.terminal) == 1, "a fill left no terminal row — the #383 gap is back"
    assert runner.terminal[0] == {"session": "2026-08-20", "symbol": "AAA",
                                  "ok": True, "detail": "filled"}


def test_a_venue_refusal_is_recorded_with_its_reason():
    """The row that matters most: without it the retry counter (`phase == "terminal" and not ok`)
    never advances and a refused symbol is silently treated as done."""
    async def go():
        runner = _RecordingRunner()
        s, _clock = _registered(session_runner=runner)
        s._loop = asyncio.get_running_loop()
        s._pending["AAA"] = "exit"
        s.on_order_rejected(_tagged_event(s, "AAA"))
        await asyncio.sleep(0.01)
        return runner

    runner = asyncio.run(go())
    assert len(runner.terminal) == 1
    rec = runner.terminal[0]
    assert rec["ok"] is False
    assert rec["detail"] == "insufficient qty available", "the venue's reason was discarded"


def test_a_runner_without_record_terminal_degrades_silently():
    """The live state until cockpit's half lands, and the behaviour the docstring promises. A
    missing method must not raise inside a Nautilus event handler — that would turn an observability
    gap into a dead subscription."""
    s, _clock = _registered(session_runner=SimpleNamespace())   # no record_terminal at all
    s._pending["AAA"] = "enter"
    s.on_order_filled(_tagged_event(s, "AAA"))                  # must not raise
    assert "AAA" in s._held


def test_book_state_moves_only_on_fills():
    """Submit is a request; a fill is a fact. Marking held at submit is how a rejected entry becomes
    a phantom position that the strategy then refuses to re-enter."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    assert "AAA" not in s._held
    s.on_order_filled(_event("AAA"))
    assert "AAA" in s._held
    assert s._pending == {}


def test_a_rejected_order_does_not_enter_the_book():
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_rejected(_event("AAA"))
    assert "AAA" not in s._held, "a rejected submit marked the position held"
    assert s._pending == {}


def test_a_denied_exit_leaves_the_position_held():
    """The mirror case. Forgetting the pending intent must not also forget that we still own it."""
    s = _strategy()
    s._held.add("AAA")
    s._pending["AAA"] = "exit"
    s.on_order_denied(_event("AAA"))
    assert "AAA" in s._held, "a denied exit dropped the position from the book"


# -- monthly cadence -----------------------------------------------------------------------------


def test_the_cadence_rule_is_the_backtests_own_function():
    """One derivation, not two.

    `_is_rebalance` delegates to the pure `rebalance_dates()` that the verified backtest uses. A
    second, adapter-local notion of "first session of the month" would be a second derivation of one
    fact — and two derivations of one fact disagree, which is how research and production drift
    apart without anyone changing either on purpose.
    """
    s = _strategy()
    sessions = pd.date_range("2026-01-01", "2026-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})
    expected = set(rebalance_dates(panel["date"]))

    for day in sessions:
        assert s._is_rebalance(day, panel) is (day in expected)


def test_only_the_first_session_of_a_month_is_a_rebalance():
    """The rule stated independently of the function, so a change to BOTH is required to break it
    silently. January 2026 opens on the 1st (a Thursday); February on the 2nd (the 1st is a Sunday)."""
    s = _strategy()
    sessions = pd.date_range("2026-01-01", "2026-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})

    assert s._is_rebalance(pd.Timestamp("2026-01-01"), panel) is True
    assert s._is_rebalance(pd.Timestamp("2026-01-02"), panel) is False
    assert s._is_rebalance(pd.Timestamp("2026-02-02"), panel) is True
    assert s._is_rebalance(pd.Timestamp("2026-02-03"), panel) is False


def test_forced_rebalance_dates_are_seeded_from_config_at_construction():
    """The seam that actually matters: `_forced_rebalance` must be populated from
    `cfg.forced_rebalance_dates` BEFORE the first `_arm()`, not only reachable by calling the msgbus
    handler directly. cockpit persists this field in the `qc345` settings domain and rebuilds `cfg`
    on every process start, so THIS is what survives a container recreate -- the msgbus topic is only
    for edits that should not require a restart. A test that only drives
    `_on_rebalance_override()` would pass even if the constructor never read `cfg` at all."""
    cfg = _live_cfg(forced_rebalance_dates=("2026-01-15", "2026-01-16"))
    s, _clock = _registered(cfg=cfg)
    assert s._forced_rebalance == {pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-16")}

    sessions = pd.date_range("2026-01-01", "2026-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})
    assert s._is_rebalance(pd.Timestamp("2026-01-15"), panel) is True   # seeded, not published


def test_forced_rebalance_date_overrides_the_monthly_rule():
    """The operator override: a session that `rebalance_dates()` would refuse is still a rebalance
    once the operator has forced it. Needed because a strategy enabled mid-month would otherwise sit
    idle until the next natural month-start -- see kumo-trading-strategies live incident, 2026-08-20."""
    s = _strategy()
    sessions = pd.date_range("2026-01-01", "2026-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})

    assert s._is_rebalance(pd.Timestamp("2026-01-15"), panel) is False       # before the override
    s._on_rebalance_override({"date": "2026-01-15"})
    assert s._is_rebalance(pd.Timestamp("2026-01-15"), panel) is True        # after it


def test_forced_rebalance_is_additive_not_a_replacement():
    """Forcing one date must not disturb the natural monthly dates -- the override is a second,
    disjoint set of dates to ALSO treat as a rebalance, not a second rule that could disagree with
    `rebalance_dates()` about a date it already covers."""
    s = _strategy()
    sessions = pd.date_range("2026-01-01", "2026-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})

    s._on_rebalance_override({"date": "2026-01-15"})
    assert s._is_rebalance(pd.Timestamp("2026-01-01"), panel) is True    # still the natural Jan date
    assert s._is_rebalance(pd.Timestamp("2026-02-02"), panel) is True    # still the natural Feb date
    assert s._is_rebalance(pd.Timestamp("2026-01-16"), panel) is False   # a day we never forced


def test_rebalance_override_ignores_a_payload_with_no_date():
    """A malformed message must not crash the strategy or silently force an undefined date."""
    s = _strategy()
    s._on_rebalance_override({})
    assert s._forced_rebalance == set()
    s._on_rebalance_override(None)
    assert s._forced_rebalance == set()


def test_rebalance_override_ignores_an_unparseable_date_without_raising():
    """A `pd.Timestamp()` parse failure inside a msgbus callback is not the same failure as one in
    this strategy's own call stack -- it can leave the SUBSCRIPTION dead rather than just the one
    message, which then looks exactly like "no override was ever sent." Must log and return, never
    propagate."""
    s = _strategy()
    s._on_rebalance_override({"date": "not-a-real-date-at-all"})   # must not raise
    assert s._forced_rebalance == set()


def test_rebalance_override_arrives_live_over_the_real_msgbus():
    """Not just a unit-tested handler -- the subscription itself, over Nautilus's real message bus,
    the same mechanism `_on_broker_account` already proves works for live-pushing state into a
    running strategy without a restart. This is the whole point of the override: an operator setting
    it must not need to bounce the process."""
    s, _clock = _registered(now="2026-01-05 12:00")
    assert s._is_rebalance(pd.Timestamp("2026-01-20"), pd.DataFrame({"date": []})) is False

    # `_registered` deliberately does not call the full `on_start()` (it would try to request bars
    # and an instrument definition with no data client attached) -- so subscribe exactly as on_start
    # does, using the strategy's OWN topic attribute rather than a hardcoded string, which is what
    # actually proves the constructor default and the subscribe call agree.
    s.msgbus.subscribe(s._rebalance_override_topic, s._on_rebalance_override)
    s.msgbus.publish(s._rebalance_override_topic, {"date": "2026-01-20"})

    assert pd.Timestamp("2026-01-20") in s._forced_rebalance


def test_a_non_rebalance_session_does_not_decide_and_is_not_recorded_as_missed():
    """The core of the monthly cadence.

    The alert fires every SESSION so a mid-month restart is never more than one session from noticing
    the next rebalance. Firing is therefore normal and must not decide, and — importantly — must not
    land in `missed_rebalances`, or the strategy would report ~20 missed rebalances a month and the
    signal that actually matters would be buried in noise.
    """
    s, _clock = _registered()
    _fill_panel(s)

    decided: list = []
    s._decide_for = lambda session, panel: decided.append(session)

    s._due = pd.Timestamp("2026-03-10")        # a Tuesday mid-month
    s._try_decide()

    assert decided == [], "decided on a non-rebalance session"
    assert s._due is None, "left the session pending, so it would be reported missed at the next alert"
    assert s.missed_rebalances == [], "counted an ordinary session as a missed rebalance"
    assert s.skipped_sessions == 1


def test_a_non_rebalance_session_can_still_run_exit_only_logic():
    s, _clock = _registered()
    _fill_panel(s)

    exits: list[pd.Timestamp] = []
    s._exit_only_for = lambda session: exits.append(session)
    s._due = pd.Timestamp("2026-03-10")
    s._try_decide()

    assert exits == [pd.Timestamp("2026-03-10")]
    assert s.missed_rebalances == []
    assert s.skipped_sessions == 1


def test_a_rebalance_session_does_decide():
    """The positive case — without it every assertion above passes on a strategy that never trades."""
    s, _clock = _registered()
    _fill_panel(s)

    decided: list = []
    s._decide_for = lambda session, panel: decided.append(session)

    s._due = pd.Timestamp("2026-03-02")        # first business day of March
    s._try_decide()

    assert decided == [pd.Timestamp("2026-03-02")], "skipped the month's own rebalance"


def test_a_rebalance_uses_the_source_universe_not_the_full_panel():
    cfg = _live_cfg(
        lookback_sessions=1,
        corporate_action_window=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        portfolio_size=2,
        exits=ExitConfig(),
    )
    s, _clock = _registered(cfg=cfg, source=StaticList(["B", "C"]))
    _fill_panel(s, sessions=["2026-02-26", "2026-02-27", "2026-03-02"], symbols="ABC")

    chosen: list[str] = []
    original = s._open
    s._open = lambda symbol, today: chosen.append(symbol)  # noqa: ARG005
    try:
        s._held.clear()
        s._decide_for(pd.Timestamp("2026-03-02"), s._panel())
    finally:
        s._open = original

    assert set(chosen) == {"B", "C"}
    assert s.skipped_sessions == 0


def test_orders_in_flight_defer_the_decision():
    """Deciding again while an entry is unfilled would size against a book that is about to change."""
    s, _clock = _registered()
    _fill_panel(s)
    decided: list = []
    s._decide_for = lambda session, panel: decided.append(session)

    s._pending["AAA"] = "enter"
    s._due = pd.Timestamp("2026-03-02")
    s._try_decide()

    assert decided == []
    assert s._due == pd.Timestamp("2026-03-02"), "dropped the rebalance instead of deferring it"


def test_exit_only_path_submits_forced_exits_between_rebalances():
    s, _clock = _registered()
    s._trail_exits = lambda session: {"A"}
    exited: list[str] = []
    s._close = lambda symbol: exited.append(symbol)

    s._exit_only_for(pd.Timestamp("2026-03-10"))

    assert exited == ["A"]


# -- session clock -------------------------------------------------------------------------------


def test_the_alert_re_arms_even_when_the_decision_raises():
    """A strategy that has stopped scheduling looks exactly like one that decided to hold — silent,
    and wrong for as long as nobody looks. So the re-arm happens BEFORE anything that can throw."""
    s, clock = _registered(now="2026-03-02 12:00")

    def _boom() -> None:
        raise RuntimeError("decision blew up")

    s._try_decide = _boom
    try:
        _fire(clock, "2026-03-02 14:35")
    except RuntimeError:
        pass
    assert s._armed_session is not None, "stopped scheduling after a failure"
    assert "qc345_session_decide" in clock.timer_names


def test_the_alert_alone_does_not_decide_without_bars():
    """The alert says WHEN TO LOOK; the bars say WHETHER WE MAY DECIDE. A 1-DAY bar for session D is
    emitted at D's close, so a late feed must defer to `on_bar` rather than be polled."""
    s, clock = _registered(now="2026-03-02 12:00")
    decided: list = []
    s._decide_for = lambda session, panel: decided.append(session)

    _fire(clock, "2026-03-02 14:35")
    assert s._due == pd.Timestamp("2026-03-02")
    assert decided == [], "decided on an empty panel"


def test_the_fixture_actually_contains_the_rebalance_it_asserts_on():
    """The premise, checked before anything is derived from it.

    The first version of the cadence fixture fed 65 sessions into a deque capped at `need + 5`, so the
    panel silently began mid-March and 2026-03-02 was not in it at all. `_is_rebalance` then answered
    False for a completely correct reason and the test failed misleadingly — but the same truncation
    with a different date would have made it PASS for a wrong reason. A test whose fixture cannot
    exhibit the property carries no information about it.
    """
    s, _clock = _registered()
    _fill_panel(s)
    panel = s._panel()

    dates = set(panel["date"])
    assert pd.Timestamp("2026-03-02") in dates, "the rebalance under test is not in the panel"
    assert pd.Timestamp("2026-03-10") in dates, "the non-rebalance under test is not in the panel"
    assert pd.Timestamp("2026-02-23") in dates, "no February session, so March's first is trivially first"


# -- live session path ---------------------------------------------------------------------------


class _FakeJournal:
    def __init__(self):
        self.writes = []

    async def write(self, kind, summary, *, session, detail=None, **kw):
        self.writes.append({"kind": kind, "summary": summary, "session": session, "detail": detail})


class _FakeRunner:
    def __init__(self, gate=None):
        self.calls = []
        self.gate = gate
        self.journal = _FakeJournal()

    async def run(self, panel, session, jobs=None, slot=None, opens=None):
        # THE FULL `SessionRunner` SIGNATURE, not the subset this test happened to need. A double
        # narrower than the protocol cannot observe a caller starting to pass an argument — it dies
        # with `TypeError` inside the adapter's own `except`, and the test reads as "the runner was
        # never called", which is indistinguishable from the session being blocked. That is the
        # 2026-08-17 failure exactly (an adapter passed `slot=`, its gateway did not accept it, two
        # trading days were lost), reproduced in a test double.
        self.calls.append({"session": session, "rows": len(panel), "panel_max": panel["date"].max(),
                           "slot": slot})
        if self.gate is not None:
            await self.gate.wait()
        return SimpleNamespace(state="SHADOW", decided=True, blocked=None, submitted=0)


def _live(runner, **kw):
    s, clock = _registered(session_runner=runner, **kw)
    s._loop = asyncio.get_running_loop()
    _fill_panel(s, sessions=_LIVE_SESSIONS)
    return s, clock


def test_live_alert_hands_the_session_to_the_runner():
    async def go():
        runner = _FakeRunner()
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
        return runner

    runner = asyncio.run(go())
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["session"] == "2026-03-02"
    assert call["panel_max"] == pd.Timestamp("2026-03-02")


def test_forced_rebalance_override_is_journalled_when_a_runner_is_attached():
    """An INFO log line alone does not survive a container recreate (kumo-trading-platform issue 378: recreation
    destroys the prior container's logs) -- the same class of gap this whole override exists to
    route around for BCTROT. A manual, live-money-affecting operator action must leave a durable
    row, not just a log an operator has to catch in real time."""
    async def go():
        runner = _FakeRunner()
        s, _clock = _live(runner)
        s._on_rebalance_override({"date": "2026-04-01"})
        await asyncio.sleep(0.01)     # let the scheduled coroutine actually run
        return runner

    runner = asyncio.run(go())
    assert len(runner.journal.writes) == 1
    w = runner.journal.writes[0]
    assert w["kind"] == "risk"
    assert w["session"] == "2026-04-01"
    assert w["detail"]["forced_rebalance_dates"] == ["2026-04-01"]


def test_forced_rebalance_override_does_not_journal_without_a_runner():
    """No `session_runner` attached (e.g. a bare `_strategy()` in most of this file's other tests) --
    must degrade to log-only, never raise for lack of somewhere to journal to."""
    s = _strategy()
    s._on_rebalance_override({"date": "2026-04-01"})   # must not raise
    assert s._forced_rebalance == {pd.Timestamp("2026-04-01")}


def test_a_second_alert_does_not_race_a_running_session():
    async def go():
        gate = asyncio.Event()
        runner = _FakeRunner(gate=gate)
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.sleep(0)
        s._run_session(pd.Timestamp("2026-04-01"), s._panel().loc[s._panel()["date"] <= pd.Timestamp("2026-03-03")])
        gate.set()
        await asyncio.wrap_future(s._session_task)
        return runner, s

    runner, s = asyncio.run(go())
    assert len(runner.calls) == 1
    assert s.missed_rebalances == [pd.Timestamp("2026-04-01")]


def test_no_event_loop_refuses_instead_of_falling_back_to_the_local_path():
    runner = _FakeRunner()
    s, clock = _registered(session_runner=runner)
    _fill_panel(s, sessions=_LIVE_SESSIONS)
    s._loop = None
    s._decide_for = lambda *a, **k: pytest.fail("local decision path ran in a live node")

    _fire(clock, "2026-03-02 14:35")

    assert runner.calls == []
    assert s.missed_rebalances == [pd.Timestamp("2026-03-02")]


def test_stopping_cancels_an_in_flight_session_not_just_the_timers():
    async def go():
        gate = asyncio.Event()
        runner = _FakeRunner(gate=gate)
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.sleep(0)
        task = s._session_task
        assert task is not None and not task.done()
        s.on_stop()
        await asyncio.sleep(0)
        return task, clock, s

    task, clock, s = asyncio.run(go())
    assert task.cancelled() or task.done(), "the in-flight session survived on_stop()"
    assert "qc345_session_decide" not in clock.timer_names
    assert s._due is None


def test_claims_are_passed_to_the_strategy_config():
    from nautilus_trader.model.identifiers import InstrumentId

    iid = InstrumentId.from_str("MET.XNYS")
    s = QC345RotationStrategy(
        cfg=_live_cfg(),
        source=StaticList(["MET"]),
        instrument_ids=[iid],
        external_order_claims=[iid],
    )
    assert s.config.external_order_claims == [iid]
    assert s.external_order_claims == [iid]


def test_no_claims_by_default():
    from nautilus_trader.model.identifiers import InstrumentId

    s = QC345RotationStrategy(
        cfg=_live_cfg(),
        source=StaticList(["MET"]),
        instrument_ids=[InstrumentId.from_str("MET.XNYS")],
    )
    assert not s.external_order_claims


# ==================================================================================================
# A FAILED REBALANCE MUST BE LOCATABLE, AND MUST REACH SOMEBODY (2026-08-21).
#
# QC345 has now gone live twice and booked nothing both times.
#
#   2026-08-20 13:35:00Z  KeyError: 'eligible'                     (fixed downstream, cockpit #385)
#   2026-08-21 13:35:00Z  TypeError: 'NoneType' object is not callable
#
# The second one cannot be diagnosed. The handler logs `f"...{type(exc).__name__}: {exc}"` and drops
# the traceback, so "'NoneType' object is not callable" arrives with no file, no line, no frame. The
# only narrowing available is by elimination — the cockpit's runner wraps `_decide()` in its own
# try/except that would have logged a different message, so it is somewhere in `_lifecycle()`,
# `broker.strategy_positions()`, `journal.decided_this_session()`, `_resume`, `_liquidate` or `_submit`.
# That is not a diagnosis, it is a shortlist, and it exists only because the stack was discarded.
#
# Second, and worse: `missed_rebalances` is appended at four sites and READ BY NOTHING outside these
# tests. A missed monthly rebalance is recorded into a list nobody consults. Two consecutive failures
# raised no alarm, and the Telegram transport they would have alarmed through had never been configured
# (kumo-trading-platform PR #426) — three independent silencers stacked on one failure.
# ==================================================================================================
import logging as _logging

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]



def test_a_failed_rebalance_logs_the_TRACEBACK_not_just_the_message(caplog):
    """Asserts the STACK is present, not the message text.

    A test that only checked for "failed" in the log passes against the broken version — that version
    logged "failed" perfectly well. What it could not do is tell you where.
    """
    import inspect
    from kumo_strategies.strategies.qc345_rotation import nautilus as qc345_rotation
    src = inspect.getsource(qc345_rotation.QC345RotationStrategy._session_coro)
    # Strip comments so this cannot be satisfied by prose describing the fix.
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    # BLOCK-BASED, not line-based: the fixed handler spans several lines, and a per-line match would
    # report the log call as "gone" the moment somebody wrapped it.
    assert "except Exception" in code, "no broad handler in _session_coro — did it get restructured?"
    handler = code[code.index("except Exception"):]
    assert "self.log.error" in handler and "failed" in handler, (
        "the rebalance failure log line is gone — did _session_coro get restructured?"
    )
    # NAUTILUS'S LOGGER TAKES A STRING AND NOTHING ELSE:
    #     Logger.error(self, str message, LogColor color=LogColor.RED) -> void
    # There is no `exc_info` and no `.exception()` on it, which is very likely WHY the original handler
    # dropped the stack. Demanding those here would assert against an API that cannot exist in this
    # runtime — the test-shaped version of a double that cannot represent production. So the invariant
    # is "the stack reaches the message", by whichever route the logger actually supports.
    assert any(k in handler for k in ("format_exception", "format_exc", "exc_info", ".exception(")), (
        "the rebalance failure handler discards the traceback. 'TypeError: NoneType object is not "
        "callable' with no stack is unlocatable."
    )


def test_every_broad_except_that_logs_an_error_keeps_the_traceback():
    """AIMED AT THE CLASS, NOT THE INSTANCE.

    The question is not "is this one handler fixed" but "what would have caught this AND its siblings".
    Any `except Exception` in this module that reports via log.error without a traceback can hide the
    next unlocatable failure exactly as this one did.
    """
    import inspect
    import re
    from kumo_strategies.strategies.qc345_rotation import nautilus as qc345_rotation
    src = inspect.getsource(qc345_rotation)
    lines = [ln.split("#")[0] for ln in src.splitlines()]
    offenders = []
    for i, ln in enumerate(lines):
        if not re.search(r"except\s+Exception", ln):
            continue
        block = "\n".join(lines[i : i + 12])
        if "self.log.error" in block or "log.error" in block:
            if not any(k in block for k in ("format_exception", "format_exc", "exc_info", ".exception(")):
                offenders.append(i + 1)
    assert not offenders, (
        f"except-Exception handlers logging an error with no traceback at line(s) {offenders}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REPRODUCED, NOT FIXED (2026-08-21). `missed_rebalances` is appended at four sites and read "
        "nowhere in the LIVE path — only by tests and by a backtest result dict, which surfaces "
        "nothing to an operator. Two consecutive live rebalance failures raised no alarm. "
        "WHERE it should surface is a design decision in this runtime (kumo-trading-platform has a "
        "`notify_strategy_degraded` switch that looks like the natural consumer), so it is not being "
        "taken unilaterally from the cockpit side. strict=True: once a reader exists this fails and the "
        "marker must be deleted."
    ),
)
def test_missed_rebalances_is_READ_by_something_and_not_just_appended_to():
    """THE SEAM, NOT THE UNIT.

    `missed_rebalances.append(...)` at four sites is green in isolation and useless in production if
    nothing consults the list. Recording a missed monthly rebalance into a variable no code reads is
    indistinguishable from not recording it. This asserts a reader exists in the SOURCE TREE, not in
    the tests — the tests were the only readers when this shipped.
    """
    import pathlib
    import re

    root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    assert (root / "kumo_strategies").is_dir(), f"source tree not found at {root}"
    appenders, readers = [], []
    for path in _src_only(root.rglob("*.py")):
        # BACKTEST HARNESSES ARE NOT READERS. `runner_qc27_nautilus.py` copies `missed_rebalances`
        # into its result dict, which flipped this marker to XPASS and would have closed the gap on a
        # technicality — a number in a backtest report surfaces nothing to an operator at 09:35. The
        # question this test asks is whether anything in the LIVE path raises an alarm, so only the
        # live path counts. Scoping added 2026-08-21 after exactly that false XPASS.
        if "backtesting" in path.parts:
            continue
        for n, raw in enumerate(path.read_text().splitlines(), 1):
            ln = raw.split("#")[0]
            if "missed_rebalances" not in ln:
                continue
            if ".missed_rebalances.append" in ln:
                appenders.append(f"{path.name}:{n}")
            elif re.search(r"missed_rebalances\s*(:[^=]*)?=", ln):
                continue          # the DECLARATION is not a reader; counting it hid the gap once already
            else:
                readers.append(f"{path.name}:{n}")
    # The fixture's own property first: if nothing appends, the assertion below is vacuous.
    assert appenders, "nothing appends to missed_rebalances — this test no longer describes the code"
    assert readers, (
        f"missed_rebalances is appended at {appenders} and read NOWHERE in src/. Two consecutive live "
        "rebalances failed and nothing surfaced it."
    )


# -- a failed session must leave a DURABLE record ---------------------------------------------------
class _RaisingRunner(_FakeRunner):
    """Reproduces 2026-08-21: the runner raises AFTER journalling its own decision row.

    The exception type is the real one — `TypeError: 'NoneType' object is not callable` — because the
    point of this failure is that the type was ALL anyone got. A double that raises `ValueError("x")`
    would test the plumbing while dodging the thing that made the incident expensive.
    """

    def __init__(self, exc=None):
        super().__init__()
        self._exc = exc or TypeError("'NoneType' object is not callable")

    async def run(self, panel, session, jobs=None, slot=None, opens=None):
        self.calls.append({"session": session})
        # The gateway writes its decision row before submitting, so the journal is NOT empty when the
        # failure lands. That is exactly why the incident was confusing: the record showed a decision
        # and then nothing, which reads as "decided and declined".
        await self.journal.write("decision", f"QC345 {session}: enter ['AAA'] exit []",
                                 session=session, detail={"enter": ["AAA"], "exit": []})
        raise self._exc


class _BrokenJournal(_FakeJournal):
    async def write(self, kind, summary, *, session, detail=None, **kw):
        raise RuntimeError("postgres is unreachable")


def _run_failing_session(runner, **kw):
    async def go():
        s, clock = _live(runner, **kw)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
        return s
    return asyncio.run(go())


def test_a_failed_rebalance_is_JOURNALLED_not_only_logged():
    """THE 2026-08-21 INCIDENT.

    QC345 decided five entries and submitted nothing. The journal held the decision row and NOTHING
    else — no order, no refusal, no error — so the durable record of the session was indistinguishable
    from a strategy that decided and chose not to act. The reason existed only as one line in a
    container log, and kumo-trading-platform issue 378 already established that recreating a container destroys the
    prior one's logs.

    MOMENTUM's adapter has journalled its failures since `_record` was added; QC345's has never
    journalled anything from this handler. That asymmetry is the defect.
    """
    runner = _RaisingRunner()
    _run_failing_session(runner)

    kinds = [w["kind"] for w in runner.journal.writes]
    assert "error" in kinds, (
        f"a failed rebalance left no durable record — journal kinds were {kinds}. "
        "The container log is not a record: it is not queryable and does not survive a recreate.")


def test_the_journalled_failure_names_the_exception_TYPE_and_message():
    """"Something went wrong" costs a night of log archaeology. The type is what let the FIRST QC345
    failure be diagnosed at all — `KeyError: 'eligible'` named its own cause."""
    runner = _RaisingRunner()
    _run_failing_session(runner)

    err = [w for w in runner.journal.writes if w["kind"] == "error"]
    assert err, "no error row to inspect"
    text = err[-1]["summary"]
    assert "TypeError" in text, f"the exception type is not in the record: {text!r}"
    assert "'NoneType' object is not callable" in text, f"the message is not in the record: {text!r}"


def test_the_journalled_failure_carries_the_SESSION_it_belongs_to():
    """A record that cannot be tied to a session cannot answer "why did nothing happen on the 21st"."""
    runner = _RaisingRunner()
    _run_failing_session(runner)
    err = [w for w in runner.journal.writes if w["kind"] == "error"]
    assert err and err[-1]["session"] == "2026-03-02"


def test_a_SUCCESSFUL_session_also_leaves_an_outcome_row():
    """"Session ran and declined" and "session never ran" must not look identical in the record —
    the single most expensive ambiguity in this system. MOMENTUM writes a `state` row every session;
    QC345 wrote none, which is why tonight's journal cannot distinguish the two."""
    runner = _FakeRunner()

    async def go():
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
    asyncio.run(go())

    kinds = [w["kind"] for w in runner.journal.writes]
    assert "state" in kinds, f"a completed session left no outcome row — kinds were {kinds}"


def test_journalling_the_failure_is_BEST_EFFORT_and_never_replaces_the_log():
    """If Postgres is what failed, the journal write cannot work either. The log line must be emitted
    FIRST so it is never lost to a second failure, and a broken journal must not turn a recorded
    failure into an unhandled exception that kills the session task."""
    runner = _RaisingRunner()
    runner.journal = _BrokenJournal()
    # Must not raise out of the task.
    s = _run_failing_session(runner)
    assert s.missed_rebalances, "the failure was not even counted as missed"


def test_no_runner_attached_degrades_to_log_only():
    """The standalone/backtest path has nowhere to journal to. It must not raise for lack of one."""
    async def go():
        s, clock = _live(_FakeRunner())
        s._runner = None
        _fire(clock, "2026-03-02 14:35")
        if s._session_task is not None:
            await asyncio.wrap_future(s._session_task)
        return s
    asyncio.run(go())    # the assertion is that this does not raise


def test_the_adapter_TELLS_the_runner_which_slot_it_is():
    """The journal's slot column and the alert time were two independent facts, and they disagreed.

    Measured live by kumo-trading-platform 2026-08-22: settings said `open+315m`, the lane fired at open+315m,
    and every journal row said `open+150m` — the runner's module default. `exec_action_log.slot` is
    also half of the `(strategy_id, session, slot)` unique index, so the label was the idempotency
    key too.
    """
    async def go():
        runner = _FakeRunner()
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
        return runner, s

    runner, s = asyncio.run(go())
    assert runner.calls, "the runner was not called at all"
    assert runner.calls[0]["slot"] == f"open+{s._open_offset}m", (
        f"the runner was told {runner.calls[0]['slot']!r}; the name must be DERIVED from the offset "
        f"the lane actually fires at, or the two can drift apart again")


# -- a skipped session leaves a row (#249, kumo-trading-platform issue 1099) ---------------------------------------


def test_a_skipped_non_rebalance_session_WRITES_a_state_row_naming_the_next_rebalance():
    """QC345 on paper read as dead for sixteen sessions because the skip branch wrote nothing — no
    log, no journal row — and that silence hid that protection had stopped out its whole book on
    09-10..09-14 (kumo-trading-platform issue 1099). "Skipped" is an outcome; it gets the row every outcome gets.
    """
    async def go():
        runner = _FakeRunner()
        s, _clock = _registered(session_runner=runner)
        s._loop = asyncio.get_running_loop()
        _fill_panel(s)                              # the same panel the skip tests above use
        s._exit_only_for = lambda session: None
        s._held = {"A", "B", "C"}
        s._due = pd.Timestamp("2026-03-10")        # a Tuesday mid-month
        s._try_decide()
        for _ in range(20):                         # the row is scheduled onto the loop; let it land
            await asyncio.sleep(0)
        rows = [w for w in runner.journal.writes if w["kind"] == "state"]
        assert len(rows) == 1, runner.journal.writes
        row = rows[0]
        assert row["session"] == "2026-03-10"
        assert "skipped: not a rebalance session" in row["summary"]
        assert "next 2026-04-01" in row["summary"]     # the first session on or after April 1st
        assert "held 3" in row["summary"]
        assert row["detail"]["state"] == "SKIPPED" and row["detail"]["held"] == 3
        # The shape every session-outcome row shares (kumo-trading-platform issue 1098): undecided, a named reason,
        # and which gate it was.
        assert row["detail"]["decided"] is False
        assert row["detail"]["blocked"] == "not a rebalance session"
        assert row["detail"]["skip"] == {"warm": True, "panel_has_due": True, "is_rebalance": False}
        assert row["detail"]["next_rebalance_on_or_after"] == "2026-04-01"
        assert row["detail"]["skipped_sessions"] == 1
        assert runner.calls == [], "a skipped session must not reach the runner"
    asyncio.run(go())


def test_the_skipped_row_names_a_FORCED_rebalance_when_one_is_nearer_than_the_month():
    async def go():
        runner = _FakeRunner()
        s, _clock = _registered(session_runner=runner)
        s._loop = asyncio.get_running_loop()
        _fill_panel(s)
        s._exit_only_for = lambda session: None
        s._forced_rebalance.add(pd.Timestamp("2026-03-16"))
        s._due = pd.Timestamp("2026-03-10")
        s._try_decide()
        for _ in range(20):
            await asyncio.sleep(0)
        rows = [w for w in runner.journal.writes if w["kind"] == "state"]
        assert len(rows) == 1 and "next 2026-03-16" in rows[0]["summary"]
        assert rows[0]["detail"]["next_rebalance_on_or_after"] == "2026-03-16"
    asyncio.run(go())


def test_a_skipped_session_with_NO_loop_still_skips_and_does_not_raise():
    # The backtest / standalone path has no loop and no journal; the skip must stay a skip.
    s, _clock = _registered()
    _fill_panel(s)
    s._exit_only_for = lambda session: None
    s._due = pd.Timestamp("2026-03-10")
    s._try_decide()
    assert s.skipped_sessions == 1 and s._due is None
