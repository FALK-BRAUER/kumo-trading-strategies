"""Tests for the Nautilus binding.

These do NOT build a BacktestEngine — a second engine per interpreter aborts natively (cockpit
documents this, and its canonical test script runs engine tests in separate processes). What is
pinned here is the contract between the pure engine and the runtime: identity, warmup, the rule
that book state moves only on order EVENTS, and the session clock.

The clock tests register the strategy against a real Nautilus `TestClock` rather than a fake. The
whole reason the trigger moved onto Nautilus's clock was that a hand-rolled one could not be
exercised by a backtest; testing it against a hand-rolled stub would give that back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.strategies.momentum_rotation.nautilus import (
    STRATEGY_NAME, STRATEGY_TAG, MomentumRotationStrategy,
)
from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig


def _strategy(**kw) -> MomentumRotationStrategy:
    return MomentumRotationStrategy(
        cfg=kw.pop("cfg", MomentumRotationConfig()),
        source=kw.pop("source", StaticList(["AAA", "BBB"])),
        instrument_ids=kw.pop("instrument_ids", []),
        **kw,
    )


def _event(symbol="AAA", side="BUY", coid="O-20260831-000001"):
    """Shaped like a real `OrderFilled`: production carries a side and a client order id, and a
    double without them cannot express the kumo-trading-platform issue 748 defect these handlers now guard."""
    return SimpleNamespace(
        instrument_id=SimpleNamespace(symbol=SimpleNamespace(value=symbol)),
        order_side=SimpleNamespace(name=side),
        client_order_id=coid,
    )


def test_identity_is_momentum_001():
    """`MOMENTUM-002` is the cycle-attribution key — NETTING position id is
    {instrument}-{strategy_id}. It must be set via the config so the order-id tag binds at
    registration; change_id() would leave the tag at its default and desync client_order_ids.

    Tag 002, not 001: Nautilus requires order_id_tag unique across every strategy in one trader, and
    MANUAL already holds 001 with live positions keyed to it. Registering both with 001 fails at
    add_strategy — which is exactly how this was found, on the first deployment that ran two."""
    s = _strategy()
    assert (STRATEGY_NAME, STRATEGY_TAG) == ("MOMENTUM", "002")
    assert s.config.strategy_id == "MOMENTUM"
    assert s.config.order_id_tag == "002"
    assert s.config.order_id_tag is not None, "tag must be explicit or registration rewrites it"


def test_refuses_to_trade_before_warmup():
    """Live history arrives asynchronously. Acting before enough bars means ranking on half-formed
    indicators — a real failure mode in production that never shows up in a preloaded backtest."""
    s = _strategy()
    assert s.warm is False


def test_warm_once_enough_symbols_have_enough_history():
    s = _strategy()
    need = s._need
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"):
        s._bars[sym].extend(range(need))
    assert s.warm is True


# -- silent-block visibility -----------------------------------------------------------------
# A strategy blocked on warmup or in-flight orders returned from `_try_decide` with NO log line at
# all before this -- indistinguishable from correctly finding nothing to trade. BCTROT's first live
# day (2026-08-19) produced zero decisions and left nothing in the log explaining why. `on_bar`
# fires per SYMBOL, so this cannot log unconditionally every call without flooding the moment
# warmup stalls; `_last_blocked_due` is the dedup guard that makes one line per blocked SESSION
# safe. These tests pin the guard's state transitions, since the log call itself is not easily
# observed through Nautilus's own logger in a unit test -- the guard's value is what actually
# decides whether a second call would log again.


def test_a_blocked_session_is_recorded_once_in_the_dedup_guard():
    """Not warm yet: `_try_decide` must record `_due` in `_last_blocked_due` on the first blocked
    call, so a repeat call for the SAME session (the common case -- one call per incoming bar,
    every symbol, every update) does not re-log."""
    s = _strategy()
    s._due = pd.Timestamp("2026-01-05")
    assert s.warm is False

    s._try_decide()
    assert s._last_blocked_due == pd.Timestamp("2026-01-05")

    s._try_decide()                                     # repeat call, same session -- still blocked
    assert s._last_blocked_due == pd.Timestamp("2026-01-05"), "must not have moved off the guard"


def test_pending_orders_also_trip_the_blocked_guard():
    """The other blocking condition -- orders in flight, not just warmup -- must set the same
    guard, since both share the one early-return branch."""
    s = _strategy()
    need = s._need
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"):
        s._bars[sym].extend(range(need))
    assert s.warm is True
    s._due = pd.Timestamp("2026-01-05")
    s._pending["AAA"] = "enter"

    s._try_decide()
    assert s._last_blocked_due == pd.Timestamp("2026-01-05")


def test_the_guard_resets_once_the_block_clears():
    """Once warmup completes (or orders clear) and a decision actually proceeds past the blocked
    branch, the guard must reset -- otherwise a LATER session that blocks again would be silently
    swallowed by a guard still holding a stale, unrelated value."""
    s = _strategy()
    need = s._need
    for sym in ("AAA", "BBB"):
        s._bars[sym].extend(range(need))
    s._due = pd.Timestamp("2026-01-05")
    assert s.warm is False

    s._try_decide()
    assert s._last_blocked_due == pd.Timestamp("2026-01-05")

    for sym in ("CCC", "DDD", "EEE", "FFF", "GGG", "HHH"):
        s._bars[sym].extend(range(need))
    assert s.warm is True
    s._due = pd.Timestamp("2026-01-05")
    # `_panel()` expects real Bar objects; the synthetic ints above only exercise `warm`'s length
    # check. Stub it empty so `_try_decide` reaches its own `if panel.empty: return` right after
    # the guard reset this test is pinning, rather than crashing on fake bar contents this test
    # has no reason to construct.
    s._panel = lambda: pd.DataFrame()
    s._try_decide()
    assert s._last_blocked_due is None, "the guard must clear once the block itself clears"


def test_book_state_moves_only_on_fills():
    """Submitting is not holding. Backtest fills are certain; live orders get rejected, and a book
    that updates on submit will believe it holds something it does not."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    assert "AAA" not in s._held                      # submitted, not filled
    # The double carries a SIDE and a client order id because production's `OrderFilled` does. It
    # did not, and a shape that cannot represent production is the bug — these handlers now read
    # both, and a sideless event moves nothing.
    s.on_order_filled(_event("AAA", side="BUY"))
    assert "AAA" in s._held


def test_rejected_order_does_not_enter_the_book():
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_rejected(_event("AAA", side="BUY"))
    assert "AAA" not in s._held and "AAA" not in s._pending


def test_canceled_exit_clears_pending():
    s = _strategy()
    s._held.add("AAA")
    s._pending["AAA"] = "exit"
    s.on_order_canceled(_event("AAA", side="SELL"))
    assert "AAA" in s._held          # the exit never happened, so it is still held
    assert "AAA" not in s._pending


def test_no_second_decision_while_orders_in_flight():
    """Deciding again with orders outstanding would double-submit against a stale book."""
    s = _strategy()
    for sym in "ABCDEFGH":
        s._bars[sym].extend(range(s._need))
    s._pending["AAA"] = "enter"
    s._decide_for(pd.Timestamp("2026-03-02"))
    assert s._last_decision is None       # bailed before recording a decision


def test_one_decision_per_session():
    s = _strategy()
    for sym in "ABCDEFGH":
        s._bars[sym].extend(range(s._need))
    s._last_decision = pd.Timestamp("2026-03-02")
    called = []
    s._panel = lambda: called.append(1) or pd.DataFrame()
    s._decide_for(pd.Timestamp("2026-03-02"))
    assert called == []


# -- the session clock -------------------------------------------------------------------------
# These register the strategy against a real Nautilus `TestClock`, so what is under test is
# Nautilus's own timer, not a stub of it. A stub would happily "fire" an alert that the real clock
# would have rejected or scheduled elsewhere, which is precisely the class of gap this trigger
# exists to close.

def _registered(now: str = "2026-03-02 12:00", **kw):
    """A strategy wired to a TestClock. Returns (strategy, clock)."""
    from nautilus_trader.common.component import TestClock
    from nautilus_trader.test_kit.stubs.component import TestComponentStubs
    from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs

    from kumo_strategies.runtime.calendar import WeekdayCalendar

    clock = TestClock()
    clock.set_time(pd.Timestamp(now, tz="UTC").value)
    s = _strategy(calendar=kw.pop("calendar", WeekdayCalendar()), **kw)
    s.register(TestIdStubs.trader_id(), TestComponentStubs.portfolio(),
               TestComponentStubs.msgbus(), TestComponentStubs.cache(), clock)
    s._arm(s.clock.utc_now())
    return s, clock


def _fire(clock, when: str) -> None:
    for handler in clock.advance_time(pd.Timestamp(when, tz="UTC").value):
        handler.handle()


def _warm(s) -> None:
    for sym in "ABCDEFGHIJKLMNOPQRST":
        s._bars[sym].extend(range(s._need))


def _bar(day: str, sym: str):
    return SimpleNamespace(ts_event=pd.Timestamp(day).value,
                           bar_type=SimpleNamespace(instrument_id=SimpleNamespace(
                               symbol=SimpleNamespace(value=sym))))


def test_alert_is_armed_for_the_next_session_at_the_open_offset():
    """Offset is from the OPEN, not a wall-clock time, so a half-day still fires correctly."""
    s, clock = _registered(now="2026-03-02 12:00")          # 07:00 ET, before the open
    assert s._armed_session == pd.Timestamp("2026-03-02")
    assert "session_decide" in clock.timer_names
    _fire(clock, "2026-03-02 14:35")                        # 09:35 ET = open + 5
    assert s._due == pd.Timestamp("2026-03-02")


def test_rearms_after_every_firing():
    """A one-shot alert that is not re-armed leaves a strategy that has silently stopped trading and
    looks identical to one that decided to hold."""
    s, clock = _registered()
    _fire(clock, "2026-03-02 14:35")
    assert s._armed_session == pd.Timestamp("2026-03-03")
    assert "session_decide" in clock.timer_names


def test_rearms_even_when_the_decision_raises():
    s, clock = _registered()
    _warm(s)
    s._panel = lambda: (_ for _ in ()).throw(RuntimeError("panel exploded"))
    with pytest.raises(RuntimeError):
        _fire(clock, "2026-03-02 14:35")
    assert s._armed_session == pd.Timestamp("2026-03-03")
    assert "session_decide" in clock.timer_names


def test_alert_alone_does_not_decide_and_bars_complete_it():
    """The alert says WHEN TO LOOK; the bars say WHETHER WE MAY DECIDE. Firing on a half-formed
    panel is what once ranked a universe of one at 09:30."""
    s, clock = _registered()
    _warm(s)
    fired, panel = [], {"df": pd.DataFrame({"date": []})}
    s._decide_for = lambda d, p=None: fired.append(d)
    s._panel = lambda: panel["df"]

    _fire(clock, "2026-03-02 14:35")
    assert fired == [], "decided before any bar for the prior session had arrived"
    assert s._due == pd.Timestamp("2026-03-02")

    panel["df"] = pd.DataFrame({"date": [pd.Timestamp("2026-02-27")] * 3})
    s.on_bar(_bar("2026-02-27", "AAA"))
    assert fired == [pd.Timestamp("2026-02-27")]    # scores the last COMPLETED session
    assert s._due is None


def test_a_session_that_never_got_data_is_recorded_not_silently_skipped():
    s, clock = _registered()
    _warm(s)
    s._panel = lambda: pd.DataFrame({"date": []})
    _fire(clock, "2026-03-02 14:35")
    _fire(clock, "2026-03-03 14:35")
    assert s.missed_sessions == [pd.Timestamp("2026-03-02")]
    assert s._due == pd.Timestamp("2026-03-03")


def test_refuses_to_decide_on_stale_data():
    """A ten-day gap is a data outage, not a long weekend. Ranking on a picture of the market that
    old is worse than holding what we have."""
    s, clock = _registered()
    _warm(s)
    fired = []
    s._decide_for = lambda d, p=None: fired.append(d)
    s._panel = lambda: pd.DataFrame({"date": [pd.Timestamp("2026-02-20")] * 3})
    _fire(clock, "2026-03-02 14:35")
    assert fired == []
    assert s.missed_sessions == [pd.Timestamp("2026-03-02")]


def test_a_late_restart_skips_to_the_next_session_rather_than_firing_stale():
    """Coming up at 15:00 must not immediately decide for a session whose open was hours ago."""
    s, _ = _registered(now="2026-03-02 20:00")      # 15:00 ET, well past the open
    assert s._armed_session == pd.Timestamp("2026-03-03")
    assert s._due is None


def test_engine_layer_imports_no_nautilus():
    """The pure layer must stay runtime-free, or the claim that research and production share one
    implementation is false."""
    import ast
    import pathlib

    from kumo_strategies.strategies import _layout
    from kumo_strategies.strategies import momentum_rotation as pkg

    root = pathlib.Path(pkg.__file__).parent
    # The lane and the runner live in this folder too now (ks#211); they are the runtime layer and
    # may import nautilus. Everything else in the folder is the pure layer and may not.
    for f in sorted(p for p in root.glob("*.py") if p.stem not in _layout.RUNTIME_STEMS):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any("nautilus" in n.lower() for n in names), \
                f"{f.name} imports nautilus - the pure layer must stay runtime-free"


def test_config_documents_partial_deployment():
    """ledger-tool confirmed from 7 verbatim posts: 'Each position averages between 1 to 2% of my
    entire portfolio'. 40 names at 1-2% means he runs 40-80% deployed, never fully invested.

    Sizing at 100% is also unrunnable: it drove the cash account negative four days into a
    128-session window and Nautilus halted, which came back looking like a real strategy losing
    money rather than a stopped run."""
    from kumo_strategies.strategies.momentum_rotation.config import PortfolioConfig

    p = PortfolioConfig()
    assert p.max_weight <= 1.0
    assert p.n_hold >= 1


def test_decision_never_sees_the_session_it_is_trading_into():
    """In backtest, `bars_for` stamps a session's bar at midnight UTC, so it is already in the panel
    when the 09:35 alert fires. Scoring the prior session with it present is lookahead that exists
    ONLY in backtest — it would flatter the number and then not show up live."""
    s, clock = _registered()
    _warm(s)
    seen = {}
    s._decide_for = lambda d, p=None: seen.update(session=d, dates=sorted(set(p.date)))
    s._panel = lambda: pd.DataFrame({"date": pd.to_datetime(
        ["2026-02-26", "2026-02-27", "2026-03-02"])})       # 03-02 = the session being traded into
    _fire(clock, "2026-03-02 14:35")
    assert seen["session"] == pd.Timestamp("2026-02-27")
    assert pd.Timestamp("2026-03-02") not in seen["dates"], "leaked the traded session into scoring"


# -- the live session path ----------------------------------------------------------------------
# With a session runner attached, the alert must hand off to it rather than run the local decision.
# The local path has no lifecycle gate, no pool-staleness block, no ownership boundary and no
# journal row -- so quietly using it in a live node would submit orders from a DISABLED strategy.

class _FakeJournal:
    def __init__(self):
        self.rows = []

    async def write(self, kind, summary, **kw):
        self.rows.append((kind, summary))
        return 1


class _FakeRunner:
    def __init__(self, gate=None):
        self.calls, self.gate = [], gate
        self.journal = _FakeJournal()

    async def run(self, panel, session, jobs=None, slot=None, opens=None):
        self.calls.append({"session": session, "jobs": jobs, "rows": len(panel),
                           "panel_max": panel.date.max(), "slot": slot, "opens": opens})
        if self.gate is not None:
            await self.gate.wait()
        return SimpleNamespace(state="SHADOW", decided=True, blocked=None)


def _live(runner, **kw):
    """Registered AND loop-attached. on_start captures the loop; these tests bypass on_start, so it
    is attached here — the strategy dispatches timer work onto it rather than calling
    get_running_loop() from the clock's thread."""
    import asyncio

    s, clock = _registered(session_runner=runner, session_jobs="JOBS", **kw)
    try:
        s._loop = asyncio.get_running_loop()
    except RuntimeError:
        s._loop = None
    _warm(s)
    s._panel = lambda: pd.DataFrame({"date": [pd.Timestamp("2026-02-27")] * 3})
    s._decide_for = lambda *a, **k: pytest.fail("local decision path ran in a live node")
    return s, clock


def test_live_alert_hands_the_session_to_the_runner():
    import asyncio

    async def go():
        runner = _FakeRunner()
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
        return runner

    runner = asyncio.run(go())
    assert len(runner.calls) == 1
    call = runner.calls[0]
    # The slot the alert actually fired at reaches the runner, and for this single-slot strategy it
    # is the one MOMENTUM-002 has always used. It is the second half of the `(session, slot)`
    # idempotency key — a decision written under the wrong slot is indistinguishable from a
    # duplicate of a different one.
    assert call["slot"] == "open+5m"
    assert call["session"] == "2026-03-02"          # the session being traded into
    assert call["jobs"] == "JOBS", "source refresh was skipped — the pool could be silently stale"
    assert call["panel_max"] == pd.Timestamp("2026-02-27")   # scored on the last COMPLETED session


def test_a_second_alert_does_not_race_a_running_session():
    """Two runners against one journal row: the database refuses the duplicate decision, but only
    after both have read the same book and computed exits from it."""
    import asyncio

    async def go():
        gate = asyncio.Event()
        runner = _FakeRunner(gate=gate)
        s, clock = _live(runner)
        _fire(clock, "2026-03-02 14:35")
        await asyncio.sleep(0)
        _fire(clock, "2026-03-03 14:35")            # previous session still in flight
        gate.set()
        await asyncio.wrap_future(s._session_task)
        return runner, s

    runner, s = asyncio.run(go())
    assert len(runner.calls) == 1
    assert s.missed_sessions == [pd.Timestamp("2026-03-03")]


def test_no_event_loop_refuses_instead_of_falling_back_to_the_local_path():
    """Falling back would run the session without the lifecycle gate — a DISABLED or SHADOW
    strategy would submit real orders."""
    runner = _FakeRunner()
    s, clock = _live(runner)
    s._loop = None                      # simulate a callback with no loop ever captured
    _fire(clock, "2026-03-02 14:35")                # sync context: no running loop
    assert runner.calls == []
    assert s.missed_sessions == [pd.Timestamp("2026-03-02")]


def test_source_refresh_runs_on_its_own_timer_independent_of_the_session():
    """Sources refresh on their own cadence. A pool that only updates when a session runs is a pool
    that is always stale on the morning it matters — and the runner treats a stale source as a hard
    block, so the first thing the session would find is a reason not to trade."""
    import asyncio

    class Jobs:
        def __init__(self):
            self.calls = 0

        async def refresh_due(self, session):
            self.calls += 1

    async def go():
        jobs = Jobs()
        s, clock = _registered(session_runner=_FakeRunner(), session_jobs=jobs)
        s._loop = asyncio.get_running_loop()
        s.clock.set_timer("source_refresh", pd.Timedelta(seconds=60),
                          callback=s._on_source_timer)
        _fire(clock, "2026-03-02 12:02")
        await asyncio.sleep(0)
        await asyncio.wrap_future(s._refresh_task)
        return jobs

    assert asyncio.run(go()).calls == 1


def test_a_failing_refresh_does_not_kill_the_timer():
    """A refresh loop that dies leaves the pool frozen at its last good set, which stays inside the
    staleness window for hours before anything says so."""
    import asyncio

    class Boom:
        def __init__(self):
            self.calls = 0

        async def refresh_due(self, session):
            self.calls += 1
            raise RuntimeError("upstream down")

    async def go():
        jobs = Boom()
        s, clock = _registered(session_runner=_FakeRunner(), session_jobs=jobs)
        s._loop = asyncio.get_running_loop()
        s.clock.set_timer("source_refresh", pd.Timedelta(seconds=60),
                          callback=s._on_source_timer)
        for t in ("2026-03-02 12:02", "2026-03-02 12:03"):
            _fire(clock, t)
            await asyncio.sleep(0)
            await asyncio.wrap_future(s._refresh_task)          # must not raise out of the task
        return jobs, clock

    jobs, clock = asyncio.run(go())
    assert jobs.calls == 2
    assert "source_refresh" in clock.timer_names, "the timer died on the first failure"


def test_stopping_cancels_an_in_flight_session_not_just_the_timers():
    """Cancelling timers is not enough. A running session is a raw asyncio task Nautilus does not
    manage, so it outlives stop() and keeps acting on the lifecycle it read when it STARTED — an
    operator who pauses mid-session watches the strategy submit anyway."""
    import asyncio

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
    assert "session_decide" not in clock.timer_names
    assert s._due is None


def test_the_holiday_unaware_calendar_is_refused_where_orders_can_be_placed(monkeypatch):
    import pytest

    from kumo_strategies.runtime.calendar import (
        AlpacaCalendar, WeekdayCalendar, build_calendar)

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert isinstance(build_calendar(), WeekdayCalendar)        # dev default unchanged
    with pytest.raises(RuntimeError):
        build_calendar(require_exchange=True)

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    assert isinstance(build_calendar(require_exchange=True), AlpacaCalendar)


def test_on_start_requests_instrument_definitions_not_only_bars():
    """Nautilus needs the instrument definition to size and price an order, and cache.instrument()
    returning None is a hard refusal at submit. On the first live run every entry was priced,
    journalled, and then rejected with "no instrument definition cached" — bars alone do not
    populate it."""
    import inspect

    from kumo_strategies.strategies.momentum_rotation import nautilus as m
    src = inspect.getsource(m.MomentumRotationStrategy.on_start)
    assert "request_instrument" in src, "subscribes to bars for instruments it never defines"
    assert src.index("request_instrument") < src.index("subscribe_bars"), \
        "the definition must be requested before the subscription that depends on it"


# -- external order claims (kumo-trading-platform issue 197 B8) -------------------------------------------------
def test_claims_are_passed_to_the_strategy_config():
    """Without a claimant, Nautilus books a reconciliation-generated flatting order under EXTERNAL,
    and NETTING's `{instrument}-{strategy}` position id makes that OPEN a phantom rather than CLOSE
    the real position. HSBC sat as a -93 short the broker had never heard of."""
    from nautilus_trader.model.identifiers import InstrumentId

    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
    from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    iid = InstrumentId.from_str("MET.XNYS")
    s = MomentumRotationStrategy(cfg=MomentumRotationConfig(), source=StaticList(["MET"]),
                                 instrument_ids=[iid], external_order_claims=[iid])
    assert s.config.external_order_claims == [iid]
    assert s.external_order_claims == [iid]


def test_no_claims_by_default():
    """Claims are exclusive node-wide, so the default must be to claim nothing and let the caller —
    which alone knows what the other strategies hold — opt in."""
    from nautilus_trader.model.identifiers import InstrumentId

    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
    from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    s = MomentumRotationStrategy(cfg=MomentumRotationConfig(), source=StaticList(["MET"]),
                                 instrument_ids=[InstrumentId.from_str("MET.XNYS")])
    assert not s.external_order_claims


def test_two_slots_in_one_session_reach_the_runner_as_DIFFERENT_slots():
    """BCTROT's whole premise, and the failure it would otherwise hit SILENTLY.

    The unique index is `(strategy_id, session, slot)`. If the adapter reports the same slot for
    both of a session's decisions, the second is rejected as a duplicate of the first — the close
    decision never happens, and the journal shows one tidy decision per session. Nothing errors.

    Mutation-bitten: replacing the fired slot with `self._slots[0]` turns this red. Every other test
    in this file is single-slot, so that mutation passes all of them.
    """
    import asyncio

    async def go():
        runner = _FakeRunner()
        s, clock = _live(runner, decision_slots=("open+150m", "close-20m"))
        _fire(clock, "2026-03-02 17:00")            # 12:00 ET — open+150m
        await asyncio.wrap_future(s._session_task)
        _fire(clock, "2026-03-02 20:40")            # 15:40 ET — close-20m
        await asyncio.wrap_future(s._session_task)
        return runner

    runner = asyncio.run(go())
    assert len(runner.calls) == 2, "the second slot never ran"
    assert [c["slot"] for c in runner.calls] == ["open+150m", "close-20m"]
    assert runner.calls[0]["session"] == runner.calls[1]["session"], "same session, two decisions"


# ==================================================================================================
# `_try_decide` MUST READ `_due` ONCE (2026-08-21 — this crashed the live paper engine).
#
# 12:00:00 ET, mid-session, the whole TradingNode terminated:
#
#   File ".../momentum_rotation.py", line 373, in _try_decide
#     prior = panel.loc[panel.date < self._due, "date"].max()
#   TypeError: Invalid comparison between dtype=datetime64[ns] and NoneType
#   [ERROR] DataEngine: System will terminate immediately to prevent operation in degraded state
#
# It raised inside `on_bar`, so Nautilus tore the node down rather than run degraded — correct
# behaviour, and it meant every position ran unmanaged until Docker restarted the container. BCTROT-004
# lost its 12:00 decision slot to the gap ("did not run and will NOT be run late").
#
# `_try_decide` guards `if self._due is None: return` at the top, and then reads `self._due` FOUR more
# times across twenty lines while calling out to `self._panel()` in between. Line 375 is the only writer
# that sets it back to None. Any path that re-enters between the guard and the comparison walks straight
# past a check that has already been satisfied.
#
# The test does NOT try to reproduce the exact re-entry path — that is what makes it a regression test
# for the CLASS rather than for one caller. It forces the window open at the one place the method yields
# control, and asserts the method survives it. A version that reads `self._due` at line 373 cannot.
# ==================================================================================================
import types


def test_due_is_read_ONCE_so_a_reentrant_clear_cannot_crash_on_bar(monkeypatch):
    import pandas as pd
    from kumo_strategies.strategies.momentum_rotation import nautilus as mr
    strat = mr.MomentumRotationStrategy.__new__(mr.MomentumRotationStrategy)
    due = pd.Timestamp("2026-08-21")

    panel = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-19", "2026-08-20", "2026-08-21"]),
        "ticker": ["AAA", "AAA", "AAA"],
    })
    # The fixture's own property first: the frame really is datetime64 (the UNIT is irrelevant — the
    # live crash was [ns], pandas builds [us] here — what matters is that the comparison below raises).
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])
    with pytest.raises(TypeError, match="Invalid comparison"):
        _ = panel.date < None

    strat._due = due
    strat._due_slot = "open"
    strat._pending = set()
    strat._bars = {}
    strat._need = 0
    strat._last_blocked_due = None
    strat._max_stale_days = 5
    strat._cfg = types.SimpleNamespace(portfolio=types.SimpleNamespace(n_hold=0, buffer=0))
    strat._runner = None            # take the non-runner branch; the decision itself is not under test
    strat._slots = ["open"]
    strat.missed_sessions = []

    def _panel_that_clears_due():
        # THE WINDOW. `_try_decide` calls out here after its None-guard has already passed; anything
        # that clears `_due` during the call leaves the later reads looking at None.
        strat._due = None
        return panel

    monkeypatch.setattr(strat, "_panel", _panel_that_clears_due, raising=False)
    monkeypatch.setattr(strat, "_report_dataless", lambda: None, raising=False)
    monkeypatch.setattr(strat, "_decide_for", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(strat, "_exit_only_for", lambda *a, **k: None, raising=False)

    # No assertion on the outcome — only that it does not raise. A TypeError here reaches `on_bar`,
    # and Nautilus terminates the node.
    strat._try_decide()


# ==================================================================================================
# ORDER EVENTS THIS STRATEGY DID NOT CAUSE (kumo-trading-platform issue 748)
#
# Cockpit is about to stamp each protective stop with the strategy_id of the lane whose shares it
# covers, because under NETTING a fill's position is derived as `{instrument}-{fill.strategy_id}` and
# a stop stamped with the display strategy resolves to a position that never existed — minting a
# phantom. The consequence on THIS side is that Nautilus routes those orders' events to us:
# `Strategy.submit_order` publishes to `events.order.{order.strategy_id}`, so `PROT-*` fills,
# cancels and rejections will arrive at handlers that assume every event is ours.
#
# They are not defensive today. `on_order_filled` pops `self._pending` by SYMBOL alone.
# ==================================================================================================
def test_a_PROTECTIVE_SELL_does_not_complete_a_pending_ENTRY():
    """The corruption, and it is silent.

    A pending "enter" is waiting for a BUY. Cockpit's protective stop is a SELL, and once it carries
    this lane's strategy_id its fill arrives here — where `_pending.pop(sym)` returns "enter" and the
    symbol is added to `_held`. The strategy would believe it holds a position that a STOP JUST SOLD,
    size its next decision off that belief, and skip re-entering a name it thinks it owns.

    The side check needs no shared vocabulary with cockpit: a SELL cannot complete an entry, whoever
    sent it.
    """
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_filled(_event("AAA", side="SELL", coid="PROT-SELL-AAA-XNAS-8ba2ec77"))
    assert "AAA" not in s._held, "a protective SELL was read as this lane's entry completing"
    assert s._pending.get("AAA") == "enter", "and it consumed the pending intent on the way through"


def test_a_PROTECTIVE_order_is_ignored_by_CLIENT_ORDER_ID_even_when_the_side_matches():
    """The side check alone is not enough. A protective SELL arriving while a pending EXIT rests
    would still be read as that exit completing — plausible-looking and wrong, because the exit is
    still working at the venue.

    Cockpit mints `PROT-` client order ids and its `cancel_attribution.OUR_STOP_PREFIXES` already
    treats that prefix as the durable marker of an order it placed. It is durable in the way tags are
    NOT: cockpit's own `engine_node.py` records that reconciled orders come back with NO tags at all,
    so a tag-based check goes blind on exactly the orders that survive a restart.

    This is EXCLUSION by positive identification of a foreign order, not attribution — the failure
    direction is safe. If the prefix ever stops matching we are back to today's behaviour; and this
    strategy never mints such an id, so it cannot exclude its own.
    """
    s = _strategy()
    s._held.add("AAA")
    s._pending["AAA"] = "exit"
    s.on_order_filled(_event("AAA", side="SELL", coid="PROT-SELL-AAA-XNAS-8ba2ec77"))
    assert "AAA" in s._held, "a foreign stop's fill was read as this lane's exit completing"
    assert s._pending.get("AAA") == "exit", "and the real exit is still working"


def test_a_PROTECTIVE_CANCEL_does_not_clear_this_lanes_pending_intent():
    """`on_order_canceled` calls `_forget`, which drops the pending intent by symbol. Cockpit cancels
    and re-arms protective stops on a 60s reconciler tick, so this would fire routinely — clearing
    the in-flight marker for an order that is still working and unblocking a second decision on a
    symbol this lane already has an order out for."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_canceled(_event("AAA", side="SELL", coid="PROT-SELL-AAA-XNAS-3c182de6"))
    assert s._pending.get("AAA") == "enter"


def test_THIS_LANES_OWN_orders_still_move_the_book():
    """The guard must not switch the handlers off. This is the behaviour every other test here pins,
    restated against the production-shaped double so it cannot pass for the wrong reason."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_filled(_event("AAA", side="BUY", coid="O-20260831-000001"))
    assert "AAA" in s._held
    assert "AAA" not in s._pending


def test_an_event_WITHOUT_A_SIDE_does_not_move_the_book():
    """Three states. A shape this handler cannot read is not an entry completing — and moving book
    state on it is how a strategy comes to believe it holds something nobody bought."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_filled(SimpleNamespace(
        instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="AAA")),
        client_order_id="O-20260831-000001",
    ))
    assert "AAA" not in s._held


def test_the_SIDE_CHECK_stands_alone_when_the_id_carries_no_foreign_prefix():
    """The two guards must each be load-bearing, not shadow each other.

    Mutating the side check passed the file, because every foreign fixture also carried a `PROT-`
    id and the prefix caught it first. That makes the side check look tested while it is not — and
    the side check is the half that survives the prefix drifting, which is the whole reason it exists
    alongside a cross-repo convention that cannot be imported.

    A discretionary sell placed by hand under this lane, or any future protective id that is not
    `PROT-`, is a SELL with an ordinary client order id. It must still not complete an entry.
    """
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_filled(_event("AAA", side="SELL", coid="O-20260831-000999"))
    assert "AAA" not in s._held
    assert s._pending.get("AAA") == "enter"


def test_a_SIDELESS_event_records_NOTHING_even_with_no_pending_intent():
    """Where the explicit None check is the only thing acting.

    With a pending intent the side comparison already refuses a sideless event, which masked the
    mutant. With NO pending intent the handler used to fall straight through to `_record_terminal`,
    writing a journal row for an event it could not read — a terminal record asserting an outcome
    nobody established.
    """
    s = _strategy()
    recorded: list = []
    s._record_terminal = lambda *a, **k: recorded.append(a)      # noqa: ARG005
    s.on_order_filled(SimpleNamespace(
        instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="AAA")),
        client_order_id="O-20260831-000001",
    ))
    assert recorded == [], "a journal row was written for an event with no readable side"


def test_the_opens_handed_to_the_runner_are_TODAYS_not_the_panels_newest():
    """THE WIRING, not the rule. `min_abs_gap_pct` needs today's open, and the panel the runner
    receives is trimmed to sessions strictly BEFORE the one being decided — so the open cannot come
    from there. It is built HERE and passed alongside.

    This test exists because the shipped defect was exactly this half: the runner read the panel's
    newest row, got YESTERDAY's open, and computed the previous session's gap at every slot. A
    behavioural test of the runner cannot catch it — supply `opens` and the runner is correct — so
    the assertion has to be on what the STRATEGY sends.

    Both numbers are present in the fixture on purpose: today's open is 120.0 and yesterday's is
    110.0, so an implementation reading the wrong row fails rather than coincidentally passing.
    """
    import asyncio

    async def go():
        runner = _FakeRunner()
        s, clock = _live(runner)
        # 03-02 is the session being traded into: today. 02-27 is the newest row the panel may
        # expose to scoring. Today's open is 120.0, yesterday's 110.0 — both present so a wrong
        # row fails rather than coincidentally matching.
        s._panel = lambda: pd.DataFrame({
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2026-02-26", "2026-02-27", "2026-03-02"]),
            "open": [105.0, 110.0, 120.0],
            "close": [106.0, 112.0, 121.0],
        })
        _fire(clock, "2026-03-02 14:35")
        await asyncio.wrap_future(s._session_task)
        return runner

    runner = asyncio.run(go())
    assert runner.calls, "the session never reached the runner"
    call = runner.calls[-1]
    assert call["opens"] == {"AAA": 120.0}, (
        f"the strategy sent {call['opens']} — today's open is 120.0; 110.0 is YESTERDAY's and is "
        f"what the shipped defect used")
    assert call["panel_max"] == pd.Timestamp("2026-02-27"), (
        "the panel handed to the runner must still exclude today — today's prices must not vote on "
        "today's ranking; only the OPEN crosses that line, and only because it is the price being "
        "traded against")


# -- kumo-trading-platform issue 829: the claim follows the venue's answer, not the submit ----------------------
# pgrunner records the claim when Nautilus accepts the submit LOCALLY. On 2026-09-09 13:35:16 UTC a
# 260-share LAND BUY was denied by cockpit's budget gate 3 ms later and MOMENTUM-002 carried a
# 260-share claim on a position that never existed. Every terminal event now re-syncs the claim to
# the lane's OWN cache position — driven through the real `_record_terminal` with a real cache and a
# real tagged order.


def _terminal_through_the_real_path(*, ok: bool, last_px=None):
    import asyncio

    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    async def go():
        synced, recorded = [], []

        class Runner:
            async def record_terminal(self, session, sym, ok, detail):
                recorded.append((session, sym, ok))

            async def sync_claim(self, sym, qty, px):
                synced.append((sym, qty, px))

        s, _clock = _registered(session_runner=Runner())
        s._loop = asyncio.get_running_loop()
        inst = TestInstrumentProvider.equity(symbol="AAA", venue="XNAS")
        s.cache.add_instrument(inst)
        order = s.order_factory.market(inst.id, OrderSide.BUY, Quantity.from_int(10),
                                       tags=["session:2026-09-09"])
        s.cache.add_order(order)
        ev = SimpleNamespace(instrument_id=inst.id, client_order_id=order.client_order_id,
                             last_px=last_px)
        s._record_terminal(ev, ok=ok, detail="denied" if not ok else "filled")
        await asyncio.sleep(0.05)
        return synced, recorded

    return asyncio.run(go())


def test_a_denied_entry_resyncs_the_claim_to_the_lanes_cache_position_which_is_flat():
    synced, recorded = _terminal_through_the_real_path(ok=False)
    assert recorded == [("2026-09-09", "AAA", False)], "fixture: the real terminal path did not run"
    # the lane holds nothing in AAA — the runner is told 0, and no price (nothing filled)
    assert synced == [("AAA", 0, None)], synced


def test_a_fill_hands_the_runner_the_fill_price_with_the_lanes_quantity():
    synced, recorded = _terminal_through_the_real_path(ok=True, last_px=12.5)
    assert recorded == [("2026-09-09", "AAA", True)]
    assert synced == [("AAA", 0, 12.5)], synced       # quantity is the CACHE's word, price the fill's


def test_a_runner_without_sync_claim_is_left_alone():
    """Optional on the runner, like `record_terminal` — an older cockpit gateway must not raise."""
    import asyncio

    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    async def go():
        recorded = []

        class Runner:
            async def record_terminal(self, session, sym, ok, detail):
                recorded.append(sym)

        s, _clock = _registered(session_runner=Runner())
        s._loop = asyncio.get_running_loop()
        inst = TestInstrumentProvider.equity(symbol="AAA", venue="XNAS")
        s.cache.add_instrument(inst)
        order = s.order_factory.market(inst.id, OrderSide.BUY, Quantity.from_int(10),
                                       tags=["session:2026-09-09"])
        s.cache.add_order(order)
        s._record_terminal(SimpleNamespace(instrument_id=inst.id, client_order_id=order.client_order_id),
                           ok=False, detail="denied")
        await asyncio.sleep(0.05)
        return recorded

    assert asyncio.run(go()) == ["AAA"]
