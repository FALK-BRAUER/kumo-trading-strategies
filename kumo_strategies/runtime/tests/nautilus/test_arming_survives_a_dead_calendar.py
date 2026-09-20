"""Starting a lane must not depend on a network call succeeding (kumo-trading-platform deploy, 2026-08-22).

BCTROT-004 raised `URLError(<urlopen error timed out>)` out of `on_start`, Nautilus re-raised it from
`Trader.START`, and QC345-003 and TECHIVOL-005 never started at all — they were queued behind it. One
lane's dead socket took three lanes down.

The chain, every frame of it inside this repo:

    on_start -> _arm / _report_missed_on_start   (momentum_rotation.py)
             -> next_slot_fire / elapsed_slots   (calendar.py)
             -> AlpacaCalendar.day -> _ensure -> _fetch
             -> urllib.request.urlopen(req, timeout=15)      BLOCKING, ON THE EVENT LOOP

Two separate defects in one line. It BLOCKS the loop for up to 15s, and when it raises it is FATAL to
the whole start sequence rather than to one lane's schedule.

WHY THIS WAS NOT CAUGHT: the only existing test of `on_start` reads its SOURCE (`inspect.getsource`)
and asserts a substring appears. Nothing drove it. A source read cannot observe a socket, so the one
call in this function that reaches the network was the one thing the test could not see.

So these tests drive the REAL `on_start` — the production function object, bound to a narrow host —
against a REAL `AlpacaCalendar` pointed at a closed port. A fake calendar that raised on demand would
prove nothing here: the defect is that a genuine `urlopen` failure escapes, and a double that agrees
with the code cannot test the code.
"""

from __future__ import annotations

import ast
import asyncio
import inspect

import pandas as pd
import pytest

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.calendar import AlpacaCalendar
from kumo_strategies.strategies.momentum_rotation import nautilus as M
from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


#: A port nothing listens on. Refuses instantly instead of hanging for the 15s production timeout —
#: same `URLError`, same escape path, a test that finishes.
DEAD = "http://127.0.0.1:1"


class _Log:
    def __init__(self):
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))

    def warning(self, msg, *a, **k):
        self.errors.append(str(msg))

    def info(self, msg, *a, **k):
        self.infos.append(str(msg))


class _Clock:
    def __init__(self, now):
        self._now = now
        self.timers: dict[str, object] = {}
        self.alerts: dict[str, object] = {}

    def utc_now(self):
        return self._now

    def set_timer(self, name, interval, callback=None, **k):
        self.timers[name] = interval

    def set_time_alert(self, name, at, callback=None, **k):
        self.alerts[name] = at

    @property
    def timer_names(self):
        return list(self.timers) + list(self.alerts)


class _MsgBus:
    def __init__(self):
        self.topics: list[str] = []

    def subscribe(self, topic, handler, **k):
        self.topics.append(topic)


class _Host(HeldSeedMixin, RegistrationMixin):
    """The narrowest host `on_start` can run on, carrying the REAL production functions.

    Nautilus's `Strategy` is Cython — `self.log`, `self.clock` and `self.msgbus` are extension-type
    attributes that cannot be populated without a live trader. Binding the unbound functions to a
    plain object runs the same bytecode with those three replaced, which is the only part of the
    call this test is not interested in. Everything below `on_start` — `_arm`, the slot resolution,
    the calendar, the socket — is production code, untouched.
    """

    # `on_start` seeds `_held` from this lane's own open positions (#194), so the host carries
    # the REAL mixin rather than a stub — the same principle as every other function here.
    # It reads `self.cache`; a host without one gets the mixin's reported failure, not a crash.

    on_start = M.MomentumRotationStrategy.on_start
    # `on_start` resolves symbols before it subscribes (kumo-trading-platform issue 622), so the host carries the
    # REAL function rather than a stub — the same principle the docstring above states. `_symbols` is
    # empty because this test is about ARMING, and an empty list means `on_start` never reaches
    # `self.cache`, which is another read-only Cython attribute this host cannot supply.
    _resolve_symbols_if_needed = M.MomentumRotationStrategy._resolve_symbols_if_needed
    _symbols: list = []
    # Moved to `RegistrationMixin` — the host inherits it via `RegistrationMixin`, which it
    # already subclasses. Naming it here would pin a copy that no longer exists.
    _arm = M.MomentumRotationStrategy._arm
    # `_arm` calls this (kumo-trading-platform issue 514). Carried as the REAL function, not stubbed out — a host
    # that faked it would stop exercising the production arm path, which is the whole point here.
    _reread_slots = M.MomentumRotationStrategy._reread_slots

    def __init__(self, calendar, now):
        self.log = _Log()
        self.clock = _Clock(now)
        self.msgbus = _MsgBus()
        self.id = "MOMENTUM-002"
        self._calendar = calendar
        self._slots = ("open+5m",)
        #: No settings reader, which is what a backtest and every current caller pass. The live
        #: re-read path has its own tests in `test_qc27_rotation.py`.
        self._read_slots = None
        # No instruments: the subscribe loop is not what is under test, and an empty list keeps the
        # host free of Nautilus data types.
        self._iids = []
        self._history_days = 0
        self._jobs = None
        self._need = 0
        self._loop = None
        self._account_topic = "account.PLATFORM-001"
        self._armed_session = None
        self._armed_slot = None
        self.missed_sessions: list[pd.Timestamp] = []
        self.missed_on_start: list[pd.Timestamp] = []

    def _on_broker_account(self, *a, **k):
        pass

    def _bar_type(self, iid):
        raise AssertionError("no instruments configured — this must not be reached")


def _dead_calendar():
    return AlpacaCalendar(key="k", secret="s", base=DEAD)


def _run(host):
    """Drive `on_start` inside a real event loop, then let any background arming take a turn."""

    async def go():
        host.on_start()
        # Give a task created by on_start a chance to run and reach its first failure.
        for _ in range(20):
            await asyncio.sleep(0)
        task = getattr(host, "_arming", None)
        if task is not None:
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):      # noqa: BLE001
                pass

    asyncio.run(go())


def test_a_calendar_that_cannot_be_reached_does_NOT_take_on_start_down():
    """The failure that cost three lanes.

    `Trader.START` calls each strategy's `on_start` in sequence and does not isolate them, so a raise
    here is not one lane failing — it is every lane after it in the sequence never starting. On
    2026-08-22 that was two strategies that had nothing wrong with them.
    """
    host = _Host(_dead_calendar(), pd.Timestamp("2026-08-24 13:45", tz="UTC"))
    _run(host)          # must not raise


def test_a_lane_that_could_not_resolve_its_schedule_SAYS_SO():
    """Unarmed must be LOUD, because unarmed and idle look identical from outside.

    A lane that starts, cannot resolve its calendar, and sits quiet is counted in
    `strategies_running` while being incapable of ever deciding. That is a worse outcome than the
    crash it replaces: the crash was at least visible. So the guard is not merely "does not raise".
    """
    host = _Host(_dead_calendar(), pd.Timestamp("2026-08-24 13:45", tz="UTC"))
    _run(host)
    said = " ".join(host.log.errors).lower()
    assert "unarmed" in said, f"nothing announced the lane cannot decide; errors={host.log.errors}"
    assert host._armed_session is None, "claimed to be armed against a calendar that never answered"


def test_arming_does_NOT_run_the_socket_on_the_event_loop():
    """The other half of the defect, and the half that survives the crash being fixed.

    Catching the exception would stop the node dying while leaving a 15s blocking `urlopen` on the
    loop during start — which stalls every other strategy's start, the data feeds and the message
    bus. The fetch has to happen off the loop, so this records WHICH THREAD it ran on.
    """
    import threading

    seen: list[int] = []
    cal = _dead_calendar()
    real_fetch = cal._fetch

    def spy(start, end):
        seen.append(threading.get_ident())
        return real_fetch(start, end)

    cal._fetch = spy
    host = _Host(cal, pd.Timestamp("2026-08-24 13:45", tz="UTC"))

    async def go():
        loop_thread = threading.get_ident()
        host.on_start()
        for _ in range(20):
            await asyncio.sleep(0)
        task = getattr(host, "_arming", None)
        if task is not None:
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):      # noqa: BLE001
                pass
        return loop_thread

    loop_thread = asyncio.run(go())
    assert seen, "the calendar was never fetched at all — the lane would never arm"
    assert all(t != loop_thread for t in seen), (
        "the calendar fetch ran on the event loop thread; a 15s timeout there stalls the whole node")


def test_a_reachable_calendar_still_arms():
    """The guard must not have been bought by never arming at all.

    A retry loop that swallows everything passes both tests above while leaving every lane
    permanently unarmed, which is the failure it exists to prevent, moved one layer in.
    """
    from datetime import datetime
    from kumo_strategies.runtime.calendar import ET, TradingDay

    class Working:
        def day(self, d):
            if d.weekday() >= 5:
                return None
            return TradingDay(d, datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
                              datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))

    host = _Host(Working(), pd.Timestamp("2026-08-24 12:00", tz="UTC"))
    _run(host)
    assert host._armed_session is not None, "a calendar that answered did not produce an armed slot"
    assert host._armed_slot == "open+5m"


def _shipped_lanes():
    from kumo_strategies.runtime.tests.nautilus.test_contract import _shipped_adapters

    return _shipped_adapters()


def _python_on_start(cls):
    """The `on_start` a lane will actually run, or None if it inherits Nautilus's Cython default."""
    for klass in cls.__mro__:
        fn = klass.__dict__.get("on_start")
        if fn is not None:
            return fn if inspect.isfunction(fn) else None
    return None


@pytest.mark.parametrize("cls", _shipped_lanes(), ids=lambda c: c.__name__)
def test_no_lane_arms_DIRECTLY_from_on_start(cls):
    """Every lane THAT ARMS, including the ones that define no `on_start` of their own.

    `BCTRotationStrategy` subclasses `MomentumRotationStrategy` and defines no `on_start`. A scan
    over classes that DEFINE one would have skipped the only lane that actually failed — so this
    resolves through the MRO, which is what the lane really runs.

    WHICH LANES ARE SUBJECT IS DERIVED, NOT LISTED: a lane arms if it has an `_arm`, and `_arm` is
    the method that reaches the calendar. `TemplateRotationStrategy` is bar-driven, owns no clock and
    has no `_arm`, so it has no calendar to block on. A lane that later grows one is covered by
    existing, which is the property a hand-written exclusion list would not have.

    All four arming lanes had this: momentum, bctrot by inheritance, qc27 and qc345 each called
    `_arm` straight from `on_start`. Only one was ever reached, because it killed the sequence.
    """
    if not hasattr(cls, "_arm"):
        assert _python_on_start(cls) is None or "begin_arming" not in inspect.getsource(
            _python_on_start(cls)), f"{cls.__name__} arms without an `_arm` — this guard is stale"
        pytest.skip(f"{cls.__name__} has no `_arm`: not calendar-driven, nothing to defer")

    fn = _python_on_start(cls)
    assert fn is not None, (
        f"{cls.__name__} has an `_arm` but no `on_start` of its own, so it arms from somewhere this "
        f"guard cannot see")
    tree = ast.parse(inspect.getsource(fn).lstrip())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "_arm" not in called, (
        f"{cls.__name__}.on_start calls `_arm` directly, which reaches the broker's calendar over a "
        f"blocking socket during start; route it through `begin_arming`")
    assert "begin_arming" in called, (
        f"{cls.__name__}.on_start never calls `begin_arming`, so this lane resolves its schedule "
        f"somewhere this guard cannot see")


def test_bctrot_really_does_INHERIT_the_shared_on_start():
    """The fact that made one defect look like two, recorded so it cannot quietly stop being true.

    If BCTROT ever grows its own `on_start`, it stops being covered by momentum's fix and every
    conclusion drawn from "they share the code" expires with it.
    """
    from kumo_strategies.strategies.bct.nautilus import BCTRotationStrategy

    assert "on_start" not in BCTRotationStrategy.__dict__
    assert _python_on_start(BCTRotationStrategy) is M.MomentumRotationStrategy.on_start


# -- a failure that is NOT the calendar's ---------------------------------------------------------

class _BadArm(_Host):
    """A lane whose `_arm_now` raises deterministically, for a reason no retry can fix.

    THE CONCRETE TRIGGER is a malformed slot spec. `slots.resolve` parses each spec through `_parse`,
    which RAISES `SlotError` on an unparseable one rather than dropping it — deliberately, so a slot
    that cannot be resolved never becomes a session that silently decides fewer times than configured.
    `*_SLOTS` is operator-editable settings, so `["open+45"]` with the `m` missing is one keystroke
    away at any time.
    """

    def __init__(self, calendar, now, exc):
        super().__init__(calendar, now)
        self._exc = exc
        self.arm_attempts = 0

    def _arm_now(self):
        self.arm_attempts += 1
        raise self._exc


def _working_calendar():
    from datetime import datetime
    from kumo_strategies.runtime.calendar import ET, TradingDay

    class Working:
        def day(self, d):
            if d.weekday() >= 5:
                return None
            return TradingDay(d, datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
                              datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))

    return Working()


def test_a_failure_that_is_not_the_calendars_is_NOT_retried_as_if_it_were():
    """A bug must not be laundered into a business condition (found in review of 7de760a).

    `warm()` is the call that can fail for network reasons. `_arm_now()` is not — it resolves slots,
    computes the next fire and sets a time alert. Catching both under one `except` means a malformed
    slot spec produces a lane that starts, never arms, never decides, and logs THE TRADING CALENDAR
    DID NOT ANSWER every 60 seconds forever, pointing every diagnosis at a socket that is fine.

    Every other guard in this file stays green through that: the lane does start, does not block, and
    does log. So it needs its own.

    The assertion is on ATTEMPT COUNT, because "does not retry" is the property. A deterministic
    failure retried is a diagnosis sent in the wrong direction for as long as nobody reads the code.
    """
    host = _BadArm(_working_calendar(), pd.Timestamp("2026-08-24 12:00", tz="UTC"),
                   ValueError("unrecognised slot 'open+45'"))
    host.ARM_RETRY_SECS = 0.01       # a retrying implementation racks up attempts in the window
    _run(host)
    assert host.arm_attempts == 1, (
        f"a deterministic non-calendar failure was retried {host.arm_attempts} times; it will be "
        f"retried forever in production")


def test_a_non_calendar_failure_is_reported_AS_ITSELF_not_as_a_network_outage():
    """The message must name what actually broke.

    "UNARMED — the trading calendar did not answer" said about a slot-parsing bug is worse than no
    message: it is a confident, specific, wrong answer, and it is the first thing anyone reads.
    """
    host = _BadArm(_working_calendar(), pd.Timestamp("2026-08-24 12:00", tz="UTC"),
                   ValueError("unrecognised slot 'open+45'"))
    host.ARM_RETRY_SECS = 0.01
    _run(host)
    said = " ".join(host.log.errors)
    assert said, "a lane that failed to arm said nothing at all"
    assert "did not answer" not in said, (
        f"a slot-parsing bug was reported as a calendar outage: {said}")
    assert "open+45" in said, f"the actual failure was not named: {said}"


def test_the_forward_SCAN_stays_inside_the_span_that_was_warmed():
    """`_ensure` is a RANGE check, not a cache check, so a scan past the warmed span refetches.

        if self._span and self._span[0] <= around <= self._span[1]: return
        self._fetch(around - WARM_BACK_DAYS, around + WARM_FORWARD_DAYS)

    `_arm_now()` runs back ON THE EVENT LOOP once `warm()` has returned, on the assumption that the
    cache can now answer without I/O. That assumption holds only while the forward scan stays inside
    the warmed span — and a scan reaching past it would put a blocking `urlopen` on the loop, which
    is the defect this module exists to remove, reached through the code path that removes it.

    It cannot fire today: `next_slot_fire` breaks at the first trading day, 0-4 days out. It was two
    independent literals (45 and 40) one refactor from being wrong, so this asserts the relation AND
    that both sites read the shared names — a test comparing two constants nothing uses would pass
    while the code still carried its own numbers.
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap

    from kumo_strategies.runtime import calendar as C

    assert C.SCAN_DAYS <= C.WARM_FORWARD_DAYS, (
        f"a scheduling scan looks {C.SCAN_DAYS} days forward but the calendar warms only "
        f"{C.WARM_FORWARD_DAYS}; a scan past the warmed span fetches ON THE EVENT LOOP")

    def names_in(fn):
        tree = _ast.parse(textwrap.dedent(_inspect.getsource(fn)))
        return {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}

    assert "SCAN_DAYS" in names_in(C.next_slot_fire), (
        "`next_slot_fire` carries its own scan distance instead of the shared one, so widening it "
        "would not move the warmed span with it")
    assert "WARM_FORWARD_DAYS" in names_in(C.AlpacaCalendar._ensure), (
        "`_ensure` carries its own warm distance instead of the shared one")
