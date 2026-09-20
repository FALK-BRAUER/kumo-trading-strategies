"""A calendar that cannot see far enough YET must not silence a lane (kumo-trading-platform issue 628).

`OutsideCalendarWindow` was added so an unknown day could not be read as a closed one. It is raised
from `cal.day()`, which means it comes out of `_arm` — and `_arm` is called from two places that both
treated any failure as permanent. Fixing one and not the other leaves the lane broken in a way that
looks fixed, so both are asserted here.

    _arm_until_resolved   -> `_arm_now` failures were NOT retried, by design: the rule is that a
                             failure while arming is deterministic (a malformed `*_SLOTS` entry) and
                             retrying it forever while blaming the venue is worse than stopping.
                             Correct for that; exactly wrong for a rolling window, which resolves by
                             itself as the venue advances.

    _on_session_alert     -> re-arms as its FIRST statement, deliberately, because everything after
                             may raise. An unguarded raise from `_arm` therefore aborts the whole
                             callback: the session that just fired is never marked due, no decision
                             is taken for it, AND no future alert is set. Silent, permanent, and
                             mid-session.

The second is the worse of the two and neither was reachable before this ticket, because
`AlpacaCalendar` answers for any date and `WeekdayCalendar` answers for all of them.

Found by codex review of 3f98a04 — the exception written to stop a lane going silent would have made
a lane go silent.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pytest

from kumo_strategies.runtime.calendar import ET, OutsideCalendarWindow, TradingDay
from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Clock:
    """Records what was scheduled, so "it retried" is a fact rather than an absence of exception."""

    def __init__(self, now: datetime):
        self._now = now
        self.alerts: list[tuple[str, datetime]] = []

    def utc_now(self) -> datetime:
        return self._now

    def set_time_alert(self, name, when, callback, override=False):
        self.alerts.append((name, when))


class _Log:
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg):
        self.errors.append(str(msg))

    def warning(self, msg):
        pass

    def info(self, msg):
        pass


class _Lane(RegistrationMixin):
    """The narrowest host the mixin needs. Nothing here is more forgiving than a real lane."""

    id = "TESTLANE-999"

    def __init__(self, now: datetime, fails: int):
        self.clock = _Clock(now)
        self.log = _Log()
        self.arm_calls = 0
        self._fails = fails
        self.armed = False

    def _arm(self, after) -> None:
        self.arm_calls += 1
        if self.arm_calls <= self._fails:
            raise OutsideCalendarWindow(
                "the venue described sessions through 2026-08-31; 2026-09-08 is beyond that window")
        self.armed = True


_NOW = datetime(2026, 8, 31, 16, 5, tzinfo=ET)


# ==================================================================================================
# THE MID-SESSION RE-ARM — the worse of the two
# ==================================================================================================

def test_a_refusing_calendar_does_not_abort_the_alert_callback():
    """THE HEADLINE. `rearm_after_alert` must RETURN, not raise, so the caller goes on to decide the
    session that just fired. An exception here loses the decision as well as the schedule."""
    lane = _Lane(_NOW, fails=1)
    lane.rearm_after_alert(_NOW)             # must not raise
    assert lane.armed is False, "the fixture must actually have refused, or this asserts nothing"


def test_the_refusal_schedules_a_RETRY_rather_than_going_quiet():
    """Surviving the exception is not enough — a lane that survives and never arms again is the same
    silent outcome reached politely. The retry must be an observable scheduled alert."""
    lane = _Lane(_NOW, fails=1)
    lane.rearm_after_alert(_NOW)
    assert lane.clock.alerts, "no retry was scheduled — the lane would never arm again"
    name, when = lane.clock.alerts[-1]
    assert when == _NOW + timedelta(seconds=lane.ARM_RETRY_SECS)


def test_the_retry_does_NOT_reuse_the_session_alert():
    """The retry must not re-enter `_on_session_alert`, which does session bookkeeping — it records a
    missed session and clears `_due`. A retry firing that would fabricate a missed session on every
    window boundary."""
    lane = _Lane(_NOW, fails=1)
    lane.rearm_after_alert(_NOW)
    name, _ = lane.clock.alerts[-1]
    assert name == RegistrationMixin.REARM_ALERT
    assert "session" not in name.lower(), (
        "the retry shares a name with the session alert and would re-run session bookkeeping")


def test_the_retry_ACTUALLY_ARMS_once_the_window_advances():
    """The loop has to close. A retry that reschedules forever is a different silent failure."""
    lane = _Lane(_NOW, fails=1)
    lane.rearm_after_alert(_NOW)
    assert not lane.armed
    lane._on_rearm_retry()                   # the venue has since extended its window
    assert lane.armed, "the retry never armed even though the calendar answered"
    assert lane.arm_calls == 2


def test_it_keeps_retrying_while_the_window_stays_short():
    """One retry is not a policy. A boundary that persists for several minutes must keep producing
    retries rather than giving up after the first."""
    lane = _Lane(_NOW, fails=3)
    lane.rearm_after_alert(_NOW)
    for _ in range(2):
        lane._on_rearm_retry()
    assert not lane.armed
    assert len(lane.clock.alerts) == 3, "each refusal must schedule its own retry"
    lane._on_rearm_retry()
    assert lane.armed


def test_UNARMED_is_announced_every_time():
    """A lane that cannot schedule is counted as running while being incapable of deciding. That has
    to be loud on every occurrence, not once."""
    lane = _Lane(_NOW, fails=2)
    lane.rearm_after_alert(_NOW)
    lane._on_rearm_retry()
    assert len(lane.log.errors) == 2
    for msg in lane.log.errors:
        assert "UNARMED" in msg
        assert "window" in msg.lower()


def test_a_REAL_arming_bug_is_still_fatal_and_still_not_retried():
    """THE GUARANTEE THAT MUST SURVIVE. Only the rolling-window refusal is transient. A malformed
    `*_SLOTS` entry raises `SlotError` out of `slots.resolve` and retrying cannot fix it — reporting
    a deterministic bug as a venue problem every 60s is the failure this classification exists to
    avoid, and it must not be widened into a blanket catch."""
    class _Broken(_Lane):
        def _arm(self, after):
            raise ValueError("open+45 is not a slot spec")

    lane = _Broken(_NOW, fails=0)
    with pytest.raises(ValueError):
        lane.rearm_after_alert(_NOW)
    assert not lane.clock.alerts, "a deterministic bug must not be retried"


# ==================================================================================================
# THE START-UP PATH
# ==================================================================================================

def test_start_up_arming_RETRIES_a_window_refusal_instead_of_giving_up():
    """`_arm_now` failures are deliberately not retried, and that classification is right for the
    failure it was written for. A rolling-window refusal is the opposite kind and must not fall into
    it — that would leave the lane unarmed until somebody restarted the node."""
    lane = _Lane(_NOW, fails=1)
    lane.ARM_RETRY_SECS = 0                  # do not make the suite wait on a real backoff
    lane._armed_ok = False

    def _arm_now():
        lane._arm(lane.clock.utc_now())

    lane._arm_now = _arm_now
    lane._calendar = object()

    async def _run():
        # `warm` is patched out: this test is about the arming half, and the fetch half already has
        # its own retry which this must not be confused with.
        import kumo_strategies.runtime.calendar as cal_mod

        real_warm = cal_mod.warm
        cal_mod.warm = lambda cal, at: None
        try:
            await asyncio.wait_for(lane._arm_until_resolved(), timeout=5)
        finally:
            cal_mod.warm = real_warm

    asyncio.run(_run())
    assert lane.armed, "the start-up loop gave up on a refusal that resolves by itself"
    assert lane.arm_calls == 2, "it must have retried exactly once, then succeeded"
    assert any("UNARMED" in m and "rolling-window" in m for m in lane.log.errors), (
        "the retry was silent, or blamed something other than the window")


def test_start_up_arming_still_GIVES_UP_on_a_deterministic_failure():
    """The other side of the same classification, so widening the catch is a test failure."""
    class _Broken(_Lane):
        def _arm(self, after):
            self.arm_calls += 1
            raise ValueError("open+45 is not a slot spec")

    lane = _Broken(_NOW, fails=0)
    lane._arm_now = lambda: lane._arm(lane.clock.utc_now())
    lane._calendar = object()

    async def _run():
        import kumo_strategies.runtime.calendar as cal_mod

        real_warm = cal_mod.warm
        cal_mod.warm = lambda cal, at: None
        try:
            await asyncio.wait_for(lane._arm_until_resolved(), timeout=5)
        finally:
            cal_mod.warm = real_warm

    asyncio.run(_run())
    assert lane.arm_calls == 1, "a deterministic arming bug was retried"
    assert any("not a network problem" in m for m in lane.log.errors)


# ==================================================================================================
# EVERY LANE, NOT THE ONES I REMEMBERED
# ==================================================================================================

def _lanes():
    """Every cockpit lane, discovered — the same rule `test_contract` uses."""
    import inspect

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    out = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, Strategy)
                    and getattr(obj, "EXTERNAL_ID", None)):
                out.append(obj)
    return sorted(set(out), key=lambda c: c.__name__)


@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_every_lane_re_arms_through_the_guarded_path(cls):
    """AST-bound, and aimed at the class rather than a list of three files.

    A lane calling `self._arm(...)` directly inside its session-alert handler has the unguarded
    ordering back. Asserted on the parse tree rather than the source text, because a grep is
    satisfied by a comment mentioning the right name and broken by one mentioning the wrong one.
    """
    import ast
    import inspect
    import textwrap

    handler = getattr(cls, "_on_session_alert", None)
    if handler is None:
        pytest.skip(f"{cls.__name__} has no session alert handler")

    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "rearm_after_alert" in called, (
        f"{cls.__name__}._on_session_alert does not re-arm through `rearm_after_alert`")
    assert "_arm" not in called, (
        f"{cls.__name__}._on_session_alert calls `_arm` directly. A rolling-window refusal there "
        f"aborts the callback: the fired session is never decided and no future alert is set.")


# ==================================================================================================
# STOPPING MUST ACTUALLY STOP IT — found by codex on 822c629
# ==================================================================================================

class _StoppableClock(_Clock):
    """Carries `timer_names` and `cancel_timer`, as Nautilus's clock does. A double without them
    would let `cancel_arming` look like it cancelled something it never touched."""

    @property
    def timer_names(self):
        return [n for n, _ in self.alerts]

    def cancel_timer(self, name):
        self.alerts = [(n, w) for n, w in self.alerts if n != name]


def test_cancel_arming_also_cancels_the_REARM_retry():
    """`_arming` (an asyncio task) and REARM_ALERT (a clock timer) are two different mechanisms and
    `on_stop` reached only the first. A stopped lane kept rescheduling itself every ARM_RETRY_SECS."""
    lane = _Lane(_NOW, fails=1)
    lane.clock = _StoppableClock(_NOW)
    lane.rearm_after_alert(_NOW)
    assert RegistrationMixin.REARM_ALERT in lane.clock.timer_names

    lane.cancel_arming()
    assert RegistrationMixin.REARM_ALERT not in lane.clock.timer_names, (
        "a stopped lane keeps retrying, and the retry that succeeds arms a session alert on a "
        "strategy an operator has stopped")


def test_a_retry_already_in_flight_does_not_arm_a_STOPPED_lane():
    """The race cancelling cannot close: this callback can already be executing on the timer thread
    when `on_stop` runs. `is_running` is Nautilus's own component state."""
    lane = _Lane(_NOW, fails=0)
    lane.clock = _StoppableClock(_NOW)
    lane.is_running = False
    lane._on_rearm_retry()
    assert lane.arm_calls == 0, "a stopped lane armed anyway"
    assert not lane.armed


def test_a_RUNNING_lane_still_retries():
    """The lifecycle guard must not be the thing that silences the lane."""
    lane = _Lane(_NOW, fails=0)
    lane.clock = _StoppableClock(_NOW)
    lane.is_running = True
    lane._on_rearm_retry()
    assert lane.armed, "the guard blocked a legitimate retry"


@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_every_lane_routes_stop_through_cancel_arming(cls):
    """Why the cancellation lives in `cancel_arming` and not in three `on_stop` bodies: every lane
    already calls it, so a fourth inherits the fix by calling one method. Bound to the AST so this
    fails if a lane stops routing through it."""
    import ast
    import inspect
    import textwrap

    # WALK THE MRO for the first Python-defined `on_stop`, rather than reading `cls.on_stop`.
    # BCTROT inherits momentum's and would be skipped by a `__dict__` check, which is the lane with
    # two slots and the one where losing the schedule matters most. TemplateRotationStrategy defines
    # none and inherits Nautilus's Cython base, which has no retrievable source — that is a genuine
    # skip, and it is separated from the inheriting case here rather than conflated with it.
    stop = next((b.__dict__["on_stop"] for b in cls.__mro__ if "on_stop" in b.__dict__), None)
    try:
        src = inspect.getsource(stop) if stop is not None else None
    except (TypeError, OSError):
        src = None
    if src is None:
        pytest.skip(f"{cls.__name__} defines no Python `on_stop` — it sets no timers of its own")
    tree = ast.parse(textwrap.dedent(src))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "cancel_arming" in called, (
        f"{cls.__name__}.on_stop does not call `cancel_arming`, so its re-arm retry timer survives "
        f"the stop and can arm a session alert on a stopped strategy")
