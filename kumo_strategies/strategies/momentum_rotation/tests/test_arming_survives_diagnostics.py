"""A missed-slot REPORT must never cost a lane its schedule.

`_arm_until_resolved` (contract.py) has two halves and only one of them retries:

    warm()      network. Fails -> announces, sleeps ARM_RETRY_SECS, RETRIES FOREVER.
    _arm_now()  "none of which can fail for a network reason". Fails -> announces, RETURNS.
                Never retried. Its own message: "This is not a network problem and will not be
                retried. This strategy CANNOT DECIDE until the cause is fixed and it is restarted."

That split is correct and deliberate: a malformed `*_SLOTS` entry is a deterministic bug, and
retrying it every 60s forever while blaming the venue is a bug laundered into a business condition.

BUT `_arm_now` NO LONGER SATISFIES ITS OWN PRECONDITION. Its docstring used to read "Arm, assuming
the calendar can answer WITHOUT I/O" over a bare `self._arm(...)`. It now calls
`report_missed_on_start`, which reaches `elapsed_slots` -> `cal.day(d)` -> `_ensure` -> `_fetch` ->
`urlopen`. So one transient calendar blip during arming lands in the NON-RETRYABLE branch and
permanently unarms the lane until restart — while reporting "this is not a network problem", which is
a confident, specific, wrong answer and the exact failure that comment exists to prevent.

`momentum_rotation` carried its own report-then-arm `_arm_now` before the mixin did, and dispatch is
virtual, so MOMENTUM-002 and BCTROT-004 have had this exposure the longest.

It has not fired in production — kumo-trading-platform grepped both tenants for the exact strings and found
zero of either announcement (2026-08-27). It is a trap, not an incident, and both are worth closing.

THE RULE: diagnostics may fail. Arming may not fail BECAUSE diagnostics failed.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.runtime.calendar import ET
from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Host(RegistrationMixin):
    EXTERNAL_ID = "TEST-001"
    LABEL = "test"

    def __init__(self, *, report_raises: bool = False, report_error: BaseException | None = None,
                 arm_error: BaseException | None = None, arm_fails: int = 0):
        self.armed_at = None
        self.logged: list[str] = []
        self._report_raises = report_raises
        self._report_error = report_error
        self.arm_calls = 0
        self._arm_error = arm_error
        self._arm_fails = arm_fails
        # SEPARATE LISTS. Funnelling `warning` and `error` into one made every severity change
        # invisible to every assertion in this file — which is how the guard shipped announcing a
        # lost-session signal at WARNING while every comparable line in the class is an ERROR, and
        # the tenants are grepped for ERROR. Found by Fable.
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self._log = SimpleNamespace(
            error=lambda m, *a, **k: (self.errors.append(str(m)), self.logged.append(str(m))),
            warning=lambda m, *a, **k: (self.warnings.append(str(m)), self.logged.append(str(m))),
            info=lambda *a, **k: None)
        self._clock = SimpleNamespace(utc_now=lambda: pd.Timestamp("2026-08-27T13:00:00Z"))

    @property
    def log(self):
        return self._log

    @property
    def clock(self):
        return self._clock

    @property
    def id(self):
        return "TEST-001"

    def report_missed_on_start(self, now):
        if self._report_error is not None:
            raise self._report_error
        if self._report_raises:
            # What `elapsed_slots` -> `cal.day` -> `urlopen` actually raises on a blip.
            raise OSError("urlopen error timed out")
        return []

    def _arm(self, after):
        # ABLE TO REFUSE. The original double only recorded, so the claim this guard rests on —
        # "a window refusal surfaces in `_arm` and is retried there" — was asserted in prose and
        # tested nowhere. A double that cannot fail the way production fails proves nothing.
        self.arm_calls += 1
        if self._arm_error is not None and self.arm_calls <= self._arm_fails:
            raise self._arm_error
        self.armed_at = after


def test_a_failing_missed_slot_REPORT_does_not_stop_the_lane_arming():
    """THE POINT. A lane that cannot report what it missed is a lane with worse diagnostics. A lane
    that cannot ARM is a lane that never decides again until someone restarts it — and the branch it
    lands in does not retry."""
    h = _Host(report_raises=True)
    h._arm_now()
    assert h.armed_at is not None, (
        "a transient calendar failure inside the missed-slot report prevented arming — one blip and "
        "the lane is permanently unarmed, reported as 'not a network problem'")


def test_the_failure_is_still_ANNOUNCED_not_swallowed():
    """Degrading quietly would trade one silent failure for another. The lane arms AND says the
    report did not happen."""
    h = _Host(report_raises=True)
    h._arm_now()
    assert any("missed" in m.lower() or "timed out" in m.lower() for m in h.logged), (
        f"the report failed and left no trace: {h.logged}")


def test_a_HEALTHY_lane_arms_and_says_nothing():
    """An alarm that fires on every boot buries the one that matters — the same reason warmup is not
    counted as a missed rebalance."""
    h = _Host(report_raises=False)
    h._arm_now()
    assert h.armed_at is not None
    assert h.logged == [], f"a clean arming logged: {h.logged}"


# ==================================================================================================
# THE INTERACTION THIS BRANCH PREDATES — a venue calendar can now REFUSE (kumo-trading-platform issue 628)
# ==================================================================================================

def test_a_ROLLING_WINDOW_refusal_from_the_REPORT_still_arms():
    """`report_missed_on_start` reaches the calendar (`elapsed_slots` -> `cal.day`), and since #628
    a venue-supplied calendar RAISES `OutsideCalendarWindow` for a date it never described. That is
    a new way for the diagnostic to fail, and it must behave like every other: report, arm anyway.

    Arming is not weakened by swallowing it here. `_arm` consults the same calendar immediately
    afterwards, so a window that genuinely cannot schedule surfaces there — where it is classified
    as retryable and retried every ARM_RETRY_SECS, rather than here where it would be read as a
    permanent fault.
    """
    from kumo_strategies.runtime.calendar import OutsideCalendarWindow

    h = _Host(report_error=OutsideCalendarWindow(
        "the venue described sessions through 2026-08-31; 2026-09-08 is beyond that window"))
    h._arm_now()
    assert h.armed_at is not None, (
        "a rolling-window refusal from the DIAGNOSTIC prevented arming. The report says what was "
        "missed; it cannot be allowed to decide whether the lane runs")
    assert any("window" in m.lower() for m in h.logged), "the refusal was swallowed silently"


def test_a_KeyboardInterrupt_is_NOT_swallowed():
    """The guard catches `Exception`, deliberately, not `BaseException`. A blanket catch here would
    absorb Ctrl-C and `SystemExit` during start-up, so a shutdown would look like a lane arming
    normally — a diagnostic guard is not a reason to become unkillable."""
    h = _Host(report_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        h._arm_now()
    assert h.armed_at is None


# ==================================================================================================
# WHAT FABLE FOUND — the guard made a retry reachable, and the report was not idempotent
# ==================================================================================================

#: A calendar whose rolling window can be MOVED, which is what a venue's actually does.
class _WindowCal:
    """Answers for weekdays inside `window_end`, REFUSES beyond it — as a venue calendar does.

    An ENVIRONMENT double, not a host double, and the distinction is the lesson of this file. The
    previous attempt faked `report_missed_on_start` itself — the subject eaten by its own fixture —
    so the idempotence test asserted on a list nothing populated. Fake what production does not own
    (the calendar, the clock, the log) and run every method under test for real.
    """

    def __init__(self, window_end):
        self.window_end = window_end

    def day(self, d):
        from kumo_strategies.runtime.calendar import OutsideCalendarWindow

        if d > self.window_end:
            raise OutsideCalendarWindow(
                f"the venue described sessions through {self.window_end}; {d} is beyond that window")
        if d.weekday() >= 5:
            return None
        return SimpleNamespace(
            session=d,
            open_at=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
            close_at=datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))


def _stalled_lane(window_end=date(2026, 8, 28), now=None):
    """A real MomentumRotationStrategy, built the way `test_missed_slot_on_start` builds one.

    THE SCENARIO, which is the one that produced the duplicates: restart after the close on the LAST
    day the venue's window describes. `report_missed_on_start` SUCCEEDS — today is in-window and its
    slot has elapsed — and `_arm` then scans forward to tomorrow, which is not, so it refuses and the
    retry loop comes back round.

    No `warm` monkeypatch: `warm` is just `cal.day(today)` and this double answers it. One less lie
    in the fixture than the rolling-window tests need.
    """
    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

    logged: list[str] = []
    at = now or datetime(2026, 8, 28, 16, 5, tzinfo=ET)
    clock = SimpleNamespace(utc_now=lambda: pd.Timestamp(at), set_time_alert=lambda *a, **k: None)
    log = SimpleNamespace(error=lambda m, *a, **k: logged.append(str(m)),
                          warning=lambda m, *a, **k: logged.append("WARN:" + str(m)),
                          info=lambda *a, **k: None)

    class _P(MomentumRotationStrategy):
        clock = property(lambda self: clock)
        log = property(lambda self: log)
        id = property(lambda self: "MOMENTUM-002")

    s = _P.__new__(_P)
    s._calendar = _WindowCal(window_end)
    s._slots = ("open+5m",)
    s._armed_session = None
    s._armed_slot = None
    # `_read_slots` is the live re-reader (#514). None means "not overridden", which is what a lane
    # without the settings hook has — set explicitly rather than left absent, because an absent
    # attribute here raises AttributeError from `_reread_slots` and would look like an arming bug.
    s._read_slots = None
    s.ARM_RETRY_SECS = 0
    # `missed_on_start` deliberately NOT injected — supplying it is what hid the original defect.
    return s, logged


def test_the_missed_record_is_written_ONCE_across_many_arming_attempts():
    """THE DEFECT THE GUARD CREATED, through the real report and the real retry loop.

    An hour of window stall used to add ~60 duplicate entries to a list operators read as "sessions
    we did not trade", plus an ERROR line a minute saying the same thing.
    """
    s, logged = _stalled_lane()

    async def _run():
        task = asyncio.ensure_future(s._arm_until_resolved())
        for _ in range(6):                       # let the loop turn several times
            await asyncio.sleep(0)
        s._calendar.window_end = date(2026, 9, 30)   # the venue extends its window
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_run())

    assert s._armed_session is not None, "the retry never armed — the loop is not being exercised"
    assert len(s.missed_on_start) == 1, (
        f"the missed-slot record was written {len(s.missed_on_start)} times for one start-up. "
        f"Arming retries; start-up happened once.")
    said = [m for m in logged if "STARTED AFTER" in m]
    assert len(said) == 1, f"the lost-session announcement repeated {len(said)} times"


def test_a_stall_that_outlasts_the_SESSION_does_not_claim_a_second_start_up():
    """THE EDGE MEMBERSHIP GOT WRONG, and why the key is a private flag rather than the record.

    The refusal this gate exists for can run past midnight. Keyed on `stamped in missed_on_start`,
    the next session's stamp differs, the gate passes, and the lane announces "STARTED AFTER … the
    session was lost in the gap" for a day on which the process did NOT start — while it has been up
    and announcing UNARMED the whole time. Those slots are missed-WHILE-UNARMED, which the retry
    branch already reports, and they are a different fact.
    """
    s, logged = _stalled_lane()
    s.report_missed_on_start(pd.Timestamp(datetime(2026, 8, 28, 16, 5, tzinfo=ET)))
    assert len(s.missed_on_start) == 1

    # The stall runs on into the next session. Same process, same start-up.
    s._calendar.window_end = date(2026, 9, 30)
    s.report_missed_on_start(pd.Timestamp(datetime(2026, 8, 31, 16, 5, tzinfo=ET)))

    assert len(s.missed_on_start) == 1, (
        "a stall that crossed a session boundary recorded a SECOND start-up miss. The process "
        "started once; the later slots were missed while UNARMED, which the retry branch reports.")
    assert len([m for m in logged if "STARTED AFTER" in m]) == 1


def test_a_report_that_RAISES_still_retries_next_time():
    """The flag is set on COMPLETION, not on entry. A report that dies before recording has said
    nothing, so the next arming attempt must try again rather than treating it as done."""
    s, _ = _stalled_lane(window_end=date(2026, 1, 1))   # today is already beyond the window
    from kumo_strategies.runtime.calendar import OutsideCalendarWindow

    with pytest.raises(OutsideCalendarWindow):
        s.report_missed_on_start(pd.Timestamp(datetime(2026, 8, 28, 16, 5, tzinfo=ET)))
    assert not getattr(s, "_missed_report_done", False), (
        "a report that raised was marked done, so the real start-up miss is never recorded")

    s._calendar.window_end = date(2026, 9, 30)
    s.report_missed_on_start(pd.Timestamp(datetime(2026, 8, 28, 16, 5, tzinfo=ET)))
    assert len(s.missed_on_start) == 1, "the retry after a failed report recorded nothing"


def test_a_report_failure_is_announced_at_ERROR_not_warning():
    """Every comparable announcement in this class is an error — UNARMED, STARTED AFTER, arming
    FAILED — and what is lost here is the same class of fact. The tenants are grepped for ERROR, so
    a warning is a line nobody sees."""
    h = _Host(report_raises=True)
    h._arm_now()
    assert any("could not report" in m for m in h.errors), (
        "the report failure was announced below ERROR; ERROR-filtered monitoring will not see it")
    assert not any("could not report" in m for m in h.warnings)


def test_the_guard_does_not_PROMISE_that_arming_worked():
    """It used to end "the schedule below is unaffected", asserted BEFORE `_arm` ran. When the report
    and the arming share a cause, that is a confident, specific, wrong claim printed directly above
    the real failure — the shape this method's own docstring condemns."""
    from kumo_strategies.runtime.calendar import OutsideCalendarWindow

    h = _Host(report_raises=True, arm_error=OutsideCalendarWindow("beyond the window"), arm_fails=99)
    with pytest.raises(OutsideCalendarWindow):
        h._arm_now()
    said = " ".join(h.errors)
    assert "unaffected" not in said, (
        "the guard claimed the schedule was unaffected, and arming then failed on the same cause")
