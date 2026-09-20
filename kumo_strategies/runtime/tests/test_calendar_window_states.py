"""A venue calendar knows a ROLLING WINDOW, so "unknown" is a third state (kumo-trading-platform issue 628).

IB returns roughly six days of `liquidHours`. Real data decoded off staging's cached SPY instrument
on 2026-08-28:

    20260826:0930-20260826:1600;20260827:0930-20260827:1600;20260828:0930-20260828:1600;
    20260829:CLOSED;20260830:CLOSED;20260831:0930-20260831:1600

29 and 30 August are the weekend, and CLOSED reads identically to a holiday — which is what makes it
holiday-aware for free, and also why the WINDOW BOUNDARY is the only thing in it that can lie.

WHY THIS IS THE NORMAL PATH AND NOT AN EDGE CASE. `next_fire` scans 14 days forward and
`next_slot_fire` scans 45, against a window of six. Every call runs off the end of what the venue
said. If unknown collapses to None it becomes "closed", and closed is SILENT by design:

    elapsed_slots(cal, at, specs)  ->  td = cal.day(now.date())
                                       if td is None: return []          # "nothing was due"

A decision that never happens and leaves no trace is the 2026-08-17 failure this codebase already
paid for once, arriving through the window this time instead of through a restart.

Absence must not be readable as permission.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from kumo_strategies.runtime.calendar import (
    ET,
    OutsideCalendarWindow,
    TradingDay,
    WeekdayCalendar,
    build_calendar,
    elapsed_slots,
    next_slot_fire,
    warm,
)


#: The real staging window, verbatim. Sessions the venue described as OPEN.
_OPEN = {date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)}
#: Described, and CLOSED. The weekend — a holiday looks exactly the same.
_CLOSED = {date(2026, 8, 29), date(2026, 8, 30)}
#: Everything the venue said anything at all about.
_WINDOW = (date(2026, 8, 26), date(2026, 8, 31))


class VenueCalendar:
    """The shape cockpit will inject: three states, and it RAISES on the third."""

    def day(self, d: date) -> TradingDay | None:
        if not (_WINDOW[0] <= d <= _WINDOW[1]):
            raise OutsideCalendarWindow(
                f"the venue described sessions through {_WINDOW[1]}; {d} is beyond that window")
        if d not in _OPEN:
            return None
        return TradingDay(d, datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
                          datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))

    def is_trading_day(self, d: date) -> bool:
        return self.day(d) is not None

    def next_fire(self, after: datetime, offset_minutes: int) -> tuple[date, datetime]:
        d = after.astimezone(ET).date()
        for _ in range(14):
            td = self.day(d)                      # propagates OutsideCalendarWindow, deliberately
            if td is not None:
                fire = td.open_at + timedelta(minutes=offset_minutes)
                if fire > after.astimezone(ET):
                    return td.session, fire
            d += timedelta(days=1)
        raise RuntimeError("no trading day found")


def test_the_fixture_distinguishes_all_three_states():
    """FIXTURE PROPERTY FIRST. A window with no CLOSED day inside it cannot tell a calendar that
    reports unknown-as-closed from one that reports it correctly, because there would be nothing
    correct for closed to look like."""
    cal = VenueCalendar()
    assert cal.day(date(2026, 8, 28)) is not None, "an open session inside the window"
    assert cal.day(date(2026, 8, 29)) is None, "a CLOSED session inside the window"
    with pytest.raises(OutsideCalendarWindow):
        cal.day(date(2026, 9, 15))
    assert _CLOSED and _OPEN, "the window must contain both, or the distinction is untested"


def test_a_CLOSED_day_and_an_UNKNOWN_day_do_not_look_the_same():
    """THE HEADLINE, stated as the disagreement it is. Both were `None` before this ticket, and the
    conflation is what makes an unscheduled decision invisible."""
    cal = VenueCalendar()
    assert cal.day(date(2026, 8, 29)) is None
    with pytest.raises(OutsideCalendarWindow):
        cal.day(date(2026, 9, 1))


def test_elapsed_slots_REFUSES_rather_than_reporting_nothing_was_due():
    """The failure this exception exists for. `elapsed_slots` returns [] for a closed day, and [] is
    the correct answer there — so an unknown day silently claiming [] is a decision that never
    happens and leaves no trace."""
    at = datetime(2026, 9, 20, 14, 0, tzinfo=ET)          # past the window
    with pytest.raises(OutsideCalendarWindow):
        elapsed_slots(VenueCalendar(), at, ("open+5m",))


def test_next_slot_fire_REFUSES_rather_than_scheduling_a_day_nobody_described():
    """Scanning past the window must not silently skip to a day the venue never confirmed. Arming
    retries at ARM_RETRY_SECS, so refusing costs a minute and guessing costs an order."""
    after = datetime(2026, 8, 31, 17, 0, tzinfo=ET)       # after the last described session
    with pytest.raises(OutsideCalendarWindow):
        next_slot_fire(VenueCalendar(), after, ("open+5m",))


def test_warm_REFUSES_too_so_the_failure_lands_before_the_loop():
    """`warm` is the one blocking call and exists to move failure OFF the event loop, not to hide
    it. Nothing here may catch — a calendar that cannot answer must stay fatal to scheduling."""
    with pytest.raises(OutsideCalendarWindow):
        warm(VenueCalendar(), datetime(2026, 12, 1, 12, 0, tzinfo=ET))


def test_inside_the_window_everything_still_works():
    """The refusal must not be the only thing that happens. A calendar that refuses everything would
    pass every assertion above and schedule nothing at all."""
    cal = VenueCalendar()
    after = datetime(2026, 8, 27, 17, 0, tzinfo=ET)
    session, name, when = next_slot_fire(cal, after, ("open+5m",))
    assert session == date(2026, 8, 28)
    assert when == datetime(2026, 8, 28, 9, 35, tzinfo=ET)
    assert name == "open+5m"


def test_the_weekend_is_SKIPPED_not_refused():
    """A described-CLOSED day is a normal answer and the scan steps over it. Only the boundary
    refuses — if closed also raised, no calendar could ever schedule across a weekend."""
    cal = VenueCalendar()
    after = datetime(2026, 8, 28, 17, 0, tzinfo=ET)       # Friday evening; 29/30 CLOSED
    session, _, when = next_slot_fire(cal, after, ("open+5m",))
    assert session == date(2026, 8, 31), "the scan must step over the described weekend"
    assert when.date() == date(2026, 8, 31)


# ==================================================================================================
# THE INJECTION SEAM
# ==================================================================================================

def test_an_injected_calendar_SATISFIES_require_exchange(monkeypatch):
    """It does not relax the requirement. Relaxing the flag to obtain a boot is how a misconfigured
    environment quietly schedules Thanksgiving; meeting it with a real calendar is the fix."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    cal = VenueCalendar()
    assert build_calendar(require_exchange=True, calendar=cal) is cal


def test_a_calendar_missing_next_fire_is_REFUSED_AT_BUILD():
    """THE ONE THAT WOULD HAVE SHIPPED. `next_fire` is a METHOD on the calendar, not a module helper,
    and only QC27 and QC345 call it — `qc27_rotation.py:312`, `qc345_rotation.py:352`.

    A calendar with only `day` and `is_trading_day` satisfies `warm`, `next_slot_fire` and
    `elapsed_slots`, so it boots, passes preflight, and raises AttributeError the first time
    TECHIVOL-005 or QC345-003 arms — inside arming, which catches and retries, leaving the lane
    silently unarmed. The first draft of cockpit's double was exactly this shape.

    Checked structurally at BUILD so the failure lands in front of whoever is deploying.
    """
    class HalfACalendar:
        def day(self, d):
            return None

        def is_trading_day(self, d):
            return False

    with pytest.raises(TypeError, match="next_fire"):
        build_calendar(require_exchange=True, calendar=HalfACalendar())


def test_the_refusal_names_every_missing_member_not_just_the_first():
    """One name per round trip makes a caller fix, redeploy and fail again."""
    with pytest.raises(TypeError) as e:
        build_calendar(calendar=object())
    for member in ("day", "next_fire", "is_trading_day"):
        assert member in str(e.value)


def test_a_non_callable_attribute_does_not_satisfy_the_contract():
    """`getattr` is not enough: an attribute that merely EXISTS passes a presence check and then
    fails at the call.

    EVERY MEMBER HERE IS NON-NONE, deliberately. The first version of this decoy set `day = None`,
    and a check written as `getattr(...) is None` rejects that just as `callable(...)` does — so the
    test passed under both and the mutation that weakens presence-to-callable survived it untouched.
    An axis no fixture varies is invisible to mutation testing, and its green looks identical to
    real coverage.
    """
    class Decoy:
        day = "yes"
        next_fire = "soon"
        is_trading_day = 3

    with pytest.raises(TypeError) as e:
        build_calendar(calendar=Decoy())
    for member in ("day", "next_fire", "is_trading_day"):
        assert member in str(e.value), f"{member} exists but is not callable, and was accepted"


def test_the_fallback_and_alpaca_paths_are_UNCHANGED(monkeypatch):
    """Both tenants run this package on one pinned ref, and paper is the tenant actually trading."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert isinstance(build_calendar(require_exchange=False), WeekdayCalendar)
    with pytest.raises(RuntimeError, match="HOLIDAY-UNAWARE"):
        build_calendar(require_exchange=True)


def test_the_shipped_calendars_satisfy_their_own_contract():
    """Derived, not listed. The contract is enforced on injected calendars; if the ones shipped here
    could not meet it, the contract would be describing something this module does not do."""
    from kumo_strategies.runtime.calendar import _CALENDAR_CONTRACT, AlpacaCalendar

    for cls in (AlpacaCalendar, WeekdayCalendar):
        for member in _CALENDAR_CONTRACT:
            assert callable(getattr(cls, member, None)), f"{cls.__name__} lacks {member}"
