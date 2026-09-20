"""A restart inside the decision window must not silently lose the session (#32).

This cost a live session on 2026-08-17: the engine restarted at 09:35 ET — exactly MOMENTUM-002's
decision minute — armed forward to the 18th, and produced no decision row for the day. The only
trace anywhere was a startup line reading "first session 2026-08-18", which looks entirely normal.

`missed_sessions` did not catch it, because that is appended only when a session was already DUE and
a restart before the alert fires means it never became due. From the outside, "skipped because we
restarted" and "held because nothing ranked" are the same picture.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from kumo_strategies.runtime.calendar import ET, WeekdayCalendar, elapsed_slots


@pytest.fixture
def cal():
    return WeekdayCalendar()


def _at(hh, mm, day=17):
    return datetime(2026, 8, day, hh, mm, tzinfo=ET)


def test_starting_before_the_slot_has_missed_nothing(cal):
    assert elapsed_slots(cal, _at(9, 0), ("open+5m",)) == []


def test_starting_ONE_MINUTE_after_the_slot_reports_it(cal):
    """The exact 2026-08-17 case: 09:35 fires, restart at 09:36 arms tomorrow, day lost."""
    missed = elapsed_slots(cal, _at(9, 36), ("open+5m",))
    assert len(missed) == 1
    assert missed[0][1] == "open+5m"


def test_the_boundary_counts_as_missed(cal):
    """A restart AT the slot instant is the case that actually happened. Treating it as not-yet-due
    would arm for a moment already past and the alert would never fire."""
    assert len(elapsed_slots(cal, _at(9, 35), ("open+5m",))) == 1


def test_it_names_WHICH_slots_were_lost_not_just_how_many(cal):
    """An operator needs to know which decision is missing. On a two-slot strategy, losing midday and
    losing the close are different problems."""
    missed = elapsed_slots(cal, _at(15, 45), ("open+150m", "close-20m"))
    assert [m[1] for m in missed] == ["open+150m", "close-20m"]

    midday_only = elapsed_slots(cal, _at(13, 0), ("open+150m", "close-20m"))
    assert [m[1] for m in midday_only] == ["open+150m"]


def test_a_non_trading_day_has_no_missed_slots(cal):
    """Starting up on a Sunday has not skipped anything, and saying so would train an operator to
    ignore the warning."""
    assert elapsed_slots(cal, _at(12, 0, day=16), ("open+5m",)) == []


def test_it_reports_TODAY_only_not_a_backlog(cal):
    """Deliberately scoped. A process down for a week has missed a week, but this runs at startup to
    answer 'did I just eat today's decision' — the backlog question needs the journal, which the
    clock cannot see."""
    missed = elapsed_slots(cal, _at(23, 0), ("open+5m",))
    assert len(missed) == 1 and missed[0][0].day == 17
