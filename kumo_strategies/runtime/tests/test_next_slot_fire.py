"""Multi-slot scheduling for the LIVE clock (#29, #32).

`next_fire` answers "when is the next session's single decision", which has no answer once a
strategy decides twice a day. The live adapter was built entirely around that assumption — one
alert, one offset — while the backtest runner and the decision store's `(session, slot)` key had
already moved on. This is the primitive that closes the gap.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from kumo_strategies.runtime.calendar import ET, WeekdayCalendar, next_slot_fire


@pytest.fixture
def cal():
    return WeekdayCalendar()


def _at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_a_single_slot_reproduces_the_old_one_decision_behaviour(cal):
    """MOMENTUM-002 is live on exactly this. Its schedule must not move."""
    session, slot, when = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+5m",))
    assert slot == "open+5m"
    assert when.hour == 9 and when.minute == 35


def test_two_slots_fire_TWICE_in_one_session(cal):
    """The property the live path could not express. Both fires belong to the same session and are
    distinguished by slot, which is what makes them separately idempotent."""
    first = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+150m", "close-20m"))
    second = next_slot_fire(cal, first[2], ("open+150m", "close-20m"))
    assert first[0] == second[0], "both decisions belong to the same session"
    assert first[1] == "open+150m" and second[1] == "close-20m"
    assert first[2] < second[2]


def test_after_the_last_slot_it_rolls_to_the_next_session(cal):
    day1 = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+150m", "close-20m"))
    last = next_slot_fire(cal, day1[2], ("open+150m", "close-20m"))
    nxt = next_slot_fire(cal, last[2], ("open+150m", "close-20m"))
    assert nxt[0] > last[0], "did not roll to the next session"
    assert nxt[1] == "open+150m", "the new session starts at its first slot"


def test_it_never_returns_a_time_at_or_before_the_moment_asked(cal):
    """A fire time in the past re-arms instantly and spins. Boundary included deliberately: asking
    exactly AT a slot must return the NEXT one, not the same one again."""
    _, _, when = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+150m", "close-20m"))
    _, slot2, when2 = next_slot_fire(cal, when, ("open+150m", "close-20m"))
    assert when2 > when and slot2 == "close-20m"


def test_the_slot_NAME_comes_back_because_it_is_the_idempotency_key(cal):
    """`2026-08-14/close-20m` survives a restart or a retry. A wall-clock timestamp cannot be told
    apart from a genuinely new decision — the #197 identity-split shape."""
    _, slot, _ = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("close-20m",))
    assert slot == "close-20m"


def test_slots_resolve_through_the_SAME_function_the_backtest_uses(cal):
    """Live and backtest disagreeing about when a decision happens would make every cadence result
    unverifiable rather than merely wrong. Asserted by construction, not by comment."""
    from kumo_strategies.strategies.momentum_rotation.slots import resolve
    td = cal.day(next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+150m",))[0])
    expected = dict(resolve(("open+150m", "close-20m"), td.open_at, td.close_at))
    _, slot, when = next_slot_fire(cal, _at(2026, 8, 17, 6, 0), ("open+150m", "close-20m"))
    assert expected[slot] == when
