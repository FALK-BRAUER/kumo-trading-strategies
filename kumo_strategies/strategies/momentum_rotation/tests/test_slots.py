"""Decision slots (#29) — the identity of a decision, so the edges matter more than the happy path.

The live journal enforces one decision per session and keys it on the date (`pgjournal.py:51`).
Cadence changes that key to (session, slot), which means a slot name is not cosmetic: it is what a
retry deduplicates against. These tests are mostly about the cases where naming could collapse or
resolution could put a decision outside the session.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from kumo_strategies.strategies.momentum_rotation.slots import (
    DEFAULT_SLOTS, SlotError, every, resolve, session_key, validate)

ET = ZoneInfo("America/New_York")


def _session(open_h=9, open_m=30, close_h=16, close_m=0, day="2026-08-14"):
    d = date.fromisoformat(day)
    return (datetime(d.year, d.month, d.day, open_h, open_m, tzinfo=ET),
            datetime(d.year, d.month, d.day, close_h, close_m, tzinfo=ET))


def test_the_default_is_exactly_todays_live_behaviour():
    """MOMENTUM-002 decides once at open+5. The control must be expressible, not special-cased."""
    o, c = _session()
    got = resolve(DEFAULT_SLOTS, o, c)
    assert len(got) == 1
    assert got[0][0] == "open+5m"
    assert got[0][1] == o + timedelta(minutes=5)


def test_anchors_resolve_against_the_actual_session():
    o, c = _session()
    got = dict(resolve(("open+5m", "12:00", "close-30m"), o, c))
    assert got["open+5m"].hour == 9 and got["open+5m"].minute == 35
    assert got["12:00"].hour == 12 and got["12:00"].minute == 0
    assert got["close-30m"].hour == 15 and got["close-30m"].minute == 30


def test_a_half_day_moves_close_relative_slots_and_drops_what_falls_past_the_close():
    """The reason anchors beat an interval. A 13:00 close is a real calendar case.

    `close-30m` stays meaningful (12:30). A 15:30 wall-clock slot is after the close and must be
    DROPPED — firing a decision into a shut venue is the failure `runtime/calendar.py` exists for.
    """
    o, c = _session(close_h=13, close_m=0)
    got = dict(resolve(("open+5m", "close-30m", "15:30"), o, c))
    assert got["close-30m"].hour == 12 and got["close-30m"].minute == 30
    assert "15:30" not in got, "a slot past the close must not resolve to a decision"
    assert len(got) == 2


def test_a_slot_past_the_close_is_dropped_not_clipped_back():
    """Clipping would collapse two slots onto one instant, and two decisions would then share an
    idempotency key — the exact failure the naming exists to prevent."""
    o, c = _session(close_h=13, close_m=0)
    got = resolve(("close-30m", "15:00", "15:30"), o, c)
    assert [n for n, _ in got] == ["close-30m"]


def test_a_slot_before_the_open_is_clipped_forward_not_dropped():
    """Opposite direction, opposite treatment, and deliberately so: a pre-open slot means "as early
    as possible", which the open satisfies. There is only one such slot in practice, so clipping
    cannot collapse two names together."""
    o, c = _session()
    got = dict(resolve(("open-15m",), o, c))
    assert got["open-15m"] == o


def test_duplicate_slots_are_refused_because_a_name_is_a_key():
    with pytest.raises(SlotError, match="idempotency key"):
        validate(("open+5m", "OPEN+5M"))


def test_an_unparseable_slot_raises_rather_than_being_skipped():
    """Silent degradation is the #26 failure. A bad slot must not become a session that quietly
    decides fewer times than configured."""
    with pytest.raises(SlotError, match="unrecognised slot"):
        validate(("open+5m", "lunchtime"))
    with pytest.raises(SlotError):
        validate(("25:00",))
    with pytest.raises(SlotError, match="at least one"):
        validate(())


def test_every_390_reproduces_the_single_daily_decision():
    """The research generator must be able to express the control, or the sweep has no baseline."""
    o, c = _session()
    assert len(resolve(every(390), o, c)) == 1


@pytest.mark.parametrize("minutes,expected", [(120, 4), (60, 7), (30, 13)])
def test_every_generates_the_expected_number_of_slots(minutes, expected):
    o, c = _session()
    assert len(resolve(every(minutes), o, c)) == expected


def test_every_keeps_the_last_slot_off_the_closing_auction():
    """The measured spread widens into the close (1.81bps at 15:55 against a 1.36 low), so the
    generator stops short rather than putting an arm's final decision in the worst window."""
    o, c = _session()
    last = resolve(every(30), o, c)[-1][1]
    assert last <= c - timedelta(minutes=5)


def test_every_yields_fewer_slots_on_a_half_day_rather_than_a_different_grid():
    """An interval mechanism would silently change the number of decisions on these sessions, which
    is an uncontrolled variable inside the sweep measuring cadence. Offsets from the open degrade to
    a prefix of the same grid instead."""
    full_o, full_c = _session()
    half_o, half_c = _session(close_h=13, close_m=0)
    full = [n for n, _ in resolve(every(60), full_o, full_c)]
    half = [n for n, _ in resolve(every(60), half_o, half_c)]
    assert len(half) < len(full)
    assert half == full[:len(half)], "a half day must be a PREFIX of the full-day grid"


def test_session_key_is_stable_and_distinct_per_slot():
    assert session_key(date(2026, 8, 14), "close-30m") == "2026-08-14/close-30m"
    keys = {session_key(date(2026, 8, 14), n)
            for n, _ in resolve(every(120), *_session())}
    assert len(keys) == 4, "slots must not share an idempotency key"


def test_slots_come_back_sorted_by_time():
    """Callers iterate them as the session progresses; out-of-order slots would decide on stale
    state and fill against bars that had already passed."""
    o, c = _session()
    got = resolve(("close-30m", "open+5m", "12:00"), o, c)
    assert [n for n, _ in got] == ["open+5m", "12:00", "close-30m"]
    assert [w for _, w in got] == sorted(w for _, w in got)
