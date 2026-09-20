"""A restart inside a lane's decision window must not silently cost it the session.

`_arm` asks the calendar for the NEXT fire. A process that is gone when its alert is due comes back
and arms tomorrow — no alert, no decision row, no journal row. The "ALREADY DECIDED — RESUME" path
does not cover it: that covers the window between the journal write and the submit loop, and a lane
that never fired never got as far as the journal write. There is nothing to resume.

IT HAS COST TWO LIVE SESSIONS, NINE DAYS APART, ON TWO DIFFERENT LANES:

  2026-08-17  MOMENTUM-002. Engine restarted at 09:35 ET, exactly the decision minute, armed forward
              to the 18th, no decision row for the day. `calendar.elapsed_slots` and
              `momentum_rotation._report_missed_on_start` were written for it.
  2026-08-25  QC345-003 produced nothing all day — `exec_action_log` empty for the lane, no
              decision row. TECHIVOL decided at 12:00, so nothing at the stack level looked wrong.
              THAT ATTRIBUTION IS REFUTED, and the silence was CORRECT (settled same day).
              QC345 is MONTHLY: `_is_rebalance` delegates to `rebalance_dates()` = first session of
              each month, plus an operator `forced_rebalance_dates` set. August's first session was
              08-03; the live forced set was ['2026-08-20','2026-08-21','2026-08-24']. So 08-24 was
              forced and decided, and **08-25 was neither forced nor a month-start** — a
              non-rebalance session, on which `_try_decide` increments `skipped_sessions`, runs
              exit-only and returns WITHOUT a journal write. Silence is the designed behaviour.
              It was first blamed on a deploy landing on an `open+100m` slot, read from
              `/strategies.last_decision_slot` — which reports where a lane LAST decided, not where
              it fires next, and that value came from an override already removed.

The second one happened because the FIX FROM THE FIRST WAS NEVER PROPAGATED. `momentum_rotation`
has the check (and BCTROT inherits it); `qc345_rotation` and `qc27_rotation` schedule off an offset
rather than a slot tuple and never got it. Third instance this week of "fixed in one runner, the
sibling grew without it" — after the ownership reads (578fbb6) and the risk limits (a48ede1).

DELIBERATELY NOT A CATCH-UP. Firing a decision hours after its slot trades a stale ranking at prices
that have moved, and the slot exists precisely to fix when that happens. The requirement is that the
skip is VISIBLE, not that it is recovered.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.runtime.calendar import ET
from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import QC27RotationStrategy
from kumo_strategies.strategies.qc345_rotation.nautilus import QC345RotationStrategy

#: A session whose open+close bracket the restart below, so today's slot is already in the past.
_SESSION = dt.date(2026, 8, 25)
_OPEN = dt.datetime(2026, 8, 25, 9, 30, tzinfo=ET)
_CLOSE = dt.datetime(2026, 8, 25, 16, 0, tzinfo=ET)
#: 11:20 ET — after QC345's open+100m (11:10), after momentum's open+5m, before QC27's open+150m.
_RESTARTED_AT = dt.datetime(2026, 8, 25, 11, 20, tzinfo=ET)


class _Cal:
    """A calendar that answers for EVERY weekday.

    `elapsed_slots` only needs `day()`, but momentum's `_arm` goes through `next_slot_fire`, which
    walks forward looking for the next trading day and raises "no trading day found in the next 45
    days" against a double that answers for one date. A double narrow enough to break the code path
    under test is not a narrower double, it is a different test.
    """

    def day(self, d):
        if d.weekday() >= 5:
            return None
        return SimpleNamespace(
            session=d,
            open_at=dt.datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
            close_at=dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET),
        )

    def next_fire(self, after, offset):
        return pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-26T16:00:00Z")


def _probe(cls, **attrs):
    logged: list[str] = []
    clock = SimpleNamespace(utc_now=lambda: pd.Timestamp(_RESTARTED_AT), set_time_alert=lambda *a, **k: None)
    log = SimpleNamespace(error=lambda m, *a, **k: logged.append(str(m)),
                          warning=lambda m, *a, **k: logged.append(str(m)),
                          info=lambda *a, **k: None)

    class _P(cls):
        clock = property(lambda self: clock)
        log = property(lambda self: log)
        id = property(lambda self: cls.__name__)

    s = _P.__new__(_P)
    s._calendar = _Cal()
    s._armed_session = None
    s._armed_slot = None
    # `missed_sessions` is set ONLY where production has it. `momentum_rotation.__init__` declares
    # it; `qc27_rotation` and `qc345_rotation` declare `missed_rebalances` instead, which means a
    # different thing ("was due and found no data"). Injecting it on every lane gave the double an
    # attribute production lacks and made all three exercise momentum's legacy-append path — codex
    # judged it not masking this defect, and it is still the thing that just cost us finding 1.
    # `missed_on_start` is DELIBERATELY NOT INJECTED. Supplying it here is what hid the defect:
    # `report_missed_on_start` did `getattr(self, "missed_on_start", []).append(...)`, so with no
    # such attribute on the real adapter the append landed on a throwaway list and the record
    # evaporated — while this test, which had provided the attribute, saw it survive. A double with
    # an attribute production lacks (found by 2026-08-26).
    for k, v in attrs.items():
        setattr(s, k, v)
    s._arm_now()
    return s, logged


@pytest.mark.parametrize("cls,attrs", [
    (MomentumRotationStrategy, {"_slots": ("open+5m",), "_read_slots": None,
                                "missed_sessions": []}),
    (QC345RotationStrategy, {"_open_offset": 100, "_read_open_offset": None}),
    (QC27RotationStrategy, {"_open_offset": 150, "_read_open_offset": None}),
])
def test_every_lane_REPORTS_a_slot_that_elapsed_while_it_was_down(cls, attrs):
    """QC27's 150m slot is 12:00 ET and the restart is 11:20, so it is NOT missed — its row asserts
    the check does not cry wolf. The other two are past their slot and must say so."""
    s, logged = _probe(cls, **attrs)
    missed_expected = cls is not QC27RotationStrategy

    said = [m for m in logged if "did not run" in m or "STARTED AFTER" in m]
    if missed_expected:
        assert said, (
            f"{cls.__name__} restarted past its decision slot and reported NOTHING — the session is "
            f"lost and indistinguishable from a lane that ranked nothing")
        assert s.missed_on_start, f"{cls.__name__} did not record the missed slot for the operator"
    else:
        assert not said, f"{cls.__name__} reported a miss for a slot that has not happened yet"
        # `getattr` on the NEGATIVE case only: a lane with nothing missed never reaches the append,
        # so "no attribute" and "empty list" both correctly mean nothing was recorded. The positive
        # case above asserts the attribute directly, which is what catches the throwaway-list form.
        assert not getattr(s, "missed_on_start", [])


def test_it_still_ARMS_after_reporting():
    """Reporting must not replace scheduling. A lane that announced the miss and then failed to arm
    would lose tomorrow as well, which is strictly worse than the defect."""
    s, _ = _probe(QC345RotationStrategy, _open_offset=100, _read_open_offset=None)
    assert s._armed_session is not None, "reported the miss and never armed the next one"
