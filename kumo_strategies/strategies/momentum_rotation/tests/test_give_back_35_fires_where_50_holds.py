"""give_back_frac 0.35 exits a real MOMENTUM-shaped trade that 0.50 rides down (#221, step 1).

2026-09-13: MOMENTUM-002's give-back goes 0.50 → 0.35, alone first (ks#221; lab's
`research/sell-in-strength/exit_combo_by_lane.py`: +3.6pp, Sharpe 1.5 → 1.9, drawdown unchanged on
129 sessions, DSR 0.6 — direction consistent, sample short, the live read-back is the test).

THIS FILE PINS THE MECHANISM ON ONE REAL TRADE, from the panel the lab measured on. MU, entered at
the 2025-12-29 close (294.36), peaked at the 2026-02-02 close (437.83, +48.7%), and closed 2026-02-04
at 379.47 (+28.9%) — 40.7% of the peak gain surrendered. At 0.35 that is an exit on 02-04; at 0.50 it
is not, and the trade is still open on 02-06 at 394.48. The fixture is a committed cut of
`research/residual-gate/daily.parquet` (28 closes), so nothing here reads `research/` at run time.

WHAT THIS DOES NOT DO: change the deployed value. kumo-trading-strategies holds NO deployed give-back —
`ExitConfig.give_back_frac` defaults to None and the 0.50 lives in kumo-trading-platform
`strategies/momentum.py:_RESEARCHED`, reachable by settings only once cockpit's `strategies` schema
carries `MOMENTUM_GIVE_BACK_FRAC` (today `additionalProperties: false` strips it — step 0, cockpit).
So "seen red before the config change" is a cockpit-side property; on this side the tests below are
EARNED by mutation (the give-back arm removed, the fraction inverted, the trail not reconstructed),
recorded in the PR.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import replace

import pytest

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import (
    RECONSTRUCTED, TrailState, evaluate_exits, reconstruct_trail)

_FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURE = _FIX / "mu_2025-12-29_to_2026-02-06_closes.csv"
#: The two sides the MU fixture cannot show (coverage review): a trade BOTH arms exit, 0.35 earlier —
#: ALB entered 2025-12-29 (144.55), 0.35 exits 2026-01-16 off a first peak, 0.50 rides to the higher
#: peak (194.23, 01-27) and exits 2026-02-02; and a LOSER that never trades above entry — ACN
#: entered 2026-01-28 (270.29), lowest 201.35 — on which give-back must never arm at either fraction.
_BOTH = _FIX / "alb_2025-12-29_both_arms_exit_closes.csv"
_LOSER = _FIX / "acn_2026-01-28_never_above_entry_closes.csv"

#: The two arms of the decision, and the probe sessions the ticket's claim rests on.
CONTROL, CANDIDATE = 0.50, 0.35
ENTRY_DATE, PEAK_DATE, EXIT_DATE, LAST_DATE = "2025-12-29", "2026-02-02", "2026-02-04", "2026-02-06"


def _load(path: pathlib.Path) -> list[tuple[str, float]]:
    with path.open() as fh:
        return [(r["date"], float(r["close"])) for r in csv.DictReader(fh)]


def _closes() -> list[tuple[str, float]]:
    rows = _load(_FIXTURE)
    assert rows[0][0] == ENTRY_DATE and rows[-1][0] == LAST_DATE, rows[:1] + rows[-1:]
    return rows


def _walk(rows, frac: float, sym: str = "MU", confirm: int | None = None):
    """Walk a fixture the way the live runner walks a session: the trail is RECONSTRUCTED from the
    closes up to yesterday, today's close is the price, first match wins. The confirmation counter
    is the one piece of state that legitimately carries across sessions, so it is carried."""
    cfg = ExitConfig(give_back_frac=frac, give_back_confirm_sessions=confirm)
    entry = rows[0][1]
    carried = 0
    for j in range(1, len(rows)):
        state = reconstruct_trail(entry, [c for _, c in rows[:j]], covers_entry=True)
        state = replace(state, sessions_in_give_back=carried)
        plan = evaluate_exits(cfg, {sym: rows[j][1]}, {sym: state})
        if sym in plan.exits:
            return rows[j][0]
        carried = plan.state[sym].sessions_in_give_back
    return None


def _first_exit(frac: float, confirm: int | None = None) -> str | None:
    return _walk(_closes(), frac, confirm=confirm)


def test_FIXTURE_is_the_trade_the_docstring_describes():
    """Fixture property first: the peak and the give-back are what the claim rests on. If the cut
    were wrong, both arms below could agree for a reason that has nothing to do with the fraction."""
    rows = dict(_closes())
    entry, peak, exit_px = rows[ENTRY_DATE], rows[PEAK_DATE], rows[EXIT_DATE]
    assert peak == max(rows[d] for d in rows if d <= EXIT_DATE)
    surrendered = (peak - exit_px) / (peak - entry)
    assert CANDIDATE < surrendered < CONTROL, f"surrendered {surrendered:.3f} of the peak gain"
    assert pytest.approx(surrendered, abs=0.005) == 0.407


def test_0_35_EXITS_on_the_give_back_day_and_0_50_does_NOT():
    """THE CLAIM. Same trade, same trail, same sessions — only the fraction differs. Cockpit builds
    `ExitConfig(give_back_frac=give_back)` and nothing else (momentum.py:606), so confirm=1 IS the
    deployed shape; this is that walk."""
    assert _first_exit(CANDIDATE) == EXIT_DATE
    assert _first_exit(CONTROL) is None, "0.50 exited a trade the lab's control arm rode"


def test_with_TWO_confirmation_sessions_0_35_fires_one_session_LATER_not_never():
    """Not the deployed shape (cockpit passes no confirm), pinned so the knob's effect is known: at
    confirm=2 the breach must hold two sessions, and on MU it does — 02-04 and 02-05 — so the exit
    moves to 02-05. A counter that failed to carry across sessions would never fire."""
    assert _first_exit(CANDIDATE, confirm=2) == "2026-02-05"
    assert _first_exit(CONTROL, confirm=2) is None


def test_the_confirmation_counter_RESETS_when_the_breach_lifts():
    """A wiggle cannot accumulate toward an exit across unrelated sessions (PEAK's lesson, ported).
    Driven directly: two sessions in breach, one out, one in again — the counter reads 1, not 3."""
    st = TrailState(entry_px=100.0, peak_px=120.0, quality=RECONSTRUCTED)
    cfg = ExitConfig(give_back_frac=CANDIDATE, give_back_confirm_sessions=3)
    breach, clear = 110.0, 119.0                 # 110 gives back 50% of the 20 run; 119 gives back 5%
    st = evaluate_exits(cfg, {"X": breach}, {"X": st}).state["X"]
    st = evaluate_exits(cfg, {"X": breach}, {"X": st}).state["X"]
    assert st.sessions_in_give_back == 2
    st = evaluate_exits(cfg, {"X": clear}, {"X": st}).state["X"]
    assert st.sessions_in_give_back == 0, "the breach lifted and the counter did not reset"
    plan = evaluate_exits(cfg, {"X": breach}, {"X": st})
    assert plan.state["X"].sessions_in_give_back == 1 and "X" not in plan.exits


def test_on_a_trade_BOTH_arms_exit_0_35_exits_EARLIER():
    """Ordering, and the cost side in the same fixture: ALB's 0.35 exit on 01-16 comes off a FIRST
    peak; the trade then runs to a higher peak (194.23 on 01-27) that 0.50 rides before exiting on
    02-02. That is the "cut a winner short" the lab's 0.35 arm was costed against — the unit pins
    the ordering, the lab table in the module docstring carries the net."""
    rows = _load(_BOTH)
    assert rows[0] == ("2025-12-29", 144.55)
    early, late = _walk(rows, CANDIDATE, "ALB"), _walk(rows, CONTROL, "ALB")
    assert early == "2026-01-16" and late == "2026-02-02", (early, late)
    assert early < late


def test_on_a_LOSER_that_never_trades_above_entry_give_back_never_arms_at_either_fraction():
    """`peak_gain > 0` is the arming condition. A position that went straight against us has given
    nothing back, and reporting one is the SU line of 6 Aug — "gave back 30825% of a 0.1% peak"."""
    rows = _load(_LOSER)
    entry = rows[0][1]
    assert max(c for _, c in rows) == entry, "the fixture traded above entry — not a loser"
    assert min(c for _, c in rows) < entry * 0.75
    assert _walk(rows, CANDIDATE, "ACN") is None
    assert _walk(rows, CONTROL, "ACN") is None


def test_the_exit_names_the_fraction_it_fired_at_so_the_journal_can_be_read_back():
    """ks#221's acceptance: 'give-back exits fire at 35% of peak gain on the journal, not 50%'. The
    reason string is what the journal carries; it must say which fraction fired."""
    rows = _closes()
    state = reconstruct_trail(rows[0][1], [c for _, c in rows[:-3]], covers_entry=True)
    plan = evaluate_exits(ExitConfig(give_back_frac=CANDIDATE), {"MU": dict(rows)[EXIT_DATE]},
                          {"MU": state})
    assert "MU" in plan.exits
    assert "(trail 35%)" in plan.exits["MU"], plan.exits["MU"]     # the exact substring, not a regex
    assert plan.exits["MU"].startswith("gave back 41% of a 48.7% peak"), plan.exits["MU"]


def test_the_JOURNAL_row_carries_the_same_reason_the_evaluator_produced():
    """The seam. `pgrunner._trail_exits` returns the evaluator's reasons and the decision row writes
    `forced.get(sym)` verbatim (pgrunner.py:1204) — a runner that rewrote reasons ("exit: give-back")
    would pass the unit above and the operator still could not read the fraction back. Driven on the
    real `PgSessionRunner._trail_exits` with the MU trail at the give-back day."""
    import asyncio

    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

    rows = _closes()
    trail = reconstruct_trail(rows[0][1], [c for _, c in rows[:-3]], covers_entry=True)

    class Runner(PgSessionRunner):
        async def _load_state(self, symbols):
            return {"MU": trail}

        async def _save_state(self, sym, st, qty=None):
            return None

    class Broker:
        def positions(self):
            return {"MU": 10}

        def equity(self):
            return 100_000.0

        def last_price(self, sym, **_):
            return dict(rows)[EXIT_DATE]

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(),
               lifecycle=Lifecycle(State.TRADING, "armed"),
               cfg=MomentumRotationConfig(exits=ExitConfig(give_back_frac=CANDIDATE)), broker=Broker())
    forced = asyncio.run(r._trail_exits({"MU": 10}, {"MU": dict(rows)[EXIT_DATE]}))
    assert "MU" in forced, forced
    assert "(trail 35%)" in forced["MU"], forced["MU"]


def test_the_trail_is_RECONSTRUCTED_from_closes_not_remembered():
    """A remembered peak that does not survive a restart is how give-back went dead in production
    (#197 B1). The fixture walk above rebuilds the trail from bars every session; a state that
    carried the peak forward from entry would see a peak of `entry` and never arm."""
    rows = _closes()
    to_peak = reconstruct_trail(rows[0][1], [c for _, c in rows[:24]], covers_entry=True)
    assert to_peak.peak_px == dict(rows)[PEAK_DATE]
    assert to_peak.peak_is_trustworthy


def test_neither_arm_fires_on_the_way_UP():
    """Give-back is peak-relative; on the climb the position is at or near its peak and nothing
    has been surrendered. An arm that fired before 2026-02-02 would be a rule other than give-back.
    This also catches a trail rebuilt from closes INCLUDING today (look-ahead): on the up sessions
    today's close IS the new peak, and a peak that already contains it reads a give-back of zero
    where the honest trail reads a fresh high."""
    rows = _closes()
    entry = rows[0][1]
    for j in range(1, 24):
        state = reconstruct_trail(entry, [c for _, c in rows[:j]], covers_entry=True)
        for frac in (CANDIDATE, CONTROL):
            plan = evaluate_exits(ExitConfig(give_back_frac=frac), {"MU": rows[j][1]}, {"MU": state})
            assert "MU" not in plan.exits, (rows[j][0], frac, plan.exits)
