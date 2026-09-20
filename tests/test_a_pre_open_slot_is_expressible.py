"""A pre-open decision must be expressible, and today it is silently clamped (#131).

CRSISHORT's entry is a limit-on-open. IB accepts an OPG order until roughly 09:28 ET, so the
DECISION has to happen before the open — the ruled pair is `("open-10m", "open+5m")`. The signal
permits it: `apply_gates` computes the entry limit from the PRIOR close, so nothing about the
decision needs the session it trades into.

`slots.resolve` clamps it away:

    resolve(("open-10m", "open+5m"), open_at=09:30, close_at=16:00)
      open-10m   -> 09:30      NOT 09:20
      open+5m    -> 09:35

That clamp is DELIBERATE and correct for the lanes that existed when it was written — "A slot
resolving before the open is clipped forward to the open", because a rotation lane cannot trade
pre-open and a decision scheduled into a shut venue is the failure `runtime/calendar.py` exists to
prevent. It is not a bug being removed here. It is a default being made explicit.

THE DEFECT IS ONLY VISIBLE WHEN THE TWO MECHANISMS ARE EXECUTED TOGETHER, which is why neither file
looks wrong. `opening_leg` accepts `open-10m` because it parses to open-with-a-non-positive-offset,
which is true. `resolve` rewrites it because a pre-open slot was meaningless for every existing
caller, which was also true. The OPG order then goes out at 09:30, the venue refuses or converts it,
and every check upstream passes while the decomposition sits inert — the exact failure
`opening_leg`'s own guard was written to prevent, one layer below the guard.

OPT-IN, NOT OPT-OUT. `allow_pre_open` defaults False so every existing caller is byte-identical and
a mistyped `open-10m` on a rotation lane still cannot schedule a decision into a shut venue. A lane
that genuinely needs the auction has to say so.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import pytest

from kumo_strategies.strategies.momentum_rotation.slots import resolve

OPEN = datetime(2026, 9, 11, 9, 30)
CLOSE = datetime(2026, 9, 11, 16, 0)


def _at(specs, **kw):
    return {name: when for name, when in resolve(specs, OPEN, CLOSE, **kw)}


def test_a_pre_open_slot_is_CLAMPED_by_default_exactly_as_today():
    """The existing behaviour, pinned FIRST. Every shipped lane relies on it, and a change that
    quietly moved a rotation lane's decision would be worse than the gap it closes."""
    assert _at(("open-10m", "open+5m")) == {"open-10m": OPEN,
                                            "open+5m": datetime(2026, 9, 11, 9, 35)}


def test_a_lane_that_ASKS_gets_its_pre_open_slot():
    got = _at(("open-10m", "open+5m"), allow_pre_open=True)
    assert got["open-10m"] == datetime(2026, 9, 11, 9, 20), (
        f"the pre-open slot resolved to {got['open-10m']:%H:%M}; an OPG order decided at or after "
        f"the open cannot reach the auction (IB accepts OPG until ~09:28)")
    assert got["open+5m"] == datetime(2026, 9, 11, 9, 35), "the intraday slot moved"


def test_allow_pre_open_DEFAULTS_to_False():
    """Opt-in. A caller that has not thought about pre-open must not get one — the clamp is
    load-bearing for a lane that cannot trade before the open."""
    p = inspect.signature(resolve).parameters["allow_pre_open"]
    assert p.default is False
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
        "allow_pre_open must be keyword-only; positionally it could be passed as close_at")


def test_a_pre_open_slot_still_may_not_resolve_past_the_CLOSE():
    """The other end of the clamp is unchanged and is NOT opt-in: a decision after the close fires
    into a shut venue. `close-30m` on a half day is still meaningful; `close+30m` is not."""
    assert "close+30m" not in _at(("close+30m",), allow_pre_open=True)


def test_two_slots_cannot_COLLAPSE_onto_one_instant():
    """`resolve`'s own reason for dropping rather than clipping the late end: two decisions
    resolving to one time would share an idempotency key. Clamping the early end can do it too —
    `open-10m` and `open-5m` both become 09:30 under the default — so a lane asking for pre-open
    must get distinct times."""
    got = resolve(("open-10m", "open-5m"), OPEN, CLOSE, allow_pre_open=True)
    times = [when for _name, when in got]
    assert len(set(times)) == len(times), f"two slots collapsed onto one instant: {got}"


def test_the_CLAMPED_case_still_collapses_and_that_is_why_it_is_opt_in():
    """Stated as a property rather than left implicit: under the default, two pre-open slots DO
    collapse. That is an argument for the opt-in, not a defect to fix — a lane that did not ask for
    pre-open scheduling should not be silently given two names for one instant either, and
    `validate` already refuses duplicate NAMES, which is a different check."""
    got = resolve(("open-10m", "open-5m"), OPEN, CLOSE)
    assert {when for _n, when in got} == {OPEN}


# -- the runtime seam: two callers that must not disagree -------------------------------------------

def test_BOTH_runtime_resolvers_can_ask_for_pre_open():
    """`next_slot_fire` arms the next decision; `elapsed_slots` detects one a restart missed. They call
    the SAME `resolve` — deliberately, because "live and backtest disagreeing about when a decision
    happens is the recurring defect shape in this repo".

    Plumbing the flag into one and not the other recreates that shape INSIDE the runtime: `next_fire`
    would arm 09:20 while `elapsed_slots` computed 09:30 for the same name, so a process starting at
    09:25 would believe the slot had not passed, arm it again, and decide twice — or start at 09:35
    and fail to notice the pre-open slot was lost. Both or neither.
    """
    from kumo_strategies.runtime import calendar as cal_mod

    for fn_name in ("next_slot_fire", "elapsed_slots"):
        fn = getattr(cal_mod, fn_name)
        p = inspect.signature(fn).parameters.get("allow_pre_open")
        assert p is not None, (
            f"{fn_name} cannot ask for a pre-open slot, so a lane whose decision must precede the "
            f"open has no way to arm one — the clamp applies and the OPG order goes out at the open")
        assert p.default is False, f"{fn_name} defaults to pre-open scheduling"
        assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_two_resolvers_AGREE_on_a_pre_open_slot():
    """Verify by disagreement: the same spec, the same session, through both runtime paths.

    A slot that armed at one instant and was judged elapsed at another is the defect this asserts
    against, and it is invisible from either side alone.
    """
    from datetime import timedelta

    from kumo_strategies.runtime import calendar as cal_mod

    class _Day:
        session = OPEN.date()
        open_at = OPEN.replace(tzinfo=cal_mod.ET)
        close_at = CLOSE.replace(tzinfo=cal_mod.ET)

    class _Cal:
        def day(self, d):
            return _Day if d == OPEN.date() else None

    specs = ("open-10m", "open+5m")
    before = _Day.open_at - timedelta(hours=1)
    _session, name, armed = cal_mod.next_slot_fire(_Cal(), before, specs, allow_pre_open=True)
    assert name == "open-10m", f"the pre-open slot was not the next to fire: {name}"

    after_it = armed + timedelta(minutes=1)
    elapsed = cal_mod.elapsed_slots(_Cal(), after_it, specs, allow_pre_open=True)
    judged = {n: w for _d, n, w in elapsed}
    assert judged["open-10m"] == armed, (
        f"armed at {armed:%H:%M} and judged elapsed at {judged['open-10m']:%H:%M} — the two "
        f"resolvers disagree about when one slot fires")


def test_the_flag_covers_the_WALL_CLOCK_branch_too():
    """`_parse` accepts a bare `HH:MM`, and `09:20` is pre-open on a normal session but is NOT
    expressible as an offset — so it reaches the same clamp by a different route.

    Asserted rather than read: the clamp sits after both branches converge on `when`, so it happens
    to cover the wall clock. "Happens to" is how a later refactor that moves the clamp into the
    offset branch passes review.
    """
    assert _at(("09:20",))["09:20"] == OPEN, "the default no longer clamps a pre-open wall clock"
    assert _at(("09:20",), allow_pre_open=True)["09:20"] == datetime(2026, 9, 11, 9, 20)


def test_the_BACKTEST_callers_are_byte_identical_under_the_default():
    """THE BACKTEST CALLER DOES NOT GO THROUGH `calendar`: `backtesting/runner_sessions.py` calls
    `resolve(slots, open, close)` bare (`runner_cadence` did too, until it was folded, #270).

    With the default False they are unchanged, which is the point — a default of True would make the
    BACKTEST resolve a pre-open slot to its real time while LIVE (through `calendar`) clamped it, so
    research and live would disagree about what a slot NAME means. That is the divergence
    `calendar.next_slot_fire`'s own docstring argues must never exist, reappearing between the
    harness and the runtime instead of inside the runtime.

    They are deliberately NOT plumbed. Nothing sweeps a pre-open slot today, and adding a parameter
    no caller passes is how an option comes to have an untested value. If CRSISHORT is ever swept
    pre-open in research, this is the third site and it must be plumbed THEN — measured, not
    assumed, because a cadence result produced under a different clamp than live is unverifiable.
    """
    import ast
    import inspect
    import textwrap

    from kumo_strategies.backtesting.families import rotation as runner_sessions   # the rotation family is where the loops live (#270)

    for mod in (runner_sessions,):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod)))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None)) == "resolve"]
        assert calls, f"{mod.__name__} no longer calls resolve — update this test's premise"
        for call in calls:
            passed = {k.arg for k in call.keywords}
            assert "allow_pre_open" not in passed, (
                f"{mod.__name__} now passes allow_pre_open. That is a real decision and it needs "
                f"its own evidence: a swept cadence resolved under a different clamp than live is "
                f"not verifiable against it.")
