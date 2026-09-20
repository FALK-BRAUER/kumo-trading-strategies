"""The quarantine trigger must CONFIRM, and must fire only on the tail it is named for (#147, #171).

Three defects, all measured by driving `assess` rather than reading it.

1. `min_windows_to_act` COUNTS WINDOWS PRODUCED, NOT WINDOWS THAT AGREE. Its own docstring says
   "How many independent live windows must AGREE before `acts` is allowed to be True"; the code is
   `if live_windows < envelope.min_windows_to_act`, and `live_windows` is the lane's TOTAL window
   count, supplied by the caller. Driven on one catastrophically bad window and nothing else:

       live_windows=1  acts=False        live_windows=3  acts=True   STAND_DOWN
       live_windows=2  acts=False        live_windows=4  acts=True   STAND_DOWN

   So after 39 sessions a SINGLE bad window quarantines the lane. "Three windows must agree" was
   never implemented — the knob is a WARMUP COUNTER wearing a confirmation rule's name.

2. A LANE ABOVE ITS 90th PERCENTILE IS QUARANTINED. `if pct < lo or pct > hi` is two-sided and both
   ends produce STAND_DOWN, which Falk defines as A STRATEGY THAT LOST ITS EDGE. Driven:

       return_pct  -30.00  far BELOW p10 (-4.55)      STAND_DOWN
       return_pct   +6.50  just above p90 (+6.38)     STAND_DOWN
       return_pct  +30.00  far ABOVE p90              STAND_DOWN

   Two-sided DETECTION is right — a lane far above its own envelope is genuinely not doing what it
   said, and a sizing bug, a data error and a config change all look like this. What is wrong is one
   action, named for one tail, serving both. The #144 shape: a flag named for blocking that
   liquidated.

3. THE FALSE RATE IS DESIGNED IN. A pre-registered (10, 90) band puts 20% of windows outside BY
   CONSTRUCTION when the lane behaves exactly as measured, on each of two banded metrics. With a
   confirmation rule that does not confirm, that is several stand-downs per lane-year on a healthy
   lane.

WARMUP AND CONFIRMATION ARE DIFFERENT QUESTIONS AND GET DIFFERENT COUNTERS. A lane that has not
produced enough windows to judge is UNKNOWN; a lane that has produced them and breached once is
OUT_OF_ENVELOPE without acting. Sharing one counter is how this defect existed at all.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.envelope import Envelope, assess
from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, UNKNOWN, SelfAction)

FP = "test-fingerprint"


def _env(**kw):
    base = dict(
        config_fingerprint=FP,
        horizon_sessions=13,
        samples=14,
        distribution={"return_pct": {10: -4.55, 25: -2.71, 50: 0.78, 75: 3.26, 90: 6.38}},
        bands={"return_pct": (10.0, 90.0)},
        min_live_sessions=13,
        min_windows_to_act=2,
    )
    base.update(kw)
    return Envelope(**base)


def _assess(value, *, windows_out=0, live_windows=3, env=None, sessions=13):
    env = env or _env()
    return assess(env, {"return_pct": value}, config_fingerprint=FP,
                  live_sessions=sessions, live_windows=live_windows,
                  consecutive_breaches=windows_out)


# -- (a) the confirmation rule must actually confirm ------------------------------------------------

def test_ONE_bad_window_does_NOT_quarantine_however_long_the_lane_has_run():
    """The defect, stated as its fix. A lane that has produced 50 windows and breached ONCE has one
    bad window — being old is not evidence."""
    for lw in (3, 10, 50):
        a = _assess(-30.0, windows_out=1, live_windows=lw)
        assert a.state == OUT_OF_ENVELOPE, "the breach must stay VISIBLE"
        assert not a.acts, f"one breach quarantined a lane with {lw} windows produced"
        assert a.action is None


def test_k_CONSECUTIVE_breaches_quarantine():
    a = _assess(-30.0, windows_out=2, live_windows=10)
    assert a.acts and a.action is SelfAction.STAND_DOWN


def test_the_counter_is_CONSECUTIVE_breaches_not_windows_produced():
    """The two were one counter, which is how "must agree" came to mean "must exist"."""
    import inspect
    params = inspect.signature(assess).parameters
    assert "consecutive_breaches" in params, (
        "assess has no way to be told how many windows AGREED — `live_windows` is how many the lane "
        "has produced, which is a warmup question, not a confirmation one")
    assert params["consecutive_breaches"].kind is inspect.Parameter.KEYWORD_ONLY


def test_WARMUP_is_still_its_own_counter_and_still_answers_UNKNOWN():
    """Three states. A lane too new to judge is UNKNOWN — never 'fine', never 'quarantined'."""
    a = _assess(-30.0, windows_out=5, live_windows=1, sessions=4)
    assert a.state == UNKNOWN and not a.acts and a.action is None


# -- (b) STAND_DOWN only on the tail it is named for ------------------------------------------------

@pytest.mark.parametrize("value,tail", [(-30.0, "below"), (-4.6, "below")])
def test_the_BAD_tail_stands_the_lane_down(value, tail):
    a = _assess(value, windows_out=2, live_windows=10)
    assert a.action is SelfAction.STAND_DOWN, f"{value} ({tail}) did not quarantine"


@pytest.mark.parametrize("value", [6.5, 30.0])
def test_the_HIGH_tail_is_VISIBLE_but_never_reduces(value):
    """A lane beating its own envelope has not lost its edge. The live hypotheses are a sizing bug,
    a data error or a config change — all reasons to LOOK, none to stop trading.

    An action that reduces risk on a lane that is winning is the most expensive thing this mechanism
    could do, and it would look correct in every log line while it cost return.
    """
    a = _assess(value, windows_out=5, live_windows=10)
    assert a.state == OUT_OF_ENVELOPE, "outperformance must stay visible — it is a real anomaly"
    assert a.action is not SelfAction.STAND_DOWN, (
        f"a lane at {value:+.1f} (above its own 90th percentile) was told to stand down")
    assert not a.acts, "`acts` means the quarantine fires; outperformance must not set it"


def test_the_HIGH_tail_has_its_OWN_name_and_that_name_does_NOT_reduce():
    """The coordinator's ruling: it gets its own name and it is NOT an action that reduces anything.

    `if assessment.action:` is the call site that makes this dangerous — a truthy action read as
    "reduce something" is exactly #144, where one flag named for blocking performed a liquidation.
    So the enum itself has to say which members reduce, rather than every caller remembering.
    """
    a = _assess(30.0, windows_out=5, live_windows=10)
    assert a.action is not None, (
        "outperformance produced no action at all, so it is indistinguishable from a lane with "
        "too little evidence — an operator cannot tell 'look at this' from 'nothing yet'")
    assert isinstance(a.action, SelfAction)
    assert a.action.reduces is False, f"{a.action} reduces risk on a lane that is WINNING"
    assert SelfAction.STAND_DOWN.reduces is True


def test_a_lane_INSIDE_its_envelope_has_no_action_at_all():
    a = _assess(0.78, windows_out=0, live_windows=10)
    assert a.state == IN_ENVELOPE and a.action is None and not a.acts


# -- the false rate the band implies ----------------------------------------------------------------

def test_the_BAND_is_the_false_rate_and_k_is_the_only_lever():
    """Stated as an assertion so the arithmetic cannot drift from the docstring.

    A pre-registered (10, 90) band puts 10% of windows below the low edge BY CONSTRUCTION when the
    lane is behaving exactly as measured. k consecutive INDEPENDENT breaches is 0.1^k. Whether live
    windows are independent is the separate measurement this number depends on
    (`research/envelopes/window_independence.py`) — if they overlap, the true rate is far higher and
    k buys much less than this implies.
    """
    env = _env(min_windows_to_act=2)
    lo, _hi = env.bands["return_pct"]
    p_bad_tail = lo / 100.0
    assert p_bad_tail == 0.10
    assert round(p_bad_tail ** env.min_windows_to_act, 4) == 0.01


# -- the tripwire on a coincidence that currently holds ---------------------------------------------

def test_every_registered_envelope_has_BANDS_AT_ITS_EXTREME_POINTS():
    """TWO MECHANISMS, ONE DEAD, AGREEING TODAY BY COINCIDENCE OF CONSTRUCTION.

    `assess` has two ways to be outside: a percentile band (`pct < lo or pct > hi`) and being BEYOND
    the registered range of points. Measured on MOMENTUM-002:

        registered points: [(10, -4.55), (25, -2.71), (50, 0.78), (75, 3.26), (90, 6.38)]
        band: (10.0, 90.0)   distribution spans p10 .. p90
        in-range values whose percentile falls outside the band: 0

    The band edges ARE the extreme registered points, so nothing inside the range can sit outside
    the band and the percentile comparison never fires. Every real breach takes the range path. Two
    surviving mutation bites found this: mutating the band comparison killed nothing, twice.

    THE NUMBERS AGREE WHILE THE EDGES COINCIDE, which is why the false-rate ruling is unaffected —
    a (10,90) band and "beyond p10/p90" are the same 19.4% today. THE DAY SOMEONE REGISTERS p5/p95,
    OR NARROWS A BAND TO (20,80), THEY DIVERGE, and the behaviour every docstring promises still
    never happens — silently, on a mechanism that is by then armed.

    This test does not fix that. It makes the departure loud: change one without the other and this
    fails, naming the problem, instead of a dead branch quietly coming alive with semantics nobody
    chose. Same move as #181's AST test on the unplumbed callers — the deliberate state is fine, the
    SILENT departure from it is not.
    """
    from kumo_strategies.strategies.momentum_rotation.envelopes import REGISTERED

    for key, env in sorted(REGISTERED.items()):
        for metric, (lo, hi) in env.bands.items():
            points = env.distribution.get(metric)
            assert points, f"{key}/{metric}: banded with no distribution"
            assert (lo, hi) == (min(points), max(points)), (
                f"{key}/{metric}: band ({lo}, {hi}) no longer equals the extreme registered points "
                f"({min(points)}, {max(points)}).\n\n"
                f"That is not necessarily wrong — but it means the PERCENTILE BAND in `assess` "
                f"(`pct < lo or pct > hi`) is now reachable, and until today it never was: every "
                f"breach took the 'BEYOND the registered range' path instead. Two mutation bites "
                f"survived against that branch because it is dead code.\n\n"
                f"So decide which rule this envelope means and make `assess` say it, rather than "
                f"letting the dead branch come alive with semantics nobody chose. See the semantic "
                f"ticket linked from #182.")
