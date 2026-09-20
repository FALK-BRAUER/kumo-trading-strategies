"""A lane's PRE-REGISTERED envelope: what it said it would do, so "not doing it" is checkable (#147).

THE ANALYSIS THIS REPLACES WAS DONE BY HAND AND THROWN AWAY. From the TECHIVOL paper post-mortem:

    TECHIVOL's paper window sits at the 13th percentile of its own backtest's 13-session
    distribution, where 38% of all such windows are negative and p10 is -6.40% — live delivered
    exactly -6.4%. Its picks are fine. Its turnover is not.

That is a self-assessment. It stopped a lane being blamed for a selection failure it did not have,
and the next incident would have had to recompute it under pressure. The lane should EXPOSE it.

SO THE HOOK ANSWERS "WHERE DOES LIVE SIT IN MY OWN DISTRIBUTION?", NOT "AM I DOWN?" — and the
consequence that matters is that **being down is not out-of-envelope if the lane's own history says
38% of windows are negative.** A lane that cannot say that gets de-risked for behaving exactly as
measured.
"""

from __future__ import annotations

import dataclasses

import pytest

from kumo_strategies.strategies.envelope import (
    Envelope, IN_ENVELOPE, OUT_OF_ENVELOPE, StaleEnvelope, assess)
from kumo_strategies.strategies.market_view import UNKNOWN, SelfAction


def _envelope(**kw) -> Envelope:
    base = dict(
        config_fingerprint="qc27:abc123",
        horizon_sessions=13,
        samples=64,
        # metric -> (low, high) acceptable percentiles of the lane's OWN backtest distribution
        bands={"return_pct": (5.0, 95.0)},
        # the lane's own distribution, as percentile -> value
        distribution={"return_pct": {5: -9.1, 10: -6.4, 25: -2.0, 50: 1.8, 75: 6.2, 95: 14.0}},
        min_live_sessions=13,
        min_windows_to_act=3,
    )
    base.update(kw)
    return Envelope(**base)


# -- the point of the whole thing ---------------------------------------------------------------------

def test_being_DOWN_is_not_out_of_envelope_when_the_lane_said_it_would_be():
    """THE TECHIVOL CASE, and the reason this exists. −6.4% over 13 sessions is the 10th percentile
    of that lane's own distribution — inside a band whose floor is the 5th. A drawdown alarm would
    have fired; the envelope does not, because the lane pre-registered that this happens."""
    a = assess(_envelope(), {"return_pct": -6.4}, config_fingerprint="qc27:abc123",
               live_sessions=13, live_windows=3)
    assert a.state == IN_ENVELOPE, a.reasons
    assert a.acts is False
    assert any("percentile" in r for r in a.reasons), a.reasons


def test_a_lane_OUTSIDE_its_own_distribution_is_flagged():
    """FLAGGED on the first breach — and NOT acted on, which is the half this test used to have
    backwards (#147).

    It asserted `acts is True` at `live_windows=3` on a SINGLE bad window, and passed, because
    `min_windows_to_act` was compared against windows PRODUCED rather than windows that AGREED. The
    assertion was true of the code and false of the documented rule, which is the most convincing
    form a wrong test takes: green, with a reason beside it.
    """
    a = assess(_envelope(), {"return_pct": -30.0}, config_fingerprint="qc27:abc123",
               live_sessions=13, live_windows=3, consecutive_breaches=1)
    assert a.state == OUT_OF_ENVELOPE, "a single breach must still be VISIBLE"
    assert a.acts is False, "one window against a distribution is a percentile, not a verdict"
    assert a.breach == "below"


def test_the_lane_is_quarantined_once_the_breaches_CONFIRM():
    """The other side of the same rule: `min_windows_to_act=3` means three in a row."""
    env = _envelope()
    for k in range(1, env.min_windows_to_act):
        assert not assess(env, {"return_pct": -30.0}, config_fingerprint="qc27:abc123",
                          live_sessions=13, live_windows=9, consecutive_breaches=k).acts
    a = assess(env, {"return_pct": -30.0}, config_fingerprint="qc27:abc123",
               live_sessions=13, live_windows=9, consecutive_breaches=env.min_windows_to_act)
    assert a.acts is True and a.action is SelfAction.STAND_DOWN


# -- addition 1: versioned against the config it was fitted on ------------------------------------------

def test_an_envelope_fitted_on_a_DIFFERENT_CONFIG_REFUSES_rather_than_warns():
    """A lane whose sizing, universe, cadence or exit rule changed has NEW behaviour and an OLD
    distribution. Assessing one against the other is a knob agreeing with its configured value
    because nothing recomputed it — this package has shipped that shape repeatedly.

    REFUSE, not warn: an assessment against the wrong envelope is worse than no assessment, because
    it carries the authority of a number.
    """
    with pytest.raises(StaleEnvelope, match="fingerprint"):
        assess(_envelope(), {"return_pct": -6.4}, config_fingerprint="qc27:CHANGED",
               live_sessions=13, live_windows=3)


def test_the_refusal_names_BOTH_fingerprints():
    """"Stale envelope" is not actionable; which config it was fitted on, against which is running,
    is what tells an operator whether to refit or to revert."""
    try:
        assess(_envelope(), {"return_pct": 0.0}, config_fingerprint="qc27:CHANGED",
               live_sessions=13, live_windows=3)
    except StaleEnvelope as e:
        assert "qc27:abc123" in str(e) and "qc27:CHANGED" in str(e)


# -- addition 3: UNKNOWN, and it never acts ------------------------------------------------------------

def test_too_few_live_sessions_is_UNKNOWN_and_UNKNOWN_NEVER_ACTS():
    """Rule 4 of the market-view protocol, and it applies here for the same reason. This is the
    COMMON CASE for months on any newly-registered lane, not an edge."""
    a = assess(_envelope(), {"return_pct": -6.4}, config_fingerprint="qc27:abc123",
               live_sessions=4, live_windows=1)
    assert a.state == UNKNOWN
    assert a.acts is False
    assert a.reasons, "UNKNOWN carried no reason — #873 needs one for the UI"


def test_ASSESSMENT_HAS_THREE_STATES_NOT_A_BOOL():
    """`healthy: bool` cannot express "I do not know yet", so a lane with too little evidence would
    have to answer False and be treated as unhealthy — which is exactly the failure the market view
    avoids with its three states. I built this hook with two and the market view with three; the
    inconsistency is the defect."""
    fields = {f.name for f in dataclasses.fields(
        __import__("kumo_strategies.strategies.market_view", fromlist=["Assessment"]).Assessment)}
    assert "state" in fields, "Assessment still answers with a bool; UNKNOWN is not expressible"
    assert "healthy" not in fields, (
        "a `healthy` bool alongside `state` is two derivations of one fact, and they will disagree")


# -- addition 2: n of live evidence needed to act ------------------------------------------------------

def test_one_bad_window_is_not_a_VERDICT():
    """n=1 against n=many is a percentile, not evidence. The 13th percentile of a distribution where
    38% of windows are negative says nothing after one window — and the first bad month is precisely
    when a lane most needs to be left alone."""
    a = assess(_envelope(min_windows_to_act=3), {"return_pct": -30.0},
               config_fingerprint="qc27:abc123", live_sessions=13, live_windows=1)
    assert a.state == OUT_OF_ENVELOPE, "the observation stands"
    assert a.acts is False, "but one window is not enough to act on"
    assert any("1 of 3" in r or "windows" in r for r in a.reasons), a.reasons


def test_the_threshold_is_PRE_REGISTERED_not_chosen_at_assessment_time():
    """If the number of windows needed to act could be supplied by the caller, it would be chosen
    after seeing the result. It lives on the envelope, with the bands."""
    import inspect
    params = set(inspect.signature(assess).parameters)
    assert "min_windows_to_act" not in params, (
        "the acting threshold is a caller argument — it can be picked to suit the answer")


# -- the pre-registration rule --------------------------------------------------------------------------

def test_an_envelope_cannot_be_refitted_to_include_what_just_happened():
    """A band that moves to contain the observation cannot fail. The envelope is frozen, so a caller
    that wants a wider band must REGISTER a new one — which changes the fingerprint and is therefore
    visible."""
    e = _envelope()
    assert dataclasses.is_dataclass(e) and e.__dataclass_params__.frozen, (
        "the envelope is mutable, so a band can be widened in place to swallow a bad window")


def test_an_envelope_must_say_how_many_samples_its_distribution_has():
    """A percentile from 4 windows and one from 64 are different claims wearing the same word."""
    with pytest.raises(ValueError, match="samples"):
        _envelope(samples=0)
