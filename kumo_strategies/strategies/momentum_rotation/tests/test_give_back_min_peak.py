"""Variant D — give-back must not arm on a peak that is only noise (#24 D, #30).

`give_back_frac` is scale-free: it treats a 0.7% peak and a 28% peak identically. Live evidence that
this is wrong, from every give-back exit MOMENTUM-002 has ever fired:

    08-06  SU    0.1% peak   -$554   (compounded by #197 B1)
    08-10  PRU   0.7% peak   -$126   NOT a bug — the rule working exactly as written
    08-13  AEM   5.3% peak           legitimate, kept falling after sale
    08-13  FSM   4.1% peak           legitimate
    08-13  WPM   2.8% peak           legitimate
    08-13  CGAU  4.6% peak           legitimate

Two of six armed on a peak smaller than one average session's range. The fixtures below are those
real cases, so the test states which live trades the guard would and would not have changed.

The band matters as much as the direction: set too high, the guard disables the 08-13 exits, which
were correct — all five names kept falling after sale.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState, evaluate_exits


def _held(entry: float, peak_pct: float) -> TrailState:
    return TrailState(entry_px=entry, peak_px=entry * (1 + peak_pct), quality=LIVE)


# entry, peak %, ATR as a fraction of price — ATR ~2% is typical for these names
PRU = ("PRU", 122.63, 0.007, 0.020)      # 0.7% peak: a third of one average session
AEM = ("AEM", 176.04, 0.053, 0.020)      # 5.3% peak: two and a half sessions


def _case(sym, entry, peak_pct, atr_pct, now_pct, cfg):
    st = _held(entry, peak_pct)
    plan = evaluate_exits(cfg, {sym: entry * (1 + now_pct)}, {sym: st},
                          atr={sym: entry * atr_pct})
    return plan.exits


def test_the_guard_off_reproduces_todays_behaviour():
    """Default is None. Every recorded result must be unchanged."""
    sym, entry, peak, atr = PRU
    cfg = ExitConfig(give_back_frac=0.5)
    st = _held(entry, peak)
    plan = evaluate_exits(cfg, {sym: entry * 1.001}, {sym: st})   # gave back most of 0.7%
    assert sym in plan.exits, "give-back should still fire with the guard off"


def test_pru_would_not_have_been_sold():
    """The live case the guard exists for. A 0.7% peak against a 2% ATR is 0.35 sessions."""
    sym, entry, peak, atr = PRU
    cfg = ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=1.0)
    assert _case(sym, entry, peak, atr, -0.015, cfg) == {}, (
        "give-back armed on a peak smaller than one average session — that is the PRU trade")


def test_aem_would_still_have_been_sold():
    """The other side of the band. The 08-13 exits were CORRECT — all five kept falling."""
    sym, entry, peak, atr = AEM
    cfg = ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=1.0)
    assert sym in _case(sym, entry, peak, atr, 0.020, cfg), (
        "the guard disabled a legitimate exit; the threshold is too high")


def test_a_threshold_high_enough_disables_the_legitimate_exits_too():
    """States the failure mode explicitly so the sweep has to find the band rather than assume one.

    At 3 ATR even AEM's 5.3% peak (2.65 sessions) is below the floor. A sweep that only reports
    return would show this as 'give-back off' without saying so.
    """
    sym, entry, peak, atr = AEM
    cfg = ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=3.0)
    assert _case(sym, entry, peak, atr, 0.020, cfg) == {}


def test_missing_atr_does_not_arm():
    """Conservative, and consistent with how an ADOPTED peak is treated: when the SCALE of a move
    cannot be established, leave the position alone rather than act on a number of unknown meaning.

    Arming instead would make the guard silently optional per symbol — a rule that looks armed while
    being inert, which is the #26 shape.
    """
    sym, entry, peak, _ = AEM
    cfg = ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=1.0)
    plan = evaluate_exits(cfg, {sym: entry * 1.02}, {sym: _held(entry, peak)}, atr={})
    assert plan.exits == {}
    plan = evaluate_exits(cfg, {sym: entry * 1.02}, {sym: _held(entry, peak)},
                          atr={sym: float("nan")})
    assert plan.exits == {}


def test_the_config_without_the_input_raises_rather_than_degrading():
    """The #26 failure mode, refused rather than documented. Setting the guard and forgetting to
    pass ATR would otherwise arm give-back exactly as before — the config would be inert."""
    cfg = ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=1.0)
    with pytest.raises(ValueError, match="no `atr` was passed"):
        evaluate_exits(cfg, {"AEM": 180.0}, {"AEM": _held(176.04, 0.053)})


def test_the_guard_does_not_touch_the_other_rules():
    """It gates give-back only. A time stop or a stall must still fire on a small-peak position —
    they are not peak-relative and have nothing to do with the scale of the run."""
    sym, entry, peak, atr = PRU
    cfg = ExitConfig(max_hold_days=1, give_back_frac=0.5, give_back_min_peak_atr=1.0)
    got = _case(sym, entry, peak, atr, -0.015, cfg)
    assert sym in got and "sessions" in got[sym]


# -- PEAK's persistence principle, ported (#30, kumo-trading-platform #46) ---------------------------------

def _run(cfg, prices, st):
    """Feed prices session by session, carrying state forward as a driver would."""
    fired = []
    state = {"AAA": st}
    for px in prices:
        plan = evaluate_exits(cfg, {"AAA": px}, state)
        state = plan.state
        fired.append("AAA" in plan.exits)
    return fired


def test_a_single_breach_does_not_fire_when_confirmation_is_required():
    """PEAK's exact lesson: one red bar off a fresh high is not an exit."""
    st = _held(100.0, 0.10)                     # peak 110, entry 100
    cfg = ExitConfig(give_back_frac=0.5, give_back_confirm_sessions=2)
    # 104 breaches (kept 40% of a 10% peak, below the 50% floor); 109 lifts it.
    assert _run(cfg, [104.0], st) == [False], "fired on a single observation"


def test_two_consecutive_breaches_do_fire():
    st = _held(100.0, 0.10)
    cfg = ExitConfig(give_back_frac=0.5, give_back_confirm_sessions=2)
    assert _run(cfg, [104.0, 104.0], st) == [False, True]


def test_a_wiggle_resets_the_counter_rather_than_accumulating():
    """The anti-noise property. Breach, recover, breach must NOT count as two consecutive.

    Without the reset a name that dips below the line every other session would accumulate its way
    to an exit it never actually sustained — which is the failure PEAK's 2-bar rule prevents.
    """
    st = _held(100.0, 0.10)
    cfg = ExitConfig(give_back_frac=0.5, give_back_confirm_sessions=2)
    assert _run(cfg, [104.0, 109.0, 104.0], st) == [False, False, False]


def test_confirmation_of_one_reproduces_todays_behaviour():
    """None and 1 must both be the single-observation rule, so every recorded result is unchanged."""
    st = _held(100.0, 0.10)
    for n in (None, 1):
        cfg = ExitConfig(give_back_frac=0.5, give_back_confirm_sessions=n)
        assert _run(cfg, [104.0], st) == [True], f"confirm={n} changed the default behaviour"
