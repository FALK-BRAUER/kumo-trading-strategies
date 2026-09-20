"""The shared exit evaluator. Each test pins a rule or a failure that cost money.

Several are the 4-6 Aug 2026 live shapes replayed as fixtures (kumo-trading-platform issue 197), because those are
the cases the two previous copies of this logic disagreed about.
"""

from __future__ import annotations


import pytest

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import (
    ADOPTED, LIVE, RECONSTRUCTED, TrailState, evaluate_exits)


def s(entry, peak=None, **kw) -> TrailState:
    return TrailState(entry_px=entry, peak_px=peak if peak is not None else entry, **kw)


# -- nothing configured --------------------------------------------------------------------------
def test_no_rules_means_no_exits():
    p = evaluate_exits(ExitConfig(), {"A": 50.0}, {"A": s(100.0)})
    assert p.exits == {}


def test_state_still_advances_when_no_rule_is_configured():
    """The trail has to keep tracking even when nothing acts on it, or enabling a rule later starts
    from a peak that ignores everything before it."""
    p = evaluate_exits(ExitConfig(), {"A": 120.0}, {"A": s(100.0)})
    assert p.state["A"].peak_px == 120.0
    assert p.state["A"].sessions_held == 1


# -- give-back ------------------------------------------------------------------------------------
def test_give_back_fires_after_surrendering_half_the_run():
    st = s(100.0, peak=120.0)
    p = evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 109.0}, {"A": st})
    assert "gave back" in p.exits["A"]


def test_give_back_holds_while_the_run_is_intact():
    st = s(100.0, peak=120.0)
    assert evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 111.0}, {"A": st}).exits == {}


def test_give_back_cannot_arm_on_a_position_that_never_rose():
    """SU, HSBC and PAA on 4-6 Aug: never traded above entry, so peak == entry and peak_gain == 0.
    give_back is a profit-protection rule and structurally cannot cut these — which is why the loss
    ran unbounded and why a correctly-seeded replay came out WORSE than what actually happened."""
    st = s(107.69)                                   # HSBC: entry == peak
    p = evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 102.91}, {"A": st})
    assert p.exits == {}, "give_back must not claim to protect a position that never rose"


def test_a_position_below_entry_reports_the_loss_not_a_nonsense_ratio():
    """SU's real state on 6 Aug was entry 64.75 / peak 64.79, and the journal recorded the exit as
    "gave back 30825% of a 0.1% peak" — which told the operator nothing except that something was
    wrong. Past 100% the useful fact is the loss, not the ratio."""
    st = s(64.75, peak=64.79)
    reason = evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 61.20}, {"A": st}).exits["A"]
    assert "below entry" in reason
    assert "5.5% below entry" in reason, reason
    assert "30825" not in reason and "8975" not in reason


def test_a_normal_give_back_still_reports_the_fraction():
    """The percentage framing is right while the position is still above entry."""
    st = s(100.0, peak=120.0)
    reason = evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 105.0}, {"A": st}).exits["A"]
    assert "gave back 75% of a 20.0% peak (trail 50%)" == reason, reason


# -- off-peak / time rules --------------------------------------------------------------------------
def test_off_peak_cuts_a_position_that_never_rose():
    """The one rule that CAN act on the SU/HSBC/PAA shape, because it measures distance from peak
    rather than surrendered profit — and when a position never rose, peak == entry."""
    p = evaluate_exits(ExitConfig(off_peak_pct=0.05), {"A": 94.0}, {"A": s(100.0)})
    assert "below its peak" in p.exits["A"]


def test_max_hold_counts_sessions_not_calendar_days():
    st = s(100.0, sessions_held=14)
    assert evaluate_exits(ExitConfig(max_hold_days=15), {"A": 100.0}, {"A": st}).exits
    st2 = s(100.0, sessions_held=13)
    assert not evaluate_exits(ExitConfig(max_hold_days=15), {"A": 100.0}, {"A": st2}).exits


def test_stall_counts_sessions_since_the_last_high():
    st = s(100.0, peak=110.0, sessions_since_high=4)
    assert evaluate_exits(ExitConfig(stall_days=5), {"A": 105.0}, {"A": st}).exits


def test_a_new_high_resets_the_stall_counter():
    st = s(100.0, peak=110.0, sessions_since_high=4)
    p = evaluate_exits(ExitConfig(stall_days=5), {"A": 115.0}, {"A": st})
    assert p.exits == {}
    assert p.state["A"].sessions_since_high == 0
    assert p.state["A"].peak_px == 115.0


# -- adopted state ---------------------------------------------------------------------------------
def test_peak_relative_rules_are_skipped_while_the_peak_is_unknown():
    """#197 B1. The old code seeded `entry, peak = (px, px)` for any position with no row, asserting
    a peak that never happened. Skipping is the honest alternative to inventing one."""
    st = s(100.0, quality=ADOPTED)
    cfg = ExitConfig(give_back_frac=0.5, off_peak_pct=0.05)
    assert evaluate_exits(cfg, {"A": 80.0}, {"A": st}).exits == {}


def test_time_rules_still_apply_to_an_adopted_position():
    """Only the PEAK is untrustworthy. Time held is knowable from the fill, so a time rule is safe —
    and skipping every rule would leave an adopted position uncoverable forever."""
    st = s(100.0, sessions_held=20, quality=ADOPTED)
    assert evaluate_exits(ExitConfig(max_hold_days=15), {"A": 80.0}, {"A": st}).exits


def test_a_new_high_promotes_an_adopted_position_to_trustworthy():
    """The peak becomes real the moment it is observed, so coverage is regained without guessing."""
    st = s(100.0, peak=100.0, quality=ADOPTED)
    p = evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 130.0}, {"A": st})
    assert p.state["A"].quality == LIVE
    assert p.state["A"].peak_px == 130.0


def test_reconstructed_state_is_trusted():
    st = s(100.0, peak=120.0, quality=RECONSTRUCTED)
    assert evaluate_exits(ExitConfig(give_back_frac=0.5), {"A": 105.0}, {"A": st}).exits


# -- missing / bad prices ---------------------------------------------------------------------------
@pytest.mark.parametrize("px", [None, float("nan"), 0.0, -1.0])
def test_a_position_with_no_usable_price_is_left_entirely_alone(px):
    """Advancing sessions_held on a day we could not see the price would let a data outage age a
    position out of the book."""
    st = s(100.0, sessions_held=5)
    prices = {} if px is None else {"A": px}
    p = evaluate_exits(ExitConfig(max_hold_days=6), prices, {"A": st})
    assert p.exits == {}
    assert p.state["A"].sessions_held == 5, "state must not advance on an unusable price"


# -- rule precedence ---------------------------------------------------------------------------------
def test_first_matching_rule_wins_and_the_reason_is_specific():
    """A position out of time should not also be reported as a give-back; the operator needs to know
    which rule acted."""
    st = s(100.0, peak=200.0, sessions_held=30)
    cfg = ExitConfig(max_hold_days=15, give_back_frac=0.5)
    reason = evaluate_exits(cfg, {"A": 110.0}, {"A": st}).exits["A"]
    assert "held" in reason and "gave back" not in reason
