"""The promoted config is what the research measured, and the caveats say the hard parts out loud."""

from __future__ import annotations

from kumo_strategies.strategies.smhgld_sleeve import (
    SmhGldSleeveConfig, live_config, live_notes)


def test_live_config_is_the_researched_form():
    assert live_config() == SmhGldSleeveConfig()


def test_the_risk_leg_is_the_DECIDED_weight_and_the_notes_say_who_decided_and_what_it_costs():
    """0.48 is a DECISION (2026-09-13, #219), not a measurement's optimum — the ulcer-index
    measurement still prefers 32%. A config that carries 0.48 without a note naming the decision
    and its measured cost is a number nobody can defend at the next review."""
    assert live_config().risk_weight == 0.48
    note = live_notes()["the_weight_was_32_and_the_measurement_behind_that_still_holds"]
    assert "#219" in note and "decision" in note.lower()
    for cost in ("-27.6%", "-16.3%", "-37.3%"):
        assert cost in note, f"the measured cost {cost} of the decision is not stated"


def test_the_weights_sum_to_the_whole_sleeve():
    weights = live_config().target_weights
    assert sum(weights.values()) == 1.0
    assert set(weights) == set(live_config().universe)


def test_the_caveats_name_the_window_where_the_lane_fails():
    """An operator who reads only the notes must still learn that the headline drawdown comes from
    a window with no bear market in it. A caveat list that omits the failure case is marketing."""
    notes = live_notes()
    stress = notes["2022_IS_THE_STRESS_CASE_AND_THE_SLEEVE_FAILS_IT"]

    assert "2022" in stress
    assert "-16.3%" in stress, "the actual 2022 loss at the decided 48% must be stated, not characterised (#219)"
    assert "no bear market" in stress


def test_the_caveats_refuse_to_call_gold_a_hedge():
    """The lane exists because eleven timing mechanisms failed. If the notes ever start describing
    the defensive leg as protection, the reason for the fixed weights has been forgotten."""
    notes = live_notes()
    assert "gold_is_not_a_hedge_here" in notes
    assert "eleven_timing_mechanisms_failed_on_this_pair" in notes


def test_the_band_is_tight_enough_to_actually_rebalance():
    """A band this lane never crosses is a fixed weight that silently became a drifting one.

    The first promoted value was 5 points, then 3; both traded 3-9 times a year and were fairly
    called inert. The sweep behind `drift_band` shows the tightest band tested is best on drawdown,
    ulcer, 2022 and return at once, so a future widening is far more likely to be someone
    economising on trading costs than a measured improvement.
    """
    assert live_config().drift_band <= 0.0025


def test_the_notes_state_the_execution_contract_in_capitals():
    """A runner author who reads only the notes must learn that enter/exit will execute nothing
    here. That is the single most expensive thing to discover after arming rather than before."""
    notes = live_notes()
    assert "THIS_LANE_TRADES_BY_TARGET_AND_DELTA_NOT_BY_ENTRY_AND_EXIT" in notes
    assert "the_four_partial_sell_invariants" in notes
    assert "housekeeping_this_lane_needs" in notes


def test_the_notes_link_the_split_ledger_defect():
    """SMH split 2:1 on 2023-05-05, inside this lane's own measured window, so a future split is
    not hypothetical for it. Until issue 179 lands, a split day under-claims and the
    lane silently rebalances at half size."""
    note = live_notes()["A_SPLIT_IN_EITHER_LEG_SILENTLY_HALVES_WHAT_THIS_LANE_MAY_TRADE"]
    assert "#179" in note
    assert "2023-05-05" in note
