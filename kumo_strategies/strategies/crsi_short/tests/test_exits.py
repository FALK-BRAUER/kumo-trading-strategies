"""CRSISHORT's covers: the flat, the reversal, and what "acted at the next open" means.

Between them these are 96% of #123's exits (flat 48%, reversal 48%, no-data 4%), so the session a
cover fires in and the price it fires at decide the trade set, not the presentation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.exits import (
    FLAT, REVERSAL, WEAKNESS, SessionBar, evaluate_short_exits, open_state)

CFG = CrsiShortConfig()


def _bar(o: float, h: float, lo: float, c: float) -> dict[str, SessionBar]:
    return {"AAA": SessionBar(open=o, high=h, low=lo, close=c)}


def _held(**kw):
    st = open_state(100.0, SessionBar(open=100.0, high=102.0, low=98.0, close=99.0))
    return {"AAA": replace(st, **kw)}


# --- flagged covers act at the NEXT open ------------------------------------------------------

def test_a_close_above_the_prior_high_only_FLAGS_it():
    """The signal is a close, so the earliest tradable moment is the next open. Covering at the
    close it was observed at needs a market-on-close order this book does not send."""
    plan = evaluate_short_exits(CFG, _bar(103.0, 106.0, 102.0, 105.0), _held(prev_high=104.0))
    assert plan.exits == {}
    assert plan.state["AAA"].flag_reversal


def test_the_flagged_cover_fires_at_the_next_open():
    plan = evaluate_short_exits(CFG, _bar(107.0, 108.0, 106.0, 107.5),
                                _held(flag_reversal=True, prev_high=104.0))
    assert plan.kind["AAA"] == REVERSAL
    assert plan.exit_px["AAA"] == 107.0
    assert plan.state["AAA"].sessions_held == 0        # a covered position does not also age


def test_a_close_below_the_prior_high_flags_nothing():
    plan = evaluate_short_exits(CFG, _bar(103.0, 104.0, 102.0, 103.5), _held(prev_high=104.0))
    assert not plan.state["AAA"].flag_reversal
    assert plan.exits == {}


def test_the_weakness_cover_is_off_in_the_frozen_spec_and_can_be_turned_on():
    """It covers a short that is WINNING. Off by default; the lab measures all four exit sets."""
    bar, st = _bar(95.0, 96.0, 90.0, 91.0), _held(prev_low=94.0)
    assert not evaluate_short_exits(CFG, bar, st).state["AAA"].flag_weakness
    on = evaluate_short_exits(replace(CFG, weakness_exit=True), bar, st)
    assert on.state["AAA"].flag_weakness
    fired = evaluate_short_exits(replace(CFG, weakness_exit=True), _bar(89.0, 90.0, 88.0, 89.5),
                                 _held(flag_weakness=True))
    assert fired.kind["AAA"] == WEAKNESS


# --- the flat cover is intraday ----------------------------------------------------------------

def test_the_flat_cover_fills_at_entry_when_the_tape_reaches_it():
    """A resting buy at the entry price. It is detected by the session HIGH and it FILLS AT ENTRY —
    booking it at the high would report a loss the rule never took."""
    plan = evaluate_short_exits(CFG, _bar(96.0, 101.0, 95.0, 99.0), _held(best_px=90.0))
    assert plan.kind["AAA"] == FLAT
    assert plan.exit_px["AAA"] == 100.0


def test_a_session_that_GAPS_through_the_trail_fills_at_the_open():
    """The exposure #123 names: "latest at flat" cannot protect against an overnight squeeze, and
    the worst observed trade is already -53%. A fill at entry here would be fiction."""
    plan = evaluate_short_exits(CFG, _bar(130.0, 140.0, 128.0, 135.0), _held(best_px=90.0))
    assert plan.exit_px["AAA"] == 130.0
    assert "gapped" in plan.exits["AAA"]


def test_the_flat_cover_needs_the_position_to_have_BEEN_in_profit():
    """A short from 100 that went straight to 130 has given nothing back."""
    plan = evaluate_short_exits(CFG, _bar(125.0, 130.0, 124.0, 129.0), _held())
    assert plan.exits == {}


def test_THIS_sessions_low_cannot_arm_THIS_sessions_flat_cover():
    """The trail is evaluated as of the PRIOR close. A name that first trades below entry and then
    back through it inside one session has not yet armed the rule — arming it here would let a
    single intraday round trip both open and close the profit that triggers the cover."""
    plan = evaluate_short_exits(CFG, _bar(99.0, 105.0, 90.0, 104.0), _held())
    assert plan.exits == {}
    assert plan.state["AAA"].best_px == 90.0           # armed for NEXT session


def test_a_partial_give_back_fires_at_the_trail_not_at_entry():
    """`give_back_frac = 0.5` on a short entered at 100 that reached 80: the trail sits at 90."""
    cfg = replace(CFG, give_back_frac=0.5)
    st = _held(best_px=80.0)
    assert evaluate_short_exits(cfg, _bar(85.0, 89.0, 84.0, 88.0), st).exits == {}
    fired = evaluate_short_exits(cfg, _bar(85.0, 91.0, 84.0, 88.0), st)
    assert fired.exit_px["AAA"] == pytest.approx(90.0)


def test_flat_is_taken_ahead_of_a_flagged_cover_when_both_apply():
    """Order is the semantics: the resting buy filled intraday, before any open-priced cover could
    have been worked. Labelling the session a reversal would misreport the exit mix."""
    plan = evaluate_short_exits(CFG, _bar(99.0, 101.0, 98.0, 100.5),
                                _held(best_px=90.0, flag_reversal=True))
    assert plan.kind["AAA"] == FLAT


def test_every_exit_can_be_turned_off():
    """One of #123's four mutations: disable the exits and the result must move."""
    cfg = replace(CFG, give_back_frac=None, reversal_exit=False, weakness_exit=False)
    plan = evaluate_short_exits(cfg, _bar(101.0, 110.0, 99.0, 109.0),
                                _held(best_px=90.0, flag_reversal=True))
    assert plan.exits == {}


# --- state ------------------------------------------------------------------------------------

def test_a_session_we_could_not_price_does_not_age_the_position():
    """#123's no-data exits are already 4% of the sample. Whether to close such a position is the
    driver's call; a data outage silently ageing it through a holding rule is nobody's."""
    before = _held()
    plan = evaluate_short_exits(CFG, {}, before)
    assert plan.state["AAA"] == before["AAA"]
    plan = evaluate_short_exits(CFG, _bar(0.0, float("nan"), 0.0, 0.0), before)
    assert plan.state["AAA"] == before["AAA"]


def test_a_position_cannot_be_flagged_out_on_the_session_it_opened_in():
    """`open_state` seeds `prev_low`/`prev_high` from the FILL session, so the first flag can only
    be set at the next close — and `best_px` starts at entry, so it is not in profit yet however
    the fill session traded."""
    st = open_state(100.0, SessionBar(open=100.0, high=120.0, low=80.0, close=118.0))
    assert not st.flag_reversal and not st.ever_in_profit
    assert (st.prev_low, st.prev_high) == (80.0, 120.0)


def test_state_advances_and_is_returned_not_mutated():
    before = _held()
    plan = evaluate_short_exits(CFG, _bar(99.0, 100.0, 95.0, 96.0), before)
    assert before["AAA"].sessions_held == 0
    st = plan.state["AAA"]
    assert (st.sessions_held, st.best_px, st.prev_low, st.prev_high) == (1, 95.0, 95.0, 100.0)
    assert st.ever_in_profit
