"""PEAK ported to a session clock (#30, kumo-trading-platform #46).

The gate fires on EITHER of two confirmed conditions — never a single observation:

    `lower_high_bars` consecutive bars each making a lower high, OR
    price sustained `off_hod_pct`+ below the high for 2+ consecutive bars

Its own ticket says why: "a single red bar off a fresh HoD is NOT an exit — the -1.5% wiggle
resumed". `off_peak_pct` in this module is the first half with the second half missing.

The two branches are OR because they catch different shapes. Tests below pin each branch firing
alone, so a refactor that quietly ANDs them would fail rather than merely returning fewer exits.
"""

from __future__ import annotations

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import (
    LIVE, TrailState, evaluate_exits, needs_highs)


def _run(cfg, bars, entry=100.0, peak=110.0):
    """Feed (close, high) sessions in order, carrying state as a driver would."""
    st = {"AAA": TrailState(entry_px=entry, peak_px=peak, quality=LIVE)}
    fired = []
    for close, high in bars:
        plan = evaluate_exits(cfg, {"AAA": close}, st, highs={"AAA": high})
        st = plan.state
        fired.append(plan.exits.get("AAA"))
    return fired


def test_a_single_close_below_the_threshold_does_not_fire():
    """The PENG lesson. One red bar off a fresh high is not an exit."""
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2)
    assert _run(cfg, [(104.0, 105.0)]) == [None]


def test_two_consecutive_sessions_below_the_threshold_do_fire():
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2)
    got = _run(cfg, [(104.0, 105.0), (103.0, 104.0)])
    assert got[0] is None and got[1] is not None
    assert "below its" in got[1] and "peak for" in got[1]


def test_a_recovery_resets_the_off_peak_run():
    """Breach, recover, breach must not accumulate — that is the whole anti-noise property."""
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2)
    assert _run(cfg, [(104.0, 105.0), (109.0, 110.0), (104.0, 105.0)]) == [None, None, None]


def test_the_lower_high_branch_fires_on_its_own():
    """Three consecutive lower HIGHS, while price never sits far below the peak. The threshold
    branch would never trigger here — this is the shape it misses."""
    cfg = ExitConfig(peak_fade_lower_highs=3)
    got = _run(cfg, [(109.0, 109.5), (108.8, 109.0), (108.6, 108.5), (108.5, 108.0)])
    assert got[-1] is not None and "lower highs" in got[-1]


def test_a_higher_high_resets_the_lower_high_run():
    cfg = ExitConfig(peak_fade_lower_highs=3)
    got = _run(cfg, [(109.0, 109.5), (108.8, 109.0), (109.2, 110.0), (108.6, 109.5)])
    assert all(g is None for g in got), f"a higher high did not reset the run: {got}"


def test_the_branches_are_OR_not_AND():
    """Either alone must be sufficient. ANDing them would miss both shapes it exists to catch."""
    only_threshold = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2,
                                peak_fade_lower_highs=99)
    assert _run(only_threshold, [(104.0, 105.0), (103.0, 106.0)])[-1] is not None

    only_sequence = ExitConfig(peak_fade_off_pct=0.99, peak_fade_confirm=2,
                               peak_fade_lower_highs=2)
    assert _run(only_sequence, [(109.0, 109.5), (108.8, 109.0), (108.6, 108.5)])[-1] is not None


def test_highs_fall_back_to_closes_and_degrade_SAFELY():
    """With no highs supplied the branch uses closes, which is a weaker signal — a name can close
    green on a lower high. It must therefore fire LESS often, never on something spurious."""
    cfg = ExitConfig(peak_fade_lower_highs=2)
    st = {"AAA": TrailState(entry_px=100.0, peak_px=110.0, quality=LIVE)}
    for close in (109.0, 108.0, 107.0):
        plan = evaluate_exits(cfg, {"AAA": close}, st)      # no highs
        st = plan.state
    assert plan.exits.get("AAA") is not None, "closes should still form a descending sequence"


def test_an_adopted_position_is_skipped():
    """No trustworthy peak means no peak-relative rule, same as everywhere else (#197 B1)."""
    from kumo_strategies.strategies.momentum_rotation.exits import ADOPTED
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2)
    st = {"AAA": TrailState(entry_px=100.0, peak_px=110.0, quality=ADOPTED)}
    for close in (104.0, 103.0):
        plan = evaluate_exits(cfg, {"AAA": close}, st, highs={"AAA": close})
        st = plan.state
    assert plan.exits == {}


def test_disabled_by_default():
    assert _run(ExitConfig(), [(80.0, 81.0), (79.0, 80.0)]) == [None, None]


def test_the_threshold_measures_against_the_HIGH_since_entry_not_the_close():
    """2026-08-16: "it needs to be high since entry. session high could be higher".

    `peak_px` tracks the highest CLOSE — right for give-back, since that is profit that could
    actually have been closed. Wrong for the fade threshold: PEAK compares against the high of day,
    and a session high routinely exceeds every close, so a close-based peak puts the line too low
    and the rule fires late or never.

    Here the high reaches 120 while no close exceeds 110. Against the close-peak, 106 is only 3.6%
    down and nothing fires. Against the true 120 high it is 11.7% down and the rule triggers.
    """
    cfg = ExitConfig(peak_fade_off_pct=0.10, peak_fade_confirm=2)
    got = _run(cfg, [(110.0, 120.0), (106.0, 112.0), (106.0, 107.0)], entry=100.0, peak=110.0)
    assert got[-1] is not None, "measured against the close-peak and missed an 11.7% fade"
    assert "120.00" in got[-1], f"reason should name the true high peak: {got[-1]}"


def test_the_high_peak_only_rises():
    """A lower subsequent high must not lower the reference — the peak is a running maximum."""
    cfg = ExitConfig(peak_fade_off_pct=0.10, peak_fade_confirm=2)
    got = _run(cfg, [(110.0, 120.0), (100.0, 101.0), (100.0, 100.5)], entry=100.0, peak=110.0)
    assert got[-1] is not None and "120.00" in got[-1]


def test_a_position_with_no_recorded_high_peak_falls_back_to_the_close_peak():
    """Carried positions and reconstructed trails have a close-peak and no high-peak.

    Starting the high-peak from zero would reset the reference to today's high, dropping the
    threshold far below where it belongs and disabling the rule exactly on the positions that have
    been held longest. Two existing tests caught this when the fallback was missing.
    """
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=2)
    st = {"AAA": TrailState(entry_px=100.0, peak_px=110.0, quality=LIVE)}   # no peak_high_px
    out = []
    for close, high in ((104.0, 105.0), (103.0, 104.0)):
        plan = evaluate_exits(cfg, {"AAA": close}, st, highs={"AAA": high})
        st = plan.state
        out.append(plan.exits.get("AAA"))
    assert out[-1] is not None, "fell back to nothing instead of the close peak"


def test_needs_highs_covers_BOTH_branches_not_just_lower_highs():
    """The threshold branch reads the high too, so it must be able to ask for one.

    This returned False for a threshold-only config, so runners computed no highs and the branch
    measured against closes — the reference this rule was explicitly corrected away from. Because a
    close is weakly below the high, the effect was one-directional: the rule fired strictly less
    often than configured, and at 8% and 10% it stopped firing at all.
    """
    assert needs_highs(ExitConfig(peak_fade_lower_highs=3))
    assert needs_highs(ExitConfig(peak_fade_off_pct=0.08)), "threshold branch cannot see highs"
    assert not needs_highs(ExitConfig(give_back_frac=0.5))


def test_supplying_highs_makes_the_threshold_branch_fire_where_closes_did_not():
    """The behavioural consequence, not just the predicate: same bars, same config, both answers.

    Guards the direction of the bug. With highs the peak is 120 and a 108 close is 10% below it;
    with closes only, the peak is 110 and 108 is barely off it, so nothing fires.
    """
    cfg = ExitConfig(peak_fade_off_pct=0.08, peak_fade_confirm=2)
    bars = [(110.0, 120.0), (108.0, 112.0), (108.0, 110.0)]

    def _go(with_highs: bool):
        st, last = {"AAA": TrailState(entry_px=100.0, peak_px=100.0, quality=LIVE)}, None
        for close, high in bars:
            plan = evaluate_exits(cfg, {"AAA": close}, st,
                                  highs={"AAA": high} if with_highs else None)
            st, last = plan.state, plan.exits.get("AAA")
        return last

    assert _go(with_highs=True) is not None, "the rule must fire when the real peak is visible"
    assert _go(with_highs=False) is None, "without highs it silently does not — that was the bug"


# --- stop_loss_atr: the floor, and the only rule that does not need a peak (#30) -----------------

def _stop(cfg, closes, entry=100.0, atr=5.0, quality=LIVE, peak=None):
    st = {"AAA": TrailState(entry_px=entry, peak_px=peak if peak is not None else entry,
                            quality=quality)}
    out = []
    for px in closes:
        plan = evaluate_exits(cfg, {"AAA": px}, st, atr={"AAA": atr})
        st = plan.state
        out.append(plan.exits.get("AAA"))
    return out


def test_the_stop_fires_at_the_configured_multiple_of_ATR_below_entry():
    cfg = ExitConfig(stop_loss_atr=2.0)
    got = _stop(cfg, [95.0, 89.9])          # ATR 5: 1.0 ATR down, then 2.02 ATR down
    assert got[0] is None, "1 ATR is not 2 ATR"
    assert got[1] is not None and "stopped out" in got[1]


def test_the_stop_measures_from_ENTRY_not_from_the_peak():
    """The distinguishing property. A position up 4 ATR and back to entry has lost nothing from
    entry, so the stop must stay silent — that fade is give-back's job, not the floor's."""
    cfg = ExitConfig(stop_loss_atr=2.0)
    assert _stop(cfg, [100.0], peak=120.0) == [None]


def test_the_stop_covers_a_position_that_never_went_up():
    """The gap it exists to close. Every other rule here is peak-relative and needs the position to
    have risen first; a name that falls from entry arms none of them."""
    falling = ExitConfig(stop_loss_atr=2.0)
    assert _stop(falling, [98.0, 94.0, 89.0])[-1] is not None
    # Same path, the live ruleset without a stop: nothing fires at all.
    assert _stop(ExitConfig(give_back_frac=0.5), [98.0, 94.0, 89.0]) == [None, None, None]


def test_the_stop_protects_an_ADOPTED_position_where_the_peak_rules_decline_to():
    """An adopted position has no trustworthy peak, so the peak-relative rules are skipped — it runs
    with almost no coverage. Entry price IS known (#197 B1 takes the broker's real entry), so this
    rule works where the others must not."""
    from kumo_strategies.strategies.momentum_rotation.exits import ADOPTED
    assert _stop(ExitConfig(stop_loss_atr=2.0), [88.0], quality=ADOPTED)[0] is not None
    assert _stop(ExitConfig(give_back_frac=0.5), [88.0], quality=ADOPTED)[0] is None


def test_a_symbol_with_no_usable_ATR_does_not_stop():
    """Same discipline as `_peak_is_big_enough`: when the scale of a move cannot be established, the
    honest response is to leave the position alone rather than act on a number of unknown meaning."""
    st = {"AAA": TrailState(entry_px=100.0, peak_px=100.0, quality=LIVE)}
    for bad in ({}, {"AAA": 0.0}, {"AAA": float("nan")}):
        assert evaluate_exits(ExitConfig(stop_loss_atr=2.0), {"AAA": 50.0}, st,
                              atr=bad).exits.get("AAA") is None


def test_needs_atr_covers_the_stop():
    """A runner that computed ATR only for the rules it knew about would leave this one inert, and
    the evaluator would refuse the run — correctly, but the caller should never have got there."""
    from kumo_strategies.strategies.momentum_rotation.exits import needs_atr
    assert needs_atr(ExitConfig(stop_loss_atr=2.0))
