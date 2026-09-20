"""The sidestep operator, both directions (#32).

Not an exit: the slot stays reserved and the position is restored on the reversal. The tests that
matter are the ones pinning WHAT EACH TRIGGER MEASURES AGAINST, because every plausible-but-wrong
variant of this rule differs only in that.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.momentum_rotation.sidestep import (
    STRONG, WEAK, SidestepConfig, SidestepState, evaluate_sidestep)


def _run(cfg, closes, entry=100.0):
    """Feed closes in order, carrying state as a driver would. Returns (sells, buys) per session."""
    st = {"AAA": SidestepState(entry_px=entry)}
    out = []
    for px in closes:
        plan = evaluate_sidestep(cfg, {"AAA": px}, st)
        st = plan.state
        out.append((plan.sells.get("AAA"), plan.buys.get("AAA")))
    return out, st


def test_weakness_sells_below_entry_and_rebuys_off_the_low():
    cfg = SidestepConfig(drop_pct=0.10, rebuy_pct=0.02)
    got, _ = _run(cfg, [95.0, 89.0, 85.0, 87.0])
    assert got[0] == (None, None), "5% down is not 10%"
    assert got[1][0] is not None and "weakness" in got[1][0]
    assert got[2] == (None, None), "still falling — a new low is not a bounce"
    assert got[3][1] is not None, "87 is 2.4% off the 85 low"


def test_strength_sells_above_entry_and_rebuys_on_the_dip():
    cfg = SidestepConfig(pop_pct=0.10, dip_pct=0.02)
    got, _ = _run(cfg, [105.0, 112.0, 120.0, 117.0])
    assert got[0] == (None, None)
    assert got[1][0] is not None and "strength" in got[1][0]
    assert got[2] == (None, None), "still running — a new high is not a dip"
    assert got[3][1] is not None, "117 is 2.5% off the 120 high"


def test_the_rebuy_measures_from_the_EXTREME_not_from_the_sell_price():
    """The distinguishing property. Waiting to get back to the sell price means a name that drops
    20% and recovers 15% is never rebought — the sidestep captures the whole decline and none of the
    recovery, which is the worst of both."""
    cfg = SidestepConfig(drop_pct=0.10, rebuy_pct=0.02)
    got, _ = _run(cfg, [88.0, 80.0, 84.0])       # sold at 88, low 80, back to 84 — still below 88
    assert got[0][0] is not None
    assert got[2][1] is not None, "rebuy must fire off the 80 low, not wait for 88"


def test_a_strength_sidestep_does_not_rebuy_on_a_bounce_off_a_low():
    """`out` records the DIRECTION. Without it a strength sidestep would apply the weakness rebuy
    rule and buy back into a name still making new highs."""
    cfg = SidestepConfig(pop_pct=0.10, dip_pct=0.05, rebuy_pct=0.01)
    got, st = _run(cfg, [115.0, 118.0, 119.5])
    assert got[0][0] is not None and st["AAA"].out == STRONG
    assert all(b is None for _, b in got[1:]), "rose off its high and must not rebuy"


def test_both_directions_can_be_configured_together():
    """A close cannot be both below entry and above it, so the two triggers cannot race."""
    cfg = SidestepConfig(drop_pct=0.10, rebuy_pct=0.02, pop_pct=0.10, dip_pct=0.02)
    weak, st_w = _run(cfg, [85.0])
    strong, st_s = _run(cfg, [115.0])
    assert st_w["AAA"].out == WEAK and st_s["AAA"].out == STRONG
    assert weak[0][0] and strong[0][0]


def test_the_position_is_never_given_up_it_only_goes_flat():
    """The whole point. A sidestep is not an exit — the slot stays reserved, so state persists for
    the symbol whether it is in or out."""
    cfg = SidestepConfig(drop_pct=0.10)
    _, st = _run(cfg, [85.0, 84.0, 83.0])
    assert "AAA" in st and st["AAA"].out == WEAK
    assert st["AAA"].entry_px == 100.0, "entry survives the round trip — it is still the position"


def test_max_sidesteps_caps_the_whipsaw():
    """Each cycle pays two spreads. On a five-name book that is 20% of the portfolio churning, and
    uncapped the rule is unbounded exactly where it is most wrong."""
    capped = SidestepConfig(drop_pct=0.05, rebuy_pct=0.02, max_sidesteps=1)
    got, st = _run(capped, [94.0, 96.0, 94.0, 96.0])
    assert got[0][0] is not None and got[1][1] is not None, "first round trip completes"
    assert st["AAA"].cycles == 1
    assert got[2][0] is None, "second sidestep must not arm once the cap is reached"


def test_uncapped_is_the_default_because_the_cap_is_a_parameter_to_test():
    cfg = SidestepConfig(drop_pct=0.05, rebuy_pct=0.02)
    got, st = _run(cfg, [94.0, 96.0, 94.0, 96.0])
    assert st["AAA"].cycles == 2, "uncapped must keep cycling so the cost can be measured"


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -1.0])
def test_an_unusable_price_leaves_the_position_entirely_alone(bad):
    """Acting on a missing price is how a data outage becomes a trade. State must not advance
    either — a stale extreme would trigger a spurious rebuy on the next good bar."""
    st = {"AAA": SidestepState(entry_px=100.0, out=WEAK, extreme_px=80.0)}
    plan = evaluate_sidestep(SidestepConfig(drop_pct=0.10), {"AAA": bad}, st)
    assert not plan.sells and not plan.buys
    assert plan.state["AAA"] == st["AAA"]


def test_disabled_by_default_so_it_changes_nothing_until_asked_for():
    got, st = _run(SidestepConfig(), [50.0, 200.0])
    assert got == [(None, None), (None, None)]
    assert st["AAA"].out is None
