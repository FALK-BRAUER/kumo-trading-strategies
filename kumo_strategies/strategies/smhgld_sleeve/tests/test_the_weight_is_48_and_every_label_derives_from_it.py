"""SMHGLD's risk leg is 48%, and every place that SAYS the weight derives it from the config (#219).

2026-09-13: the risk leg goes from 32% to 48%. The lab costed it (research-lab
research/gld-hedge/bars_long.json, 0.25pp band, 2 bps, fixed-weight walk, reproduced 2026-09-13):

    window               weight   CAGR    Sharpe  maxDD    2022     trades/yr
    2018-01 -> 2026-09   0.32     +21.8%  1.23    -23.0%   -11.0%   146
                         0.48     +25.2%  1.21    -27.6%   -16.3%   156
    2005-01 -> 2026-09   0.32     +14.7%  0.92    -32.4%   -11.0%   132
                         0.48     +16.1%  0.93    -37.3%   -16.3%   143
    2024-01 -> 2026-09   0.32     +41.9%  1.76    -15.7%            151
                         0.48     +46.4%  1.74    -14.8%            160

THE LABEL WAS ALREADY WRONG (#186). `runtime/nautilus/smhgld_sleeve.py` said "a fixed 48/52 sleeve"
while `risk_weight` was 0.32, and `live_notes` said "A fixed 32/68 sleeve" — three copies of one
number, two of them disagreeing. After the flip the label would AGREE BY COINCIDENCE, which is worse
than disagreeing: nobody would notice the next drift. So the percentages are derived from the
config, in one function, and asserted with a weight no default can produce.

Every test here was seen red on 9079067 before the fix, for the reason its docstring names.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest

from kumo_strategies.strategies.smhgld_sleeve import nautilus as adapter_module
from kumo_strategies.strategies.smhgld_sleeve import (
    Decision, SmhGldSleeveConfig, live_config, live_notes, order_plan)

#: A weight no default, no note and no label in this package can produce.
_PROBE_WEIGHT = 0.37


# ------------------------------------------------------------------ the decision ---------------

def test_the_live_config_carries_the_DECIDED_weight():
    cfg = live_config()
    assert cfg.risk_weight == 0.48
    assert cfg.target_weights == {"SMH": pytest.approx(0.48), "GLD": pytest.approx(0.52)}


def test_the_default_config_IS_the_live_config():
    """`live_config()` returns `SmhGldSleeveConfig()` on purpose; the decision lives on the default."""
    assert SmhGldSleeveConfig().risk_weight == live_config().risk_weight == 0.48


# ------------------------------------------------------------------ one derivation of "48/52" --

def test_the_percentages_are_DERIVED_from_the_weight():
    from kumo_strategies.strategies.smhgld_sleeve import weight_split

    assert weight_split(SmhGldSleeveConfig()) == "48/52"
    assert weight_split(SmhGldSleeveConfig(risk_weight=_PROBE_WEIGHT)) == "37/63"
    assert weight_split(SmhGldSleeveConfig(risk_weight=0.5)) == "50/50"
    # GLD is 100 - SMH%, never rounded on its own: rounding each half independently sums to 99 or
    # 101 at 16 of the 999 weights below (0.425, 0.445, 0.465, 0.685, ... — float error on x.5).
    for w in range(5, 996):   # the config refuses a leg narrower than the 0.25pp band
        a, b = weight_split(SmhGldSleeveConfig(risk_weight=w / 1000)).split("/")
        assert int(a) + int(b) == 100, f"risk_weight={w / 1000}: {a}/{b}"


def test_the_runtime_LABEL_agrees_with_the_config_and_is_not_a_literal():
    """On 9079067 the label said 48/52 and the config said 0.32. Both halves: the value agrees,
    and the assignment is not a bare string constant a future edit can leave behind."""
    from kumo_strategies.strategies.smhgld_sleeve import weight_split

    label = adapter_module.STRATEGY_LABEL
    # Anchored, and EXACTLY one: `found[0]` on an unanchored pattern would compare "24/7" or "1/2"
    # from the prose to the config (coverage review, #219).
    found = re.findall(r"\b(\d{1,2})/(\d{1,2})\b", label)
    assert len(found) == 1, f"the label must name exactly one weight split: {label!r} -> {found}"
    assert int(found[0][0]) + int(found[0][1]) == 100
    assert "/".join(found[0]) == weight_split(live_config()), (
        f"label says {'/'.join(found[0])}, config says {weight_split(live_config())} — two "
        f"derivations of one fact, disagreeing (#186)")

    # THE DERIVATION PATH, not only the value: a second f-string formatter that agrees today is the
    # coincidence being killed. The assignment must CALL `weight_split`.
    tree = ast.parse(textwrap.dedent(inspect.getsource(adapter_module)))
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "STRATEGY_LABEL" for t in n.targets)]
    assert assigns, "STRATEGY_LABEL is not assigned at module level"
    calls = [n for n in ast.walk(assigns[0].value) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "weight_split"]
    assert calls, (
        "STRATEGY_LABEL does not call `weight_split` — a literal or a private formatter agrees "
        "with the config by coincidence and drifts the next time either is edited (#186)")


def test_the_adapter_class_LABEL_is_the_derived_one():
    from kumo_strategies.strategies.smhgld_sleeve import weight_split

    assert weight_split(live_config()) in adapter_module.SmhGldSleeveStrategy.LABEL


def test_live_notes_state_the_weight_FROM_the_config():
    """`live_notes` said "A fixed 32/68 sleeve" — the third copy. It takes the config now, so a
    probe weight proves the note reads it rather than restating it."""
    assert "48/52" in live_notes()["what_this_lane_is"]
    assert "32/68" not in live_notes()["what_this_lane_is"]
    assert "37/63" in live_notes(SmhGldSleeveConfig(risk_weight=_PROBE_WEIGHT))["what_this_lane_is"]


def test_no_note_still_argues_for_32():
    """The 32%/ulcer argument was replaced by the decision and its measured cost. A note that still
    says the weight is chosen against drawdown on 2021-2026 describes a lane that no longer exists."""
    joined = " ".join(live_notes().values())
    assert "promoted 32%" not in joined and "32% weight is set" not in joined, joined[:400]
    assert "-16.3%" in live_notes()["2022_IS_THE_STRESS_CASE_AND_THE_SLEEVE_FAILS_IT"], (
        "the 2022 loss at 48% must be stated, not the 32% one")


# ------------------------------------------------------------------ the band still validates ---

def test_the_drift_band_is_still_inside_the_smaller_leg_at_48():
    """0.25pp < min(0.48, 0.52). Unchanged rule, re-asserted at the new weight."""
    cfg = live_config()
    assert cfg.drift_band < min(cfg.risk_weight, 1.0 - cfg.risk_weight)
    with pytest.raises(ValueError):
        SmhGldSleeveConfig(drift_band=0.48)
    SmhGldSleeveConfig(drift_band=0.47)  # inside the smaller leg: accepted


# ------------------------------------------------------------------ the whole-share floor (#186)

#: The snapshot's last closes (research/smhgld_sleeve/bars.json, 2026-09-10). Fixed here so the
#: arithmetic below is reproducible; the PR restates it at the deploy-day prices.
_PX = {"SMH": 560.28, "GLD": 396.36}


def _plan(sleeve: float, drift_pp: float):
    """A book worth `sleeve` whose SMH WEIGHT is exactly `risk_weight + drift_pp`, planned.

    The drift is built on weight, not by adding whole shares: adding shares understates the drift,
    which makes "nothing moves" EASIER to satisfy — the direction that lets a broken floor pass
    (coverage review, #219). Fractional shares are legal in the drifted book; rounding lives on the
    delta inside `order_plan`."""
    cfg = live_config()
    w = cfg.risk_weight + drift_pp
    drifted = {"SMH": w * sleeve / _PX["SMH"], "GLD": (1.0 - w) * sleeve / _PX["GLD"]}
    decision = Decision(hold=("GLD", "SMH"), weights=cfg.target_weights, regime="rebalance",
                        reasons=("test",))
    return order_plan(decision, drifted, _PX, sleeve)


@pytest.mark.parametrize("sleeve,expected", [
    (100_000.0, {1: (), 2: (("GLD", 1.0),), 4: (("SMH", -1.0), ("GLD", 2.0))}),
    (20_000.0, {1: (), 2: (), 4: ()}),
    (250_000.0, {1: (("SMH", -1.0), ("GLD", 1.0)), 2: (("SMH", -2.0), ("GLD", 3.0)),
                 4: (("SMH", -4.0), ("GLD", 6.0))}),
])
def test_the_whole_share_floor_at_48_for_a_100k_a_20k_and_a_250k_sleeve(sleeve, expected):
    """#186's finding re-checked at the new weight, driven through the real `order_plan` at the
    snapshot closes with the drift built on weight EXACTLY. One SMH share is 0.56% of a 100k sleeve
    and 2.8% of a 20k one against a 0.25pp band: at the band NO sleeve below ~224k moves a whole
    SMH share (250k does: SMH -1 / GLD +1), the 100k sleeve moves its first SMH share at four
    times the band, and the 20k sleeve (paper's) moves NOTHING at four times the band. That is the
    floor (#186), unchanged in kind by this PR and restated at 48% because the risk leg is larger:
    the same drift in points is a larger dollar delta than at 32%."""
    for k, plan in expected.items():
        got = tuple((o.symbol, o.delta) for o in _plan(sleeve, k * live_config().drift_band))
        assert got == plan, f"sleeve {sleeve:,.0f} at {k}x band: {got} != {plan}"


def test_the_floor_for_one_whole_share_at_exactly_the_band():
    """The sleeve size below which a band-triggered rebalance cannot move one whole share of each
    leg: price / band. At the snapshot closes: SMH 224,112; GLD 158,544."""
    cfg = live_config()
    assert round(_PX["SMH"] / cfg.drift_band) == 224_112
    assert round(_PX["GLD"] / cfg.drift_band) == 158_544
