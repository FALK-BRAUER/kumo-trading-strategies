"""`max_weight` — a position cap must redistribute, not shrink the book (#26).

ONE defect, not the two the audit first reported.

REAL: the cap was `min(x, cap)` with no redistribution. The weights then summed to less than 1, so
setting `max_weight` moved capital to CASH rather than to the other names. That does not look like a
bug — gross exposure quietly falls, the return falls with it, and the shortfall is indistinguishable
from the strategy underperforming. The #25 ablation hit the same shape from another direction: a
4-point deployment drop read as a risk effect until it was isolated.

NOT REAL, and retracted: "the cap sits only on the inverse-vol branch, so an equal-weighted book
ignores its own position cap." That was read off control flow and is mathematically empty — equal
weights are exactly 1/n, so a cap above 1/n cannot bind and a cap below it cannot be satisfied by
any allocation summing to 1. A mutation bite reverting the branch left every test green, which is
how it was caught. See `test_the_equal_weight_branch_was_never_actually_a_defect`.

The second-order case is the one worth the code: redistributing an excess raises the remaining
names and can carry a SECOND over the cap. A single pass leaves that breach while the weights still
sum to 1, so the obvious test passes and the cap is still violated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig, PortfolioConfig)
from kumo_strategies.strategies.momentum_rotation.engine import _apply_max_weight, _weights

NAMES = ("AAA", "BBB", "CCC", "DDD")


def _cfg(max_weight: float = 1.0, inverse_vol: bool = False) -> MomentumRotationConfig:
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, max_weight=max_weight,
                                  inverse_vol_sizing=inverse_vol))


def test_weights_sum_to_one_when_the_cap_binds():
    """The headline. A cap that leaves the book under-invested is not a cap."""
    vol = pd.Series({"AAA": 0.01, "BBB": 0.04, "CCC": 0.04, "DDD": 0.04})
    w = _weights(NAMES, vol, _cfg(max_weight=0.40, inverse_vol=True))
    assert sum(w.values()) == pytest.approx(1.0), (
        f"weights sum to {sum(w.values()):.4f} — the excess went to cash instead of to the other "
        "names, so the cap shrank gross exposure rather than redistributing it")


def test_the_cap_is_actually_respected():
    vol = pd.Series({"AAA": 0.005, "BBB": 0.05, "CCC": 0.05, "DDD": 0.05})
    w = _weights(NAMES, vol, _cfg(max_weight=0.40, inverse_vol=True))
    assert max(w.values()) <= 0.40 + 1e-9, f"cap breached: {w}"


def test_redistribution_can_push_a_second_name_over_and_must_iterate():
    """One pass is not enough, and the single-pass bug is invisible in the sum.

    Redistributing the first name's excess raises the others. If that carries a second name over the
    cap, a single-pass implementation leaves the breach in place while the weights still sum to 1 —
    so the obvious test (sum == 1) passes and the cap is still violated.
    """
    w = _apply_max_weight({"AAA": 0.70, "BBB": 0.20, "CCC": 0.06, "DDD": 0.04}, cap=0.35)
    assert sum(w.values()) == pytest.approx(1.0)
    assert max(w.values()) <= 0.35 + 1e-9, f"second-order breach survived: {w}"


def test_the_equal_weight_branch_was_never_actually_a_defect():
    """Correction, kept as a test so the false claim cannot come back.

    The audit first reported the early return at the top of `_weights` as a second `max_weight`
    defect — the cap sat only on the inverse-vol branch, so an equal-weighted book "ignored its own
    position cap". A mutation bite killed that claim: reverting the early return left every test
    green, because equal weights are exactly 1/n and a cap above 1/n can never bind on them. Below
    1/n no allocation summing to 1 satisfies the cap at all, so both branches degrade identically.

    The branch is now shared anyway, which is simpler and removes the question. But the code was
    correct and the finding was wrong, and that is worth pinning: a defect asserted from reading
    control flow rather than from a failing case is a hypothesis, not a bug.
    """
    for cap in (0.30, 0.25, 0.20):
        w = _weights(NAMES, None, _cfg(max_weight=cap))
        assert all(x == pytest.approx(0.25) for x in w.values()), (
            f"equal weight is 1/n regardless of cap={cap}; got {w}")


def test_a_cap_below_one_over_n_degrades_to_equal_weight_not_to_cash():
    """No allocation summing to 1 can satisfy it. Equal weight is the honest answer.

    Returning capped-and-short would put the book in cash again, which is the defect this whole
    file exists for.
    """
    w = _weights(NAMES, None, _cfg(max_weight=0.10))      # 0.10 < 1/4
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(x == pytest.approx(0.25) for x in w.values()), w


def test_default_config_is_untouched():
    """max_weight defaults to 1.0, so every recorded result must reproduce byte-identically."""
    assert _weights(NAMES, None, _cfg()) == {n: 0.25 for n in NAMES}
    vol = pd.Series({"AAA": 0.01, "BBB": 0.02, "CCC": 0.03, "DDD": 0.04})
    w = _weights(NAMES, vol, _cfg(inverse_vol=True))
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["AAA"] > w["DDD"], "inverse-vol ordering lost"


def test_redistribution_preserves_ranking_among_uncapped_names():
    w = _apply_max_weight({"AAA": 0.60, "BBB": 0.25, "CCC": 0.10, "DDD": 0.05}, cap=0.40)
    assert w["BBB"] > w["CCC"] > w["DDD"], f"pro-rata redistribution flattened the ranking: {w}"
