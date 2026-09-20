"""Selling INTO STRENGTH — the exit category the strategy has never had (#30 item 3).

Every other rule fires on weakness: give-back, off-peak, stall, time, and the ranking exit itself.
Nothing sells while a position is still rising. The operator raised the gap on 2026-08-15.

The measured cost of not having it, from the 2026-08-13 journal — give-back sells AFTER the run is
surrendered, by construction: AEM left 2.7% behind, CGAU 3.5%, FSM 4.1%, WPM 2.0%. About $1,192 in
one session.

The argument against is equally real and is why this is measured rather than assumed: the strategy
is tail-driven (+90.4% of P&L in the wildest volatility quartile), and a profit target caps the right
tail by construction.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import LIVE, TrailState, evaluate_exits

ATR = {"AAA": 2.0}          # $2 of true range on a $100 name


def _st(entry=100.0, peak=None):
    return TrailState(entry_px=entry, peak_px=peak if peak is not None else entry, quality=LIVE)


def test_fires_while_the_position_is_still_rising():
    """The whole point. Nothing else in this module can sell a winner that has not yet turned."""
    cfg = ExitConfig(take_profit_atr=3.0)
    plan = evaluate_exits(cfg, {"AAA": 106.5}, {"AAA": _st()}, atr=ATR)   # +6.5 = 3.25 ATR
    assert "AAA" in plan.exits
    assert "took profit" in plan.exits["AAA"]


def test_does_not_fire_below_the_target():
    cfg = ExitConfig(take_profit_atr=3.0)
    assert evaluate_exits(cfg, {"AAA": 105.0}, {"AAA": _st()}, atr=ATR).exits == {}


def test_it_beats_give_back_to_the_exit_and_the_reason_says_so():
    """Order is the semantics. A position at its target is sold for a reason unrelated to fading;
    letting give-back claim it would mislabel the exit in the journal an operator reads."""
    cfg = ExitConfig(take_profit_atr=2.0, give_back_frac=0.5)
    # Peak 112 (entry 100), now 104 — give-back would fire; but 104 is also +2 ATR.
    plan = evaluate_exits(cfg, {"AAA": 104.0}, {"AAA": _st(peak=112.0)}, atr=ATR)
    assert "took profit" in plan.exits["AAA"], f"mislabelled: {plan.exits['AAA']}"


def test_scales_with_the_name_rather_than_using_a_fixed_percent():
    """#14's rule. The same 3 ATR target is a different price move on a quiet name and a wild one —
    which is the point, since this book spans a ~5x volatility range."""
    cfg = ExitConfig(take_profit_atr=3.0)
    quiet, wild = {"AAA": 1.0}, {"AAA": 5.0}
    assert "AAA" in evaluate_exits(cfg, {"AAA": 103.5}, {"AAA": _st()}, atr=quiet).exits
    assert evaluate_exits(cfg, {"AAA": 103.5}, {"AAA": _st()}, atr=wild).exits == {}


def test_missing_atr_does_not_fire():
    """Consistent with every other ATR-scaled rule: when the scale is unknown, do nothing rather
    than act on a number of unknown meaning."""
    cfg = ExitConfig(take_profit_atr=3.0)
    assert evaluate_exits(cfg, {"AAA": 200.0}, {"AAA": _st()}, atr={}).exits == {}


def test_the_config_without_atr_raises():
    """#26 shape, refused rather than degraded."""
    with pytest.raises(ValueError, match="no `atr` was passed"):
        evaluate_exits(ExitConfig(take_profit_atr=3.0), {"AAA": 110.0}, {"AAA": _st()})


def test_disabled_by_default():
    assert evaluate_exits(ExitConfig(), {"AAA": 500.0}, {"AAA": _st()}).exits == {}


def test_a_loser_is_never_taken_as_profit():
    cfg = ExitConfig(take_profit_atr=3.0)
    assert evaluate_exits(cfg, {"AAA": 94.0}, {"AAA": _st()}, atr=ATR).exits == {}


def test_needs_atr_covers_every_atr_scaled_rule():
    """One predicate, so a driver cannot be left behind when a rule is added.

    The re-test of every candidate crashed because `runner_verified` computed ATR only when
    `give_back_min_peak_atr` was set, so `take_profit_atr` alone reached `evaluate_exits` without
    it. The guard refused the run — correctly — but the caller should never have got there.
    """
    from dataclasses import fields

    from kumo_strategies.strategies.momentum_rotation.exits import needs_atr

    assert needs_atr(ExitConfig(take_profit_atr=3.0))
    assert needs_atr(ExitConfig(give_back_min_peak_atr=1.0))
    assert not needs_atr(ExitConfig(give_back_frac=0.5))

    # Every field whose name ends in _atr must be covered, or this predicate has fallen behind.
    atr_fields = [f.name for f in fields(ExitConfig) if f.name.endswith("_atr")]
    for name in atr_fields:
        assert needs_atr(ExitConfig(**{name: 1.0})), f"needs_atr missed {name}"
