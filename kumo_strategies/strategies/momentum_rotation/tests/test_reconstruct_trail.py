"""Rebuilding a position's trail from bars instead of remembering it (#26 / #197 B1).

This is what lets the Nautilus adapters honour `ExitConfig` without a persistent store. The
property that matters is not "it computes a peak" — it is that a position whose history does NOT
reach back to its entry is marked ADOPTED, so peak-relative rules skip it.

#197 B1 is the reason. The old code seeded `entry, peak = (px, px)` for any position without a
record, which asserts a peak that was never observed: it left give-back unarmed on seven positions
and fired it on a 0.06% artefact for the eighth. A reconstructed trail must never invent a peak it
did not see.
"""

from __future__ import annotations

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
from kumo_strategies.strategies.momentum_rotation.exits import (
    ADOPTED, LIVE, RECONSTRUCTED, evaluate_exits, reconstruct_trail)


def test_full_history_gives_a_trustworthy_peak():
    t = reconstruct_trail(100.0, [101.0, 108.0, 104.0, 103.0], covers_entry=True)
    assert t.quality == RECONSTRUCTED
    assert t.peak_px == 108.0
    assert t.sessions_held == 4
    assert t.sessions_since_high == 2          # 104, 103 came after the 108 high
    assert t.peak_is_trustworthy


def test_partial_history_is_adopted_so_peak_rules_skip_it():
    """The #197 B1 case. History starts after the entry, so the true peak is unknowable."""
    t = reconstruct_trail(100.0, [104.0, 103.0], covers_entry=False)
    assert t.quality == ADOPTED
    assert not t.peak_is_trustworthy


def test_no_bars_never_claims_the_entry_was_the_peak():
    """Seeding peak == entry is precisely the bug that left give-back unarmed on seven positions."""
    t = reconstruct_trail(100.0, [], covers_entry=False)
    assert t.quality == ADOPTED
    assert not t.peak_is_trustworthy


def test_a_position_underwater_since_entry_keeps_entry_as_its_peak():
    """Covered history that never traded above entry: the peak IS the entry, and legitimately so.

    This is the SU/HSBC/PAA shape from the live loss — names that never printed above entry. The
    peak is real here, unlike the adopted case, so it must stay trustworthy or give-back could never
    arm on a name that only ever fell.
    """
    t = reconstruct_trail(100.0, [98.0, 95.0, 96.0], covers_entry=True)
    assert t.quality == RECONSTRUCTED
    assert t.peak_px == 100.0
    assert t.peak_is_trustworthy


def test_give_back_fires_on_a_reconstructed_trail():
    """End to end: reconstruction feeds the shared evaluator and the rule actually triggers."""
    t = reconstruct_trail(100.0, [110.0, 120.0, 112.0], covers_entry=True)
    # Peak 120 on a 100 entry is +20 of profit; give_back 0.5 arms at 110.
    plan = evaluate_exits(ExitConfig(give_back_frac=0.5), {"AAA": 109.0}, {"AAA": t})
    assert "AAA" in plan.exits, f"give-back did not fire on a reconstructed trail: {plan.exits}"


def test_give_back_does_not_fire_on_an_adopted_trail():
    """The guard that makes reconstruction safe. Same prices, unknown peak, no exit.

    Without this a restart mid-hold would resurrect the #197 B1 behaviour: a peak nobody saw,
    driving a real sell.
    """
    t = reconstruct_trail(100.0, [110.0, 120.0, 112.0], covers_entry=False)
    plan = evaluate_exits(ExitConfig(give_back_frac=0.5), {"AAA": 109.0}, {"AAA": t})
    assert "AAA" not in plan.exits, "peak-relative rule fired on a peak that was never observed"


def test_reconstruction_is_deterministic_across_restarts():
    """The whole justification for reconstructing rather than remembering.

    A fresh process holding the same bars must derive the same trail — that is why this needs no
    store, and why it cannot lose state the way a remembered peak did.
    """
    closes = [101.0, 108.0, 104.0, 103.0]
    a = reconstruct_trail(100.0, closes, covers_entry=True)
    b = reconstruct_trail(100.0, list(closes), covers_entry=True)
    assert a == b


def test_sessions_since_high_counts_sessions_not_bars():
    """`ExitConfig.stall_days` counts SESSIONS, so a trail built from session closes must too."""
    t = reconstruct_trail(50.0, [60.0, 59.0, 58.0, 57.0, 56.0], covers_entry=True)
    assert t.sessions_held == 5
    assert t.sessions_since_high == 4
    plan = evaluate_exits(ExitConfig(stall_days=3), {"AAA": 56.0}, {"AAA": t})
    assert "AAA" in plan.exits, "stall rule did not see the session count"
