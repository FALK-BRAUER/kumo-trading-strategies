"""The shared give-back arithmetic, both sides (#110, #123).

These are the tests that make "the same mechanism, arrived at from the opposite direction" a
checkable claim rather than a sentence in a ticket.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.give_back import (
    LONG, SHORT, favourable_extreme, favourable_gain, gave_back)


def test_a_short_that_fell_is_in_profit_and_a_long_that_fell_is_not():
    """The sign convention is the POSITION's, not the tape's. This is the whole module."""
    assert favourable_gain(100.0, 90.0, side=SHORT) == pytest.approx(0.10)
    assert favourable_gain(100.0, 90.0, side=LONG) == pytest.approx(-0.10)


def test_the_favourable_extreme_of_a_short_is_the_low():
    assert favourable_extreme(100.0, 90.0, side=SHORT) == 90.0
    assert favourable_extreme(100.0, 90.0, side=LONG) == 100.0


def test_a_return_to_exactly_entry_breaches_a_full_give_back():
    """`frac = 1.0` at exactly entry. The `<` this replaced returns False here.

    That is the flat exit inverted by one character: it could then fire only once the position was
    already a LOSS, which is the opposite of "never give back a profitable trade".
    """
    breached, peak_gain, now_gain = gave_back(
        entry_px=100.0, extreme_px=80.0, price=100.0, frac=1.0, side=SHORT)
    assert breached
    assert peak_gain == pytest.approx(0.20)
    assert now_gain == pytest.approx(0.0)


def test_a_position_that_was_never_in_profit_has_given_nothing_back():
    """Arming on `peak_gain > 0`. Without it every losing trade reports a give-back of a peak that
    never existed — the "gave back 30825% of a 0.1% peak" journal line in #197."""
    breached, peak_gain, _ = gave_back(
        entry_px=100.0, extreme_px=100.0, price=130.0, frac=1.0, side=SHORT)
    assert not breached
    assert peak_gain == 0.0


def test_a_partial_give_back_fires_at_the_configured_fraction_on_both_sides():
    short = dict(entry_px=100.0, extreme_px=80.0, frac=0.5, side=SHORT)   # peak_gain 20%
    assert gave_back(price=92.0, **short)[0]          # kept 8%: below half of 20%
    assert not gave_back(price=88.0, **short)[0]      # kept 12%: still above half
    long = dict(entry_px=100.0, extreme_px=120.0, frac=0.5, side=LONG)
    assert gave_back(price=108.0, **long)[0]
    assert not gave_back(price=112.0, **long)[0]


def test_an_unknown_side_raises_rather_than_picking_one():
    with pytest.raises(ValueError, match="side must be"):
        favourable_gain(100.0, 90.0, side="flat")     # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entry_px must be positive"):
        favourable_gain(0.0, 90.0, side=SHORT)
