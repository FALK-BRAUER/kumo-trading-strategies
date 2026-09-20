"""A locate that EXISTS but carries no fee is a third fact, and it needs its own word (#123).

`_borrow_refusal` reads the borrow map as `symbol -> annual fee or None`, where `None` and absence
both mean "no locate" and both refuse. That is right for a venue that publishes a fee. Interactive
Brokers does not: measured on ibkr-paper, TWS exposes a shortability CODE (tick 46) and SHARES
AVAILABLE (tick 89), and no fee-rate tick at all — IB's fee file is a separate FTP feed.

So on the venue where this lane goes FIRST, every name is "borrowable, fee unknown", and the map as
typed can only say either "no locate" (refuse everything) or a number (invent one).

WHY NOT JUST PASS 0.0. It typechecks, it clears any ceiling, and it reads as FREE BORROW forever —
in the decision, in the journal, and in anything later reconciled against it. A fabricated zero is
the "default that is a real value" failure: undetectable downstream, because nothing distinguishes
it from a venue that genuinely quoted zero.

WHY NOT JUST SET max_borrow_fee_annual=None. That turns the gate off entirely, which also turns off
AVAILABILITY, so an unborrowable name would consume a slot with an order that cannot fill — what
TECHIVOL-005 did eight times in one session. It would also move the refusals out of the lane's
journal, and #123's acceptance requires refused entries to be logged: a gate that silently removes
names is indistinguishable from a signal that never fired.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import LOCATABLE, decide


def _day(symbols, *, crsi=95.0, limit=103.0):
    """One session's already-gated rows — what `decide` actually consumes."""
    return pd.DataFrame([{"ticker": s, "signal": True, "eligible": True, "manageable": True,
                          "crsi": crsi + i, "limit_px": limit, "asof_close": 100.0}
                         for i, s in enumerate(symbols)])


def test_a_LOCATABLE_name_is_entered_although_no_fee_is_known():
    dec = decide(_day(["AAA"]), CrsiShortConfig(), set(), borrow={"AAA": LOCATABLE})
    assert dec.enter == ("AAA",), dec.refused


def test_a_LOCATABLE_name_is_not_compared_against_the_ceiling():
    """The ceiling is a statement about FEES. With no fee there is nothing to compare, and the
    comparison must be skipped rather than resolved in either direction."""
    cfg = CrsiShortConfig(max_borrow_fee_annual=0.01)      # 1%/yr, tighter than almost any borrow
    dec = decide(_day(["AAA"]), cfg, set(), borrow={"AAA": LOCATABLE})
    assert dec.enter == ("AAA",), dec.refused


def test_None_still_means_NO_LOCATE_and_still_refuses():
    """The existing meaning is unchanged. LOCATABLE adds a state; it does not soften one."""
    dec = decide(_day(["AAA"]), CrsiShortConfig(), set(), borrow={"AAA": None})
    assert dec.enter == () and "AAA" in dec.refused


def test_an_ABSENT_name_still_refuses():
    dec = decide(_day(["AAA"]), CrsiShortConfig(), set(), borrow={})
    assert dec.enter == () and "AAA" in dec.refused


def test_a_real_fee_over_the_ceiling_still_refuses():
    cfg = CrsiShortConfig(max_borrow_fee_annual=1.0)
    dec = decide(_day(["AAA"]), cfg, set(), borrow={"AAA": 2.5})
    assert dec.enter == () and "over the" in dec.refused["AAA"]


def test_the_refusal_is_RECORDED_not_silent():
    """A borrow gate that silently removes names is indistinguishable from a signal that never
    fired — #123 requires the refusals, and this is why the gate stays in the lane rather than
    moving upstream into the pool."""
    dec = decide(_day(["AAA", "BBB"]), CrsiShortConfig(), set(),
                 borrow={"AAA": LOCATABLE, "BBB": None})
    assert dec.enter == ("AAA",)
    assert "BBB" in dec.refused and "no locate" in dec.refused["BBB"]


def test_LOCATABLE_is_not_a_number_and_cannot_be_mistaken_for_one():
    """It must not be 0.0, or anything that compares like a fee. A sentinel that arithmetic accepts
    would be indistinguishable from a venue quoting zero everywhere it is later read."""
    assert not isinstance(LOCATABLE, (int, float))
    assert LOCATABLE is not None
    assert "locat" in repr(LOCATABLE).lower(), "the repr must say what it is in a journal row"


def test_LOCATABLE_survives_a_round_trip_through_the_map_type():
    """Providers build this dict per session; the sentinel has to be the SAME object on the way out,
    since identity is how it is recognised."""
    got = {"AAA": LOCATABLE}.get("AAA")
    assert got is LOCATABLE


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_NON_FINITE_fee_is_still_no_locate_and_is_NOT_read_as_locatable(bad):
    """`nan` passes `is not None` and `float()` happily. It must not become the new way to say
    'available' — that is the same fabrication as 0.0, wearing a different mask."""
    dec = decide(_day(["AAA"]), CrsiShortConfig(), set(), borrow={"AAA": bad})
    assert dec.enter == () and "AAA" in dec.refused
