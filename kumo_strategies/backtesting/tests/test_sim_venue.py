"""The replay Venue and Book ports (#270 step 2, docs/runner-architecture.md §3).

The extraction's gate was three recorded numbers unchanged (ledger-book-daily-2024-2026 baseline 13.6297 %, the
bctrot variation, the cadence reference 9.2976 %). These pin the port's own contract: the two gates
it applies, and the arithmetic every fill goes through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.sim_venue import Book, SimVenue

TS = pd.Timestamp("2026-03-02 09:35")


def _venue(bps: float = 10.0) -> SimVenue:
    defs = {"AAA": SimpleNamespace(id=SimpleNamespace(venue=SimpleNamespace(value="XNAS")))}
    return SimVenue(CostModel(half_spread_bps={"AAA": bps}, default_bps=bps, time_of_day=False), defs)


def test_a_buy_charges_the_measured_cost_and_records_the_fill():
    v, b = _venue(10.0), Book(10_000.0)
    cost = v.buy(b, "AAA", 10, 100.0, TS, slot="open+5m")
    b.open("AAA", 10, 100.0)
    assert cost == pytest.approx(1.0)                      # 10 bps of $1,000
    assert b.cash == pytest.approx(10_000.0 - 1_000.0 - 1.0)
    assert b.held == {"AAA": 10} and b.state["AAA"].entry_px == 100.0
    assert b.fills == [{"ts": TS, "symbol": "AAA", "side": "BUY", "qty": 10, "price": 100.0,
                        "cost": pytest.approx(1.0), "venue": "XNAS", "slot": "open+5m"}]


def test_a_sell_credits_net_of_cost_and_the_exit_drops_the_trail():
    v, b = _venue(10.0), Book(0.0)
    b.open("AAA", 10, 100.0)
    qty = b.close("AAA")
    v.sell(b, "AAA", qty, 110.0, TS)
    assert b.cash == pytest.approx(1_100.0 - 1.1)
    assert b.held == {} and "AAA" not in b.state


def test_budget_allows_counts_the_cost_not_just_the_notional():
    """The replay budget gate: cash must cover notional PLUS the cost of the fill. Off by the cost
    is exactly the overspend the inline rule refused ("never spend cash we do not have")."""
    v, b = _venue(10.0), Book(1_000.5)
    assert v.budget_allows(b, "AAA", 10, 100.0, TS) is False     # $1,000 + $1.00 > $1,000.50
    b.cash = 1_001.0
    assert v.budget_allows(b, "AAA", 10, 100.0, TS) is True


def test_owns_is_the_ownership_gate_for_a_sell():
    v, b = _venue(), Book(0.0)
    assert v.owns(b, "AAA", 1) is False                            # holds nothing
    b.open("AAA", 10, 100.0)
    assert v.owns(b, "AAA", 10) is True
    assert v.owns(b, "AAA", 11) is False                           # more than the book holds
    assert v.owns(b, "AAA", 0) is False


def test_affordable_is_cash_net_of_the_cost_rate():
    """Through the model's own rate at the instant (time-of-day multiplier included), so the test
    cannot pass by assuming a flat rate the model does not charge."""
    v, b = _venue(10.0), Book(1_001.0)
    rate = v._cm.bps("AAA", TS.hour, TS.minute) / 1e4
    assert rate > 0
    assert v.affordable(b, "AAA", TS) == pytest.approx(1_001.0 / (1.0 + rate))
    # and buying exactly that notional is affordable, one cent more is not
    qty_notional = v.affordable(b, "AAA", TS)
    assert qty_notional + v.charge("AAA", qty_notional, TS) == pytest.approx(b.cash)


def test_a_trim_and_an_add_leave_the_trail_alone():
    v, b = _venue(), Book(100_000.0)
    b.open("AAA", 10, 100.0)
    st = b.state["AAA"]
    b.trim("AAA", 4); v.sell(b, "AAA", 4, 105.0, TS, rebalance=True)
    b.add("AAA", 2); v.buy(b, "AAA", 2, 105.0, TS, rebalance=True)
    assert b.held == {"AAA": 8}
    assert b.state["AAA"] is st, "a rebalance re-armed the trail"
    assert [f["side"] for f in b.fills] == ["SELL", "BUY"] and all(f["rebalance"] for f in b.fills)


def test_marked_is_nan_when_a_held_name_has_no_mark():
    b = Book(100.0); b.open("AAA", 1, 10.0)
    assert b.marked({"AAA": 12.0}) == pytest.approx(112.0)
    assert b.marked({}) != b.marked({})                            # NaN
