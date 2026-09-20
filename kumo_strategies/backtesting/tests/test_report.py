"""Pins the reporting maths. Each of these is a way a KPI has been wrong or could silently lie."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.report import Report, deflated_sharpe, round_trips


def fills(rows):
    return pd.DataFrame(rows, columns=["ts", "symbol", "side", "qty", "price", "cost"])


def test_round_trip_matches_fifo_and_nets_cost():
    f = fills([
        (pd.Timestamp("2026-01-02"), "AAA", "BUY", 10, 100.0, 1.0),
        (pd.Timestamp("2026-01-05"), "AAA", "SELL", 10, 110.0, 1.1),
    ])
    t = round_trips(f)
    assert len(t) == 1
    assert t.gross_pnl.iloc[0] == pytest.approx(100.0)
    assert t.net_pnl.iloc[0] == pytest.approx(100.0 - 2.1)
    assert t.return_pct.iloc[0] == pytest.approx(10.0)
    assert t.hold_days.iloc[0] == pytest.approx(3.0)


def test_partial_exit_splits_the_lot():
    f = fills([
        (pd.Timestamp("2026-01-02"), "AAA", "BUY", 10, 100.0, 1.0),
        (pd.Timestamp("2026-01-03"), "AAA", "SELL", 4, 110.0, 0.4),
    ])
    t = round_trips(f)
    assert len(t) == 1 and t.qty.iloc[0] == 4
    assert t.cost.iloc[0] == pytest.approx(0.4 + 0.4)


def test_kpis_ignore_the_untraded_prefix():
    """A panel starting before the candidate source has coverage leaves a flat equity prefix. Those
    sessions are not zero-return days the strategy chose, and including them dragged Sharpe from
    1.94 to 1.22 on the real run."""
    idx = pd.bdate_range("2026-01-01", periods=60)
    eq = pd.Series([100_000.0] * 30 + list(100_000 * np.cumprod(1 + np.full(30, 0.002))), index=idx)
    f = fills([(idx[30], "AAA", "BUY", 1, 100.0, 0.0)])
    r = Report(equity=eq, trades=pd.DataFrame(), fills=f, starting_cash=100_000.0)
    k = r.kpis()
    assert k["sessions_traded"] == 30
    assert k["sessions_in_panel"] == 60
    assert k["ann_vol_pct"] > 0


def test_deflated_sharpe_falls_as_more_configs_are_tried():
    a = deflated_sharpe(0.13, 163, 1, -0.2, 4.0)
    b = deflated_sharpe(0.13, 163, 60, -0.2, 4.0)
    assert a > b, "trying more configurations must lower confidence, not raise it"
    assert 0.0 <= b <= 1.0


def test_cost_model_uses_the_symbol_not_an_average():
    cm = CostModel(half_spread_bps={"TIGHT": 0.5, "WIDE": 30.0}, default_bps=5.0, time_of_day=False)
    assert cm.charge("TIGHT", 10_000) == pytest.approx(0.5)
    assert cm.charge("WIDE", 10_000) == pytest.approx(30.0)
    assert cm.charge("UNKNOWN", 10_000) == pytest.approx(5.0)


def test_cost_model_is_dearer_at_the_open():
    """Measured (SIP, ~200k quotes): 5.53bps a side at 09:35, 2.12 at 11:30, 1.81 at 15:55 against
    a 2.46 all-day median. The open is roughly 3x midday — a strategy that trades the open pays the
    worst spread of the session, and a flat cost model hides exactly that."""
    cm = CostModel(half_spread_bps={"AAA": 4.0}, default_bps=4.0)
    assert cm.bps("AAA", 9, 35) > cm.bps("AAA", 11, 30) > cm.bps("AAA", 13, 0)
    assert cm.bps("AAA", 9, 35) / cm.bps("AAA", 13, 0) > 2.0
    # and it does not fall monotonically to the bell — spreads widen slightly into the close
    assert cm.bps("AAA", 15, 0) >= cm.bps("AAA", 13, 30)
