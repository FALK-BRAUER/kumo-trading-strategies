"""ATR-scaled exit rules must REACH behaviour from the QC345 driver (kumo-trading-platform issue 872).

`evaluate_exits` refuses a config that needs ATR without being given it — correctly, and loudly.
But `runner_qc345_verified` never computed ATR at all, so every ATR-scaled rule was unreachable
from this driver: `stop_loss_atr` could not be measured on QC345, and the attempt raised rather
than running. That is the #26 shape at one remove, the one `needs_atr` exists to prevent, and it
is why platform issue 872's arms C/D/E had to be built before they could be run.

Aimed at the CLASS, not the instance: every field `needs_atr` names is exercised, so adding a
fourth ATR-scaled rule cannot leave this driver silently behind. `test_the_stop_actually_fires`
is the one that carries the weight — a driver that passed `atr={}` would satisfy "does not raise"
while leaving the rule as dead as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, monthly family (#270)
)
from kumo_strategies.strategies.qc345_rotation import ExitConfig, QC345RotationConfig

#: Sessions per month in the fixture. ATR(14) needs `max(2, 14 // 2)` = 7 observations before it is
#: anything but NaN, and a NaN ATR is indistinguishable from an unwired one — the fixture has to
#: outlast the warmup or the test cannot tell the two apart.
SESSIONS_PER_MONTH = 20


def _bars() -> pd.DataFrame:
    """AAA climbs through January, is bought at the February open, then collapses.

    BBB never moves, so the ranking is never in doubt and the only reason AAA can leave the book
    mid-month is an exit rule firing.
    """
    rows: list[dict[str, object]] = []
    price = 10.0
    for month, day0 in ((1, "2026-01-"), (2, "2026-02-")):
        for i in range(SESSIONS_PER_MONTH):
            date = pd.Timestamp(f"{day0}{i + 1:02d}")
            if month == 1:
                price += 0.5                      # a clean uptrend: AAA ranks first in February
            else:
                price -= 0.6 if i else 0.0        # the collapse the stop exists for
            rows.append(dict(ticker="AAA", date=date, open=price, high=price + 0.25,
                             low=price - 0.25, close=price))
            rows.append(dict(ticker="BBB", date=date, open=20.0, high=20.1,
                             low=19.9, close=20.0))
    frame = pd.DataFrame(rows)
    for col in ("close_adj", "close_split", "close_split_dividend"):
        frame[col] = frame["close"]
    frame["volume"] = 1_000_000
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _assets() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "name": ["AAA Inc.", "BBB Inc."],
        "exchange": ["NASDAQ", "NYSE"],
        "status": ["active", "active"],
    })


def _instruments(path: Path) -> Path:
    path.write_text(json.dumps({
        "AAA": {"symbol": "AAA", "mic": "XNAS", "price_increment": "0.01"},
        "BBB": {"symbol": "BBB", "mic": "XNYS", "price_increment": "0.01"},
    }))
    return path


def _cfg(exits: ExitConfig) -> QC345RotationConfig:
    return QC345RotationConfig(
        lookback_sessions=5,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        corporate_action_window=5,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close",
        asset_universe_mode="all",
        market_cap_mode="skip",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
        exits=exits,
    )


def _run(exits: ExitConfig, tmp_path: Path, bars: pd.DataFrame | None = None):
    return run(
        _bars() if bars is None else bars,
        _assets(),
        cfg=_cfg(exits),
        instruments_path=_instruments(tmp_path / "instruments.json"),
        cost_model=CostModel(half_spread_bps={}, default_bps=0.0),
        starting_cash=10_000.0,
    )


def _mid_month_sells(fills: pd.DataFrame) -> pd.DataFrame:
    """AAA sells that are NOT a rebalance. The first session of a month is the only date the
    rotation itself can trade, so anything else is an exit rule and nothing else."""
    if fills.empty:
        return fills
    ts = pd.to_datetime(fills["ts"])
    return fills[(fills["side"] == "SELL") & (fills["symbol"] == "AAA") & (ts.dt.day != 1)]


def test_the_stop_actually_fires_and_the_control_holds(tmp_path):
    """The load-bearing one. Not "it does not raise" — a driver passing `atr={}` clears that bar
    while the rule stays exactly as dead as it was."""
    stopped = _run(ExitConfig(stop_loss_atr=1.5), tmp_path)
    held = _run(ExitConfig(), tmp_path)

    fired = _mid_month_sells(stopped.report.fills)
    assert not fired.empty, (
        "`stop_loss_atr=1.5` produced no mid-month exit: AAA falls ~11 ATR below its entry and the "
        "stop never fired, so the rule does not reach behaviour from this driver"
    )
    assert _mid_month_sells(held.report.fills).empty, (
        "the CONTROL exited mid-month too, so the fixture cannot attribute the exit to the stop"
    )


@pytest.mark.parametrize("exits", [
    pytest.param(ExitConfig(stop_loss_atr=1.5), id="stop_loss_atr"),
    pytest.param(ExitConfig(take_profit_atr=2.0), id="take_profit_atr"),
    pytest.param(ExitConfig(give_back_frac=0.5, give_back_min_peak_atr=1.0),
                 id="give_back_min_peak_atr"),
])
def test_every_rule_needs_atr_names_is_reachable(exits, tmp_path):
    """`needs_atr` is the predicate the driver must satisfy, so the test enumerates what it names.
    A fourth ATR-scaled rule added to `ExitConfig` without a case here should be caught by
    `test_needs_atr_has_no_field_this_file_forgets`, below."""
    _run(exits, tmp_path)          # raises ValueError from `evaluate_exits` when ATR is unwired


def test_needs_atr_has_no_field_this_file_forgets():
    """Guards the parametrisation above against `ExitConfig` growing a fourth ATR rule.

    Enumerating cases by hand is how a driver gets left behind one rule at a time — the exact
    failure `needs_atr` was written to end. This asserts the list is complete rather than trusting
    whoever adds the next field to remember."""
    from dataclasses import fields

    from kumo_strategies.strategies.momentum_rotation.exits import needs_atr

    atr_fields = {
        f.name for f in fields(ExitConfig)
        if needs_atr(ExitConfig(**{f.name: 1.0}))
    }
    assert atr_fields == {"stop_loss_atr", "take_profit_atr", "give_back_min_peak_atr"}, (
        f"ExitConfig's ATR-scaled rules changed to {sorted(atr_fields)} — add the new one to "
        f"test_every_rule_needs_atr_names_is_reachable, or this driver keeps running it inert"
    )


def test_a_panel_with_no_high_low_is_refused_by_name(tmp_path):
    """ATR is true RANGE, so it needs high and low. A panel without them must be REFUSED naming
    the missing columns — `trailing_atr` would otherwise raise a bare KeyError from three frames
    down, which reads as a bug in the runner rather than a gap in the caller's data."""
    bars = _bars().drop(columns=["high", "low"])
    with pytest.raises(ValueError, match="high.*low|low.*high"):
        _run(ExitConfig(stop_loss_atr=1.5), tmp_path, bars=bars)


def test_a_non_atr_config_runs_on_a_panel_with_no_high_low(tmp_path):
    """The mirror of the test above, and the one that makes `_atr_for`'s early return load-bearing.

    Found by review (issue 139): deleting `if not needs_atr(cfg.exits): return None`
    leaves numeric results identical on a panel that HAS high/low, so no fingerprint moves — but a
    non-ATR run on a panel WITHOUT them starts raising, with a message asserting an ATR rule is
    configured when none is.

    That is not hypothetical. `research/paper-fidelity/qc345_fid.py` selects
    ticker/date/open/close/close_adj/volume and no high/low, because the live config configures no
    ATR rule and does not need them. The early return is what lets a caller supply only the columns
    its own config actually requires.

    Two pre-existing tests in `test_runner_qc345_verified.py` happen to fail if it is removed, since
    their fixtures also carry no high/low. That is INCIDENTAL — it evaporates the day someone adds
    high/low to those fixtures for an unrelated reason — so the property gets its own assertion here
    rather than resting on a side effect of somebody else's data.
    """
    bars = _bars().drop(columns=["high", "low"])
    result = _run(ExitConfig(stall_days=12), tmp_path, bars=bars)
    assert not result.report.fills.empty, (
        "a non-ATR config on a panel with no high/low produced no fills at all — the run did not "
        "get far enough for this test to be about ATR"
    )
