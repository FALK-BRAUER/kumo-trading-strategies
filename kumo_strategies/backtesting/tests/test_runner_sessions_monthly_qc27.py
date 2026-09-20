from __future__ import annotations

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, monthly family (#270)
)
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig


def _bars(symbol: str, closes: list[float], *, start: str = "2025-01-01", volume: int = 1_000_000) -> pd.DataFrame:
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame(
        {
            "ticker": symbol,
            "date": pd.bdate_range(start, periods=len(closes)),
            "open": opens,
            "close": closes,
            "close_adj": closes,
            "volume": [volume] * len(closes),
        }
    )


def test_runner_sweeps_residual_cash_into_gld():
    # 130 bars, not 90. Like the engine test, this fixture passed only BECAUSE `warmup_sessions` was
    # unenforced: 90 covers every feature (momentum needs 65) while falling short of QC27's stated
    # 100-session warmup, so nothing is eligible now and the sweep never fires. The fixture had
    # encoded the defect (#33).
    dates = pd.bdate_range("2025-01-01", periods=130)
    up = [10.0 + 0.2 * i for i in range(len(dates))]
    down = [20.0 - 0.1 * i for i in range(len(dates))]
    gld = [30.0 + 0.01 * i for i in range(len(dates))]
    bars = pd.concat(
        [
            _bars("AAPL", up, volume=5_000_000),
            _bars("MSFT", down, volume=4_000_000),
            _bars("GLD", gld, volume=10_000_000),
        ],
        ignore_index=True,
    )
    sectors = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GLD"],
            "name": ["Apple Inc. Common Stock", "Microsoft Corporation Common Stock", "SPDR Gold Shares"],
            "sector": ["Technology", "Technology", "Finance"],
            "country": ["United States", "United States", "United States"],
        }
    )
    cfg = QC27TechInverseVolConfig(portfolio_size=2, liquidity_filter_size=2, lookback_sessions=20)
    res = run(
        bars,
        sectors,
        cfg=cfg,
        cost_model=CostModel(half_spread_bps={}, default_bps=0.0, time_of_day=False),
        n_trials=1,
    )
    assert res.diagnostics["gld_sweep_buys"] >= 1
    assert "GLD" in set(res.report.fills["symbol"])


def test_runner_triggers_stop_exit():
    # THE CRASH HAS TO LAND AFTER THE WARMUP. The original fixture ramped for 70 bars then crashed,
    # which worked only while `warmup_sessions` was unenforced -- with the 100-session warmup in
    # force the position is not open yet when the drawdown happens, so no stop can trigger and the
    # test asserts on an event that cannot occur (#33). Ramp past 100 first, then crash.
    dates = pd.bdate_range("2025-01-01", periods=140)
    ramp = [10.0 + 0.2 * i for i in range(110)]          # ends ~31.8, warmup served at bar 100
    crash = [30.0, 27.0, 24.0, 21.0, 19.0]              # >2% of portfolio, after the position exists
    tech = ramp + crash + [19.0] * (len(dates) - len(ramp) - len(crash))
    gld = [30.0 + 0.01 * i for i in range(len(dates))]
    bars = pd.concat(
        [
            _bars("AAPL", tech[: len(dates)], volume=5_000_000),
            _bars("GLD", gld, volume=10_000_000),
        ],
        ignore_index=True,
    )
    sectors = pd.DataFrame(
        {
            "symbol": ["AAPL", "GLD"],
            "name": ["Apple Inc. Common Stock", "SPDR Gold Shares"],
            "sector": ["Technology", "Finance"],
            "country": ["United States", "United States"],
        }
    )
    cfg = QC27TechInverseVolConfig(
        portfolio_size=1,
        liquidity_filter_size=1,
        lookback_sessions=20,
        stop_loss_portfolio_frac=0.02,
    )
    res = run(
        bars,
        sectors,
        cfg=cfg,
        cost_model=CostModel(half_spread_bps={}, default_bps=0.0, time_of_day=False),
        n_trials=1,
    )
    assert res.diagnostics["stop_trigger_count"] >= 1
    assert res.diagnostics["stop_exit_count"] >= 1
