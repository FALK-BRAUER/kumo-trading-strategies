from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.families.monthly import _mark_position
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, monthly family (#270)
)
from kumo_strategies.strategies.qc345_rotation import ExitConfig, QC345RotationConfig


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "date": [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-02"),
            ],
            "open": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_adj": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_split": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_split_dividend": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "volume": [1_000_000] * 6,
        }
    )


def _assets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["AAA Inc.", "BBB Inc."],
            "exchange": ["NASDAQ", "NYSE"],
            "status": ["active", "active"],
        }
    )


def _instruments(path: Path) -> Path:
    payload = {
        "AAA": {"symbol": "AAA", "mic": "XNAS", "price_increment": "0.01"},
        "BBB": {"symbol": "BBB", "mic": "XNYS", "price_increment": "0.01"},
    }
    path.write_text(json.dumps(payload))
    return path


def test_qc345_verified_runner_buys_at_month_open_and_marks_close(tmp_path):
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        corporate_action_window=1,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close",
        asset_universe_mode="all",
        market_cap_mode="skip",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )
    result = run(
        _bars(),
        _assets(),
        cfg=cfg,
        instruments_path=_instruments(tmp_path / "instruments.json"),
        cost_model=CostModel(half_spread_bps={"AAA": 0.0, "BBB": 0.0}, default_bps=0.0, time_of_day=False),
        starting_cash=100_000.0,
    )
    first_rebalance = pd.Timestamp("2026-03-02")
    assert result.report.equity.loc[first_rebalance] == 100_000.0
    assert list(result.report.fills["side"]) == ["BUY"]
    assert result.report.fills.iloc[0]["price"] == 12.0
    assert result.report.fills.iloc[0]["venue"] == "XNAS"


def test_qc345_runner_keeps_inactive_symbol_without_instrument_metadata(tmp_path):
    bars = pd.DataFrame(
        {
            "ticker": ["DEAD", "LIVE", "DEAD", "LIVE", "DEAD", "LIVE"],
            "date": [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-02"),
            ],
            "open": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_adj": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_split": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "close_split_dividend": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0],
            "volume": [1_000_000] * 6,
        }
    )
    assets = pd.DataFrame(
        {
            "symbol": ["DEAD", "LIVE"],
            "name": ["Dead Co.", "Live Co."],
            "exchange": ["NYSE", "NASDAQ"],
            "status": ["inactive", "active"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        corporate_action_window=1,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close",
        asset_universe_mode="all",
        market_cap_mode="skip",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )
    path = tmp_path / "instruments.json"
    path.write_text(json.dumps({"LIVE": {"symbol": "LIVE", "mic": "XNAS", "price_increment": "0.01"}}))
    result = run(
        bars,
        assets,
        cfg=cfg,
        instruments_path=path,
        cost_model=CostModel(half_spread_bps={"DEAD": 0.0, "LIVE": 0.0}, default_bps=0.0, time_of_day=False),
        starting_cash=100_000.0,
    )
    assert list(result.report.fills["symbol"]) == ["DEAD"]
    assert result.report.fills.iloc[0]["venue"] == "XNYS"
    assert result.diagnostics["symbols_missing_instrument_metadata"] == 1
    assert result.diagnostics["symbols_using_fallback_metadata"] == 1
    assert result.diagnostics["symbols_dropped_missing_assets_metadata"] == 0


def test_bankruptcy_marks_to_zero_even_under_lastpx():
    assert _mark_position(
        row=None,
        qty=1,
        terminal_bucket="bankruptcy",
        delisting_mode="lastpx",
        last_known_close=42.0,
    ) == 0.0


def test_qc345_runner_can_force_exit_mid_month_at_next_open(tmp_path):
    bars = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"] * 5,
            "date": [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-15"),
                pd.Timestamp("2026-01-15"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-03"),
                pd.Timestamp("2026-02-03"),
                pd.Timestamp("2026-02-04"),
                pd.Timestamp("2026-02-04"),
            ],
            "open": [10.0, 20.0, 11.0, 20.0, 12.0, 20.0, 14.0, 20.0, 11.0, 20.0],
            "close": [10.0, 20.0, 11.0, 20.0, 15.0, 20.0, 13.0, 20.0, 11.0, 20.0],
            "close_adj": [10.0, 20.0, 11.0, 20.0, 15.0, 20.0, 13.0, 20.0, 11.0, 20.0],
            "close_split": [10.0, 20.0, 11.0, 20.0, 15.0, 20.0, 13.0, 20.0, 11.0, 20.0],
            "close_split_dividend": [10.0, 20.0, 11.0, 20.0, 15.0, 20.0, 13.0, 20.0, 11.0, 20.0],
            "volume": [1_000_000] * 10,
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        corporate_action_window=1,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close",
        asset_universe_mode="all",
        market_cap_mode="skip",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
        exits=ExitConfig(off_peak_pct=0.10),
    )
    result = run(
        bars,
        _assets(),
        cfg=cfg,
        instruments_path=_instruments(tmp_path / "instruments.json"),
        cost_model=CostModel(half_spread_bps={"AAA": 0.0, "BBB": 0.0}, default_bps=0.0, time_of_day=False),
        starting_cash=100_000.0,
    )
    fills = result.report.fills.reset_index(drop=True)
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["symbol"]) == ["AAA", "AAA"]
    assert fills.iloc[1]["ts"] == pd.Timestamp("2026-02-04 09:35:00")
    assert fills.iloc[1]["price"] == 11.0


def test_qc345_forced_exit_on_rebalance_is_not_bought_back_same_open(tmp_path):
    bars = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"] * 6,
            "date": [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-15"),
                pd.Timestamp("2026-01-15"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-02"),
                pd.Timestamp("2026-02-10"),
                pd.Timestamp("2026-02-10"),
                pd.Timestamp("2026-02-27"),
                pd.Timestamp("2026-02-27"),
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-02"),
            ],
            "open": [10.0, 20.0, 12.0, 19.0, 12.0, 20.0, 22.0, 18.0, 21.0, 15.0, 20.0, 9.0],
            "close": [10.0, 20.0, 12.0, 19.0, 20.0, 20.0, 22.0, 18.0, 21.0, 15.0, 20.0, 9.0],
            "close_adj": [10.0, 20.0, 12.0, 19.0, 20.0, 20.0, 22.0, 18.0, 21.0, 15.0, 20.0, 9.0],
            "close_split": [10.0, 20.0, 12.0, 19.0, 20.0, 20.0, 22.0, 18.0, 21.0, 15.0, 20.0, 9.0],
            "close_split_dividend": [10.0, 20.0, 12.0, 19.0, 20.0, 20.0, 22.0, 18.0, 21.0, 15.0, 20.0, 9.0],
            "volume": [1_000_000] * 12,
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        corporate_action_window=1,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close_adj",
        asset_universe_mode="all",
        market_cap_mode="skip",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
        exits=ExitConfig(stall_days=1),
    )
    result = run(
        bars,
        _assets(),
        cfg=cfg,
        instruments_path=_instruments(tmp_path / "instruments.json"),
        cost_model=CostModel(half_spread_bps={"AAA": 0.0, "BBB": 0.0}, default_bps=0.0, time_of_day=False),
        starting_cash=100_000.0,
    )
    mar2 = result.report.fills[result.report.fills["ts"] == pd.Timestamp("2026-03-02 09:35:00")]
    assert list(mar2["side"]) == ["SELL", "BUY"]
    assert list(mar2["symbol"]) == ["AAA", "BBB"]
    assert not (
        (result.report.fills["ts"] == pd.Timestamp("2026-03-02 09:35:00"))
        & (result.report.fills["side"] == "BUY")
        & (result.report.fills["symbol"] == "AAA")
    ).any()
