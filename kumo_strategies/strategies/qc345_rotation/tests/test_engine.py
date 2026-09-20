from __future__ import annotations

import math

import pandas as pd

from kumo_strategies.strategies.momentum_rotation.candidates import build as build_source
from kumo_strategies.strategies.qc345_rotation.engine import is_fundamental_like_asset
from kumo_strategies.strategies.qc345_rotation import (
    QC345ComputedSource,
    QC345RotationConfig,
    build_feature_panel,
    decide,
    filter_asset_universe,
    preserving_preselection_thresholds,
    preselection_bounds,
    rebalance_dates,
    select_universe,
    select_portfolio,
    terminal_buckets,
)


def _series(
    ticker: str,
    closes: list[float],
    *,
    start: str = "2025-01-01",
    volume: int = 1_000_000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range(start, periods=len(closes)),
            "open": closes,
            "close": closes,
            "close_adj": closes,
            "close_split": closes,
            "close_split_dividend": closes,
            "volume": [volume] * len(closes),
        }
    )


def test_fundamental_like_filter_removes_funds_and_products():
    bars = pd.concat(
        [
            _series("PLTR", [10.0, 10.0]),
            _series("SQQQ", [20.0, 20.0]),
            _series("DELL", [30.0, 30.0]),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": ["PLTR", "SQQQ", "DELL"],
            "name": [
                "Palantir Technologies Inc. Class A Common Stock",
                "ProShares UltraPro Short QQQ",
                "Dell Technologies Inc.",
            ],
            "exchange": ["NASDAQ", "NASDAQ", "NYSE"],
        }
    )
    filtered = filter_asset_universe(bars, assets, QC345RotationConfig())
    assert set(filtered["ticker"]) == {"DELL", "PLTR"}


def test_fundamental_like_filter_removes_live_alpaca_fund_and_product_survivors():
    bars = pd.concat(
        [
            _series("AAPL", [10.0, 10.0]),
            _series("NMI", [20.0, 20.0]),
            _series("OXLCN", [30.0, 30.0]),
            _series("BBDC", [40.0, 40.0]),
            _series("VGASW", [50.0, 50.0]),
            _series("OXLC", [60.0, 60.0]),
            _series("MER.PRK", [70.0, 70.0]),
            _series("XELLL", [80.0, 80.0]),
            _series("MTAKU", [90.0, 90.0]),
            _series("EURKR", [100.0, 100.0]),
            _series("DYNCU", [105.0, 105.0]),
            _series("AXINR", [106.0, 106.0]),
            _series("LEO", [107.0, 107.0]),
            _series("GJP", [108.0, 108.0]),
            _series("GHI", [109.0, 109.0]),
            _series("ATIIU", [109.5, 109.5]),
            _series("AMPGZ", [109.6, 109.6]),
            _series("PCTTU", [109.7, 109.7]),
            _series("MANU", [110.0, 110.0]),
            _series("U", [120.0, 120.0]),
            _series("O", [130.0, 130.0]),
            _series("RLX", [140.0, 140.0]),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": [
                "AAPL",
                "NMI",
                "OXLCN",
                "BBDC",
                "VGASW",
                "OXLC",
                "MER.PRK",
                "XELLL",
                "MTAKU",
                "EURKR",
                "DYNCU",
                "AXINR",
                "LEO",
                "GJP",
                "GHI",
                "ATIIU",
                "AMPGZ",
                "PCTTU",
                "MANU",
                "U",
                "O",
                "RLX",
            ],
            "name": [
                "Apple Inc. Common Stock",
                "Nuveen Municipal Income",
                "Oxford Lane Capital Corp. 7.125% Series 2029 Term Preferred Stock",
                "Barings BDC, Inc.",
                "Verde Clean Fuels, Inc. Warrant",
                "Oxford Lane Capital Corp. Common Stock",
                "Bank of America Corporation Income Capital Obligation Notes initially due December 15, 2066",
                "Xcel Energy Inc. 6.25% Junior Subordinated Notes, Series due 2085",
                "Market Technology Acquisition Corp Unit",
                "Eureka Acquisition Corp Right",
                "Dynamix Corporation Unit",
                "Axiom Intelligence Acquisition Corp 1 Right",
                "BNY Mellon Strategic Municipals, Inc.",
                "Synthetic Fixed Income Securities on the behalf of STRATS for Dominion Resources Series 2005-6",
                "Greystone Housing Impact Investors LP Beneficial Unit Certificates representing assignments of limited partnership interests",
                "Archimedes Tech SPAC Partners II Co. Unit",
                "Amplitech Group, Inc. Series B Right",
                "PureCycle Technologies, Inc. Unit",
                "MANCHESTER UNITED PLC",
                "Unity Software Inc.",
                "Realty Income Corporation",
                "RLX Technology Inc. American Depositary Shares, each representing the right to receive one (1) Class A ordinary share",
            ],
            "exchange": [
                "NASDAQ",
                "NYSE",
                "NASDAQ",
                "NYSE",
                "NASDAQ",
                "NASDAQ",
                "NYSE",
                "NASDAQ",
                "NASDAQ",
                "NASDAQ",
                "NASDAQ",
                "NASDAQ",
                "NYSE",
                "NYSE",
                "NYSE",
                "NASDAQ",
                "NASDAQ",
                "NASDAQ",
                "NYSE",
                "NYSE",
                "NYSE",
                "NYSE",
            ],
        }
    )
    filtered = filter_asset_universe(bars, assets, QC345RotationConfig())
    assert set(filtered["ticker"]) == {"AAPL", "MANU", "O", "RLX", "U"}


def test_rebalance_dates_pick_first_trading_day_of_each_month():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02", "2026-02-03", "2026-03-02"])
    assert rebalance_dates(pd.Index(dates)) == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-02-02"),
        pd.Timestamp("2026-03-02"),
    ]


def test_split_block_is_trailing_and_point_in_time():
    prices = [100.0] * 30 + [20.0] * 40
    panel, diag = build_feature_panel(
        _series("AAA", prices),
        QC345RotationConfig(momentum_price_field="close_split_dividend", lookback_sessions=5),
    )
    split_day = panel.iloc[31]
    assert bool(split_day["split_blocked"])
    assert diag.split_blocked_name_months >= 1
    assert diag.mania_blocked_name_months == 0


def test_mania_guard_blocks_extreme_momentum_and_volatility():
    closes = [10.0] * 12 + [11.0, 9.0] * 14
    panel, diag = build_feature_panel(
        _series("HOT", closes, volume=5_000_000),
        QC345RotationConfig(
            lookback_sessions=5,
            momentum_price_field="close",
            realized_vol_window=10,
            min_realized_vol_history=5,
            mania_momentum_threshold=0.05,
            mania_volatility_threshold=0.09,
        ),
    )
    tail = panel.iloc[-1]
    assert bool(tail["mania_blocked"])
    assert not bool(tail["eligible"])
    assert diag.mania_blocked_name_months >= 1


def test_raw_momentum_requires_split_window_covering_lookback():
    try:
        build_feature_panel(
            _series("AAA", [100.0] * 260),
            QC345RotationConfig(
                momentum_price_field="close", lookback_sessions=252, corporate_action_window=41
            ),
        )
    except ValueError as exc:
        assert "requires corporate_action_window >= lookback_sessions" in str(exc)
    else:
        raise AssertionError("expected raw momentum config to be rejected")


def test_select_universe_prefers_price_x_dv_large_names():
    data = pd.DataFrame(
        {
            "ticker": ["MEGA", "MID", "SPEC"],
            "date": [pd.Timestamp("2026-01-02")] * 3,
            "eligible": [True] * 3,
            "asof_close": [500.0, 300.0, 10.0],
            "liquidity_proxy": [100.0, 95.0, 110.0],
            "market_cap_proxy": [100.0, 95.0, 110.0],
            "market_cap_proxy_price_x_dv": [50_000.0, 28_500.0, 1_100.0],
            "momentum": [0.2, 0.6, 0.9],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=2, portfolio_size=2)
    assert select_universe(data, cfg) == ["MEGA", "MID"]


def test_preselection_bounds_measure_the_smallest_observed_preserving_floors():
    data = pd.DataFrame(
        {
            "ticker": ["MEGA", "MID", "SPEC", "LOW"],
            "date": [pd.Timestamp("2026-02-02")] * 4,
            "eligible": [True] * 4,
            "asof_close": [500.0, 300.0, 10.0, 8.0],
            "liquidity_proxy": [100.0, 95.0, 110.0, 20.0],
            "market_cap_proxy": [100.0, 95.0, 110.0, 20.0],
            "market_cap_proxy_price_x_dv": [50_000.0, 28_500.0, 1_100.0, 160.0],
            "momentum": [0.2, 0.6, 0.9, 1.0],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=2, portfolio_size=2)

    bounds = preselection_bounds(data, cfg)
    floors = preserving_preselection_thresholds(bounds)

    assert bounds.loc[0, "rankable_names"] == 4
    assert bounds.loc[0, "liquidity_candidates"] == 3
    assert bounds.loc[0, "selected_names"] == 2
    assert floors == {
        "price_floor": 300.0,
        "liquidity_proxy_floor": 95.0,
        "market_cap_proxy_price_x_dv_floor": 28_500.0,
    }


def test_select_portfolio_ranks_only_inside_the_source_universe():
    data = pd.DataFrame(
        {
            "ticker": ["MEGA", "MID", "SPEC"],
            "date": [pd.Timestamp("2026-01-02")] * 3,
            "eligible": [True] * 3,
            "asof_close": [500.0, 300.0, 10.0],
            "liquidity_proxy": [100.0, 95.0, 110.0],
            "market_cap_proxy": [100.0, 95.0, 110.0],
            "market_cap_proxy_price_x_dv": [50_000.0, 28_500.0, 1_100.0],
            "momentum": [0.2, 0.6, 0.9],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=2, portfolio_size=2)
    names, scores = select_portfolio(data, cfg, universe={"MEGA", "MID"})
    assert names == ["MID", "MEGA"]
    assert "SPEC" not in scores


def test_decide_preserves_rank_order_and_exits_removed_names():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": [pd.Timestamp("2026-04-01")] * 3,
            "eligible": [True] * 3,
            "liquidity_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy_price_x_dv": [1000.0, 990.0, 980.0],
            "momentum": [0.9, 0.8, 0.7],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=3, portfolio_size=2)
    decision = decide(panel, cfg, held={"CCC"})
    assert decision.hold == ("AAA", "BBB")
    assert decision.enter == ("AAA", "BBB")
    assert decision.exit == ("CCC",)
    assert math.isclose(decision.scores["AAA"], 0.9)


def test_decide_does_not_sell_a_held_name_solely_because_the_source_drops_it():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": [pd.Timestamp("2026-04-01")] * 3,
            "eligible": [True] * 3,
            "asof_close": [10.0, 10.0, 10.0],
            "liquidity_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy_price_x_dv": [1000.0, 990.0, 980.0],
            "momentum": [0.9, 0.8, 0.7],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=3, portfolio_size=2)
    decision = decide(panel, cfg, held={"AAA"}, universe={"BBB", "CCC"})
    assert decision.hold == ("AAA", "BBB")
    assert decision.enter == ("BBB",)
    assert decision.exit == ()


def test_decide_can_exit_a_source_dropped_holding_when_the_strategy_replaces_it():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": [pd.Timestamp("2026-04-01")] * 3,
            "eligible": [True] * 3,
            "asof_close": [10.0, 10.0, 10.0],
            "liquidity_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy": [100.0, 99.0, 98.0],
            "market_cap_proxy_price_x_dv": [1000.0, 990.0, 980.0],
            "momentum": [0.1, 0.9, 0.8],
        }
    )
    cfg = QC345RotationConfig(liquidity_filter_size=3, universe_size=3, portfolio_size=2)
    decision = decide(panel, cfg, held={"AAA"}, universe={"BBB", "CCC"})
    assert decision.hold == ("BBB", "CCC")
    assert decision.enter == ("BBB", "CCC")
    assert decision.exit == ("AAA",)


def test_computed_source_matches_point_in_time_truncated_history():
    bars = pd.concat(
        [
            _series(
                "AAA", [10.0, 10.2, 10.4, 10.7, 11.0, 11.2], start="2026-01-01", volume=5_000_000
            ),
            _series(
                "BBB", [20.0, 20.1, 20.3, 20.4, 20.6, 20.8], start="2026-01-01", volume=3_000_000
            ),
            _series("CCC", [5.0, 5.1, 5.3, 5.2, 5.4, 5.6], start="2026-01-01", volume=1_000_000),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["AAA Inc.", "BBB Inc.", "CCC Inc."],
            "exchange": ["NASDAQ", "NYSE", "NASDAQ"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=2,
        liquidity_window=2,
        min_liquidity_history=2,
        realized_vol_window=2,
        min_realized_vol_history=2,
        liquidity_filter_size=3,
        universe_size=2,
        portfolio_size=1,
        asset_universe_mode="fundamental_like",
        market_cap_mode="price_x_dv",
        momentum_price_field="close_split_dividend",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )
    full = QC345ComputedSource(bars, assets, cfg)
    for day in [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08")]:
        truncated = QC345ComputedSource(bars.loc[bars["date"] <= day].copy(), assets, cfg)
        assert full.eligible(day) == truncated.eligible(day)


def test_computed_source_uses_the_shared_candidate_source_registry():
    bars = pd.concat(
        [
            _series("AAA", [10.0, 11.0, 12.0, 13.0, 14.0], start="2026-02-26", volume=5_000_000),
            _series("BBB", [10.0, 10.5, 11.0, 11.5, 12.0], start="2026-02-26", volume=4_000_000),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["AAA Inc.", "BBB Inc."],
            "exchange": ["NASDAQ", "NYSE"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        liquidity_filter_size=1,
        universe_size=1,
        portfolio_size=1,
        asset_universe_mode="fundamental_like",
        market_cap_mode="skip",
        momentum_price_field="close_split_dividend",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )

    source = build_source("qc345_computed", bars=bars, assets=assets, cfg=cfg)

    assert isinstance(source, QC345ComputedSource)
    assert source.eligible(pd.Timestamp("2026-03-02")) == {"AAA"}


def test_computed_source_carries_the_last_monthly_scan_forward():
    bars = pd.concat(
        [
            _series("AAA", [10.0, 11.0, 12.0, 13.0, 14.0], start="2026-02-26", volume=5_000_000),
            _series("BBB", [10.0, 10.1, 10.2, 10.3, 10.4], start="2026-02-26", volume=4_000_000),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["AAA Inc.", "BBB Inc."],
            "exchange": ["NASDAQ", "NYSE"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        liquidity_filter_size=1,
        universe_size=1,
        portfolio_size=1,
        asset_universe_mode="fundamental_like",
        market_cap_mode="skip",
        momentum_price_field="close_split_dividend",
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )
    source = QC345ComputedSource(bars, assets, cfg)
    assert source.eligible(pd.Timestamp("2026-03-02")) == {"AAA"}
    assert source.eligible(pd.Timestamp("2026-03-03")) == {"AAA"}


def test_computed_source_top50_lookup_matches_same_panel_selection_live_config():
    dates = pd.bdate_range("2026-01-26", periods=8)
    frames = []
    assets = []
    for idx in range(60):
        symbol = f"S{idx:02d}"
        base = 10.0 + idx
        volume = 1_000_000 + idx * 10_000
        frames.append(
            _series(
                symbol,
                [base + step * 0.1 for step in range(len(dates))],
                start="2026-01-26",
                volume=volume,
            )
        )
        assets.append({"symbol": symbol, "name": f"{symbol} Inc.", "exchange": "NASDAQ"})
    bars = pd.concat(frames, ignore_index=True)
    assets_df = pd.DataFrame(assets)
    cfg = QC345RotationConfig(
        lookback_sessions=2,
        liquidity_window=2,
        min_liquidity_history=2,
        realized_vol_window=2,
        min_realized_vol_history=2,
        liquidity_filter_size=60,
        universe_size=50,
        portfolio_size=5,
        asset_universe_mode="fundamental_like",
        market_cap_mode="price_x_dv",
        momentum_price_field="close",
        corporate_action_window=2,
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )

    source = QC345ComputedSource(bars, assets_df, cfg)
    rebalance = pd.Timestamp("2026-02-02")
    feature_day = source.panel.loc[source.panel["date"] == rebalance].copy()
    expected = set(select_universe(feature_day, cfg))

    assert len(expected) == 50
    assert source.eligible(rebalance) == expected


def test_computed_source_exposes_preselection_bounds():
    bars = pd.concat(
        [
            _series(
                "MEGA", [500.0, 501.0, 502.0, 503.0, 504.0], start="2026-01-29", volume=10_000_000
            ),
            _series(
                "MID", [300.0, 301.0, 302.0, 303.0, 304.0], start="2026-01-29", volume=9_000_000
            ),
            _series("SPEC", [10.0, 11.0, 12.0, 13.0, 14.0], start="2026-01-29", volume=11_000_000),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        {
            "symbol": ["MEGA", "MID", "SPEC"],
            "name": ["Mega Inc.", "Mid Inc.", "Spec Inc."],
            "exchange": ["NYSE", "NASDAQ", "NASDAQ"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        liquidity_filter_size=3,
        universe_size=2,
        portfolio_size=1,
        momentum_price_field="close",
        corporate_action_window=1,
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )
    source = QC345ComputedSource(bars, assets, cfg)
    row = source.preselection_bounds().loc[lambda df: df["selected_names"].gt(0)].iloc[0]

    assert row["liquidity_candidates"] == 3
    assert row["selected_names"] == 2
    assert row["min_selected_price"] >= 300.0


def test_terminal_buckets_and_source_terminal_symbols_signal_inactive_stopped_names():
    dates = pd.bdate_range("2026-01-01", periods=5)
    bankrupt = _series("DEADQ", [10.0, 7.0, 2.0], start="2026-01-01")
    alive = _series("LIVE", [20.0, 20.5, 21.0, 21.5, 22.0], start="2026-01-01")
    bars = pd.concat([bankrupt, alive], ignore_index=True)
    assets = pd.DataFrame(
        {
            "symbol": ["DEADQ", "LIVE"],
            "name": ["Dead Corp.", "Live Corp."],
            "exchange": ["NASDAQ", "NYSE"],
            "status": ["inactive", "active"],
        }
    )
    cfg = QC345RotationConfig(
        lookback_sessions=1,
        liquidity_window=1,
        min_liquidity_history=1,
        realized_vol_window=1,
        min_realized_vol_history=1,
        liquidity_filter_size=2,
        universe_size=2,
        portfolio_size=1,
        mania_momentum_threshold=None,
        mania_volatility_threshold=None,
    )

    assert terminal_buckets(bars, assets)["DEADQ"] == "bankruptcy"
    source = QC345ComputedSource(bars.loc[bars["date"].isin(dates)], assets, cfg)

    assert source.terminal_symbols(pd.Timestamp("2026-01-03"), {"DEADQ", "LIVE"}) == {}
    assert source.terminal_symbols(pd.Timestamp("2026-01-07"), {"DEADQ", "LIVE"}) == {
        "DEADQ": "bankruptcy"
    }


# -- a NAME THAT IS NOT A STRING is not a name -----------------------------------------------------
def test_a_NaN_name_does_not_crash_the_asset_filter():
    """`(name or "").upper()` does NOT guard a NaN — **NaN is truthy**, so `nan or ""` is the NaN and
    `.upper()` raises `AttributeError: 'float' object has no attribute 'upper'`.

    Reproduced by kumo-trading-platform 2026-08-24 by feeding QC345 the 107-name momentum pool: BLLLN, GTLAB,
    IQVIA, JEPO and OVVI are not names Alpaca knows and came back as NaN. Production passes each lane
    its own universe, so it has never seen this — the defect is invisible until the component is run
    against data it was not built for.

    Same defect, same line, in `qc27_tech_inverse_vol.engine._is_common_stock_like`. One repair, two
    engines, because a second copy is a second bug.
    """
    import math
    assert is_fundamental_like_asset(math.nan, "XNAS") is False


def test_a_MISSING_name_is_EXCLUDED_rather_than_admitted():
    """THE DIRECTION MATTERS MORE THAN THE CRASH. `True` here means KEEP.

    Coercing NaN to "" — the naive repair — finds no markers in the empty string and falls through
    every rejection to `return True`, ADMITTING an instrument whose name we do not have into the
    tradeable universe. The exception was hiding an admit-the-unknown path.
    """
    import math
    for missing in (math.nan, None, 3.7):
        assert is_fundamental_like_asset(missing, "XNAS") is False, f"admitted {missing!r}"


def test_a_MISSING_EXCHANGE_is_still_tolerated_because_it_is_not_the_identity():
    """Deliberately NOT symmetric with the name. The exchange is used to REJECT (`venue == "ARCA"`),
    so a missing one cannot admit anything on its own — and rejecting every asset whose exchange is
    absent would shrink the universe for a field this predicate only reads negatively."""
    assert is_fundamental_like_asset("SNOWFLAKE INC CLASS A COMMON STOCK", None) is True
    import math
    assert is_fundamental_like_asset("SNOWFLAKE INC CLASS A COMMON STOCK", math.nan) is True


def test_a_REAL_asset_still_passes_and_an_ARCA_one_still_fails():
    """The control. Excluding everything would satisfy the tests above."""
    assert is_fundamental_like_asset("SNOWFLAKE INC CLASS A COMMON STOCK", "XNAS") is True
    assert is_fundamental_like_asset("SNOWFLAKE INC CLASS A COMMON STOCK", "ARCA") is False
