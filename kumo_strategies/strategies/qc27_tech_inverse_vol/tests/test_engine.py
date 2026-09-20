from __future__ import annotations

import math

import pandas as pd

from kumo_strategies.strategies.qc27_tech_inverse_vol.engine import _is_common_stock_like
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    QC27TechInverseVolConfig,
    build_feature_panel,
    decide,
    filter_tech_universe,
    rebalance_dates,
    select_portfolio,
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
            "volume": [volume] * len(closes),
        }
    )


def test_filter_tech_universe_uses_snapshot_sector_and_excludes_etfs():
    bars = pd.concat(
        [
            _series("AAPL", [10.0, 11.0]),
            _series("XLK", [20.0, 21.0]),
            _series("JPM", [30.0, 31.0]),
        ],
        ignore_index=True,
    )
    sectors = pd.DataFrame(
        {
            "symbol": ["AAPL", "XLK", "JPM"],
            "name": [
                "Apple Inc. Common Stock",
                "Technology Select Sector SPDR ETF",
                "JPMorgan Chase & Co. Common Stock",
            ],
            "sector": ["Technology", "Technology", "Finance"],
            "country": ["United States", "United States", "United States"],
        }
    )
    filtered, diag = filter_tech_universe(bars, sectors, QC27TechInverseVolConfig())
    assert set(filtered["ticker"]) == {"AAPL"}
    assert diag.tech_snapshot_symbols == 1


def test_rebalance_dates_pick_first_trading_day_of_each_month():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02", "2026-02-03", "2026-03-02"])
    assert rebalance_dates(pd.Index(dates)) == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-02-02"),
        pd.Timestamp("2026-03-02"),
    ]


def test_select_portfolio_keeps_positive_names_and_leaves_residual_cash():
    panel = pd.DataFrame(
        {
            "ticker": ["FAST", "STEADY", "NEG"],
            "date": [pd.Timestamp("2026-04-01")] * 3,
            "eligible": [True, True, True],
            "positive_momentum": [True, True, False],
            "liquidity_proxy": [300.0, 200.0, 500.0],
            "momentum": [0.30, 0.10, -0.05],
            "realized_volatility": [0.30, 0.10, 0.20],
        }
    )
    cfg = QC27TechInverseVolConfig(liquidity_filter_size=3, portfolio_size=4)
    names, scores, weights, cash_proxy_weight = select_portfolio(panel, cfg)
    assert names == ["FAST", "STEADY"]
    assert "NEG" not in scores
    assert math.isclose(sum(weights.values()), 0.5)
    assert math.isclose(cash_proxy_weight, 0.5)
    assert weights["STEADY"] > weights["FAST"]


def test_build_feature_panel_marks_positive_momentum_and_eligibility():
    # 120 bars, not 90. The original fixture used 90 and passed only BECAUSE `warmup_sessions` was
    # unenforced -- momentum needs lookback+2 = 65, so 90 was plenty for the features while being
    # short of QC27's stated 100-session warmup. Once the warmup gate went in, this went red, which
    # is the fixture doing its job: it had encoded the defect.
    bars = _series("AAPL", [10.0 + i for i in range(120)], volume=2_000_000)
    filtered, snapshot_diag = filter_tech_universe(
        bars,
        pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "name": ["Apple Inc. Common Stock"],
                "sector": ["Technology"],
                "country": ["United States"],
            }
        ),
        QC27TechInverseVolConfig(),
    )
    panel, diag = build_feature_panel(filtered, QC27TechInverseVolConfig(), snapshot_diag)
    tail = panel.iloc[-1]
    assert bool(tail["eligible"])
    assert bool(tail["positive_momentum"])
    assert diag.eligible_name_days > 0


def test_decide_preserves_rank_order_and_cash_proxy():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [pd.Timestamp("2026-04-01")] * 2,
            "eligible": [True, True],
            "positive_momentum": [True, True],
            "liquidity_proxy": [100.0, 99.0],
            "momentum": [0.9, 0.7],
            "realized_volatility": [0.4, 0.2],
        }
    )
    cfg = QC27TechInverseVolConfig(liquidity_filter_size=2, portfolio_size=4)
    decision = decide(panel, cfg, held={"OLD"})
    assert decision.hold == ("AAA", "BBB")
    assert decision.enter == ("AAA", "BBB")
    assert decision.exit == ("OLD",)
    assert decision.weights["BBB"] > decision.weights["AAA"]
    assert math.isclose(decision.cash_proxy_weight, 0.5)


# -- warmup enforcement (#33) ---------------------------------------------------------------------
def _tech_snapshot(sym: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame({"symbol": [sym], "name": [f"{sym} Inc. Common Stock"],
                         "sector": ["Technology"], "country": ["United States"]})


def _panel_with(n_bars: int, cfg: QC27TechInverseVolConfig | None = None):
    cfg = cfg or QC27TechInverseVolConfig()
    bars = _series("AAPL", [10.0 + i for i in range(n_bars)], volume=2_000_000)
    filtered, diag = filter_tech_universe(bars, _tech_snapshot(), cfg)
    return build_feature_panel(filtered, cfg, diag)[0], cfg


def test_a_name_is_not_eligible_before_the_stated_warmup():
    """THE DEFECT THIS PINS. `warmup_sessions` carries QC27's own 100-session warmup and was
    declared but never read — `grep -rn warmup_sessions` returned only its definition. So the
    backtest started trading once `momentum` stopped being NaN at ~65 sessions and rebalanced
    2025-05-01 on roughly 82 sessions of history, a month early.

    Found by diffing the pandas runner against the Nautilus adapter, which DOES enforce it. 80 bars
    is enough for every feature and short of the warmup, so it is exactly the window where the old
    behaviour traded and the new one must not."""
    panel, cfg = _panel_with(80)
    tail = panel.iloc[-1]
    assert pd.notna(tail["momentum"]), "fixture is too short to isolate warmup from feature NaNs"
    assert not bool(tail["eligible"]), "traded before the stated warmup — the gate is not enforced"


def test_a_name_becomes_eligible_once_the_warmup_is_served():
    panel, _ = _panel_with(120)
    assert bool(panel.iloc[-1]["eligible"])


def test_the_warmup_boundary_is_exactly_warmup_sessions():
    """Off-by-one matters: it decides whether the first live rebalance is a month early."""
    cfg = QC27TechInverseVolConfig()
    panel, _ = _panel_with(cfg.warmup_sessions + 1, cfg)
    seen = panel["sessions_seen"].to_numpy()
    warm = panel["warm"].to_numpy()
    assert not warm[cfg.warmup_sessions - 1], "warm one session too early"
    assert warm[cfg.warmup_sessions], "not warm at the stated warmup"
    assert seen[cfg.warmup_sessions] == cfg.warmup_sessions


def test_warmup_is_per_name_not_per_panel():
    """A name added to the universe later must serve its OWN warmup. Counting panel rows instead of
    per-ticker rows would let a fresh listing inherit an established name's history."""
    cfg = QC27TechInverseVolConfig()
    old = _series("AAPL", [10.0 + i for i in range(150)], volume=2_000_000)
    new = _series("NEWCO", [10.0 + i for i in range(40)], volume=2_000_000)
    snap = pd.DataFrame({"symbol": ["AAPL", "NEWCO"],
                         "name": ["Apple Inc. Common Stock", "Newco Inc. Common Stock"],
                         "sector": ["Technology", "Technology"],
                         "country": ["United States", "United States"]})
    filtered, diag = filter_tech_universe(pd.concat([old, new], ignore_index=True), snap, cfg)
    panel = build_feature_panel(filtered, cfg, diag)[0]
    last = panel.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    assert bool(last.loc["AAPL", "eligible"]), "established name should be warm"
    assert not bool(last.loc["NEWCO", "eligible"]), "new listing inherited another name's warmup"


# -- a NAME THAT IS NOT A STRING is not a name -----------------------------------------------------
def test_a_NaN_name_does_not_crash_the_filter():
    """`(name or "").upper()` does NOT guard a NaN, because **NaN is truthy**.

    `float("nan") or ""` evaluates to the NaN, and `.upper()` on it raises
    `AttributeError: 'float' object has no attribute 'upper'`. A missing name arrives as NaN and not
    as None whenever the frame came through pandas — a left join that did not match, a CSV with an
    empty cell, or a symbol the name source does not carry.

    Found 2026-08-24 by kumo-trading-platform, which crashed QC345 exactly this way by feeding it the
    107-name momentum pool: BLLLN, GTLAB, IQVIA, JEPO and OVVI are not names Alpaca knows, so they
    came back with NaN names. Production has never seen it because production passes each lane its
    own universe — the defect is invisible until a component is run against data it was not built
    for.
    """
    import math
    assert _is_common_stock_like(math.nan) is False


def test_a_MISSING_name_is_EXCLUDED_rather_than_admitted():
    """THE DIRECTION MATTERS MORE THAN THE CRASH.

    `True` from this predicate means KEEP. With the naive repair — coerce NaN to "" — an unknown
    name has no markers in it, so `not any(...)` is True and the instrument is ADMITTED to the
    tradeable universe. The crash was hiding an admit-the-unknown path, and fixing only the exception
    would have shipped that path silently.

    A filter whose job is "is this a common stock" cannot answer yes about an instrument whose name
    it does not have. Missing means EXCLUDE.
    """
    import math
    for missing in (math.nan, None, float("nan"), 3.7):
        assert _is_common_stock_like(missing) is False, f"admitted an instrument named {missing!r}"


def test_a_REAL_name_still_passes_and_a_marked_one_still_fails():
    """The control. Excluding everything would satisfy the two tests above."""
    assert _is_common_stock_like("SNOWFLAKE INC. CLASS A COMMON STOCK") is True
    assert _is_common_stock_like("SPDR GOLD SHARES ETF") is False
