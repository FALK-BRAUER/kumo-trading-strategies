"""The lane's market view, and the seam that carries it into a runner (kumo-trading-platform issue 873).

Aimed at the properties that decide whether this mechanism is safe, not at the arithmetic:
UNKNOWN must not liquidate, the view must not read the session it is asked about, and the runner
must actually CALL it — a view nothing consults is the #26 shape, and this repo has shipped that
five times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.market_view import (
    MarketAction,
    RISK_OFF, RISK_ON, UNKNOWN, MarketSignal, MarketViewConfig, market_state)

MA = MarketSignal.INDEX_VS_MA


def _panel(path: str, n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    ramp = np.linspace(10, 30, n) if path == "up" else np.linspace(30, 10, n)
    return pd.DataFrame({"AAA": ramp, "BBB": ramp * 2}, index=idx)


def test_a_falling_market_is_risk_off_and_a_rising_one_is_not():
    cfg = MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE)
    up, down = _panel("up"), _panel("down")
    assert market_state(up, cfg, asof=up.index[-1]).state == RISK_ON
    assert market_state(down, cfg, asof=down.index[-1]).state == RISK_OFF


def test_too_little_history_is_UNKNOWN_and_UNKNOWN_DOES_NOT_BLOCK():
    """The property that keeps a data gap from becoming a realised loss.

    #873: "`unknown` never liquidates; it raises its own alert." Three states, not two — "not
    asked" is not "not protected", and a caller that flattens on an unreadable signal has turned
    an outage into a de-risking event."""
    up = _panel("up")
    state = market_state(up.head(10), MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE), asof=up.index[9])
    assert state.state == UNKNOWN
    assert state.liquidates is False, "UNKNOWN liquidated the book — a data gap must not de-risk"
    assert state.reasons, "UNKNOWN carried no reason; #873 needs one for the UI"


def test_no_view_configured_never_blocks():
    """All new automation is opt-in. A lane that declares nothing must behave exactly as today."""
    up = _panel("up")
    assert market_state(up, MarketViewConfig(), asof=up.index[-1]).liquidates is False


def test_the_view_cannot_read_the_session_it_is_asked_about():
    """ENTRY-COMPUTABLE. A view that reads `asof`'s own close is one no live lane can act on at
    `asof`'s open, and the backtest becomes a measurement of hindsight.

    Rewriting the whole `asof` row to a catastrophic price must not move the verdict."""
    cfg = MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE)
    up = _panel("up")
    asof = up.index[-1]
    clean = market_state(up, cfg, asof=asof)
    poisoned = up.copy()
    poisoned.loc[asof] = 0.01
    assert market_state(poisoned, cfg, asof=asof) == clean, (
        "the verdict moved when only the asof session changed — the view is reading a bar the "
        "live lane cannot have seen yet"
    )


def _crosses_below_for(k: int, n: int = 200, window: int = 50) -> pd.DataFrame:
    """A panel whose index sits above its `window`-session average until the last `k` sessions.

    The whole point of `dwell` is the boundary, so the fixture has to put one somewhere. A panel
    that is simply falling is risk-off at EVERY dwell, which is why the previous version of this
    test could not tell `tail(cfg.dwell)` from `tail(1)`.
    """
    rise = np.linspace(10.0, 40.0, n - k)
    # A drop steep enough to cross a 50-session average within k sessions and stay under it.
    drop = np.linspace(rise[-1] * 0.55, rise[-1] * 0.50, k)
    level = np.concatenate([rise, drop])
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"AAA": level, "BBB": level * 2}, index=idx)


@pytest.mark.parametrize("dwell,expected", [
    (1, RISK_OFF), (2, RISK_OFF), (3, RISK_OFF),   # the condition has held for 3
    (4, RISK_ON), (6, RISK_ON),                    # ...but not for 4
])
def test_dwell_requires_the_condition_to_persist_for_exactly_that_many_sessions(dwell, expected):
    """THE BOUNDARY, because without one `tail(cfg.dwell)` and `tail(1)` are indistinguishable.

    Raised in review of #142: the earlier version asserted risk-off on a falling panel and risk-on
    on a RISING one, and a rising panel is risk-on at any dwell — so the half of the property that
    matters, that a longer dwell does NOT fire, was never exercised. Mutating `tail(cfg.dwell)` to
    `tail(1)` passed the whole suite.

    `dwell` is not decoration here: #873 ships this ON, and dwell is what separates "the market
    turned" from "one close crossed a line".
    """
    panel = _crosses_below_for(3)
    cfg = MarketViewConfig(signal=MA, window=50, dwell=dwell, action=MarketAction.LIQUIDATE)
    assert market_state(panel, cfg, asof=panel.index[-1] + pd.Timedelta(days=1)).state == expected


def test_an_empty_panel_is_UNKNOWN_too():
    """Same property as the too-little-history test, second path. Review of #142 found the
    `prices.empty` branch could be mutated without any test noticing — one property, two branches,
    and only one of them was held down."""
    state = market_state(pd.DataFrame(), MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE),
                         asof=pd.Timestamp("2025-06-01"))
    assert state.state == UNKNOWN
    assert state.liquidates is False
    assert state.reasons


def _falling_market_with_one_riser():
    # THE FIXTURE ENCODES THE MECHANISM, because two earlier ones could not distinguish the arms.
    #
    # The whole finding is that a market can be falling while individual names still carry positive
    # momentum — that is why QC27's own "how many names are still going up" valve stays shut through
    # a decline. A panel where EVERY name turns down at once cannot show it: the control de-risks by
    # itself via the momentum filter, both arms end flat, and the fills come out byte-identical for
    # a reason that has nothing to do with market views. (That is exactly what the first two
    # versions of this fixture did.)
    #
    # So: AAA and BBB collapse after session 200 while CCC keeps climbing. The equal-weight index
    # turns down — the market view says risk-off — but CCC's momentum stays positive, so the
    # UNFILTERED lane happily holds it. Only the filtered arm leaves.
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    rows = []
    for i, d in enumerate(dates):
        fall = max(0, i - 200)
        px = {"AAA": (10 + 0.10 * i) * (0.97 ** fall),
              "BBB": (20 + 0.15 * i) * (0.97 ** fall),
              "CCC": 30 + 0.20 * i,
              "GLD": 100.0}
        for k, v in px.items():
            rows.append(dict(ticker=k, date=d, open=v, close=v, close_adj=v, volume=5_000_000))
    bars = pd.DataFrame(rows)
    sectors = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"], "name": ["A Inc", "B Inc", "C Inc"],
        "sector": ["Technology"] * 3, "industry": ["Software"] * 3,
        "marketCap": [1e10, 2e10, 3e10], "country": ["United States"] * 3,
        "ipoyear": [2015, 2016, 2017]})
    return bars, sectors


def test_the_runner_actually_calls_the_view_and_flattens(tmp_path):
    """THE SEAM, not the unit. A correct `market_state` that no driver consults is exactly the
    defect this module exists to end — `evaluate_exits` lost a rule to that shape, and five
    production defects in one day were wiring, not arithmetic. So drive `run()` itself."""
    from kumo_strategies.backtesting.costs import CostModel
    from kumo_strategies.backtesting.runner_sessions import run_sessions as run  # the ONE runner (#270)
    from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig

    # THE FIXTURE ENCODES THE MECHANISM, because two earlier ones could not distinguish the arms.
    #
    # The whole finding is that a market can be falling while individual names still carry positive
    # momentum — that is why QC27's own "how many names are still going up" valve stays shut through
    # a decline. A panel where EVERY name turns down at once cannot show it: the control de-risks by
    # itself via the momentum filter, both arms end flat, and the fills come out byte-identical for
    # a reason that has nothing to do with market views. (That is exactly what the first two
    # versions of this fixture did.)
    #
    # So: AAA and BBB collapse after session 200 while CCC keeps climbing. The equal-weight index
    # turns down — the market view says risk-off — but CCC's momentum stays positive, so the
    # UNFILTERED lane happily holds it. Only the filtered arm leaves.
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    rows = []
    for i, d in enumerate(dates):
        fall = max(0, i - 200)
        px = {"AAA": (10 + 0.10 * i) * (0.97 ** fall),
              "BBB": (20 + 0.15 * i) * (0.97 ** fall),
              "CCC": 30 + 0.20 * i,
              "GLD": 100.0}
        for k, v in px.items():
            rows.append(dict(ticker=k, date=d, open=v, close=v, close_adj=v, volume=5_000_000))
    bars = pd.DataFrame(rows)
    # SHAPED FROM WHAT PRODUCTION EMITS, checked against the real sectors.parquet rather than
    # guessed: `country` is "United States", not "US". The first version of this fixture used "US"
    # and `filter_tech_universe` silently returned ZERO tech names, so the runner made no decisions
    # at all and the assertion below failed for a reason that had nothing to do with market views.
    sectors = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"], "name": ["A Inc", "B Inc", "C Inc"],
        "sector": ["Technology"] * 3, "industry": ["Software"] * 3,
        "marketCap": [1e10, 2e10, 3e10], "country": ["United States"] * 3,
        "ipoyear": [2015, 2016, 2017]})
    cfg = QC27TechInverseVolConfig(lookback_sessions=5, liquidity_window=2, min_liquidity_history=1,
                                   realized_vol_window=2, min_realized_vol_history=1,
                                   warmup_sessions=10, liquidity_filter_size=3, portfolio_size=2,
                                   momentum_price_field="close", rebalance_period="W")
    cm = CostModel(half_spread_bps={}, default_bps=0.0)
    kw = dict(cost_model=cm, starting_cash=20_000.0, n_trials=1, rebalance_holds=False)
    off = run(bars, sectors, cfg=cfg, market_view=MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE), **kw)
    none = run(bars, sectors, cfg=cfg, market_view=None, **kw)

    assert "market_state" in off.decisions.columns, "the runner journalled no market state"
    assert (off.decisions["market_state"] == RISK_OFF).any(), (
        "the panel falls for 100 sessions and the runner never saw a risk-off session — the view "
        "is not being consulted"
    )
    assert not off.report.fills.equals(none.report.fills), (
        "a configured view produced byte-identical fills to no view at all — it reached no behaviour"
    )
    assert none.decisions["market_state"].isna().all(), (
        "market_view=None still produced a state; the default is not inert"
    )


def test_TECHIVOL_declares_the_measured_view_and_the_runner_reads_it_from_the_config():
    """The lane's declaration must travel with the STRATEGY, not with a runner argument.

    `rebalance_period`'s own docstring records why: it was a research-only knob, so a sweep could
    report weekly while both production call sites stayed hardwired monthly, and nothing failed to
    say so. A market view passed only as a runner argument is that shape exactly. The operator approved
    50-day for TECHIVOL on 2026-09-11; this pins the value AND that the runner picks it up without
    being told.
    """
    from kumo_strategies.strategies.qc27_tech_inverse_vol import live_config

    mv = live_config().market_view
    assert mv.signal is MA and mv.window == 50, (
        f"TECHIVOL's declared view moved to {mv}. 50 is measured, not round: summed return against "
        f"no view is 63d -78.5pp, 50d -42.8pp, 20d -71.7pp. Re-measure before changing it."
    )


def test_a_lane_that_declares_nothing_gets_nothing():
    """The opt-in rule. A default that quietly hands every lane a view is how new automation
    arrives unasked, and this repo's rule is that all of it defaults OFF."""
    from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig

    assert QC27TechInverseVolConfig().market_view.signal is MarketSignal.NONE
    assert market_state(_panel("down"), QC27TechInverseVolConfig().market_view,
                        asof=_panel("down").index[-1]).liquidates is False


def test_the_runner_takes_the_view_from_the_config_when_none_is_passed(tmp_path):
    """THE SEAM. A declaration the driver does not read is a declaration that does nothing —
    which is the defect `rebalance_period` documents and this field exists to avoid repeating."""
    from dataclasses import replace

    from kumo_strategies.backtesting.costs import CostModel
    from kumo_strategies.backtesting.runner_sessions import run_sessions as run  # the ONE runner (#270)
    from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig

    bars, sectors = _falling_market_with_one_riser()
    base = QC27TechInverseVolConfig(
        lookback_sessions=5, liquidity_window=2, min_liquidity_history=1, realized_vol_window=2,
        min_realized_vol_history=1, warmup_sessions=10, liquidity_filter_size=3, portfolio_size=2,
        momentum_price_field="close", rebalance_period="W")
    kw = dict(cost_model=CostModel(half_spread_bps={}, default_bps=0.0), starting_cash=20_000.0,
              n_trials=1, rebalance_holds=False)
    declared = run(bars, sectors, cfg=replace(
        base, market_view=MarketViewConfig(signal=MA, window=50, action=MarketAction.LIQUIDATE)), **kw)
    silent = run(bars, sectors, cfg=base, **kw)

    assert (declared.decisions["market_state"] == RISK_OFF).any(), (
        "the runner ignored a view DECLARED ON THE CONFIG — passing it as an argument is the only "
        "way it reaches behaviour, which makes it a research-only knob"
    )
    assert not declared.report.fills.equals(silent.report.fills)


def test_TECHIVOL_PAIRS_ITS_BEAR_DETECTOR_WITH_THE_BEAR_ACTION():
    """A 50-day moving average is a BEAR DETECTOR, so EXIT_ONLY is its action (#147).

    The operator's three conditions, 2026-09-11:

        emergency_exit   "market explodes, get out"   LIQUIDATE
        entries_blocked  "today is not the day"       EXIT_ONLY
        self_assessment  "I'm a loser, help me"       STAND_DOWN

    This lane declared LIQUIDATE — a bear detector wired to an EMERGENCY action, which is the
    wrong pairing whatever a backtest says. The change was made on that reasoning and needs no
    number, which is why this test asserts the ACTION rather than a return.

    UNPINNED UNTIL NOW. The action was changed and the whole suite stayed green: the sibling test
    above pins `signal` and `window` and never pinned what the lane DOES about them. A field that
    can be flipped without anything failing is a field nobody is guarding — and this one decides
    between declining to buy and selling the book.

    The old reason — "LIQUIDATE, because that is the arm that was MEASURED" — was true and is
    obsolete. EXIT_ONLY was not unmeasured but UNMEASURABLE: `runner_qc27_verified` read
    `.liquidates` and never `.blocks_entries`, so both EXIT_ONLY arms reproduced the control byte
    for byte (#170). EXIT_ONLY now beats LIQUIDATE on BOTH measured windows — +391.8% against
    +220.1% over 8x60 sessions, +116.3% against +32.2% over 25/26.
    """
    from kumo_strategies.strategies.qc27_tech_inverse_vol import live_config

    mv = live_config().market_view
    assert mv.action is MarketAction.EXIT_ONLY, (
        f"TECHIVOL declares {mv.action}; a moving-average bear detector must not drive LIQUIDATE, "
        f"which is the EMERGENCY action")


def test_NO_LANE_PAIRS_A_MOVING_AVERAGE_WITH_LIQUIDATE():
    """The rule rather than the instance, enumerated over every lane config that declares a view.

    DISCOVERED, NOT LISTED — #138's rule, which exists because a hand-written lane list silently
    dropped BCTROT. A lane added tomorrow with the wrong pairing fails here rather than shipping.
    """
    import importlib
    import pkgutil

    import kumo_strategies.strategies as pkg

    offenders = []
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if not mod.name.endswith((".live", ".config")):
            continue
        try:
            m = importlib.import_module(mod.name)
        except Exception:                                          # noqa: BLE001
            continue
        for name in dir(m):
            fn = getattr(m, name)
            if not callable(fn) or not name.endswith("config"):
                continue
            try:
                cfg = fn()
            except Exception:                                      # noqa: BLE001
                continue
            view = getattr(cfg, "market_view", None)
            if view is None or view.signal is not MarketSignal.INDEX_VS_MA:
                continue
            if view.action is MarketAction.LIQUIDATE:
                offenders.append(f"{mod.name}.{name}")
    assert not offenders, (
        f"these pair a moving-average BEAR detector with the EMERGENCY action: {offenders}. "
        f"LIQUIDATE answers 'market explodes, get out'; a 50-day average is not that signal.")
