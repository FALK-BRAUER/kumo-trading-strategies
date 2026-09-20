"""What this lane must never do, stated as tests.

Each assertion here was watched failing against a deliberately broken engine before being kept --
a test that has only ever been green is an assertion about nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kumo_strategies.strategies.smhgld_sleeve import (
    SmhGldSleeveConfig, build_feature_panel, current_weights, decide, rebalance_dates)

CFG = SmhGldSleeveConfig()


def bars(sessions: int = 40, smh: float = 500.0, gld: float = 400.0,
         drift: float = 1.002) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=sessions)
    return pd.DataFrame([
        {"date": date, "ticker": ticker, "close": start * drift ** step}
        for ticker, start in (("SMH", smh), ("GLD", gld))
        for step, date in enumerate(dates)])


def last_day(frame: pd.DataFrame, cfg: SmhGldSleeveConfig = CFG) -> pd.DataFrame:
    panel = build_feature_panel(frame, cfg)
    return panel[panel["date"] == panel["date"].max()]


def shares_at(weights: dict[str, float], prices: dict[str, float],
              equity: float = 100_000.0) -> dict[str, float]:
    return {sym: weight * equity / prices[sym] for sym, weight in weights.items()}


def prices_from(day: pd.DataFrame) -> dict[str, float]:
    return {row.ticker: float(row.asof_close) for row in day.itertuples()}


def test_the_decision_never_uses_the_session_being_traded():
    """The one defect that silently inflates every number this lane reports.

    The switching lane this replaces built a weekly signal with `resample().last()` and applied it
    across its own week, which turned a Sharpe of 0.78 into 2.57. Nothing here forecasts anything,
    so a lookahead would produce no absurd result to notice -- it would just quietly be wrong.
    """
    frame = bars()
    panel = build_feature_panel(frame, CFG)
    closes = frame.pivot(index="date", columns="ticker", values="close")

    for row in panel[panel["asof_close"].notna()].itertuples():
        assert row.asof_close == pytest.approx(
            closes[row.ticker].shift(1).loc[row.date]), (
            f"{row.ticker} on {row.date.date()} priced off its own session's close")


def test_a_sleeve_on_target_does_not_trade():
    """The band is the whole cost control. A lane that rebalances anyway turns 3 trades a year
    into 252, and at 10bp a side that is the difference between 46% CAGR and materially less."""
    day = last_day(bars())
    prices = prices_from(day)
    decision = decide(day, CFG, set(CFG.universe), shares_at(CFG.target_weights, prices))

    assert decision.regime == "on_target"
    assert decision.enter == ()
    assert decision.exit == ()


def test_a_drifted_sleeve_rebalances_without_exiting_either_leg():
    """A resize is not an exit. Reporting one as exit-then-enter makes the journal show a lane
    flattening and reopening its entire book every time it drifts a few points -- and would invite
    a runner to actually do that."""
    day = last_day(bars())
    prices = prices_from(day)
    drifted = shares_at({"SMH": 0.62, "GLD": 0.38}, prices)
    decision = decide(day, CFG, set(CFG.universe), drifted)

    assert decision.regime == "rebalance"
    assert decision.exit == (), "a resize must never be expressed as an exit"
    assert set(decision.hold) == set(CFG.universe)
    assert decision.weights == CFG.target_weights


def test_drift_is_measured_against_the_sleeve_not_the_account():
    """TECHIVOL-005 proposed exiting eight names belonging to two other strategies on its first
    live session. A sleeve that measures its weights against the whole book rebalances itself
    every time an unrelated lane opens a position."""
    prices = {"SMH": 500.0, "GLD": 400.0}
    sleeve = shares_at(CFG.target_weights, prices)

    weights = current_weights(sleeve, prices)
    assert weights["SMH"] == pytest.approx(CFG.risk_weight)

    # An unrelated holding is simply not passed in; if it leaks in, the weights move.
    polluted = current_weights({**sleeve, "SMH": sleeve["SMH"] * 3}, prices)
    assert polluted["SMH"] != pytest.approx(CFG.risk_weight)


def test_an_uncomputable_drift_holds_rather_than_trading():
    """A gap in the feed is not evidence the weights moved. Rotating on it makes every data
    outage a trade."""
    day = last_day(bars(sessions=2))
    decision = decide(day, CFG, set(CFG.universe), {"SMH": 1.0, "GLD": 1.0})

    assert decision.regime == "unknown"
    assert decision.enter == ()
    assert decision.exit == ()


def test_an_empty_sleeve_opens_both_legs():
    day = last_day(bars())
    decision = decide(day, CFG, set())

    assert decision.regime == "opening"
    assert set(decision.enter) == set(CFG.universe)


def test_a_missing_leg_is_entered_even_inside_the_band():
    """Holding one leg at 100% is inside no band worth the name -- but the drift arithmetic can
    report a small number when the absent leg's weight is simply missing from the book."""
    day = last_day(bars())
    prices = prices_from(day)
    decision = decide(day, CFG, {"SMH"}, {"SMH": 100_000.0 / prices["SMH"]})

    assert decision.regime == "rebalance"
    assert decision.enter == ("GLD",)


def test_rebalance_dates_take_the_first_session_of_a_bucket():
    """Last-of-bucket is the look-ahead. First-of-bucket, decided on the prior close, is
    computable live."""
    dates = pd.bdate_range("2026-01-01", periods=40)
    weekly = rebalance_dates(dates, "W")

    for stamp in weekly:
        week = [d for d in dates if d.to_period("W") == stamp.to_period("W")]
        assert stamp == min(week)


def test_missing_a_leg_entirely_is_refused_not_degraded():
    frame = bars()
    with pytest.raises(ValueError, match="concentrated bet"):
        build_feature_panel(frame[frame["ticker"] == "SMH"], CFG)


def test_a_rebalance_is_not_expressible_as_enter_or_exit():
    """THE DEPLOYMENT BLOCKER, pinned as a test so it cannot be forgotten.

    Every other lane in this repo trades by entering and exiting names, and the runtime consumes
    exactly that vocabulary: `for sym in d.exit: self._close(sym)` then `for sym in d.enter:
    self._open(sym, ...)`. This lane's only trade is a RESIZE of two positions it already holds, so
    a rebalance decision carries an empty `enter` and an empty `exit` and a runner written to that
    vocabulary does nothing at all -- the sleeve would drift without limit while the journal showed
    a lane deciding every session.

    The fix is a resize path that reads `weights`, not a change here: expressing a resize as
    exit-then-enter would flatten and reopen the whole sleeve about 93 times a year.
    """
    day = last_day(bars())
    prices = prices_from(day)
    drifted = shares_at({"SMH": 0.40, "GLD": 0.60}, prices)
    decision = decide(day, CFG, set(CFG.universe), drifted)

    assert decision.regime == "rebalance"
    assert decision.enter == () and decision.exit == (), (
        "if this starts failing the decision schema changed; re-check the runtime resize path")
    assert decision.weights == CFG.target_weights, (
        "weights are the ONLY expression of this lane's trade; a runner that ignores them "
        "executes nothing")
