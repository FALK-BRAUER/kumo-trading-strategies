"""A verified backtest must size at ENTRY ONLY, because no live runner in this repo can rebalance.

THE HOUSE RULE, stated in `runner_verified.py:215-219` where momentum's sizing happens:

    Sizing happens at ENTRY ONLY and is never rebalanced afterwards, so a name enters at a size
    proportional to 1/sigma and then drifts. That is the implementable form -- the live runner has
    no rebalance path -- and implementability is the whole point of the Cederburg critique this is
    being tested against. It is NOT "the book is continuously held at inverse-vol weights"; do not
    read the result as that claim.

Momentum's backtest obeys it: `scale` is built over `dec.enter` only (`:226-227`), byte-identical to
`pgrunner._submit`'s `{s2: weights[s2] * len(weights) for s2 in enters ...}` (`:1195-1197`). Two
derivations of one sizing rule that AGREE, which is the point.

QC27's verified backtest does not. `runner_qc27_verified.py:164-197` builds `target_qty` for every
name in `target_weights` each rebalance and trades the difference to it -- and adds the cash proxy at
`dec.cash_proxy_weight` (`:165-166`), which no live path buys at all. So it makes exactly the claim
momentum's runner refuses to make: the book continuously held at inverse-vol weights.

That matters beyond tidiness. The daily-vs-weekly cadence decision (#63) was taken on numbers this
runner produced -- including the $23.7k of churn `live.py` says "this config pays deliberately". The
LIVE lane does not pay it, because `qc27_runner._submit` sends `d.exit` and `d.enter` and never
touches a hold. Tracked as #68.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, monthly family (#270)
)
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig

_NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
_CASH_PROXY = "GLD"


def _bars(symbol: str, closes: list[float], volume: int) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": symbol, "date": pd.bdate_range("2025-01-01", periods=len(closes)),
        "open": [closes[0], *closes[:-1]], "close": closes, "close_adj": closes,
        "volume": [volume] * len(closes),
    })


def _stable_universe(n: int = 200):
    """Six names all trending UP together at different volatilities.

    Membership is therefore near-constant while the inverse-vol weights move every session — the
    single configuration in which re-weighting churn is separable from rotation. A fixture where
    the ranking churned would confound the two and could not answer this question at all.
    """
    rng = np.random.default_rng(0)
    frames = [_bars(sym, [max(1.0, 20.0 + 0.25 * t + noise * rng.normal()) for t in range(n)],
                    5_000_000)
              for sym, noise in zip(_NAMES, (0.0, 0.3, 0.6, 0.9, 1.2, 1.5))]
    frames.append(_bars("GLD", [30.0 + 0.005 * t for t in range(n)], 10_000_000))
    bars = pd.concat(frames, ignore_index=True)
    sectors = pd.DataFrame({
        "symbol": [*_NAMES, "GLD"],
        "sector": ["TECHNOLOGY"] * len(_NAMES) + ["ETF"],
        "name": [f"{s} INC COMMON STOCK" for s in _NAMES] + ["SPDR GOLD SHARES"],
        "country": ["UNITED STATES"] * (len(_NAMES) + 1),
    })
    return bars, sectors


def _rotation_legs_and_fills():
    bars, sectors = _stable_universe()
    res = run(bars, sectors,
              cfg=QC27TechInverseVolConfig(momentum_price_field="close", portfolio_size=5),
              cost_model=CostModel(half_spread_bps={}, default_bps=0.0, time_of_day=False),
              rebalance_period="D")
    prev, rotation = None, 0
    for _, row in res.decisions.iterrows():
        hold = set(row["hold"]) if isinstance(row["hold"], (list, tuple)) else set()
        if prev is not None:
            rotation += len(hold ^ prev)          # names that entered or left
        prev = hold
    fills = res.report.fills
    # THE CASH PROXY IS EXCLUDED, and the first version of this test was wrong for including it.
    # GLD never appears in `dec.hold` — it is added separately from `cash_proxy_weight` — so its
    # trades can never be "explained by rotation" no matter how the runner sizes. Counting it made
    # the assertion unsatisfiable even by a correct entry-only runner, which a mutation bite
    # exposed: sizing held names at their existing quantity still left 191 fills unexplained, all
    # of them GLD. A strict xfail whose fixed state also fails is not a detector.
    # That the live path never buys GLD at all is a real and separate defect (#68), not this one.
    equity_fills = fills[fills["symbol"] != _CASH_PROXY]
    return rotation, len(equity_fills)


@pytest.mark.xfail(strict=True, reason=(
    "issue 68: qc27's verified backtest rebalances the whole book to target weights every "
    "session, so it trades names that never left the ranking. The live runner cannot do this. "
    "STRICT so it turns the suite red the day the runner is made entry-only — at which point the "
    "cadence numbers behind #63 have to be re-derived, not silently inherited."))
def test_the_qc27_backtest_only_trades_names_that_ROTATE():
    """Membership is near-constant here, so every fill beyond a membership change is re-weighting.

    Measured on this fixture: 159 rotation legs against 394 equity fills — **235 trades (60%) are
    re-weighting**. Sizing held names at their existing quantity instead drops it to 149, below the
    rotation count. Independently consistent with the research's own figure (7.95 legs a session
    while rotating about one name, `research/qc27/BUILD_STATE.md:220-224`), reached from a different
    direction. Two derivations agreeing about a number nobody wanted.
    """
    rotation, fills = _rotation_legs_and_fills()
    assert fills <= rotation, (
        f"{fills} equity fills against {rotation} rotation legs — {fills - rotation} trades "
        f"({100 * (fills - rotation) / max(fills, 1):.0f}%) are re-weighting the backtest can do and "
        f"the live runner cannot")


def test_the_fixture_can_actually_SEE_churn():
    """The control, and it is not optional. `fills <= rotation` also passes when the fixture produces
    no rebalances, no eligible names, or no fills — a green that would mean nothing. This asserts the
    setup is capable of showing the thing the xfail above is measuring."""
    rotation, fills = _rotation_legs_and_fills()
    assert fills > 0, "no equity fills at all — the fixture is not exercising the runner"
    assert rotation > 0, "membership never changed — rotation and re-weighting are not separable here"


def test_rebalance_holds_FALSE_actually_changes_what_the_runner_trades():
    """The knob added in 36c2326 was never positively tested (2026-08-26).

    The strict xfail above documents that the SHIPPED runner rebalances. It says nothing about
    whether the new mode WORKS — and an API knob nothing exercises is the dead-knob shape this repo
    has hit three times now (`ALLOCATED_EQUITY`, `*_SLOTS`, and nine unread `RiskLimits` fields).
    A parameter that is accepted and discarded is worse than a missing one.

    Asserted as a DIFFERENCE, not a fixed number: the point is that the flag reaches the sizing loop
    at all, and pinning an exact fill count would break on any unrelated fixture change.
    """
    bars, sectors = _stable_universe()
    cfg = QC27TechInverseVolConfig(momentum_price_field="close", portfolio_size=5)
    cm = CostModel(half_spread_bps={}, default_bps=0.0, time_of_day=False)

    on = run(bars, sectors, cfg=cfg, cost_model=cm, rebalance_period="D", rebalance_holds=True)
    off = run(bars, sectors, cfg=cfg, cost_model=cm, rebalance_period="D", rebalance_holds=False)

    n_on, n_off = len(on.report.fills), len(off.report.fills)
    assert n_off < n_on, (
        f"rebalance_holds=False traded {n_off} against {n_on} with it on — the flag is not reaching "
        f"the sizing loop, which is a knob that turns and moves nothing")


def test_the_DEFAULT_is_unchanged_so_every_existing_result_stands():
    """`rebalance_holds` defaults True. If omitting it ever stopped matching `True`, every number
    already published from this runner — including the cadence choice in #63 — would silently
    describe a different strategy."""
    bars, sectors = _stable_universe()
    cfg = QC27TechInverseVolConfig(momentum_price_field="close", portfolio_size=5)
    cm = CostModel(half_spread_bps={}, default_bps=0.0, time_of_day=False)

    implicit = run(bars, sectors, cfg=cfg, cost_model=cm, rebalance_period="D")
    explicit = run(bars, sectors, cfg=cfg, cost_model=cm, rebalance_period="D", rebalance_holds=True)
    assert len(implicit.report.fills) == len(explicit.report.fills)
