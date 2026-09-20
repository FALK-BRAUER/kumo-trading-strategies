"""#10 rule 2 — the backtest must not read bars that had not completed yet.

The ticket asks for exactly one check and calls it the strongest available:

    A classifier given an `as_of` must produce identical output whether the dataset contains
    future bars or has been truncated at `as_of`. This makes look-ahead detectable rather than
    reasoned about.

That is what this file does, end to end rather than per classifier: run the whole backtest on the
full panel, run it again on the same panel truncated at date X, and assert every fill before X is
identical. It covers the gates, the score, the ranking, the exits and the correlation/volatility
estimates in one assertion, because all of them feed the fills.

The check earns its place. This repo has already shipped two look-ahead defects of exactly this
shape and caught both by hand:

  liquidity gate   the dollar-volume median was `transform("median")` over every row handed in,
                   which in a backtest is the entire history INCLUDING THE FUTURE. A name that
                   became liquid later cleared the gate on dates when its actual volume was below
                   the floor. Worth ~20 points of backtest return.
  split gate       a centered corporate-action window also blocked the days BEFORE a split, which
                   live cannot know is coming.

Both would have failed this test on the day they were written.

It is deliberately run with `inverse_vol_sizing` and `max_correlation` ON as well as off. Those are
the newest inputs to the decision — the correlation and volatility estimates slice a returns pivot
built over the whole panel — and a pivot built over everything then sliced is precisely the shape
that goes wrong. The equal-weight arm alone would not exercise them at all.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, daily path (#270)
)
from kumo_strategies.strategies.momentum_rotation.config import (
    ExitConfig,
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

SESSIONS = 340
NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "LIQ")
#: LIQ is illiquid before the cutoff and liquid after it. It exists because the first version of
#: this fixture held volume constant at 1M shares on ~$100 names, so dollar volume never came
#: near `GateConfig.min_dollar_volume` ($5M) and the liquidity gate could not bind in either
#: direction. The historical look-ahead bug was reintroduced deliberately and the test still
#: passed — undiscriminating, the exact failure it exists to catch. With LIQ, a full-history
#: median lifts its EARLY bars above the floor on the strength of volume that had not happened
#: yet, and the trailing median does not.
ILLIQUID_NAME = "LIQ"
#: LIQ turns liquid HERE, well before the cutoff at SPLIT_AT. The position matters and a first
#: attempt got it wrong: with the step at the cutoff, the illiquid half was the LONGER one, so a
#: full-history median still landed below the floor and the buggy and correct gates agreed. The
#: bug only shows when median(bars before the cutoff) < floor < median(all bars) — i.e. illiquid
#: bars must be a minority of the whole panel and a majority of its first part.
LIQUID_FROM = 130
#: Truncate here. Far enough in that the book is fully formed and the vol/corr windows are warm.
SPLIT_AT = 220


class _AllEligible:
    name = "test_all"

    def eligible(self, d: pd.Timestamp) -> set[str]:
        return set(NAMES)


def _panel(seed: int = 5) -> pd.DataFrame:
    """Names whose behaviour CHANGES partway through, so future data is worth peeking at.

    A stationary panel is a weak fixture here: if every name looks the same before and after the
    split point, a look-ahead bug has nothing to gain and the test can pass while the bug is
    present. So the second half reverses the ranking — the early laggards become the late leaders,
    and the volatility ordering flips too. Any peek at the future is then visibly wrong.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=SESSIONS)
    rows = []
    for i, t in enumerate(NAMES):
        # Drift spread is deliberately far larger than anything real. Realism is not the point:
        # the fixture has to make the future WORTH peeking at, and with 120 post-cutoff sessions
        # at these volatilities a realistic spread is swamped by noise — the first version of
        # this fixture produced a rank correlation of +0.10 across the cutoff instead of the
        # intended inversion, and the guard below caught it.
        if t == ILLIQUID_NAME:
            # Strongest drift in the panel, so if the gate wrongly admits it, it ranks straight
            # into the book and the fills visibly change. A mediocre name would be gated out by
            # the RANKING regardless and the bug would stay invisible.
            early_mu = late_mu = 0.0050
            early_sig = late_sig = 0.012
        else:
            early_mu, late_mu = 0.0060 - 0.0017 * i, 0.0060 - 0.0017 * (len(NAMES) - 1 - i)
            early_sig, late_sig = 0.010 + 0.0015 * i, 0.010 + 0.0015 * (len(NAMES) - 1 - i)
        mu = np.r_[np.full(SPLIT_AT, early_mu), np.full(SESSIONS - SPLIT_AT, late_mu)]
        sig = np.r_[np.full(SPLIT_AT, early_sig), np.full(SESSIONS - SPLIT_AT, late_sig)]
        px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sig)))
        if t == ILLIQUID_NAME:
            # ~$1-2M/session while illiquid, ~$1B after. The floor is $5M.
            volume = np.r_[np.full(LIQUID_FROM, 10_000.0),
                           np.full(SESSIONS - LIQUID_FROM, 5_000_000.0)]
        else:
            volume = np.full(SESSIONS, 1_000_000.0)
        rows.append(pd.DataFrame({
            "ticker": t, "date": dates, "open": px, "high": px * (1 + 1.5 * sig),
            "low": px * (1 - 1.5 * sig), "close": px, "volume": volume}))
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def run_kw(tmp_path_factory) -> dict:
    p = tmp_path_factory.mktemp("instr") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return {"instruments_path": p,
            "cost_model": CostModel(half_spread_bps={t: 2.0 for t in NAMES}, default_bps=2.0),
            "starting_cash": 100_000.0}


CONFIGS = {
    "equal weight": MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, buffer=2),
        score=ScoreConfig(lookback=20, vol_window=60),
        exits=ExitConfig(give_back_frac=0.5)),
    # The corr/vol estimates slice a pivot built over the WHOLE panel. That is the shape that goes
    # wrong, so it has to be under test, not merely believed correct.
    "inverse-vol + corr cap": MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, buffer=2, inverse_vol_sizing=True,
                                  max_correlation=0.9, corr_window=60),
        score=ScoreConfig(lookback=20, vol_window=60),
        exits=ExitConfig(give_back_frac=0.5)),
}


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_fills_before_the_cutoff_do_not_depend_on_bars_after_it(label, run_kw):
    cfg = CONFIGS[label]
    full = _panel()
    cutoff = sorted(full["date"].unique())[SPLIT_AT]
    truncated = full[full["date"] <= cutoff]

    a = run(full, _AllEligible(), cfg=cfg, **run_kw).report.fills
    b = run(truncated, _AllEligible(), cfg=cfg, **run_kw).report.fills
    assert len(a) and len(b), f"{label}: fixture produced no fills, so the test proves nothing"

    cols = ["ts", "symbol", "side", "qty", "price"]

    def before(f: pd.DataFrame) -> pd.DataFrame:
        # Strictly before the cutoff. The decision made ON the cutoff fills at the NEXT session's
        # open, which the truncated panel does not have — that absence is the truncation, not a
        # discrepancy, so it is excluded rather than asserted over.
        return (f[pd.to_datetime(f.ts) < pd.Timestamp(cutoff)][cols]
                .sort_values(["ts", "symbol", "side"]).reset_index(drop=True))

    xa, xb = before(a), before(b)
    assert len(xa), f"{label}: no fills before the cutoff — move SPLIT_AT later"
    pd.testing.assert_frame_equal(
        xa, xb, check_dtype=False,
        obj=f"{label}: fills before {pd.Timestamp(cutoff).date()} changed when future bars were "
            f"removed, so something in the decision path is reading bars that had not completed")


def test_the_fixture_would_actually_expose_a_peek(run_kw):
    """Guard on the guard: prove the panel's future differs enough to be worth peeking at.

    If the second half looked like the first, a look-ahead bug would gain nothing and the test
    above could pass with the bug present — the undiscriminating-test failure this repo keeps
    hitting. So assert the ranking really does invert across the cutoff.
    """
    full = _panel()
    dates = sorted(full["date"].unique())
    cutoff = dates[SPLIT_AT]
    early = full[full.date <= cutoff].groupby("ticker")["close"].apply(
        lambda c: c.pct_change().mean())
    late = full[full.date > cutoff].groupby("ticker")["close"].apply(
        lambda c: c.pct_change().mean())
    common = early.index.intersection(late.index).drop(ILLIQUID_NAME, errors="ignore")
    rho = early[common].rank().corr(late[common].rank())
    assert rho < -0.5, (
        f"fixture's late ranking is not the inverse of its early one (rank corr {rho:.2f}) — a "
        "look-ahead bug would have little to gain and this test would not detect it")
