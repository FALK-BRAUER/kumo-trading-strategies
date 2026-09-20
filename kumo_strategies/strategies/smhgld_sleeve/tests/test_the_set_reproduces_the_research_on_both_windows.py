"""The sleeve's tracked results reproduce the lab's research numbers on BOTH pinned windows, inside
the tolerance the verification harness stated (research/smhgld_sleeve/verify_engine_matches_research
before ks#270 folded that walk into the one runner's sleeve family).

Both windows, because the recent one is what the lane was promoted on and the full one contains 2022,
where the sleeve draws down nearly twice as deep. Checking only the flattering window is how a lane
ships with its stress case unmeasured. The metrics are computed HERE with the harness's formulas from
the tracked `equity.csv`, so the gate is independent of the Report's own KPI definitions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SET = Path(__file__).resolve().parents[1] / "backtests" / "smh-gld-daily-2021-2026"

#: What the lab reported for the promoted configuration (risk_weight 0.48, 10 bps), per window.
RESEARCH = {
    "best": {"cagr": 45.9, "sharpe": 1.73, "max_drawdown": -14.8},
    "since-2021": {"cagr": 25.8, "sharpe": 1.19, "max_drawdown": -27.6},
}
#: Wider than floating point and integer-share rounding, far narrower than the 0.19 of Sharpe that
#: once went unexplained between a research figure and the engine that was supposed to produce it.
TOLERANCE = {"cagr": 1.5, "sharpe": 0.08, "max_drawdown": 1.0}


def _harness_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = len(returns) / 252
    return {
        "cagr": ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100,
        "sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(252),
        "max_drawdown": (equity / equity.cummax() - 1).min() * 100,
    }


@pytest.mark.parametrize("variation", sorted(RESEARCH))
def test_the_tracked_equity_curve_lands_inside_the_research_tolerance(variation: str):
    eq = pd.read_csv(SET / "variations" / variation / "results" / "equity.csv", index_col=0, parse_dates=True)["equity"]
    got = _harness_metrics(eq)
    off = {k: (got[k], want) for k, want in RESEARCH[variation].items() if abs(got[k] - want) > TOLERANCE[k]}
    assert not off, f"{variation}: engine vs research outside tolerance — {off}"


def test_a_20k_book_rebalances_more_often_than_the_research_book():
    """Integer shares at $20k round to ~1 % of the sleeve, above the 0.25 % band — the rounding trades.
    A property of a small book the set measures rather than hides; if it vanishes, the family has
    started holding fractional shares and the number no longer describes what the lane can do."""
    import json
    small = json.loads((SET / "variations" / "book-20k" / "results" / "diagnostics.json").read_text())["rebalances"]
    big = json.loads((SET / "variations" / "best" / "results" / "diagnostics.json").read_text())["rebalances"]
    assert small > big, (small, big)
