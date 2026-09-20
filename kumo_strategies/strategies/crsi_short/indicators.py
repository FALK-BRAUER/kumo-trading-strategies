"""ConnorsRSI and the two universe statistics, as SERIES over one symbol's ordered closes.

PURE and per-symbol. Every function takes a chronologically sorted `pd.Series` for ONE ticker and
returns a series of the same index. The panel layer does the grouping; nothing here knows what a
ticker is, which is what makes each piece testable against a hand-computed number.

CONFORMED TO THE LAB, NOT TO THE TEXTBOOK. #123's acceptance is reproducing
`research-lab/qc-strategy-review/replication/s411_build_signals.py` on 2025-01-01 -> 2026-09-08, so
where that file and the canonical definition differ, THIS FILE FOLLOWS THAT FILE and says where:

  seeding      Wilder's own recursion seeds from the MEAN of the first `period` changes; the lab
               uses `ewm(adjust=False)`, which seeds from the FIRST change. They converge after
               roughly 5 periods and disagree before that.
  flat input   the lab returns 100 when there are no losses, INCLUDING when there are no gains
               either. Wilder's convention for a perfectly flat series is 50. The streak series is
               flat through any run of unchanged closes, so this is reachable, and it pushes
               ConnorsRSI UP rather than leaving it neutral.
  windows      `percent_rank(r, 100)` compares against 99 prior returns, and the "100-day"
               volatility is a 99-observation window. Both are off by one from their names.

Each of those is a candidate explanation if the Nautilus curve and the lab curve disagree, so they
are reproduced deliberately and listed here rather than silently corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def wilder_rsi(values: pd.Series, period: int) -> pd.Series:
    """RSI with Wilder's smoothing constant, seeded the way the lab seeds it.

    `avg_loss == 0` returns 100 even when `avg_gain` is also 0 — see the module docstring. The guard
    exists at all because `100 - 100/(1 + gain/0)` is NaN, and a NaN component voids the whole
    ConnorsRSI for that session rather than producing a wrong number, which is the harder defect to
    notice: the name simply stops signalling.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    delta = values.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).where(avg_loss != 0, 100.0)


def streak(closes: pd.Series) -> pd.Series:
    """Consecutive up (+n) / down (-n) closes; 0 on an unchanged close, which RESETS the run.

    Connors' definition. The unchanged case is the one implementations get wrong: it is neither an
    up day nor a continuation of a down run, and treating it as either lets a run survive a flat
    session and reports a longer streak than happened.
    """
    direction = np.sign(closes.astype(float).diff().fillna(0.0)).to_numpy()
    out = np.zeros(len(direction))
    run = 0.0
    for i, d in enumerate(direction):
        run = 0.0 if d == 0 else (run + d if np.sign(run) == d or run == 0 else d)
        out[i] = run
    return pd.Series(out, index=closes.index)


def percent_rank(values: pd.Series, period: int) -> pd.Series:
    """Percent of the PRIOR `period - 1` observations strictly below the current one, 0-100.

    Strictly below, and the current observation is not part of its own comparison set — otherwise
    100 is unreachable and every reading is depressed by a third of the gap, at the exact threshold
    the entry rule tests.

    The window is `period` observations of which `period - 1` are the comparison set. That is the
    lab's arithmetic and it is what "PercentRank(1-day return, 100)" means in the replication: 99
    prior returns, not 100.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    return values.astype(float).rolling(period).apply(
        lambda w: float((w[:-1] < w[-1]).sum()) / (len(w) - 1) * 100.0, raw=True)


def connors_rsi(closes: pd.Series, rsi_period: int = 3, streak_period: int = 2,
                rank_period: int = 100) -> pd.Series:
    """ConnorsRSI(3, 2, 100) — the mean of three components, each 0-100.

    price momentum     `wilder_rsi(close, 3)`
    streak duration    `wilder_rsi(streak(close), 2)`
    relative magnitude `percent_rank(1-session return, 100)`

    PLAIN ADDITION, deliberately: NaN propagates, so a session missing any component is NaN — which
    is the rule. `pd.concat([...], axis=1).mean(axis=1)` is the version to avoid, because it SKIPS
    NaN and returns a two-of-three average on a different scale during warmup, where a 0-100
    threshold still reads as a valid signal.
    """
    px = closes.astype(float)
    return (wilder_rsi(px, rsi_period)
            + wilder_rsi(streak(px), streak_period)
            + percent_rank(px.pct_change(), rank_period)) / 3.0


def annualised_log_vol(closes: pd.Series, window: int = 100) -> pd.Series:
    """Annualised stdev of LOG returns. 1.0 is 100%/yr.

    LOG returns, not simple: the filter selects names that routinely double and halve, and at that
    amplitude the two are not interchangeable — a simple return is unbounded above and floored at
    -100%, so its stdev overstates.

    `window - 1` OBSERVATIONS and `ddof=0`, both from the lab. The strategy's "100-day volatility"
    is a 99-observation population stdev. Reproduced rather than corrected: switching either moves
    a 100% threshold by roughly half a point, which reclassifies names on the boundary.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    lr = np.log(closes.astype(float)).diff()
    return lr.rolling(window - 1).std(ddof=0) * np.sqrt(TRADING_DAYS)
