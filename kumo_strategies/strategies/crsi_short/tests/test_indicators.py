"""ConnorsRSI and the universe statistics, against values computed BY HAND.

Every expected number is derived on paper in the test's own docstring. A test that recomputes the
expectation with the code under test asserts only that the code is deterministic.

Where the lab's implementation differs from the textbook one, THESE TESTS FOLLOW THE LAB — see the
module docstring of `indicators.py` for the three places and why. `test_conformance_to_lab.py`
checks the whole set against that file directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short.indicators import (
    annualised_log_vol, connors_rsi, percent_rank, streak, wilder_rsi)


def _accelerating_up(n: int) -> pd.Series:
    """Each session gains MORE than the last, so the newest return is the largest in any window."""
    return pd.Series(np.cumprod(np.r_[100.0, 1.0 + 0.001 * np.arange(1, n)]))


def _accelerating_down(n: int) -> pd.Series:
    return pd.Series(np.cumprod(np.r_[100.0, 1.0 - 0.001 * np.arange(1, n)]))


def test_wilder_rsi_matches_the_hand_computation():
    """closes 10, 11, 12, 11 at period 2 (alpha 1/2, `adjust=False`, `min_periods=2`).

    changes: -, +1, +1, -1.
    index 2: avg_gain = 0.5*1 + 0.5*1 = 1.0; avg_loss = 0.5*0 + 0.5*0 = 0 -> no loss -> 100.
    index 3: avg_gain = 0.5*0 + 0.5*1.0 = 0.5; avg_loss = 0.5*1 + 0.5*0 = 0.5
             -> RS 1.0 -> 100 - 100/2 = 50.
    """
    out = wilder_rsi(pd.Series([10.0, 11.0, 12.0, 11.0]), 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(100.0)
    assert out.iloc[3] == pytest.approx(50.0)


def test_a_flat_series_is_ONE_HUNDRED_here_not_fifty():
    """No losses -> 100, even with no gains either. Wilder's convention for a perfectly flat series
    is 50, and this is the lab's deviation, reproduced deliberately: the STREAK series is flat
    through any run of unchanged closes, so it is reachable, and it pushes ConnorsRSI UP rather than
    leaving it neutral — toward the threshold this strategy shorts on."""
    assert wilder_rsi(pd.Series([7.0] * 6), 2).iloc[-1] == pytest.approx(100.0)


def test_streak_counts_runs_and_an_unchanged_close_resets_it():
    """closes 10, 11, 12, 12, 11, 10 -> 0, +1, +2, 0, -1, -2.

    The unchanged session is the case implementations get wrong: it is neither an up day nor a
    continuation, so the run restarts at the next move rather than carrying on to +3.
    """
    out = streak(pd.Series([10.0, 11.0, 12.0, 12.0, 11.0, 10.0]))
    assert list(out) == [0.0, 1.0, 2.0, 0.0, -1.0, -2.0]


def test_percent_rank_excludes_the_current_observation_and_uses_strict_inequality():
    """period 4 -> a 4-observation window whose COMPARISON SET is the other 3.

    [1,2,3,4,5]: window [2,3,4,5], all three prior below 5 -> 100.
    [1,2,3,4,4]: window [2,3,4,4], two of three strictly below 4 -> 66.7.

    If the current value were part of its own set, 100 would be unreachable and every reading would
    be depressed — at the exact threshold the entry rule tests.
    """
    assert percent_rank(pd.Series([1.0, 2, 3, 4, 5]), 4).iloc[-1] == pytest.approx(100.0)
    assert percent_rank(pd.Series([1.0, 2, 3, 4, 4]), 4).iloc[-1] == pytest.approx(200.0 / 3.0)


def test_connors_rsi_is_nan_until_all_three_components_exist():
    """A two-of-three average is a different indicator on a different scale, and it would appear
    exactly during warmup — letting the earliest sessions signal on a statistic the rest of the
    sample never sees."""
    out = connors_rsi(_accelerating_up(60), 3, 2, 50)
    # The rank period counts RETURNS, and the first return costs a session: at period 50 the first
    # readable value is index 50. `CrsiShortConfig.warmup_sessions` derives this (+2, the lab's own
    # guard), and a stored warmup would be off by it.
    assert out.iloc[:50].isna().all()
    assert out.iloc[50:].notna().all()


def test_connors_rsi_separates_a_melt_up_from_a_sell_off():
    """Every component maxes on an accelerating advance and bottoms on an accelerating decline:
    RSI(3) = 100/0, the streak RSI = 100/0, and today's return is the largest/smallest in the set."""
    assert connors_rsi(_accelerating_up(60), 3, 2, 50).iloc[-1] == pytest.approx(100.0)
    assert connors_rsi(_accelerating_down(60), 3, 2, 50).iloc[-1] == pytest.approx(0.0)


def test_a_perfectly_steady_advance_does_NOT_reach_the_entry_threshold():
    """Identical daily returns rank at ZERO against each other — strict inequality, no ties — so
    the relative-magnitude component is 0 and ConnorsRSI caps at 66.7 however long the run.

    That is Connors' definition and it is load-bearing for #123: the entry needs an OUTSIZED move,
    not merely an extended one. A `<=` PercentRank would be a materially different entry rule.
    """
    steady = pd.Series(np.cumprod(np.r_[100.0, np.full(59, 1.02)]))
    assert connors_rsi(steady, 3, 2, 50).iloc[-1] == pytest.approx(200.0 / 3.0)


def test_annualised_log_vol_is_the_hand_computation():
    """closes 100, 100*e^0.1, 100 -> log returns +0.1, -0.1.

    `window=3` means a 2-observation window and `ddof=0`: stdev = sqrt((0.1^2 + 0.1^2)/2) = 0.1,
    annualised 0.1 * sqrt(252) = 1.5875, i.e. 158.75%/yr.
    """
    px = pd.Series([100.0, 100.0 * np.exp(0.1), 100.0])
    assert annualised_log_vol(px, 3).iloc[-1] == pytest.approx(0.1 * np.sqrt(252), rel=1e-9)


def test_log_returns_not_simple_returns_at_the_amplitude_this_strategy_trades():
    """A name that doubles and halves. Simple returns are +100% / -50%; log returns are symmetric.
    The two disagree by more than the gap between an 80% and a 120% threshold, which is the width
    of #123's whole robustness grid."""
    px = pd.Series([100.0, 200.0, 100.0, 200.0, 100.0])
    simple = px.pct_change().rolling(4).std(ddof=0).iloc[-1] * np.sqrt(252)
    assert annualised_log_vol(px, 5).iloc[-1] < simple
