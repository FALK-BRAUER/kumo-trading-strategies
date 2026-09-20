"""Strength scoring — pure functions, no Nautilus, no I/O.

Every measure is scale-free: standardised against the name's OWN history, never compared to a fixed
threshold. The universe spans roughly 5x in volatility (15% to 72% annualised across the sector
ETFs alone), so "moved more than 1.5%" means something different at each end. The same reasoning
killed a real trade: a 2% stop on a 151-IV name was guaranteed to be hit.

Deliberately NOT percentile-based. A rolling percentile saturates at 1.0 for every name making a new
extreme, and ties then break by whatever order the frame happens to be in — which once turned the
ranking into "first ten alphabetically", 77% A/B/C tickers against a 23% universe share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_normalised_momentum(close: pd.Series, lookback: int, vol_window: int,
                            min_history: int) -> pd.Series:
    """Return over `lookback` sessions, divided by the name's own return SD scaled to that horizon.

    Dividing by sd*sqrt(lookback) puts a quiet name and a volatile one on one scale: the result is
    "how many of its own standard deviations has it moved", not "how many percent".
    """
    ret = close.pct_change()
    sd = ret.rolling(vol_window, min_periods=min_history).std()
    raw = close / close.shift(lookback) - 1.0
    return raw / (sd * np.sqrt(lookback))


def distance_above_mean(close: pd.Series, window: int, vol_window: int,
                        min_history: int) -> pd.Series:
    """How far above its own rolling mean, in the name's own volatility units."""
    ret = close.pct_change()
    sd = ret.rolling(vol_window, min_periods=min_history).std()
    ma = close.rolling(window, min_periods=max(2, window // 2)).mean()
    return (close / ma - 1.0) / sd


def relative_volume(volume: pd.Series, window: int, min_history: int) -> pd.Series:
    """Volume against this name's own trailing median. A ratio, so it has no ties and no ceiling.

    NOTE: measured residual IC was positive but the effect is concentrated in the extreme, not
    spread across the distribution — treat it as a tail flag, not a monotonic quality score.
    """
    med = volume.rolling(window, min_periods=min_history).median()
    return volume / med.replace(0, np.nan)


def rank_within_day(scores: pd.Series) -> pd.Series:
    """Cross-sectional rank, ties averaged. Ranking makes the score comparable across days without
    assuming the raw scale is stable."""
    return scores.rank(pct=True)
