"""Every indicator, against the lab implementation it must reproduce (#123).

#123's acceptance is that this port reproduces the lab equity curve, and an indicator that differs
in the fourth decimal moves a 90-threshold on the names that sit at it. So the components are
compared to `research-lab/qc-strategy-review/replication/s411_build_signals.py` DIRECTLY, on a series
long and noisy enough to exercise every branch, rather than to remembered numbers.

The lab tree is a sibling checkout, not a dependency: when it is absent this SKIPS, loudly, naming
the path. A skip here means the conformance claim is unverified in this environment — it is not a
pass, and the acceptance run must not be declared on a suite where this skipped.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short.indicators import (
    annualised_log_vol, connors_rsi, percent_rank, streak, wilder_rsi)

LAB_REL = "lab/qc-strategy-review/replication/s411_build_signals.py"


@pytest.fixture(scope="module")
def lab():
    from kumo_strategies.data_root import ENV, DataRootUnset, data_path
    try:
        LAB = data_path(LAB_REL, must_exist=False)
    except DataRootUnset:
        pytest.skip(f"{ENV} unset — wanted {LAB_REL}; conformance UNVERIFIED here")
    if not LAB.exists():
        pytest.skip(f"lab replication not at {LAB} — conformance UNVERIFIED here")
    spec = importlib.util.spec_from_file_location("s411_build_signals", LAB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def closes() -> pd.Series:
    """400 sessions at 5% daily vol: enough for both 100-windows, volatile enough to cross the
    entry threshold in both directions, and with repeated closes so the streak reset is exercised."""
    rng = np.random.default_rng(7)
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.05, 400))
    px[50:53] = px[50]                       # a flat run: streak reset and the no-loss RSI branch
    return pd.Series(px)


def _identical(a: pd.Series, b: pd.Series) -> bool:
    return np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       equal_nan=True, rtol=0.0, atol=1e-10)


def test_wilder_rsi_is_identical(lab, closes):
    assert _identical(wilder_rsi(closes, 3), lab.wilder_rsi(closes, 3))


def test_streak_is_identical(lab, closes):
    assert _identical(streak(closes), lab.streaks(closes))


def test_the_streak_rsi_is_identical(lab, closes):
    assert _identical(wilder_rsi(streak(closes), 2), lab.wilder_rsi(lab.streaks(closes), 2))


def test_percent_rank_is_identical(lab, closes):
    assert _identical(percent_rank(closes.pct_change(), 100),
                      lab.percent_rank(closes.pct_change(), 100))


def test_connors_rsi_is_identical(lab, closes):
    theirs = (lab.wilder_rsi(closes, 3)
              + lab.wilder_rsi(lab.streaks(closes), 2)
              + lab.percent_rank(closes.pct_change(), 100)) / 3.0
    assert _identical(connors_rsi(closes, 3, 2, 100), theirs)


def test_annualised_volatility_is_identical(lab, closes):
    """The lab reports volatility x100 and compares to 100.0; this port keeps it as a fraction and
    compares to 1.0. Same number, stated once here so the two thresholds cannot drift apart."""
    theirs = np.log(closes).diff().rolling(99).std(ddof=0) * np.sqrt(252) * 100.0
    assert _identical(annualised_log_vol(closes, 100) * 100.0, theirs)
