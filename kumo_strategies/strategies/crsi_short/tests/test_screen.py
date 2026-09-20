"""`screen_universe` asks `apply_gates` the same question the backtest asks, and refuses rather than
returning an empty list (#242).

The live lane was handed the 130 names the acceptance run had TRADED; on 2026-09-08, 34 of them
were eligible and 575 eligible names were not on the list. The screen is what the backtest ran on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short import ADJUSTED, CrsiShortConfig, apply_gates, build_feature_panel
from kumo_strategies.strategies.crsi_short.screen import Screen, ScreenRefused, screen_universe

SESSIONS = 130


def _cfg(**kw) -> CrsiShortConfig:
    base = dict(vol_window=60, crsi_rank_period=60, min_annual_vol=None,
                min_dollar_volume=1_000_000.0, max_borrow_fee_annual=None)
    return CrsiShortConfig(**{**base, **kw})


def _bars(volumes: dict[str, float | np.ndarray], sessions: int = SESSIONS,
          history: dict[str, int] | None = None) -> pd.DataFrame:
    """{ticker: volume (scalar or per-session array)}. Prices wiggle around 50 so every name is above
    the floor. `history` gives a ticker FEWER sessions (its last N), for a name too new to be warm."""
    dates = pd.bdate_range("2025-01-01", periods=sessions)
    rng = np.random.default_rng(1)
    rows = []
    for tkr, vol in volumes.items():
        closes = 50.0 * np.cumprod(1.0 + rng.normal(0, 0.02, sessions))
        n = (history or {}).get(tkr, sessions)
        rows.append(pd.DataFrame({"ticker": tkr, "date": dates[-n:], "open": closes[-n:], "high": closes[-n:] * 1.01,
                                  "low": closes[-n:] * 0.99, "close": closes[-n:],
                                  "volume": np.broadcast_to(np.asarray(vol, dtype=float), sessions)[-n:]}))
    return pd.concat(rows, ignore_index=True)


def test_the_screen_is_exactly_apply_gates_eligible_over_the_lookback():
    # A liquid name, an illiquid one, one that crosses the floor late in the window, one that was
    # liquid only BEFORE the window, and one liquid but too new to be warm.
    late = np.r_[np.full(SESSIONS - 5, 1e3), np.full(5, 1e6)]
    early = np.r_[np.full(100, 1e3), np.full(5, 1e6), np.full(SESSIONS - 105, 1e3)]
    bars = _bars({"LIQ": 1e6, "THIN": 1e3, "LATE": late, "EARLY": early, "NEW": 1e6}, history={"NEW": 30})
    cfg = _cfg()
    s = screen_universe(bars, cfg, adjustment=ADJUSTED, lookback_sessions=20)
    panel = apply_gates(build_feature_panel(bars, cfg, adjustment=ADJUSTED), cfg)
    window = sorted(panel["date"].unique())[-20:]
    expect = tuple(sorted(panel.loc[panel["date"].isin(window) & panel["eligible"], "ticker"].unique()))
    assert s.symbols == expect
    assert "LIQ" in s.symbols and "LATE" in s.symbols
    assert "THIN" not in s.symbols            # never above the floor
    assert "EARLY" not in s.symbols           # liquid only before the window
    assert "NEW" not in s.symbols             # liquid, but 30 sessions cannot be warm
    assert s.asof == pd.Timestamp(window[-1])
    assert set(s.eligible_asof) <= set(s.symbols)


def test_lookback_one_is_eligible_asof():
    bars = _bars({"LIQ": 1e6, "THIN": 1e3})
    s = screen_universe(bars, _cfg(), adjustment=ADJUSTED, lookback_sessions=1)
    assert s.symbols == s.eligible_asof == ("LIQ",)


def test_a_name_that_left_the_floor_is_still_WATCHED_inside_the_lookback():
    # Eligible for the first 120 sessions, thin for the last 10: not eligible as-of, still watched.
    gone = np.r_[np.full(SESSIONS - 10, 1e6), np.full(10, 1e3)]
    s = screen_universe(_bars({"LIQ": 1e6, "GONE": gone}), _cfg(), adjustment=ADJUSTED, lookback_sessions=20)
    assert "GONE" in s.symbols and "GONE" not in s.eligible_asof


def test_raw_prices_are_refused_before_anything_is_screened():
    with pytest.raises(ValueError, match="split-adjusted"):
        screen_universe(_bars({"LIQ": 1e6}), _cfg(), adjustment="raw")


def test_too_little_history_refuses_by_name_rather_than_screening_nobody():
    cfg = _cfg()
    with pytest.raises(ScreenRefused, match=f"need {cfg.warmup_sessions}"):
        screen_universe(_bars({"LIQ": 1e6}, sessions=cfg.warmup_sessions - 1), cfg, adjustment=ADJUSTED)


def test_an_empty_screen_refuses_rather_than_returning_a_list():
    with pytest.raises(ScreenRefused, match="no name was eligible"):
        screen_universe(_bars({"THIN": 1e3, "ALSO": 1e2}), _cfg(), adjustment=ADJUSTED)


def test_asof_after_the_data_is_refused_not_clamped():
    with pytest.raises(ScreenRefused, match="after the last session"):
        screen_universe(_bars({"LIQ": 1e6}), _cfg(), adjustment=ADJUSTED, asof="2030-01-01")


def test_diagnostics_state_the_number_a_venue_has_to_carry():
    s = screen_universe(_bars({"LIQ": 1e6, "THIN": 1e3}), _cfg(), adjustment=ADJUSTED, lookback_sessions=5)
    d = s.diagnostics
    assert d["watched"] == len(s.symbols) == 1 and d["eligible_asof"] == 1
    assert d["sessions_in_window"] == 5 and len(d["eligible_per_session"]) == 5
    assert "volatility gate" in d["gates"]["note"]
    assert isinstance(s, Screen)
