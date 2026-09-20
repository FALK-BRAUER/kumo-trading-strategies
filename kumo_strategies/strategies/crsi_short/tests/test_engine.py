"""CRSISHORT's decision layer: the price-adjustment boundary, look-ahead, and entry selection."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import (
    ADJUSTED, build_feature_panel, decide)

SESSIONS = 140


def _cfg(**kw) -> CrsiShortConfig:
    base = dict(vol_window=60, crsi_rank_period=60,
                min_annual_vol=None, min_dollar_volume=0.0, max_borrow_fee_annual=None,
                n_slots=20)
    return CrsiShortConfig(**{**base, **kw})


def _bars(spec: dict[str, np.ndarray], volume: float = 1e7) -> pd.DataFrame:
    """One frame from {ticker: closes}. Highs are 1% above the close, lows 1% below."""
    dates = pd.bdate_range("2025-01-01", periods=SESSIONS)
    rows = []
    for tkr, closes in spec.items():
        rows.append(pd.DataFrame({"ticker": tkr, "date": dates[:len(closes)],
                                  "open": closes, "high": closes * 1.01,
                                  "low": closes * 0.99, "close": closes,
                                  "volume": volume}))
    return pd.concat(rows, ignore_index=True)


def _accelerating_up(n: int, start: float = 100.0) -> np.ndarray:
    return np.cumprod(np.r_[start, 1.0 + 0.001 * np.arange(1, n)])


def _wiggly(n: int, seed: int = 0, scale: float = 0.07) -> np.ndarray:
    """A high-volatility drifting walk.

    The accelerating series above pins ConnorsRSI at exactly 100 on every session, so a threshold
    sweep over it cannot move — an axis no fixture varies is invisible to mutation testing, and its
    green is indistinguishable from real coverage. This fixture varies that axis on purpose.
    """
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.004, scale, n - 1)])


# --- the split-adjusted requirement ----------------------------------------------------------

def test_the_adjustment_argument_is_keyword_only_and_has_no_default():
    """Asserted on the SIGNATURE, not on a docstring or a comment.

    A default would be inherited by every call site silently, and #123's expensive lesson is that
    the RAW store inverted the 2025-26 verdict on 7.4% of trades. A grep-the-source test would be
    satisfied by deleting a warning and broken by writing one; this binds the interface.
    """
    p = inspect.signature(build_feature_panel).parameters["adjustment"]
    assert p.default is inspect.Parameter.empty
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_raw_prices_are_refused_rather_than_gated():
    """This repo's long lanes answer corporate actions with an ENTRY-time gate. That cannot reach a
    reverse split on a position already open, which is where the -1000% short losses were booked."""
    bars = _bars({"AAA": _accelerating_up(SESSIONS)})
    with pytest.raises(ValueError, match="split-adjusted"):
        build_feature_panel(bars, _cfg(), adjustment="raw")


# --- look-ahead ------------------------------------------------------------------------------

def test_a_sessions_own_bar_cannot_change_that_sessions_features():
    """VERIFY BY DISAGREEMENT: two panels identical except for the LAST session's bar must produce
    identical features on that session. If any feature reads the session it decides in, they differ.

    This is the third of the four lab defects ("a rotation variant ranking on the session it
    executed in"), and it is invisible to inspection — the code reads correctly either way.
    """
    closes = _accelerating_up(SESSIONS)
    a = _bars({"AAA": closes})
    b = a.copy()
    last = b.index[-1]
    b.loc[last, ["open", "high", "low", "close"]] = [1e6, 1e6, 1e6, 1e6]
    b.loc[last, "volume"] = 1e12

    cols = ["asof_close", "asof_high", "asof_dollar_volume", "crsi", "ann_vol", "signal", "limit_px"]
    fa = build_feature_panel(a, _cfg(), adjustment=ADJUSTED).iloc[-1][cols]
    fb = build_feature_panel(b, _cfg(), adjustment=ADJUSTED).iloc[-1][cols]
    pd.testing.assert_series_equal(fa, fb)


def test_the_limit_rests_three_percent_above_the_SIGNAL_close():
    panel = build_feature_panel(_bars({"AAA": _accelerating_up(SESSIONS)}), _cfg(),
                                adjustment=ADJUSTED)
    row = panel.iloc[-1]
    assert row["limit_px"] == pytest.approx(row["asof_close"] * 1.03)


# --- gates -----------------------------------------------------------------------------------

def test_the_dollar_volume_floor_gates_ENTRIES_and_never_exits_a_held_name():
    """#123 holds through liquidity dips. Dropping a held name when it leaves the universe is the
    fourth lab defect — the position vanished without the trade being booked."""
    panel = build_feature_panel(_bars({"AAA": _accelerating_up(SESSIONS)}),
                                _cfg(min_dollar_volume=1e15), adjustment=ADJUSTED)
    day = panel[panel["date"] == panel["date"].max()]
    assert not day["eligible"].any()
    dec = decide(day, _cfg(min_dollar_volume=1e15), held={"AAA"})
    assert dec.exit == ()
    assert dec.hold == ("AAA",)
    assert dec.enter == ()


def test_decide_never_returns_an_exit():
    """CRSISHORT's exits are structural and per-position. Two exit authorities that must agree
    cannot be tested together, and one of them always goes quietly inert."""
    panel = build_feature_panel(_bars({"AAA": _accelerating_up(SESSIONS)}), _cfg(),
                                adjustment=ADJUSTED)
    day = panel[panel["date"] == panel["date"].max()]
    assert decide(day, _cfg(), held={"ZZZ"}).exit == ()


def test_the_volatility_filter_changes_which_names_signal():
    """One of the four mutations #123's acceptance requires to MOVE the result. A filter that can be
    switched off without the signal set changing is not doing anything."""
    bars = _bars({"AAA": _accelerating_up(SESSIONS)})
    off = build_feature_panel(bars, _cfg(min_annual_vol=None), adjustment=ADJUSTED)["signal"].sum()
    on = build_feature_panel(bars, _cfg(min_annual_vol=1.0), adjustment=ADJUSTED)["signal"].sum()
    assert off > 0
    assert on < off


def test_raising_the_crsi_threshold_to_99_removes_the_signals():
    bars = _bars({"AAA": _wiggly(SESSIONS)})
    hot = build_feature_panel(bars, _cfg(crsi_entry=90.0), adjustment=ADJUSTED)["signal"].sum()
    cold = build_feature_panel(bars, _cfg(crsi_entry=99.999), adjustment=ADJUSTED)["signal"].sum()
    assert hot > 0
    assert cold < hot


# --- slots and borrow ------------------------------------------------------------------------

def _three_signals():
    cfg = _cfg()
    bars = _bars({t: _accelerating_up(SESSIONS, start=s)
                  for t, s in (("AAA", 100.0), ("BBB", 50.0), ("CCC", 20.0))})
    panel = build_feature_panel(bars, cfg, adjustment=ADJUSTED)
    day = panel[panel["date"] == panel["date"].max()]
    assert day["signal"].all()
    return cfg, day


def test_more_signals_than_slots_fills_the_slots_and_LOGS_the_rest():
    cfg, day = _three_signals()
    cfg = replace(cfg, n_slots=2)
    dec = decide(day, cfg, held=set())
    assert len(dec.enter) == 2
    assert len(dec.refused) == 1
    assert "no slot" in next(iter(dec.refused.values()))


def test_slots_already_taken_leave_no_room():
    cfg, day = _three_signals()
    dec = decide(day, replace(cfg, n_slots=1), held={"ZZZ"})
    assert dec.enter == ()


def test_a_held_name_is_not_entered_twice():
    cfg, day = _three_signals()
    dec = decide(day, cfg, held={"AAA"})
    assert "AAA" not in dec.enter


def test_a_fee_ceiling_without_locate_data_raises_instead_of_running_inert():
    """The #26 shape: a rule that is configured, looks armed, and never fires."""
    cfg, day = _three_signals()
    with pytest.raises(ValueError, match="max_borrow_fee_annual"):
        decide(day, replace(cfg, max_borrow_fee_annual=1.0), held=set(), borrow=None)


def test_no_locate_and_an_expensive_borrow_are_both_refused_and_named():
    """Absence from the file and a `None` fee are the same fact. Treating an unknown name as
    borrowable consumes a slot with an order that cannot fill."""
    cfg, day = _three_signals()
    dec = decide(day, replace(cfg, max_borrow_fee_annual=1.0), held=set(),
                 borrow={"AAA": 0.01, "BBB": 2.5})
    assert dec.enter == ("AAA",)
    assert "over the 100% ceiling" in dec.refused["BBB"]
    assert dec.refused["CCC"] == "no locate"


# --- the order rests overnight, so the universe is checked TWICE ------------------------------

def _panel_with_a_liquidity_dip(*, on_signal_session: bool):
    """A name that signals every session, whose dollar volume collapses on one of the two sessions
    that matter for the LAST row of the panel.

    That row is the session the limit order RESTS in. Dollar volume is always the prior session's
    tape, so its own universe check reads the bar one back, and the signal session's check reads the
    bar two back. Both are knowable before the open; neither is look-ahead.
    """
    cfg = _cfg(min_dollar_volume=1e8)
    bars = _bars({"AAA": _accelerating_up(SESSIONS)})
    bars.loc[bars.index[-3 if on_signal_session else -2], "volume"] = 1.0
    panel = build_feature_panel(bars, cfg, adjustment=ADJUSTED)
    return cfg, panel[panel["date"] == panel["date"].max()]


def test_a_resting_order_is_cancelled_when_the_name_leaves_the_universe():
    """The lab drops a pending name that is no longer in the entry pool. Dollar volume is always the
    PRIOR session's tape, so this is knowable before the open and is not look-ahead — and skipping
    it enters names the measured strategy never entered."""
    cfg, day = _panel_with_a_liquidity_dip(on_signal_session=False)
    assert day["signalled_yesterday"].all()
    assert not day["eligible"].any()
    assert decide(day, cfg, held=set()).enter == ()


def test_a_name_that_was_thin_on_the_SIGNAL_session_never_signals():
    cfg, day = _panel_with_a_liquidity_dip(on_signal_session=True)
    assert day["eligible"].all()
    assert not day["signalled_yesterday"].any()
    assert decide(day, cfg, held=set()).enter == ()


def test_a_resting_order_consumes_a_slot():
    """`room = slots - positions - pending`. Counting only filled positions lets the book commit to
    more names than it can hold, and the last fills to arrive are the ones rejected."""
    cfg, day = _three_signals()
    dec = decide(day, replace(cfg, n_slots=2), held=set(), pending={"BBB"})
    assert len(dec.enter) == 1
    assert "BBB" not in dec.enter
