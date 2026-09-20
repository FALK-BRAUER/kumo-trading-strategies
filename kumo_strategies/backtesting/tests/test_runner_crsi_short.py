"""The CRSISHORT runner: the ledger check, the short round trip, and the three lab defects it exists
to make impossible.

The panel is HAND-BUILT here rather than derived from synthetic bars, so each test controls exactly
which session signals and what the tape does next. `apply_gates` still runs over it, so the gating
under test is the real one.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.families.short import RECONCILE_TOLERANCE
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions,  # the ONE runner, short family (#270)
)
from kumo_strategies.strategies.crsi_short import CrsiShortConfig

CFG = CrsiShortConfig(max_borrow_fee_annual=None, min_dollar_volume=1e6, n_slots=4)
FLAT_25BPS = CostModel(half_spread_bps={}, default_bps=25.0, time_of_day=False)
DATES = pd.bdate_range("2025-01-06", periods=8)


def _panel(bars: dict[str, list[tuple]], crsi: dict[str, list[float]],
           vol: float = 200.0, dv: float = 1e9) -> pd.DataFrame:
    """`bars[ticker]` is one (open, high, low, close) per session in `DATES`.

    `crsi[ticker][i]` is the ConnorsRSI AS OF THE PRIOR CLOSE — the value the panel carries on row
    `i`, which is what decides whether an order rests during session `i`.
    """
    rows = []
    for tkr, ohlc in bars.items():
        for i, (o, h, lo, c) in enumerate(ohlc):
            rows.append(dict(ticker=tkr, date=DATES[i], open=o, high=h, low=lo, close=c,
                             volume=1e6, asof_close=ohlc[i - 1][3] if i else o,
                             asof_high=ohlc[i - 1][1] if i else h,
                             asof_dollar_volume=dv, crsi=crsi[tkr][i], ann_vol=vol,
                             sessions_seen=1000, adjustment="all"))
    return pd.DataFrame(rows)


def _run(panel: pd.DataFrame, cfg: CrsiShortConfig = CFG, **kw):
    return run_sessions(None, panel=panel, cfg=cfg, cost_model=FLAT_25BPS,
                          start=str(DATES[0].date()), end=str(DATES[-1].date()),
                          starting_cash=100_000.0, **kw)


#: The session the short is opened in: the limit is 103.00 (session 0 closes at 100) and the high
#: reaches it. The LOW does not go below the entry, so the position is NOT in profit yet — a fixture
#: whose entry session dips through the entry price arms the flat exit immediately and can then
#: only ever express one of the two covers.
ENTRY_BAR = (103.0, 104.0, 103.0, 103.5)
#: Every session after the story ends. ABOVE the entry, so an unresolved position is covered at a
#: loss rather than at a profit — the sign has to be visible. Its OPEN and CLOSE differ on purpose:
#: a bar whose open equals its close cannot tell a cover priced at the open from one priced at the
#: close, and that is the whole content of "flagged at a close, acted at the next open".
TAIL_BAR = (111.0, 113.0, 110.0, 112.0)
#: A quiet session: in profit intraday, never back at the trail, never above the prior high.
QUIET_BAR = (102.0, 102.5, 101.0, 102.0)


def _one_short(*bars: tuple, crsi_after: float = 0.0) -> pd.DataFrame:
    """A book that shorts AAA at 103 on session 1, then does whatever `bars` say."""
    seq = [(100.0, 100.0, 100.0, 100.0), ENTRY_BAR, *bars]
    seq += [TAIL_BAR] * (len(DATES) - len(seq))
    return _panel({"AAA": seq}, {"AAA": [0.0, 95.0] + [crsi_after] * (len(DATES) - 2)})


# --- the assertion this runner exists for ------------------------------------------------------

def test_the_ledger_reconciles_on_an_ordinary_run():
    """`equity change == booked + open - costs`, checked on every run. It raises rather than
    warning: a run whose two accounts disagree is void, not approximate."""
    r = _run(_one_short())
    eq = r.report.equity
    booked = float(r.covers["pnl"].sum())
    costs = float(r.report.fills["cost"].sum()) + r.borrow_paid
    assert abs((eq.iloc[-1] - 100_000.0) - (booked - costs)) < RECONCILE_TOLERANCE


def test_a_position_dropped_without_being_booked_is_CAUGHT_by_the_ledger():
    """The third lab defect: a name that leaves the universe is deleted, its P&L stays in equity and
    vanishes from the trade table. The curve looks right and the statistics are wrong, or the
    reverse — and only the two together see it.

    Simulated by breaking the ledger the same way the defect does.
    """
    import kumo_strategies.backtesting.families.short as mod

    panel = _one_short()
    real = mod.RECONCILE_TOLERANCE
    try:
        mod.RECONCILE_TOLERANCE = -1.0        # nothing reconciles to within a negative dollar
        with pytest.raises(AssertionError, match="does not reconcile"):
            _run(panel)
    finally:
        mod.RECONCILE_TOLERANCE = real


# --- the short round trip ----------------------------------------------------------------------

def test_a_short_round_trip_reports_a_LOSS_when_the_price_rises():
    """Sign, once, on the number #123's headline is made of. A reporting layer that reads BUY as
    'open' inverts this and reports +6% where the book lost 6%."""
    r = _run(_one_short())
    t = r.report.trades.iloc[0]
    assert t["side"] == "SHORT"
    assert t["entry_px"] == pytest.approx(103.0)
    assert t["return_pct"] < 0
    assert t["return_pct"] == pytest.approx(100.0 * (1 - t["exit_px"] / 103.0))


def test_the_entry_is_the_LIMIT_and_the_slot_is_sized_off_it():
    r = _run(_one_short())
    f = r.report.fills.iloc[0]
    assert (f["side"], f["price"]) == ("SELL", 103.0)
    assert f["qty"] == int(100_000.0 / 4 / 103.0)         # equity / n_slots, whole shares


def test_a_session_that_never_reaches_the_limit_does_not_fill():
    panel = _one_short()
    panel.loc[panel.date == DATES[1], "high"] = 102.0     # never trades up to 103
    assert not len(_run(panel).report.fills)


def test_a_session_that_GAPS_above_the_limit_fills_at_the_open():
    """Worse than the limit, and honest: the order could not have been filled at 103 on a session
    that opened at 120."""
    panel = _one_short()
    panel.loc[panel.date == DATES[1], ["open", "high"]] = [120.0, 125.0]
    assert _run(panel).report.fills.iloc[0]["price"] == 120.0


# --- exits, through the runner -----------------------------------------------------------------

def test_the_reversal_cover_is_taken_at_the_NEXT_open():
    """Session 2 closes above session 1's high, which FLAGS it; session 3's open is the fill."""
    r = _run(_one_short((105.0, 106.0, 104.0, 105.0)))
    c = r.covers.iloc[0]
    assert (c["kind"], c["price"]) == ("reversal", TAIL_BAR[0])


def test_a_name_covered_THIS_MORNING_is_not_re_shorted_THIS_SESSION():
    """The order was placed last night, when the position was still open — so it cannot exist. The
    rotation lanes reproduce a same-session re-entry on purpose (`runner_sessions`); this book must
    not have one, because its entry is a limit resting from the previous close."""
    r = _run(_one_short((95.0, 96.0, 90.0, 92.0), (101.0, 104.0, 100.0, 102.0), crsi_after=95.0))
    cover = r.covers.iloc[0]
    same_session = r.report.fills[(r.report.fills.ts == cover["date"])
                                  & (r.report.fills.side == "SELL")]
    assert not len(same_session)


def test_the_flat_cover_is_taken_at_entry_once_the_position_has_been_in_profit():
    r = _run(_one_short((95.0, 96.0, 90.0, 92.0), (101.0, 104.0, 100.0, 102.0)))
    c = r.covers.iloc[0]
    assert (c["kind"], c["price"]) == ("flat", 103.0)
    assert c["pnl"] == pytest.approx(0.0)


def test_a_held_name_that_leaves_the_MANAGEABLE_set_is_booked_not_dropped():
    """The lab books it at the last mark as `nodata` — 4% of #123's trades. Dropping it instead is
    the defect the ledger check exists for."""
    panel = _one_short()
    panel.loc[(panel.ticker == "AAA") & (panel.date >= DATES[2]), "asof_dollar_volume"] = 1.0
    r = _run(panel)
    assert r.covers.iloc[0]["kind"] == "nodata"
    assert len(r.report.trades) == 1                       # booked, with a P&L


def test_a_liquidity_dip_does_NOT_close_a_position_under_hold_through():
    """The entry floor gates entries only. Same dip, above the manageable floor: the position is
    still there, and `hold_through=False` is what closes it."""
    panel = _one_short()
    panel.loc[(panel.ticker == "AAA") & (panel.date >= DATES[2]), "asof_dollar_volume"] = 2e6
    held = _run(panel, replace(CFG, min_dollar_volume=1e8, hold_through=True))
    closed = _run(panel, replace(CFG, min_dollar_volume=1e8, hold_through=False))
    # Held: managed to a real exit. Closed: booked out on the dip, for a reason that has nothing to
    # do with the trade — which is what #123 calls the accident of the earlier code.
    assert held.covers.iloc[0]["kind"] == "reversal"
    assert closed.covers.iloc[0]["kind"] == "nodata"
    assert closed.covers.iloc[0]["date"] < held.covers.iloc[0]["date"]


def test_the_borrow_carry_is_charged_daily_and_is_not_a_fill_cost():
    """20%/yr over a 2.7-session hold is 0.2%, so it cannot be folded into a per-side cost model —
    #123's break-even borrow of 216%/yr only means anything if the carry scales with TIME."""
    short_hold = _run(_one_short(QUIET_BAR))
    long_hold = _run(_one_short(*[QUIET_BAR] * 4))
    assert long_hold.borrow_paid > short_hold.borrow_paid > 0
    assert "borrow" not in "".join(long_hold.report.fills.columns)


# --- inputs -------------------------------------------------------------------------------------

def test_raw_bars_need_an_adjustment_and_a_panel_does_not():
    bars = pd.DataFrame({"ticker": ["AAA"], "date": [DATES[0]], "open": [1.0], "high": [1.0],
                         "low": [1.0], "close": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError, match="adjustment"):
        run_sessions(bars, cfg=CFG, cost_model=FLAT_25BPS, start=str(DATES[0].date()),
                       end=str(DATES[-1].date()))


def test_exactly_one_of_bars_and_panel():
    p = _one_short()
    for bars, kw in ((None, {}), (p, {"panel": p})):
        with pytest.raises(ValueError, match="exactly one"):
            run_sessions(bars, cfg=CFG, cost_model=FLAT_25BPS, start=str(DATES[0].date()),
                         end=str(DATES[-1].date()), **kw)


def test_a_panel_built_under_different_periods_is_refused():
    """The indicator PERIODS are baked into the columns; only thresholds can be re-applied. A panel
    re-gated under other periods is a different panel wearing these column names."""
    from kumo_strategies.strategies.crsi_short import panel_signature

    p = _one_short()
    p["panel_signature"] = panel_signature(replace(CFG, crsi_rank_period=50))
    with pytest.raises(ValueError, match="rebuild it"):
        _run(p)
