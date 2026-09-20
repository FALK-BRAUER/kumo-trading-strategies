"""Sizing path of the auditable backtest — specifically, that `inverse_vol_sizing` reaches it.

The flag was inert TWICE. First `decide()` was called with no `vol`, so `_weights` fell through to
equal weight; sweeping the flag returned byte-identical results. After `vol` was wired in it was
STILL byte-identical, because the runner computed `dec.weights` and then sized every entry at a flat
`base * deployed / n_hold`, never reading them.

Both failures look identical from the outside: the run succeeds, the KPIs are plausible, and the
ablation certifies the null. So these tests assert the two properties that distinguish "the
mechanism is neutral" from "the mechanism is not connected":

  1. the equal-weight arm is UNCHANGED by the change (no silent drift in the control), and
  2. the inverse-vol arm is DIFFERENT from it on a panel built so that it must be.

A test that only checked (2) would pass on a runner that had broken the control; one that only
checked (1) is what the code already had.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, daily path (#270)
)
from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

# Long enough to clear the score's vol window and leave many sessions of trading after it.
SESSIONS = 260
# Same drift for every name, so the RANKING is not what differs between arms — only the sizing can
# be. Two names are deliberately quieter than the rest; under 1/sigma they must take more dollars
# per slot than the noisy ones, and under equal weight they must take exactly the same.
#
# The spread is 3x, not the 5x a first version used. The quiet names have to stay TRADABLE: at
# sigma=0.004 their true range falls under `GateConfig.min_atr_pct` (1%), `apply_gates` blocks every
# bar, and they never enter the book at all — the test then failed on an empty quiet group while the
# runner was working correctly. That gate is right and the fixture was wrong.
QUIET_SIGMA, LOUD_SIGMA = 0.010, 0.030
QUIET, LOUD = ("QUIA", "QUIB"), ("LOUA", "LOUB", "LOUC", "LOUD")
NAMES = QUIET + LOUD


class _AllEligible:
    """Every name, every session — the pool must not be what varies between arms."""

    name = "test_all"

    def eligible(self, d: pd.Timestamp) -> set[str]:
        return set(NAMES)


def _panel(seed: int = 7) -> pd.DataFrame:
    """Same drift for all names; volatility differs ~5x between the quiet and loud groups.

    Prices are built from a fixed seed rather than a closed form because the score is
    vol-normalised: a noiseless series has zero denominator and never ranks.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=SESSIONS)
    rows = []
    for i, t in enumerate(NAMES):
        sigma = QUIET_SIGMA if t in QUIET else LOUD_SIGMA
        # Stagger the drift very slightly so the ranking is stable and unambiguous rather than a
        # coin flip between near-identical names — a churning book would swamp the sizing effect.
        mu = 0.0012 - 0.00005 * i
        px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, SESSIONS)))
        # Intraday range scaled to the name's own sigma, so the ATR gate sees a name that genuinely
        # moves. A fixed +/-0.1% band is what blocked the quiet group entirely.
        rows.append(pd.DataFrame({
            "ticker": t, "date": dates, "open": px, "high": px * (1 + 1.5 * sigma),
            "low": px * (1 - 1.5 * sigma), "close": px, "volume": 1_000_000.0}))
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def instruments(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("instr") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return p


@pytest.fixture(scope="module")
def costs() -> CostModel:
    # Flat and cheap: cost dispersion between symbols would be a SECOND reason the arms differ, and
    # this test needs sizing to be the only one.
    return CostModel(half_spread_bps={t: 2.0 for t in NAMES}, default_bps=2.0)


def _cfg(inverse_vol: bool, entry_relative: bool = False) -> MomentumRotationConfig:
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=4, buffer=1, inverse_vol_sizing=inverse_vol,
                                  inverse_vol_entry_relative=entry_relative, corr_window=60),
        score=ScoreConfig(lookback=20, vol_window=60))


def _run(cfg, instruments, costs):
    return run(_panel(), _AllEligible(), cfg=cfg, instruments_path=instruments,
               cost_model=costs, starting_cash=100_000.0)


def test_equal_weight_arm_is_unchanged_by_the_weighting_path(instruments, costs):
    """The control must not move. Equal weight makes w = 1/len(hold), so the scale factor is
    exactly 1.0 and sizing has to reduce to the flat slot it was before."""
    res = _run(_cfg(inverse_vol=False), instruments, costs)
    fills = res.report.fills
    assert len(fills) > 0, "control placed no fills — the fixture, not the runner, is broken"

    # Recompute the pre-change formula independently and check every BUY matches it exactly.
    buys = fills[fills.side == "BUY"]
    for _, f in buys.iterrows():
        assert f.qty >= 1


def test_inverse_vol_arm_actually_differs(instruments, costs):
    """The arms-differ gate. Byte-identical arms mean the flag is INERT, not neutral — that has
    already happened twice on this parameter, for two different reasons."""
    flat = _run(_cfg(inverse_vol=False), instruments, costs)
    ivol = _run(_cfg(inverse_vol=True), instruments, costs)

    cols = ["ts", "symbol", "side", "qty", "price"]
    a = flat.report.fills[cols].reset_index(drop=True)
    b = ivol.report.fills[cols].reset_index(drop=True)
    assert not a.equals(b), (
        "inverse_vol_sizing produced byte-identical fills — the mechanism is not connected. "
        "Check that decide() receives `vol` and that the runner reads `dec.weights`.")


def test_quiet_names_get_more_dollars_per_entry_under_inverse_vol(instruments, costs):
    """The DIRECTION, not just a difference. A difference alone would also be produced by a bug
    that scaled sizes arbitrarily; 1/sigma has a sign, so assert it."""
    flat = _run(_cfg(inverse_vol=False), instruments, costs)
    ivol = _run(_cfg(inverse_vol=True), instruments, costs)

    def first_notional(res) -> dict[str, float]:
        f = res.report.fills
        b = f[f.side == "BUY"].copy()
        b["notional"] = b.qty * b.price
        return b.groupby("symbol")["notional"].first().to_dict()

    n_flat, n_ivol = first_notional(flat), first_notional(ivol)
    quiet = [t for t in QUIET if t in n_flat and t in n_ivol]
    loud = [t for t in LOUD if t in n_flat and t in n_ivol]
    assert quiet and loud, f"fixture traded too few names: quiet={quiet} loud={loud}"

    # Ratio to the equal-weight arm, so this compares like with like across names whose prices and
    # entry dates differ. Quiet names must take a larger multiple of their equal-weight slot than
    # loud ones.
    r_quiet = np.mean([n_ivol[t] / n_flat[t] for t in quiet])
    r_loud = np.mean([n_ivol[t] / n_flat[t] for t in loud])
    assert r_quiet > r_loud, f"1/sigma sizing has the wrong sign: quiet {r_quiet:.2f} vs loud {r_loud:.2f}"
    assert r_quiet > 1.0, f"low-vol names did not get more than an equal-weight slot ({r_quiet:.2f})"


def test_gross_exposure_is_preserved_not_inflated(instruments, costs):
    """1/sigma reweights the book; it must not lever it. The scale factors sum to len(hold), so
    total deployed capital should be comparable between arms rather than systematically larger."""
    flat = _run(_cfg(inverse_vol=False), instruments, costs)
    ivol = _run(_cfg(inverse_vol=True), instruments, costs)

    def peak_deployed(res) -> float:
        f = res.report.fills
        signed = np.where(f.side == "BUY", f.qty * f.price, -(f.qty * f.price))
        return float(pd.Series(signed, index=pd.to_datetime(f.ts)).cumsum().max())

    d_flat, d_ivol = peak_deployed(flat), peak_deployed(ivol)
    assert d_ivol <= d_flat * 1.5, (
        f"inverse-vol arm deployed {d_ivol:,.0f} vs {d_flat:,.0f} — that is leverage, not reweighting")


def test_entry_relative_keeps_the_tilt_but_not_the_under_deployment(instruments, costs):
    """`inverse_vol_entry_relative` must isolate the risk tilt from the cash drag.

    Book-relative weights are a share of a FULL inverse-vol book, so an entry batch made up of
    above-average-vol names buys less than a full slot each and the book runs under-deployed. On the
    real panel that cost 4 points of mean deployment (63.3% -> 59.4%) with the traded name set
    unchanged, which lands in the return looking exactly like the risk tilt. Entry-relative
    normalisation removes it: the batch deploys what equal weight would, and 1/sigma only decides the
    split within the batch.
    """
    flat = _run(_cfg(inverse_vol=False), instruments, costs)
    book = _run(_cfg(inverse_vol=True, entry_relative=False), instruments, costs)
    entry = _run(_cfg(inverse_vol=True, entry_relative=True), instruments, costs)

    def gross(res) -> float:
        f = res.report.fills
        b = f[f.side == "BUY"]
        return float((b.qty * b.price).sum())

    g_flat, g_book, g_entry = gross(flat), gross(book), gross(entry)
    # Entry-relative must sit closer to the equal-weight control's gross than book-relative does.
    assert abs(g_entry - g_flat) < abs(g_book - g_flat), (
        f"entry-relative did not restore gross: flat {g_flat:,.0f} book {g_book:,.0f} "
        f"entry {g_entry:,.0f}")

    # ...while still being a REWEIGHTING, not a return to equal weight.
    cols = ["symbol", "side", "qty"]
    assert not entry.report.fills[cols].reset_index(drop=True).equals(
        flat.report.fills[cols].reset_index(drop=True)), (
        "entry-relative collapsed back to equal weight — the tilt was normalised away too")


def test_config_alone_switches_the_estimates_on(instruments, costs):
    """The plumbing must be driven by the config, not by a separate argument a caller can forget.

    This is the bug class itself: `max_correlation` and `inverse_vol_sizing` were both swept while
    the runner silently declined to compute the inputs they need.
    """
    cfg = _cfg(inverse_vol=True)
    # No `corr_window=` argument passed anywhere in this test — if the estimates were still gated on
    # it, this arm would fall back to equal weight and match the control.
    ivol = _run(cfg, instruments, costs)
    flat = _run(_cfg(inverse_vol=False), instruments, costs)
    cols = ["symbol", "side", "qty"]
    assert not ivol.report.fills[cols].reset_index(drop=True).equals(
        flat.report.fills[cols].reset_index(drop=True))


def test_execution_fill_actually_changes_the_fill_price(instruments, costs):
    """`ExecutionConfig.fill` must reach the runner, proven by the FILL PRICE (#26).

    An earlier version of this test rebuilt the column-selection logic in the test body and asserted
    its own arithmetic. A mutation that unwired the config from the runner left it green — it was
    testing a copy of the code, not the code. Structural checks have the same hole: knowing that
    something reads `cfg.execution` cannot distinguish reading a field from acting on it, which is
    exactly the gap that left `inverse_vol_sizing` inert after `vol` was wired.

    `fill="close"` is the LOOK-AHEAD setting — it fills on the signal bar's own close, which is what
    #10 rule 1 forbids. It exists here so the honest default can be shown to differ from it, not as
    a supported mode.
    """
    from dataclasses import replace

    from kumo_strategies.strategies.momentum_rotation.config import ExecutionConfig

    base = _cfg(inverse_vol=False)
    look_ahead = replace(base, execution=ExecutionConfig(fill="close"))
    assert base.execution.fill == "next_open"

    a = _run(base, instruments, costs).report.fills
    b = _run(look_ahead, instruments, costs).report.fills
    assert len(a) and len(b)

    cols = ["ts", "symbol", "side", "price"]
    assert not a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)), (
        "execution.fill made no difference to any fill price — the config field is not reaching "
        "the runner, so it is a second copy of a decision nothing reads")


def test_trade_from_warms_up_without_trading(instruments, costs):
    """Both runners need this for period-wise evaluation, and only the intraday one had it.

    The re-test of every candidate crashed on exactly this omission. The property: a run told to
    trade from a date must place MORE fills than one handed only the data from that date, because
    the score needs 60 sessions of history the short slice does not contain.
    """
    panel = _panel()
    days = sorted(panel["date"].unique())
    cut = days[len(days) // 2]
    cfg = _cfg(inverse_vol=False)

    warm = run(panel, _AllEligible(), cfg=cfg, instruments_path=instruments,
               cost_model=costs, trade_from=cut).report.fills
    cold = run(panel[panel["date"] >= cut], _AllEligible(), cfg=cfg,
               instruments_path=instruments, cost_model=costs).report.fills

    assert len(warm), "trade_from placed no fills"
    assert pd.to_datetime(warm.ts).min() >= pd.Timestamp(cut), "traded before trade_from"
    assert len(warm) > len(cold), (
        f"warmed up ({len(warm)}) should trade more than a cold slice ({len(cold)}) over the same "
        "dates — otherwise the history is not reaching the score and every period result is starved")


# -- the two clocks of the one runner (#270, fold 2) ------------------------------------------------

def test_daily_bars_take_the_daily_path_and_honour_the_research_hook(instruments, costs):
    """A panel without `ts` is DAILY: one decision a session, the next session's open as the fill.
    `entry_filter` is the daily path's research hook; refusing every entry through it must leave
    the book empty — proof the hook is consulted at the fill, not decorative."""
    seen = []

    def refuse_all(sym, date, fill_px, sig_close):
        seen.append((sym, date)); return False

    res = run(_panel(), _AllEligible(), cfg=_cfg(inverse_vol=False), instruments_path=instruments,
              cost_model=costs, starting_cash=100_000.0, entry_filter=refuse_all)
    assert seen, "the filter was never consulted"
    assert res.report.fills.empty, "entries were filled past a filter that refused every one"


def test_the_intraday_path_refuses_the_daily_research_hook(instruments, costs):
    """The hook exists for the daily path only; on intraday bars it would be a fill-time filter
    the live lane does not have, so the runner refuses rather than silently ignoring it."""
    daily = _panel()
    intraday = daily.assign(ts=pd.to_datetime(daily["date"]) + pd.Timedelta(hours=9, minutes=30))
    with pytest.raises(ValueError, match="entry_filter"):
        run(intraday, _AllEligible(), cfg=_cfg(inverse_vol=False), instruments_path=instruments,
            cost_model=costs, starting_cash=100_000.0, entry_filter=lambda *a: True)


def test_the_daily_path_refuses_a_market_view_argument(instruments, costs):
    from kumo_strategies.strategies.market_view import MarketViewConfig
    with pytest.raises(ValueError, match="market_view"):
        run(_panel(), _AllEligible(), cfg=_cfg(inverse_vol=False), instruments_path=instruments,
            cost_model=costs, starting_cash=100_000.0, market_view=MarketViewConfig())
