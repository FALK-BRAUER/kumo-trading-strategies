"""`runner_sessions` — the one backtest engine, and the properties that make it one.

Ported from `test_runner_cadence.py` and then narrowed to what the single engine promises:

  1. the schedule comes from the CONFIG. There is no `slots`, `signal_lag` or `moo_exits` argument,
     because each of those was a harness-local default that let measured and deployed drift (the
     `signal_lag=0` default produced a published, retracted number on 2026-09-08);
  2. the ranking NEVER sees the session being traded. Live ranks `panel[panel.date <= prior]` at
     every slot, so a name's rank cannot move between 09:35 and 15:40 — and here it must not either;
  3. a decision at t fills at the next bar after t, never the bar containing t (#10 rule 1), and
     the cost is charged at THAT bar's time of day;
  4. the overnight gap dead band is applied at EVERY slot and reaches the SAME verdict at each,
     because its inputs are fixed at the open. `runner_cadence` never applied it at all;
  5. more slots must actually change the fills, and deciding more often must cost more — otherwise
     the cadence axis is inert and a sweep certifies the null (#26, twice).
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import run_sessions
from kumo_strategies.strategies.momentum_rotation.config import (
    ExecutionConfig,
    ExitConfig,
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
SESSIONS = 120
BARS_PER_DAY = 78          # 6.5h of 5-minute bars


class _AllEligible:
    name = "test_all"

    def eligible(self, d) -> set[str]:
        return set(NAMES)


def _bars(seed: int = 11) -> pd.DataFrame:
    """Intraday bars with a real intra-session shape, so extra slots have something to see."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=SESSIONS)
    rows = []
    for i, t in enumerate(NAMES):
        sigma = 0.004 + 0.001 * i
        drift = 0.00035 - 0.00009 * i
        px = 100.0
        for d in days:
            open_ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30)
            steps = rng.normal(drift / BARS_PER_DAY, sigma, BARS_PER_DAY)
            for k, s in enumerate(steps):
                prev = px
                px = px * float(np.exp(s))
                hi, lo = max(prev, px) * (1 + 0.4 * sigma), min(prev, px) * (1 - 0.4 * sigma)
                rows.append((t, open_ts + pd.Timedelta(minutes=5 * k), pd.Timestamp(d),
                             prev, hi, lo, px, 200_000.0))
    return pd.DataFrame(rows, columns=["ticker", "ts", "date", "open", "high", "low",
                                       "close", "volume"])


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _bars()


@pytest.fixture(scope="module")
def run_kw(tmp_path_factory) -> dict:
    p = tmp_path_factory.mktemp("instr") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return {"instruments_path": p,
            "cost_model": CostModel(half_spread_bps={t: 5.0 for t in NAMES}, default_bps=5.0),
            "starting_cash": 100_000.0}


def _cfg(slots: tuple[str, ...] = ("open+5m",), gap: float | None = None) -> MomentumRotationConfig:
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=3, buffer=1),
        score=ScoreConfig(lookback=20, vol_window=40, min_history=25),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=slots, min_abs_gap_pct=gap))


def _run(bars, run_kw, cfg):
    return run_sessions(bars, _AllEligible(), cfg=cfg, **run_kw)


# --- 1. the schedule is the strategy's, not the call's ------------------------------------------

def test_the_runner_has_no_harness_knobs():
    """`slots`, `signal_lag` and `moo_exits` were each a place the backtest could diverge from the
    deployed lane. They are not arguments here, and this test keeps them out."""
    params = set(inspect.signature(run_sessions).parameters)
    for knob in ("slots", "signal_lag", "moo_exits", "entry", "decision_slot"):
        assert knob not in params, f"{knob!r} is a harness knob; put it on the config or drop it"


def test_slots_come_from_the_config_and_an_empty_schedule_is_refused(bars, run_kw):
    with pytest.raises(ValueError, match="decision_slots"):
        _run(bars, run_kw, _cfg(slots=()))
    res = _run(bars, run_kw, _cfg(slots=("open+5m",)))
    assert set(res.report.fills["slot"]) == {"open+5m"}


def test_the_control_slot_trades(bars, run_kw):
    res = _run(bars, run_kw, _cfg())
    assert len(res.report.fills) > 0, "the one-slot control placed no fills; fixture is broken"


# --- 2. the ranking never sees the traded session -------------------------------------------------

def test_the_ranking_is_identical_at_every_slot_of_a_session(bars, run_kw):
    """Live ranks the prior completed session at 09:35, 12:00 and 15:40 alike. So within one
    session every slot's `hold` set must be reachable from the same ranking — which means the set
    of names any slot ENTERS must have been in the morning slot's `hold` or its candidates, and no
    name may enter in the afternoon that the morning could not have chosen on ranking grounds.

    The cheapest observable: the DECISION rows' `hold` for a session differ across slots only by
    exits and re-entries, never by a name appearing that no earlier slot could rank."""
    res = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m")))
    d = res.decisions
    assert set(d["slot"]) == {"open+5m", "open+150m", "close-20m"}
    # A partial-bar ranking would let a name that ranked out at 09:35 rank back IN at 15:40 on
    # today's move alone. Under a fixed ranking, a name that was neither held nor entered at the
    # morning slot can only be entered later if the book made room by exiting — so every later
    # entry must coincide with a strictly smaller-or-equal book at that slot.
    for (_, session), g in d.groupby(["date", "date"]):
        g = g.sort_values("at")
        morning = g.iloc[0]
        for _, row in g.iloc[1:].iterrows():
            late = set(filter(None, row["enter"].split(",")))
            if not late:
                continue
            held_before = set(filter(None, morning["hold"].split(",")))
            # room existed: something left the book between the morning hold and this slot
            assert len(set(filter(None, row["hold"].split(",")))) <= len(held_before) + len(late), (
                f"{session.date()} {row['slot']}: entries without room — a re-rank leaked in")


def test_the_traded_session_is_not_in_its_own_ranking(bars, run_kw):
    """Poison ONE session's intraday bars after the open with an absurd price, then decide LATE in
    that session. A partial-bar ranking (what `runner_cadence` defaulted to) would see the poison
    and change the decision; a ranking on the prior completed session cannot.

    Exit rules are OFF for this test, deliberately: the give-back trail prices off the live quote
    and is SUPPOSED to see today's bars, so with it armed the poison would change exits for the
    right reason and the test would not isolate the ranking. Fills also see today (correct), so
    compare DECISIONS, not fills."""
    # buffer=3 so the recorded `scores` (top n_hold + buffer) cover all six names. The assertion is
    # on the SCORES the slot ranked on, not on whether a trade happened: the vol-normalised score
    # damps a 30% jump (sd rises with it), and a buffer absorbs a rank change without a trade, so
    # a decision-level check was vacuous twice over before this version.
    cfg = replace(_cfg(slots=("close-20m",)), exits=ExitConfig(give_back_frac=None),
                  portfolio=PortfolioConfig(n_hold=3, buffer=3))
    cut = pd.Timestamp("2025-04-15")
    clean = _run(bars, run_kw, cfg).decisions.set_index("date")
    victim = next(t for t in NAMES if t not in set(filter(None, clean.loc[cut, "hold"].split(","))))
    poisoned = bars.copy()
    hit = ((poisoned["date"] == cut) & (poisoned["ticker"] == victim)
           & (poisoned["ts"].dt.time > pd.Timestamp("09:30").time()))
    assert hit.any()
    # x1.30, NOT x1000: a move over `gates.corporate_action_move=0.4` blanks the score for 41
    # sessions, so an absurd poison is silently gated out and the test proves nothing.
    poisoned.loc[hit, ["open", "high", "low", "close"]] *= 1.30
    dirty = _run(poisoned, run_kw, cfg).decisions.set_index("date")

    cols = ["hold", "enter", "exit", "scores"]
    a, b = clean.loc[cut, cols].to_dict(), dirty.loc[cut, cols].to_dict()
    assert a == b, (
        f"the 15:40 decision on the poisoned session changed {a} -> {b} — today's own bar reached "
        f"the ranking. Live cannot do that: it ranks panel[date <= prior] at every slot.")
    # The NEXT session ranks the poisoned bar as a COMPLETED session, so the victim's score there
    # MUST move. If it does not, the poison never reached the score and the assertion above proved
    # nothing.
    nxt = clean.index[clean.index > cut][0]
    s0, s1 = clean.loc[nxt, "scores"][victim], dirty.loc[nxt, "scores"][victim]
    assert s0 != s1, f"{victim}'s score on {nxt.date()} is {s0} clean and {s1} poisoned — inert"


# --- 3. fills are the next bar, costs at that bar's time --------------------------------------

def test_every_fill_is_the_next_bar_after_its_decision(bars, run_kw):
    res = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m")))
    f = res.report.fills
    d = res.decisions.set_index(["date", "slot"])["at"]
    at = pd.to_datetime(f["ts"]).dt.normalize()
    for (_, row), day in zip(f.iterrows(), at):
        when = d.loc[(day, row["slot"])]
        assert pd.Timestamp(row["ts"]) > when, f"{row['symbol']} filled at or before its decision"
        assert pd.Timestamp(row["ts"]) - when <= pd.Timedelta(minutes=5), (
            f"{row['symbol']} filled later than the next bar")


def test_the_time_of_day_cost_profile_is_applied(bars, run_kw):
    """Same notional at the open pays ~2.25x what it pays at the close (measured profile). A runner
    charging one blended rate would make the late slot look no cheaper, which is the exact blindness
    the cadence question exists to remove."""
    res = _run(bars, run_kw, _cfg(slots=("open+5m", "close-20m")))
    f = res.report.fills
    f = f.assign(bps=1e4 * f["cost"] / (f["qty"] * f["price"]))
    early = f[f["slot"] == "open+5m"]["bps"].median()
    late = f[f["slot"] == "close-20m"]["bps"].median()
    assert early > late * 1.5, f"open {early:.2f}bps vs close {late:.2f}bps — no time-of-day profile"


# --- 4. the gap dead band, at every slot, same verdict ------------------------------------------

def test_the_gap_rule_is_applied_and_declines_something(bars, run_kw):
    """`runner_cadence` never read `min_abs_gap_pct`. Here a wide dead band must decline entries."""
    off = _run(bars, run_kw, _cfg(slots=("open+5m",), gap=None))
    on = _run(bars, run_kw, _cfg(slots=("open+5m",), gap=0.02))
    declined = on.decisions["gap_declined"].str.len().gt(0).sum()
    assert declined > 0, "a 2% dead band on a 0.4%-sigma panel declined nothing — the rule is inert"
    assert not on.report.fills.equals(off.report.fills), "fills identical with the gap rule armed"


def test_the_gap_verdict_is_the_same_at_every_slot(bars, run_kw):
    """The gap is today's open over yesterday's close, fixed at 09:30. A name declined at 09:35 must
    be declined at 12:00 and 15:40 if it is a candidate again — and a name admitted in the morning
    is guaranteed admission all day, which is exactly why the filter cannot guard re-entry."""
    res = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m"), gap=0.01))
    d = res.decisions
    for session, g in d.groupby("date"):
        declined_by_slot = [set(filter(None, x.split(","))) for x in g["gap_declined"]]
        entered_by_slot = [set(filter(None, x.split(","))) for x in g["enter"]]
        declined_any = set().union(*declined_by_slot)
        entered_any = set().union(*entered_by_slot)
        assert not (declined_any & entered_any), (
            f"{session.date()}: {sorted(declined_any & entered_any)} declined at one slot and "
            f"entered at another — the gap verdict moved within the day")


# --- 5. the cadence axis is live ----------------------------------------------------------------

def test_more_slots_actually_change_the_fills(bars, run_kw):
    one = _run(bars, run_kw, _cfg(slots=("open+5m",))).report.fills
    three = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m"))).report.fills
    assert len(three) != len(one) or not three.equals(one), "extra slots changed nothing"


def test_deciding_more_often_costs_more(bars, run_kw):
    one = _run(bars, run_kw, _cfg(slots=("open+5m",))).report.kpis()["total_cost_usd"]
    three = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m"))).report.kpis()[
        "total_cost_usd"]
    assert three > one, f"three slots cost ${three:.0f} vs one ${one:.0f} — turnover is free here"


def test_same_session_reentries_are_counted_not_hidden(bars, run_kw):
    """The live runner can sell on give-back and re-buy the same name at the next slot. This engine
    reproduces that on purpose and must report it, so the count cannot vanish into the fills."""
    res = _run(bars, run_kw, _cfg(slots=("open+5m", "open+150m", "close-20m")))
    assert hasattr(res, "same_session_reentries")
    from_decisions = res.decisions["reentry"].str.len().gt(0).sum()
    assert (res.same_session_reentries > 0) == (from_decisions > 0)


# --- 6. the design: pause + redistribute (target-based rebalancing), and the weight cap ---------

def _cfg_design(cap: float = 1.0) -> MomentumRotationConfig:
    c = _cfg(slots=("open+5m",))
    return replace(c, portfolio=replace(c.portfolio, size_to_book=True, rebalance_band=0.10,
                                        max_weight=cap))


def test_rebalancing_trims_with_partial_sells_and_never_to_zero(bars, run_kw):
    """2026-09-08: "the capital redistributes when something is unpaused." A held name that
    drifts above its target is TRIMMED — a partial SELL that leaves the position open — and the
    freed cash funds the buys in the same decision. `runner_cadence` could not do this at all."""
    res = _run(bars, run_kw, _cfg_design())
    f = res.report.fills
    assert "rebalance" in f.columns and f["rebalance"].fillna(False).any(), "no rebalance fills"
    rb_sells = f[(f["rebalance"] == True) & (f["side"] == "SELL")]
    assert len(rb_sells) > 0, "rebalancing never trimmed anything"
    # A trim is not an exit: replay holdings and check no rebalance SELL takes a name to zero.
    held: dict[str, float] = {}
    for _, row in f.sort_values("ts").iterrows():
        q = held.get(row["symbol"], 0.0) + (row["qty"] if row["side"] == "BUY" else -row["qty"])
        if row.get("rebalance") == True and row["side"] == "SELL":
            assert q >= 1, f"rebalance sold {row['symbol']} to zero — that is an exit, not a trim"
        held[row["symbol"]] = q


def test_rebalancing_changes_the_result_and_fills_the_book(bars, run_kw):
    plain = _run(bars, run_kw, _cfg(slots=("open+5m",)))
    design = _run(bars, run_kw, _cfg_design())
    assert not plain.report.fills.equals(design.report.fills), "rebalance_band was inert"


def test_max_weight_caps_entry_notional(bars, run_kw):
    """`max_weight` only reshaped `dec.weights` before, which cannot cap a one-name book: under
    size_to_book a single admitted name was sized to equity/1 (NTR at 101% of equity, 2026-03-12).
    The cap must hold at sizing."""
    res = _run(bars, run_kw, _cfg_design(cap=0.20))
    f = res.report.fills
    e = res.report.active_equity()
    buys = f[f["side"] == "BUY"].copy()
    buys["d"] = pd.to_datetime(buys["ts"]).dt.normalize()
    w = buys["qty"] * buys["price"] / buys["d"].map(e).astype(float)
    # The cap is applied to equity AT THE DECISION INSTANT (cash + marks at that bar); this divides
    # by end-of-day equity, which has moved by then, so allow a few percent of drift. Uncapped, an
    # entry on this fixture takes ~33% (n_hold=3, size_to_book), so 22% is the cap binding.
    assert w.max() <= 0.20 * 1.10, f"an entry took {100*w.max():.0f}% of equity with max_weight=0.20"
    uncapped = _run(bars, run_kw, _cfg_design(cap=1.0)).report.fills
    assert not uncapped.equals(f), "max_weight changed nothing — the cap is not binding in this fixture"


def test_trade_from_withholds_trading_but_not_scoring(bars, run_kw):
    """Slicing the panel starves the score (93 trades where the full panel makes 241); `trade_from`
    must score the warmup and simply not trade it."""
    cut = pd.Timestamp("2025-04-01")
    res = run_sessions(bars, _AllEligible(), cfg=_cfg(), trade_from=cut, **run_kw)
    f = res.report.fills
    assert len(f) > 0
    assert pd.to_datetime(f["ts"]).min() >= cut
    sliced = run_sessions(bars[bars["date"] >= cut], _AllEligible(), cfg=_cfg(), **run_kw)
    assert len(f) > len(sliced.report.fills), "warmup made no difference — scoring was not warm"


# -- ATR-scaled exits, ported from `test_runner_cadence.py` when that runner was folded (#270) -------

def _cfg_stop(atr: float | None) -> MomentumRotationConfig:
    return replace(_cfg(), exits=ExitConfig(give_back_frac=0.5, stop_loss_atr=atr))


def test_the_runner_supplies_ATR_so_ATR_RULES_ARE_NOT_INERT(bars, run_kw):
    """The cadence runner once computed no ATR at all, so every ATR-scaled exit rule was dead in it:
    four stop arms from 1.5 to 3.0 ATR produced byte-identical fills to a no-stop control across
    three quarters, and only the arms-differ gate caught it. Asserts behaviour, not plumbing — a stop
    tight enough to fire has to move trades."""
    from kumo_strategies.backtesting.ablation import fill_signature
    tight = run_sessions(bars, _AllEligible(), cfg=_cfg_stop(0.5), **run_kw)
    none_ = run_sessions(bars, _AllEligible(), cfg=_cfg_stop(None), **run_kw)
    assert len(tight.report.fills), "the tight-stop arm traded nothing at all"
    assert not fill_signature(tight).equals(fill_signature(none_)), \
        "the stop was inert — this runner is not supplying ATR"


def test_a_wider_stop_fires_less_than_a_tighter_one(bars, run_kw):
    """Orders the band. A stop whose distance does not change what it catches is not measuring ATR,
    it is measuring something constant."""
    from kumo_strategies.backtesting.ablation import fill_signature
    tight = fill_signature(run_sessions(bars, _AllEligible(), cfg=_cfg_stop(0.5), **run_kw))
    wide = fill_signature(run_sessions(bars, _AllEligible(), cfg=_cfg_stop(6.0), **run_kw))
    assert not tight.equals(wide)


def test_ATR_cannot_read_the_session_that_is_still_forming(bars, run_kw):
    """Bars that arrive AFTER the decision instant must not change that decision.

    A truncation test cannot show this — trimming the panel leaves the current session identical in
    both runs, so it only catches look-ahead into FUTURE sessions. This perturbs the tail of the
    LAST session instead, where no later decision can legitimately consume it: every fill must be
    byte-identical. It matters because ATR sets the stop DISTANCE: a partial range understates it,
    the stop tightens toward the open and fires on moves it was never configured to catch.
    """
    from kumo_strategies.backtesting.ablation import fill_signature
    cfg = _cfg_stop(0.5)
    base = fill_signature(run_sessions(bars, _AllEligible(), cfg=cfg, **run_kw))

    poisoned = bars.copy()
    last = pd.to_datetime(poisoned["date"]).max()
    ts = pd.to_datetime(poisoned["ts"])
    late = (pd.to_datetime(poisoned["date"]) == last) & (ts.dt.hour * 60 + ts.dt.minute > 9 * 60 + 40)
    assert late.any(), "fixture has no post-decision bars to poison"
    poisoned.loc[late, "high"] = poisoned.loc[late, "high"] * 5.0
    poisoned.loc[late, "low"] = poisoned.loc[late, "low"] * 0.2

    after = fill_signature(run_sessions(poisoned, _AllEligible(), cfg=cfg, **run_kw))
    assert base.equals(after), \
        "bars from after the decision instant changed a decision — ATR is reading the forming session"
