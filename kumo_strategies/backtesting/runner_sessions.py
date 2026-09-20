"""THE backtest runner. N decisions a session, live execution semantics, config-driven throughout.

    from kumo_strategies.backtesting.runner_sessions import run_sessions

WHY THIS REPLACES TWO ENGINES
-----------------------------
On 2026-09-08 the same config produced 15.28% through `runner_verified` and 9.30% through
`runner_cadence` — six points, on one strategy, and neither engine could express what the live lane
actually does. `FINDINGS-cadence.md` had recorded that disagreement three weeks earlier and
pre-registered it as invalidating; nothing reconciled it, and every downstream number kept using the
engine that had not been flagged. Two engines that can disagree must not both be authoritative.

The split also had a harder cost: BCTROT has traded three slots a session since 2026-09-04 and
NEITHER engine could model it as deployed. `runner_verified` has no intra-session concept at all;
`runner_cadence` has slots but has never applied `min_abs_gap_pct`, which is the filter BCTROT's
morning slot exists for. The repo could not backtest its own live strategy.

WHAT THE SIX POINTS WERE, so nobody re-derives it: `runner_verified` fills at the NEXT SESSION'S
09:30 opening print. That is not look-ahead — the signal is the prior session's close, so the fill
strictly follows its information — but it is obtainable only with a MARKET-ON-OPEN order, and
`runtime/nautilus/broker.py:81` submits `order_factory.market(...)` with `TimeInForce.DAY` when the
slot fires. Measured on the traded fills, the 09:30 -> 09:40 drift is +16.28bps on buys and
-17.06bps on sells, and notional-weighted it is -6.03pp against an observed gap of -5.98pp. So the
daily engine priced an order type this strategy does not send.

THIS RUNNER FILLS THE WAY THE LANE FILLS: a market order at the slot, taking the next bar.

THE THREE KNOBS THAT ARE NOT KNOBS HERE
---------------------------------------
Every harness-local default in the old engines was a place where measured and deployed could drift,
and each one did.

`slots` COMES FROM THE CONFIG, not from an argument. `cfg.execution.decision_slots` is what cockpit
deploys, so a backtest that took its schedule from a call site could measure a schedule nothing runs.
One decision a session is `("open+5m",)` — the degenerate case of this code path, not another engine.

`signal_lag` DOES NOT EXIST. `runner_cadence` defaulted it to 0, which puts today's PARTIAL bar in
the ranking; live cannot do that, because `momentum_rotation._try_decide` sets
`prior = panel.loc[panel.date < due].max()` and ranks `panel[panel.date <= prior]` at EVERY slot. A
daily-bar ranking cannot move within a session live, and it must not here. That default produced a
published, retracted result — a lane reported as losing 4.78pp when the corrected comparison had it
ahead. The correct behaviour is not a setting.

`moo_exits` DOES NOT EXIST, for the same reason: live sends day market orders, never
market-on-open. It was a diagnostic for the gap above, and the gap is now explained.

WHAT IS DELIBERATELY NOT FIXED HERE
-----------------------------------
Across slots a name sold by the give-back trail can be re-entered by the next slot's ranking: the
trail runs in the runner on live prices, `decide` receives only `set(held)`, and the ranking is
identical all day, so the name is still inside `n_hold + buffer` and the book has just made room.
The re-entry also resets `TrailState`.

THAT IS WHAT THE LIVE RUNNER DOES. `PgSessionRunner` re-reads `held_qty` each slot and has no
same-session re-entry guard, so BCTROT can sell on give-back at 09:35 and buy back at 12:00 on real
money. Reproducing it is how it was found. The gap dead band cannot catch it either, and provably
so: the gap is `today's open / yesterday's close`, fixed at the open, so a name that passed the
filter to be bought this morning is GUARANTEED to pass it again this afternoon.

So this runner reproduces the behaviour and COUNTS it, in `decisions` and in the returned summary.
The guard belongs in the live runner, not in a backtest that exists to tell the truth about live.

THE SECOND THING REPRODUCED ON PURPOSE (codex review of PR #121, P2): `evaluate_exits` advances
`sessions_held`, `sessions_since_high` and the confirmation counters ON EVERY CALL, and this runner
calls it once per SLOT, so on a three-slot lane a position accrues three "sessions" per calendar
session. That is what `PgSessionRunner` does too: `_trail_exits` loads the persisted trail each
slot, evaluates, parks the advanced state in `_pending_trail`, and `run()` persists it after the
slot's decision row — once per slot. `max_hold_days`, `stall_days` and `give_back_confirm_sessions`
therefore mean "slots", not "sessions", on BCTROT live. None of those rules is armed on the deployed
config (only `give_back_frac`), so today's numbers are unaffected; the defect is filed on #120 and
the fix belongs in the live runner and here together, in that order.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.result import (
    BacktestResult,
)
from kumo_strategies.strategies.market_view import (
    MarketViewConfig,
)
from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
from kumo_strategies.strategies.momentum_rotation.slots import validate


def run_sessions(
    bars: pd.DataFrame | None,
    source: CandidateSource | None = None,
    *,
    cfg: MomentumRotationConfig,
    instruments_path: str | Path | None = None,
    cost_model: CostModel,
    instrument_type: dict[str, str] | None = None,
    starting_cash: float = 100_000.0,
    deployed: float | None = None,
    compound: bool = True,
    n_trials: int = 1,
    trade_from: pd.Timestamp | None = None,
    benchmarks: dict[str, pd.Series] | None = None,
    market_view: MarketViewConfig | None = None,
    entry_filter: Callable[[str, pd.Timestamp, float, float], bool] | None = None,
    fill_price_col: str = "open",
    fill_hour: int = 9,
    fill_minute: int = 35,
    rebalance_period: str | None = None,
    rebalance_holds: bool = True,
    delisting_mode: str = "lastpx",
    adjustment: str | None = None,
    panel: pd.DataFrame | None = None,
    start=None,
    end=None,
    borrow_annual: float = 0.20,
    borrow: dict[str, float | None] | None = None,
    entry_fill=None,
) -> BacktestResult:
    """`bars`: ticker/date/open/high/low/close/volume, plus `ts` when the bars are intraday.

    FOUR FAMILIES, ONE ENTRY POINT (#270). The config's TYPE picks the family: a
    `MomentumRotationConfig` runs the rotation family below (two clocks); a `CrsiShortConfig` the
    SHORT family; a `SmhGldSleeveConfig` the SLEEVE family (`sleeve.py` — two fixed-weight legs
    rebalanced at the close when the drift leaves the band); a
    `QC27TechInverseVolConfig` or `QC345RotationConfig` runs the MONTHLY family (`monthly.py` —
    rebalance calendar, target weights, the QC27 stop and cash sweep, the QC345 ATR trail), where
    `source` is the universe table that family filters on (QC27: sectors; QC345: assets) and the
    last six keyword arguments are that family's research knobs (fill column and instant, cadence
    override, hold-resizing, delisting marks). On the rotation family those six must stay at their
    defaults — they are refused, not ignored.

    TWO CLOCKS, ONE RUNNER (#270). With a `ts` column the bars are intraday: a slot is an instant
    inside the session and a fill is the next bar after it — the live lane's shape, N decisions a
    session. Without `ts` the bars are DAILY: one decision a session on the close, filled at the
    next session's opening print (`_run_daily`, the former `runner_verified`). The daily path is
    where every MOMENTUM number before 2026-09-08 came from and it stays reproducible to the cent;
    the intraday path is what the deployed lane does. They share everything but the clock.

    `trade_from` withholds trading while still SCORING everything earlier, so the 25-session
    `min_history` and the 60-session volatility window are warm. Slicing the panel instead starves
    the score and reads as a strategy result: 93 trades where the full panel makes 241.

    Everything else comes from `cfg`. There is no argument here that a deployed config cannot set.
    """
    from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
    from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig
    from kumo_strategies.strategies.qc345_rotation import QC345RotationConfig

    if isinstance(cfg, CrsiShortConfig):
        # THE SHORT FAMILY: daily bars (or a built panel), a resting limit, the short side, borrow
        # carry, a self-reconciling ledger. `start`/`end` bound the TRADED window; everything before
        # is scored so the 100-session windows are warm. `source` is unused: the universe is a screen.
        from kumo_strategies.backtesting.families.short import CrsiShort
        from kumo_strategies.backtesting.result import ShortBacktestResult
        if start is None or end is None:
            raise ValueError("the short family needs `start` and `end` — the traded window")
        fam = CrsiShort(bars, cfg=cfg, cost_model=cost_model, adjustment=adjustment, panel=panel,
                        start=start, end=end, starting_cash=starting_cash, borrow_annual=borrow_annual,
                        borrow=borrow, n_trials=n_trials, benchmarks=benchmarks, entry_fill=entry_fill)
        res = fam.engine.run(fam)
        d = res.diagnostics
        return ShortBacktestResult(report=res.report, decisions=res.decisions, diagnostics={},
                                   covers=d["covers"], borrow_paid=d["borrow_paid"], refused=d["refused"])

    from kumo_strategies.strategies.smhgld_sleeve import SmhGldSleeveConfig
    if isinstance(cfg, SmhGldSleeveConfig):
        # THE SLEEVE FAMILY: two fixed-weight legs rebalanced at the close on the config's cadence
        # when the drift leaves the band. `source` is unused: the universe IS the config.
        from kumo_strategies.backtesting.families.sleeve import SmhGldSleeve
        fam = SmhGldSleeve(bars, cfg=cfg, cost_model=cost_model, starting_cash=starting_cash,
                           n_trials=n_trials, benchmarks=benchmarks)
        return fam.engine.run(fam)

    monthly_kw = {"fill_price_col": fill_price_col, "fill_hour": fill_hour, "fill_minute": fill_minute}
    # THE DEPLOYED FRACTION IS A FAMILY DEFAULT, not one number: the rotation family was measured at
    # 0.75 (a cash buffer — a fully-deployed book once drove the account negative), QC345 at 1.0.
    # A single default here would silently re-measure one of them; the gate for the fold caught
    # exactly that (QC345 176.7 % → 121.9 % under 0.75). None means "what this family recorded".
    if deployed is None:
        deployed = 1.0 if isinstance(cfg, QC345RotationConfig) else 0.75
    if isinstance(cfg, QC27TechInverseVolConfig):
        from kumo_strategies.backtesting.families.monthly import QC27Monthly
        fam = QC27Monthly(bars, source, cfg=cfg, cost_model=cost_model, starting_cash=starting_cash,
                          n_trials=n_trials, benchmarks=benchmarks, rebalance_period=rebalance_period,
                          rebalance_holds=rebalance_holds, market_view=market_view, **monthly_kw)
        return fam.engine.run(fam)
    if instruments_path is None:
        raise ValueError("instruments_path is required — every family but QC27 (which sizes and"
                         " fills without Nautilus instruments) resolves symbols through it")
    if isinstance(cfg, QC345RotationConfig):
        from kumo_strategies.backtesting.families.monthly import QC345Monthly
        fam = QC345Monthly(bars, source, cfg=cfg, instruments_path=instruments_path,
                           cost_model=cost_model, starting_cash=starting_cash, deployed=deployed,
                           delisting_mode=delisting_mode, n_trials=n_trials, benchmarks=benchmarks,
                           market_view=market_view, **monthly_kw)
        return fam.engine.run(fam)
    if (fill_price_col, fill_hour, fill_minute, rebalance_period, rebalance_holds, delisting_mode) \
            != ("open", 9, 35, None, True, "lastpx"):
        raise ValueError("fill_price_col/fill_hour/fill_minute/rebalance_period/rebalance_holds/"
                         "delisting_mode are the monthly family's knobs; the rotation family reads "
                         "its fill and schedule from cfg.execution")
    slots = tuple(cfg.execution.decision_slots or ())
    if not slots:
        raise ValueError(
            "cfg.execution.decision_slots is empty. The schedule is part of the strategy, not of "
            "the call — set it on the config the lane deploys.")
    validate(slots)
    from kumo_strategies.backtesting.families.rotation import DailyRotation, IntradayRotation
    if "ts" not in bars.columns:
        if market_view is not None:
            raise ValueError("market_view is not modelled on the daily path; pass intraday bars")
        fam = DailyRotation(bars, source, cfg=cfg, instruments_path=instruments_path,
                            cost_model=cost_model, instrument_type=instrument_type,
                            starting_cash=starting_cash, deployed=deployed, compound=compound,
                            n_trials=n_trials, trade_from=trade_from, benchmarks=benchmarks,
                            entry_filter=entry_filter)
        return fam.engine.run(fam)
    if entry_filter is not None:
        raise ValueError("entry_filter is a daily-path research hook; the intraday path has no "
                         "fill-time filter beyond the configured gap rule")
    fam = IntradayRotation(bars, source, cfg=cfg, instruments_path=instruments_path,
                           cost_model=cost_model, instrument_type=instrument_type,
                           starting_cash=starting_cash, deployed=deployed, compound=compound,
                           n_trials=n_trials, trade_from=trade_from, benchmarks=benchmarks,
                           market_view=market_view)
    res = fam.engine.run(fam)
    # Surfaced rather than buried: this is a LIVE behaviour the runner reproduces on purpose.
    res.same_session_reentries = fam.reentries          # type: ignore[attr-defined]
    return res
