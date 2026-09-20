"""ENGINE-PARITY REPLAY (not a runner, #270): QC27 through a real Nautilus BacktestEngine (#33). Lives
beside the parity tests: its value is DISAGREEMENT with the one runner, never a number of its own. Must run in its own process — a second
BacktestEngine per interpreter aborts natively.

WHY THIS EXISTS ALONGSIDE `runner_qc27_verified.py`. That one is a pandas simulation: it computes
what the decision WOULD have been and marks positions arithmetically. This one runs the actual
`QC27RotationStrategy` — the same class cockpit would register — through the order engine, so
submits, fills, rejections and position accounting are the engine's rather than the simulator's.

THE COMPARISON IS THE DELIVERABLE, not this run's numbers. Two implementations of one strategy that
agree tell you the pandas figure is trustworthy; where they disagree, one of them is wrong and the
disagreement is the finding. That is how #40's liquidity look-ahead surfaced — the Nautilus build
produced zero fills and chasing why exposed a gate reading the full session's volume to decide a
09:35 entry. A pandas number that has never been reproduced by the engine that trades is a claim,
not a measurement.

SAME INPUTS, DELIBERATELY. Bars and sector snapshot come from the identical paths the pandas runner
reads, so a difference between the two cannot come from the data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from kumo_strategies.data_root import data_path
from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import QC27RotationStrategy
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    QC27TechInverseVolConfig,
    filter_tech_universe,
)

BAR_SUFFIX = "-1-DAY-LAST-EXTERNAL"
#: The lab's daily panel and sector map, under `$KUMO_DATA_ROOT/lab/snapback-research/` (#211).
#: `QC27_BARS_PATH` / `QC27_SECTORS_PATH` still override per run; with neither set the path comes
#: from the data root and an unset root refuses by name. The absolute machine paths this replaced
#: were a default nobody could tell from a configuration.
BARS_REL = "lab/snapback-research/alpaca_bars_full.parquet"
SECTORS_REL = "lab/snapback-research/sectors.parquet"


def _input(env: str, rel: str) -> Path:
    raw = os.environ.get(env)
    return Path(raw).expanduser() if raw else data_path(rel)


def load_inputs(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The SAME files and columns the pandas runner reads — see the module note."""
    bars = pd.read_parquet(
        _input("QC27_BARS_PATH", BARS_REL),
        columns=["ticker", "date", "open_raw", "high_raw", "low_raw", "close_raw",
                 "close_adj", "volume"],
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.rename(columns={"open_raw": "open", "high_raw": "high",
                                "low_raw": "low", "close_raw": "close"})
    sectors = pd.read_parquet(_input("QC27_SECTORS_PATH", SECTORS_REL))
    return bars, sectors


def run(start: str = "2025-01-02", end: str = "2026-08-10", *,
        cfg: QC27TechInverseVolConfig | None = None, venue: str = "XNAS",
        starting_cash: float = 100_000.0, equity: float = 20_000.0,
        order_id_tag: str = "005", max_symbols: int | None = None,
        log_level: str = "ERROR") -> dict:
    cfg = cfg or QC27TechInverseVolConfig()
    bars, sectors = load_inputs(start, end)

    # Universe narrowed by the PURE filter, not a local re-derivation — a second notion of "which
    # names are tech" would be a second derivation of a fact the strategy already owns.
    tech, diag = filter_tech_universe(bars, sectors, cfg)
    if tech.empty:
        raise ValueError("tech universe is empty for the requested window")

    syms = sorted(tech["ticker"].unique())
    if max_symbols:
        # Deterministic truncation for smoke runs. Highest median dollar volume first, so a small
        # run still exercises the names the strategy would actually rank rather than an alphabetic
        # slice — and it is disclosed in the result rather than silently applied.
        dv = (tech.assign(dv=tech["close"] * tech["volume"])
                  .groupby("ticker")["dv"].median().sort_values(ascending=False))
        syms = list(dv.head(max_symbols).index)
        tech = tech[tech["ticker"].isin(set(syms))]

    insts = {s: TestInstrumentProvider.equity(symbol=s, venue=venue) for s in syms}
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id="TECHIVOL-005", logging=LoggingConfig(log_level=log_level)))
    engine.add_venue(venue=Venue(venue), oms_type=OmsType.NETTING, account_type=AccountType.CASH,
                     base_currency=USD, starting_balances=[Money(starting_cash, USD)])

    for s, inst in insts.items():
        engine.add_instrument(inst)
        bt = BarType.from_str(f"{inst.id}{BAR_SUFFIX}")
        pp, sp = inst.price_precision, inst.size_precision
        sub = tech[tech["ticker"] == s].sort_values("date")
        engine.add_data([
            # REAL high/low from the substrate. A first version passed open/close in their place
            # and Nautilus rejected it outright ("high was < open") -- the engine validates OHLC
            # ordering where a pandas simulator never looks, which is one more thing the two runs
            # disagree about only because one of them checks.
            Bar(bt, Price(r.open, pp), Price(r.high, pp), Price(r.low, pp), Price(r.close, pp),
                Quantity(max(r.volume, 0), sp),
                # A 1-DAY bar is stamped at its CLOSE. Stamping at the session start would hand the
                # strategy that day's close before the day happened.
                int(pd.Timestamp(r.date).tz_localize("UTC").value),
                int(pd.Timestamp(r.date).tz_localize("UTC").value))
            for r in sub.itertuples(index=False)
        ])

    strat = QC27RotationStrategy(cfg=cfg, instrument_ids=[i.id for i in insts.values()],
                                 order_id_tag=order_id_tag, equity=equity)
    engine.add_strategy(strat)
    try:
        engine.run()
        fills = engine.trader.generate_order_fills_report()
        # A halted engine still returns whatever it filled before dying, formatted exactly like a
        # complete run — that is how a 39-fill partial once read as a real result beside 287-fill
        # runs. Fail loudly rather than return a comparable-looking number that is not comparable.
        if len(fills):
            reached = pd.Timestamp(fills["ts_init"].max()).tz_localize(None).normalize()
            last = pd.Timestamp(tech["date"].max()).normalize()
            if reached < last - pd.Timedelta(days=45):
                raise RuntimeError(
                    f"backtest stopped early: last fill {reached.date()} but bars run to "
                    f"{last.date()}. Partial results are NOT comparable to complete ones.")
        return {
            "fills": fills,
            "positions": engine.trader.generate_positions_report(),
            "account": engine.trader.generate_account_report(Venue(venue)),
            "symbols": len(syms),
            "truncated_to": max_symbols,
            "missed_rebalances": list(strat.missed_rebalances),
            "warmup_deferrals": strat.warmup_deferrals,
            "skipped_sessions": strat.skipped_sessions,
            "diagnostics": diag,
        }
    finally:
        engine.dispose()
