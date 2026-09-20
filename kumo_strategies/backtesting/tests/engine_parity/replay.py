"""Replay one strategy through a real Nautilus BacktestEngine and write its fills as JSON.

    python -m kumo_strategies.backtesting.tests.engine_parity.replay <panel.parquet> <config.json> <out.json>

MUST run in its own process — a second BacktestEngine per interpreter aborts natively, which is why
this is a script and not a fixture. The engine wiring is `backtesting/runner.py`'s, moved here when
that module was folded (#270); the only strategy it knows today is momentum rotation, and the
parametrisation lives in the test that drives it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def replay_momentum(panel_path: str, cfg_json: str, out_path: str) -> None:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money

    from kumo_strategies.backtesting.data import bars_for, instruments_for, load_panel
    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExitConfig,
        MomentumRotationConfig,
        PortfolioConfig,
        ScoreConfig,
    )

    spec = json.loads(Path(cfg_json).read_text())
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(**spec["portfolio"]),
                                 score=ScoreConfig(**spec["score"]), exits=ExitConfig(**spec["exits"]))
    panel = load_panel(panel_path)
    syms = sorted(panel.ticker.unique())
    venue = spec.get("venue", "XNAS")
    insts = instruments_for(syms, venue)

    class _AllEligible:
        name = "parity_all"

        def eligible(self, d) -> set[str]:
            return set(syms)

    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id="PARITY-001", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=Venue(venue), oms_type=OmsType.NETTING, account_type=AccountType.CASH,
                     base_currency=USD, starting_balances=[Money(spec["starting_cash"], USD)])
    for s in syms:
        engine.add_instrument(insts[s])
        engine.add_data(bars_for(panel[panel.ticker == s], insts[s]))
    engine.add_strategy(MomentumRotationStrategy(
        cfg=cfg, source=_AllEligible(), instrument_ids=[i.id for i in insts.values()],
        equity_per_position=spec["equity_per_position"]))
    try:
        engine.run()
        fills = engine.trader.generate_order_fills_report()
        acct = engine.trader.generate_account_report(Venue(venue))
        rows = []
        for r in fills.itertuples():
            rows.append({"symbol": str(r.instrument_id).split(".")[0], "side": str(r.side),
                         "qty": float(r.filled_qty), "price": float(r.avg_px),
                         "ts": pd.Timestamp(r.ts_init).tz_convert("America/New_York").isoformat()})
        end_bal = float(acct["total"].iloc[-1]) if len(acct) else spec["starting_cash"]
        Path(out_path).write_text(json.dumps({"fills": rows, "account_end": end_bal}))
    finally:
        engine.dispose()


if __name__ == "__main__":
    replay_momentum(*sys.argv[1:4])
