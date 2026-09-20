"""ledger-book-daily-2024-2026 / <variant> — momentum rotation over the ledger book, verified runner (ks#240).

Everything the run needs sits in the set beside this file's grandparent; every parameter is in
parameters.json; every result goes to ./results/. The same run.py serves every variation of this
set — the variation IS its parameters.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import run_sessions as run
from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook
from kumo_strategies.strategies.momentum_rotation.config import (
    ExecutionConfig, ExitConfig, MomentumRotationConfig, PortfolioConfig, ScoreConfig)

HERE = Path(__file__).resolve().parent
SET = layout.set_dir(HERE)


def load_bars(p: Path) -> pd.DataFrame:
    d = pd.read_parquet(p)
    tc = next(c for c in d.columns if c.lower() in ("timestamp", "ts", "time", "date", "datetime"))
    sc = next(c for c in d.columns if c.lower() in ("symbol", "ticker"))
    d["date"] = (pd.to_datetime(d[tc], utc=True, format="mixed").dt.tz_convert("America/New_York")
                 .dt.normalize().dt.tz_localize(None))
    return d.rename(columns={sc: "ticker"})[["ticker", "date", "open", "high", "low", "close", "volume"]] \
        .sort_values(["ticker", "date"])


def main() -> None:
    p = layout.load_parameters(HERE)
    inputs = {k: SET / v for k, v in p["inputs"].items()}
    px = load_bars(inputs["bars"])
    days = LedgerBook.calibrate_expiry(str(inputs["ledger"]), str(inputs["ledger_snapshots"]),
                                       anchors=["2026-03-15", "2026-07-10"])
    src = LedgerBook(str(inputs["ledger"]), lag_days=p["source"]["lag_days"], max_open_days=p["source"]["max_open_days"])
    cfg = MomentumRotationConfig(
        score=ScoreConfig(lookback=p["score"]["lookback"]),
        portfolio=PortfolioConfig(**p["portfolio"]), exits=ExitConfig(**p["exits"]),
        execution=ExecutionConfig(min_abs_gap_pct=p["execution"]["min_abs_gap_pct"],
                                  decision_slots=tuple(p["execution"]["decision_slots"])))
    cm = CostModel.from_file(inputs["half_spreads"])
    bench = {}
    for sym in p["run"]["benchmarks"]:
        s = px[px.ticker == sym].set_index("date")["close"].sort_index()
        if len(s) > 50:
            bench[sym] = s
    print(f"panel {len(px):,} rows  {px.ticker.nunique()} symbols  {px.date.min().date()} -> {px.date.max().date()}  "
          f"expiry calibrated={days}d  cost model covers {100 * cm.coverage(sorted(px.ticker.unique())):.0f}%")
    res = run(px, src, cfg=cfg, instruments_path=inputs["instruments"], cost_model=cm,
              starting_cash=p["run"]["starting_cash"], n_trials=p["run"]["n_trials"], benchmarks=bench)
    out = layout.write_results(HERE, res.report, decisions=res.decisions, parameters=p, inputs=list(inputs.values()),
                               diagnostics={"cost_model_coverage": cm.coverage(sorted(px.ticker.unique())),
                                            "calibrated_expiry_days": days, "panel_rows": len(px),
                                            "panel_symbols": int(px.ticker.nunique())})
    k = json.loads((out / "metrics.json").read_text())
    print(f"wrote {out.relative_to(SET)}   total_return {k.get('total_return_pct')}%  sharpe {k.get('sharpe')}  "
          f"max_dd {k.get('max_drawdown_pct')}%  round_trips {k.get('round_trips')}")
    layout.write_set_report(SET)


if __name__ == "__main__":
    main()
