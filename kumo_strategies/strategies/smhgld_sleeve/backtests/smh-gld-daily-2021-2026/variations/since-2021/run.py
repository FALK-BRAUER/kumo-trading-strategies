"""smh-gld-daily-2021-2026 / <variant> — the SMH/GLD sleeve, the ONE runner's sleeve family (ks#240, ks#270).

The set holds the two legs' daily closes; every parameter is in parameters.json — the strategy config
is rebuilt from its `strategy` block, so a variation IS its parameters.json and this file is identical
across variations. Results go to ./results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.params import from_parameters
from kumo_strategies.backtesting.runner_sessions import run_sessions as run
from kumo_strategies.strategies.smhgld_sleeve import SmhGldSleeveConfig

HERE = Path(__file__).resolve().parent
SET = layout.set_dir(HERE)


def main() -> None:
    p = layout.load_parameters(HERE)
    inputs = {k: SET / v for k, v in p["inputs"].items()}
    cfg = from_parameters(SmhGldSleeveConfig, p["strategy"])
    bars = pd.read_parquet(inputs["bars"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[(bars.date >= pd.Timestamp(p["run"]["start"])) & (bars.date <= pd.Timestamp(p["run"]["end"]))]
    bars = bars.sort_values(["ticker", "date"]).reset_index(drop=True)
    bench = {}
    for sym in p["run"]["benchmarks"]:
        s = bars[bars.ticker == sym].set_index("date")["close"].sort_index()
        if len(s) > 50:
            bench[sym] = s
    legs = bars[bars.ticker.isin(cfg.universe)]
    print(f"bars {len(legs):,} rows  {legs.ticker.nunique()} legs  {legs.date.min().date()} -> {legs.date.max().date()}  cfg={cfg.tag()}  benchmarks {sorted(bench)}")
    res = run(legs, cfg=cfg, cost_model=CostModel(half_spread_bps={}, default_bps=p["run"]["cost_bps_per_side"], time_of_day=False),
              starting_cash=p["run"]["starting_cash"], n_trials=p["run"]["n_trials"], benchmarks=bench)
    diag = dict(res.diagnostics)
    diag.update({"bars_rows": len(legs), "fills": int(len(res.report.fills))})
    out = layout.write_results(HERE, res.report, decisions=res.decisions, parameters=p,
                               inputs=list(inputs.values()), diagnostics=diag)
    k = json.loads((out / "metrics.json").read_text())
    print(f"wrote {out.relative_to(SET)}   total_return {k.get('total_return_pct')}%  cagr {k.get('cagr_pct')}%  sharpe {k.get('sharpe')}  "
          f"max_dd {k.get('max_drawdown_pct')}%  rebalances {diag['rebalances']}")
    layout.write_set_report(SET)


if __name__ == "__main__":
    main()
