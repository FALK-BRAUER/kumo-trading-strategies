"""liquid-top100-daily-2024-2026 / <variant> — QC345 monthly momentum rotation with the ATR trail, the ONE runner's
monthly family (ks#240, ks#270).

Everything the run needs sits in the set two levels up; every parameter is in parameters.json — the
strategy config is rebuilt from its `strategy` block, so a variation IS its parameters.json and this
file is identical across variations. Results go to ./results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.params import from_parameters
from kumo_strategies.backtesting.runner_sessions import run_sessions as run
from kumo_strategies.strategies.qc345_rotation import QC345RotationConfig

HERE = Path(__file__).resolve().parent
SET = layout.set_dir(HERE)


def load_bars(p: Path, start: str, end: str) -> pd.DataFrame:
    d = pd.read_parquet(p, filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    p = layout.load_parameters(HERE)
    inputs = {k: SET / v for k, v in p["inputs"].items()}
    cfg = from_parameters(QC345RotationConfig, p["strategy"])
    px = load_bars(inputs["bars"], p["run"]["start"], p["run"]["end"])
    assets = pd.read_parquet(inputs["assets"])
    cm = CostModel.from_file(inputs["half_spreads"])
    bench = {}
    for sym in p["run"]["benchmarks"]:
        s = px[px.ticker == sym].set_index("date")[p["run"]["benchmark_price_field"]].sort_index()
        if len(s) > 50:
            bench[sym] = s
    print(f"panel {len(px):,} rows  {px.ticker.nunique()} symbols  {px.date.min().date()} -> {px.date.max().date()}  "
          f"cfg={cfg.tag()}  cost model covers {100 * cm.coverage(sorted(px.ticker.unique())):.0f}% of the panel")
    res = run(px, assets, cfg=cfg, instruments_path=inputs["instruments"], cost_model=cm,
              starting_cash=p["run"]["starting_cash"], deployed=p["run"]["deployed"], n_trials=p["run"]["n_trials"],
              benchmarks=bench)
    diag = dict(res.diagnostics)
    diag.update({"panel_rows": len(px), "panel_symbols": int(px.ticker.nunique())})
    if not res.report.fills.empty:
        diag["cost_model_coverage_traded_symbols"] = cm.coverage(sorted(set(res.report.fills["symbol"])))
    out = layout.write_results(HERE, res.report, decisions=res.decisions, parameters=p,
                               inputs=list(inputs.values()), diagnostics=diag)
    k = json.loads((out / "metrics.json").read_text())
    print(f"wrote {out.relative_to(SET)}   total_return {k.get('total_return_pct')}%  sharpe {k.get('sharpe')}  "
          f"max_dd {k.get('max_drawdown_pct')}%  round_trips {k.get('round_trips')}")
    layout.write_set_report(SET)


if __name__ == "__main__":
    main()
