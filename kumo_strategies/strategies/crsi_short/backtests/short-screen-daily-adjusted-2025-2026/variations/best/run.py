"""short-screen-daily-adjusted-2025-2026 / <variant> — CRSISHORT, the ONE runner's short family (ks#240, ks#270).

The set holds split/dividend-adjusted daily bars; this file rebuilds the strategy's feature panel from
them with the strategy's own `build_feature_panel` (prior-close features, the manageable floor) and
hands the PANEL to the runner — the same path the acceptance run took. Every parameter is in
parameters.json; a variation IS its parameters.json and this file is identical across variations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.params import from_parameters
from kumo_strategies.backtesting.runner_sessions import run_sessions as run
from kumo_strategies.strategies.crsi_short import CrsiShortConfig, build_feature_panel

HERE = Path(__file__).resolve().parent
SET = layout.set_dir(HERE)


def main() -> None:
    p = layout.load_parameters(HERE)
    inputs = {k: SET / v for k, v in p["inputs"].items()}
    cfg = from_parameters(CrsiShortConfig, p["strategy"])
    bars = pd.read_parquet(inputs["bars"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars.date >= pd.Timestamp(p["panel"]["warmup_start"])].sort_values(["ticker", "date"])
    bench = {}
    for sym in p["run"]["benchmarks"]:
        s = bars[bars.ticker == sym].set_index("date")["close"].sort_index()
        if len(s) > 50:
            bench[sym] = s
    universe = bars[~bars.ticker.isin(set(p["run"]["benchmarks"]))]
    panel = build_feature_panel(universe, cfg, adjustment=p["panel"]["adjustment"])
    panel = panel[panel["manageable"]]
    print(f"bars {len(bars):,} rows  {bars.ticker.nunique()} symbols  {bars.date.min().date()} -> {bars.date.max().date()}  "
          f"panel {len(panel):,} manageable rows  {int(panel.signal.sum()):,} signals")
    res = run(None, panel=panel, cfg=cfg, start=p["run"]["start"], end=p["run"]["end"],
              cost_model=CostModel(half_spread_bps={}, default_bps=p["run"]["cost_bps_per_side"], time_of_day=False),
              starting_cash=p["run"]["starting_cash"], borrow_annual=p["run"]["borrow_annual"], benchmarks=bench)
    diag = {"panel_rows": len(panel), "panel_symbols": int(panel.ticker.nunique()), "signals": int(panel.signal.sum()),
            "borrow_paid_usd": float(res.borrow_paid), "exit_mix": res.exit_mix(),
            "refused_entries": res.refused.to_dict(orient="records") if len(res.refused) else []}
    out = layout.write_results(HERE, res.report, decisions=getattr(res, "decisions", pd.DataFrame()), parameters=p,
                               inputs=list(inputs.values()), diagnostics=diag)
    k = json.loads((out / "metrics.json").read_text())
    print(f"wrote {out.relative_to(SET)}   total_return {k.get('total_return_pct')}%  sharpe {k.get('sharpe')}  "
          f"max_dd {k.get('max_drawdown_pct')}%  round_trips {k.get('round_trips')}  borrow ${res.borrow_paid:,.0f}")
    layout.write_set_report(SET)


if __name__ == "__main__":
    main()
