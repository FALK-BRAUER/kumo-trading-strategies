"""Two derivations of one strategy: the ONE runner and a real Nautilus engine replay (#270, fold 3).

`backtesting/runner.py` (deleted in the same change) was the second derivation for momentum. The
comparison is what mattered, so it survives here: the same panel, the same config, once through
`run_sessions`' daily path and once through `MomentumRotationStrategy` inside a BacktestEngine
(own process, `engine_parity/replay.py`). Identical fills mean the pandas number is a measurement;
different fills mean one of the two is wrong, and the disagreement is the finding — never a
tolerance to widen.

Sizing is matched by construction: the engine sizes a fixed $ per position; the daily path is run
with `deployed = equity_per_position * n_hold / starting_cash` and `compound=False`, so both put the
same dollars into each slot on day one. Costs are zero on both sides.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.runner_sessions import run_sessions
from kumo_strategies.strategies.momentum_rotation.config import (
    ExitConfig,
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("QUIA", "QUIB", "LOUA", "LOUB", "LOUC", "LOUD")
SESSIONS = 90
SPEC = {"portfolio": {"n_hold": 3, "buffer": 1},
        "score": {"lookback": 20, "vol_window": 40, "min_history": 25},
        "exits": {"give_back_frac": 0.5},
        "starting_cash": 100_000.0, "equity_per_position": 10_000.0, "venue": "XNAS"}


def _panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=SESSIONS)
    rows = []
    for i, t in enumerate(NAMES):
        sigma = 0.006 + 0.004 * (t.startswith("LOU"))
        drift = 0.0006 - 0.0002 * i
        px = 100.0
        for d in days:
            o = px
            px = px * float(np.exp(rng.normal(drift, sigma)))
            hi, lo = max(o, px) * (1 + 0.3 * sigma), min(o, px) * (1 - 0.3 * sigma)
            rows.append((t, pd.Timestamp(d), o, hi, lo, px, 500_000.0))
    return pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])


class _AllEligible:
    name = "parity_all"

    def eligible(self, d) -> set[str]:
        return set(NAMES)


def _daily_fills(panel: pd.DataFrame, instruments: Path) -> pd.DataFrame:
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(**SPEC["portfolio"]),
                                 score=ScoreConfig(**SPEC["score"]), exits=ExitConfig(**SPEC["exits"]))
    deployed = SPEC["equity_per_position"] * SPEC["portfolio"]["n_hold"] / SPEC["starting_cash"]
    res = run_sessions(panel, _AllEligible(), cfg=cfg, instruments_path=instruments,
                       cost_model=CostModel(half_spread_bps={t: 0.0 for t in NAMES}, default_bps=0.0),
                       starting_cash=SPEC["starting_cash"], deployed=deployed, compound=False)
    f = res.report.fills
    return pd.DataFrame({"date": pd.to_datetime(f["ts"]).dt.normalize(), "symbol": f["symbol"],
                         "side": f["side"], "qty": f["qty"].astype(float)})


def _engine_fills(panel: pd.DataFrame, tmp: Path) -> pd.DataFrame:
    panel_path, cfg_path, out = tmp / "panel.parquet", tmp / "spec.json", tmp / "fills.json"
    panel.to_parquet(panel_path, index=False)
    cfg_path.write_text(json.dumps(SPEC))
    proc = subprocess.run([sys.executable, "-m", "kumo_strategies.backtesting.tests.engine_parity.replay",
                           str(panel_path), str(cfg_path), str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"engine replay failed:\n{proc.stderr[-3000:]}"
    got = json.loads(out.read_text())
    f = pd.DataFrame(got["fills"])
    if f.empty:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty"])
    return pd.DataFrame({"date": pd.to_datetime(f["ts"]).dt.tz_localize(None).dt.normalize(),
                         "symbol": f["symbol"], "side": f["side"], "qty": f["qty"].astype(float)})


@pytest.fixture(scope="module")
def instruments(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("parity") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return p


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DISAGREEMENT, 2026-09-18 (#270 fold 3, ticket #273): on this panel the daily path "
    "makes 35 fills from session 26 and the engine replay 8 fills from session 81; zero shared "
    "(date, symbol, side) fills; the engine ends at $80,105.80. One of the two is wrong and it is not "
    "yet known which — the adapter's warmup, its bar clock (daily bars stamped at the close, slot at "
    "09:35), sizing, or the engine's fill model. Strict: the day they agree this must be turned green, "
    "not left as an xfail that passes."))
def test_the_engine_replay_and_the_daily_path_fill_the_same_trades(instruments, tmp_path):
    panel = _panel()
    daily = _daily_fills(panel, instruments).sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    engine = _engine_fills(panel, tmp_path).sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    assert len(daily) and len(engine), "one side did not trade at all — the comparison is empty"
    pd.testing.assert_frame_equal(daily, engine, check_dtype=False)
