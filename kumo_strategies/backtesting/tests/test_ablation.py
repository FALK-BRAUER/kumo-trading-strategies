"""The arms-differ gate (#26) — the harness must withhold KPIs from an inert parameter.

The property under test is unusual: it is that the harness FAILS. Two `PortfolioConfig` flags have
been swept in this repo while doing nothing, and both times the sweep completed and reported a
plausible null. So the test that matters is "an ablation whose arms cannot be told apart raises
instead of returning numbers", and it has to be written against a parameter that is genuinely inert.
`buffer` on a single-name panel is used for that: it changes nothing observable, exactly like
`max_correlation` did.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.backtesting.ablation import Arm, InertArmError, fill_signature, run_ablation
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

SESSIONS = 260
NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


class _AllEligible:
    name = "test_all"

    def eligible(self, d: pd.Timestamp) -> set[str]:
        return set(NAMES)


def _panel(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=SESSIONS)
    rows = []
    for i, t in enumerate(NAMES):
        sigma = 0.012 + 0.004 * i
        px = 100.0 * np.exp(np.cumsum(rng.normal(0.0012 - 0.00004 * i, sigma, SESSIONS)))
        rows.append(pd.DataFrame({
            "ticker": t, "date": dates, "open": px, "high": px * (1 + 1.5 * sigma),
            "low": px * (1 - 1.5 * sigma), "close": px, "volume": 1_000_000.0}))
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def run_kw(tmp_path_factory) -> dict:
    p = tmp_path_factory.mktemp("instr") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return {"instruments_path": p,
            "cost_model": CostModel(half_spread_bps={t: 2.0 for t in NAMES}, default_bps=2.0),
            "starting_cash": 100_000.0}


def _cfg(**portfolio) -> MomentumRotationConfig:
    base = dict(n_hold=4, buffer=1, corr_window=60)
    return MomentumRotationConfig(portfolio=PortfolioConfig(**{**base, **portfolio}),
                                  score=ScoreConfig(lookback=20, vol_window=60))


def test_an_inert_parameter_raises_instead_of_reporting_a_null(run_kw):
    """The whole point. An arm that changes nothing must not come back as 'no effect'."""
    arms = [Arm("control", _cfg()), Arm("inert", _cfg())]     # identical configs: nothing can differ
    with pytest.raises(InertArmError) as exc:
        run_ablation(arms, _panel(), _AllEligible(), **run_kw)
    msg = str(exc.value)
    assert "'control' == 'inert'" in msg
    assert "INERT, not neutral" in msg


def test_the_gate_catches_a_pair_that_excludes_the_control(run_kw):
    """Two arms inert in the same way must not slip through by both differing from the control.

    Comparing every arm only against the control would miss this, and it is the realistic shape:
    two parameters that are both unwired look different from the baseline for some unrelated reason
    and identical to each other.
    """
    arms = [Arm("control", _cfg(n_hold=4)),
            Arm("x", _cfg(n_hold=3)),
            Arm("y", _cfg(n_hold=3, max_weight=0.9))]   # max_weight is inert without inverse-vol
    with pytest.raises(InertArmError) as exc:
        run_ablation(arms, _panel(), _AllEligible(), **run_kw)
    assert "'x' == 'y'" in str(exc.value)


def test_distinguishable_arms_return_normally(run_kw):
    ab = run_ablation([Arm("control", _cfg()), Arm("ivol", _cfg(inverse_vol_sizing=True))],
                      _panel(), _AllEligible(), **run_kw)
    assert ab.labels == ["control", "ivol"]
    k = ab.kpi_frame()
    assert len(k) == 2
    assert {"total_return_pct", "sharpe", "max_drawdown_pct", "deployed_pct"} <= set(k.columns)
    assert not fill_signature(ab["control"]).equals(fill_signature(ab["ivol"]))


def test_every_arm_shares_the_same_run_inputs(run_kw):
    """`run_kw` is passed once and reused, so panel/costs/capital cannot drift between arms — a
    difference in any of those would be attributed to the parameter under test."""
    ab = run_ablation([Arm("control", _cfg()), Arm("ivol", _cfg(inverse_vol_sizing=True))],
                      _panel(), _AllEligible(), **run_kw)
    starts = {a.result.report.starting_cash for a in ab.arms}
    trials = {a.result.report.n_trials for a in ab.arms}
    assert starts == {100_000.0}
    assert trials == {2}, "n_trials must default to the arm count so DSR corrects for this ablation"


def test_population_report_shows_divergence_rather_than_forbidding_it(run_kw):
    # `n_hold`, not `max_correlation`. The names in this fixture are independent random walks, so
    # their trailing correlations sit near zero and no cap above ~0.1 ever binds — a cap arm here is
    # genuinely indistinguishable and the gate rejects it, correctly. Exercising the cap needs a
    # panel with co-moving names built for it, which is a different fixture than this one.
    ab = run_ablation([Arm("control", _cfg(n_hold=4)), Arm("narrow", _cfg(n_hold=2))],
                      _panel(), _AllEligible(), **run_kw)
    pop = ab.population_report()
    assert list(pop.arm) == ["control", "narrow"]
    assert {"n_names", "only_here", "missing_vs_control"} <= set(pop.columns)
    assert pop.loc[pop.arm == "control", "only_here"].iloc[0] == 0, "control differs from itself"


def test_walk_forward_frame_reports_both_halves_and_sign_agreement(run_kw):
    """The check that would have caught 2026-08-15's curve fit.

    A give-back sweep looked cleanly monotone across the full panel and was called the strongest
    result of the day. Split in two, the monotonicity appeared in NEITHER half — aggregation had
    manufactured it. `kpi_frame` cannot show that; this can.
    """
    ab = run_ablation([Arm("control", _cfg()), Arm("ivol", _cfg(inverse_vol_sizing=True))],
                      _panel(), _AllEligible(), **run_kw)
    wf = ab.walk_forward_frame()
    assert len(wf) == 2
    for col in ("is_ret", "oos_ret", "is_sharpe", "oos_sharpe", "sign_agrees"):
        assert col in wf.columns, f"{col} missing — the split is the evidence, not a nicety"
    # The halves must actually be different windows, not the same numbers twice.
    assert not (wf.is_ret == wf.oos_ret).all(), "both halves identical — the split did not happen"


def test_walk_forward_split_point_is_deterministic(run_kw):
    """The split must not move between calls. A split chosen after seeing results fits the split
    itself, which is the failure the method exists to catch."""
    ab = run_ablation([Arm("control", _cfg()), Arm("ivol", _cfg(inverse_vol_sizing=True))],
                      _panel(), _AllEligible(), **run_kw)
    a, b = ab.walk_forward_frame(), ab.walk_forward_frame()
    pd.testing.assert_frame_equal(a, b)
    # A different split must give a different answer, or the parameter is being ignored.
    assert not a.equals(ab.walk_forward_frame(split=0.4))
