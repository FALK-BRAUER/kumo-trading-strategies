"""`evaluate_periods` must withhold results from indistinguishable arms, like `run_ablation` does.

Written because it did not, and the omission produced a wrong research result rather than an error:
a PEAK sweep ran two arms that had both inherited the same `give_back_frac`, so the rule under test
was never varied. Nothing failed. Two columns of plausible numbers came back.
"""

from __future__ import annotations

import json

import pytest

from kumo_strategies.backtesting.ablation import InertArmError
from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.periods import evaluate_periods, session_blocks
from kumo_strategies.backtesting.runner_sessions import (
    run_sessions as run,  # the ONE runner, daily path (#270)
)
from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig,
    PortfolioConfig,
    ScoreConfig,
)

from .test_ablation import NAMES, _AllEligible, _panel


@pytest.fixture(scope="module")
def kw(tmp_path_factory) -> dict:
    p = tmp_path_factory.mktemp("i") / "instruments.json"
    p.write_text(json.dumps({t: {"symbol": t, "mic": "XNAS", "price_increment": "0.01"}
                             for t in NAMES}))
    return {"instruments_path": p,
            "cost_model": CostModel(half_spread_bps={t: 2.0 for t in NAMES}, default_bps=2.0)}


def _cfg(**pf) -> MomentumRotationConfig:
    return MomentumRotationConfig(
        portfolio=PortfolioConfig(**{**dict(n_hold=4, buffer=1, corr_window=60), **pf}),
        score=ScoreConfig(lookback=20, vol_window=60))


def test_duplicate_arms_raise_instead_of_reporting_two_columns(kw):
    panel = _panel()
    periods = session_blocks(panel["date"], size=60)
    with pytest.raises(InertArmError) as exc:
        evaluate_periods({"peak": _cfg(), "peak_dup": _cfg()}, panel, _AllEligible(),
                         run, periods, kw, workers=2)
    assert "'peak' == 'peak_dup'" in str(exc.value)
    assert "INERT, not neutral" in str(exc.value)


def test_distinguishable_arms_return_a_frame(kw):
    panel = _panel()
    periods = session_blocks(panel["date"], size=60)
    df = evaluate_periods({"control": _cfg(), "narrow": _cfg(n_hold=2)}, panel, _AllEligible(),
                          run, periods, kw, workers=2)
    assert set(df.arm) == {"control", "narrow"}
    assert "_sig" not in df.columns, "the hash is machinery, not a reported column"
    assert len(df) == 2 * len(periods)


def test_arm_specs_carry_per_arm_runner_kwargs(kw):
    """A schedule arm and a config arm must be comparable in one call, or the gate never sees them."""
    from kumo_strategies.backtesting.periods import ArmSpec
    panel = _panel()
    periods = session_blocks(panel["date"], size=60)
    df = evaluate_periods(
        {"control": ArmSpec(_cfg(), {}),
         "less_cash": ArmSpec(_cfg(), {"starting_cash": 50_000.0})},
        panel, _AllEligible(), run, periods, kw, workers=2)
    assert set(df.arm) == {"control", "less_cash"}


def test_a_per_arm_kwarg_that_changes_nothing_still_raises(kw):
    """The gate must judge the FILLS, not whether the kwargs dicts looked different."""
    from kumo_strategies.backtesting.periods import ArmSpec
    panel = _panel()
    periods = session_blocks(panel["date"], size=60)
    with pytest.raises(InertArmError):
        evaluate_periods({"a": ArmSpec(_cfg(), {}), "b": ArmSpec(_cfg(), {"n_trials": 7})},
                         panel, _AllEligible(), run, periods, kw, workers=2)
