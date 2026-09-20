"""The `best` variation of the tech-universe set runs EXACTLY `live_config()` — the package's promoted
configuration — so the number the set leads with is the configuration the package ships. A change to
live.py without a rerun of the variation, or an edit to the variation's parameters, fails here by field."""
from __future__ import annotations

from pathlib import Path

from kumo_strategies.backtesting import layout
from kumo_strategies.backtesting.params import from_parameters, to_parameters
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig
from kumo_strategies.strategies.qc27_tech_inverse_vol.live import live_config

SET = Path(__file__).resolve().parents[1] / "backtests" / "tech-universe-daily-2025-2026"


def test_best_parameters_are_the_promoted_config():
    p = layout.load_parameters(SET / "variations" / "best")
    assert p["strategy"] == to_parameters(live_config())
    assert from_parameters(QC27TechInverseVolConfig, p["strategy"]) == live_config()


def test_monthly_is_the_package_default_and_no_view():
    p = layout.load_parameters(SET / "variations" / "monthly")
    assert p["strategy"] == to_parameters(QC27TechInverseVolConfig())
    assert p["strategy"]["market_view"]["signal"] == "none"
