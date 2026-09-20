"""`parameters.json` ⇄ config round-trips exactly, and a key the config does not declare is refused
by name — a variation cannot silently run a default because its author misspelled a field."""
from __future__ import annotations

import json

import pytest

from kumo_strategies.backtesting.params import from_parameters, to_parameters
from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig
from kumo_strategies.strategies.qc27_tech_inverse_vol import QC27TechInverseVolConfig
from kumo_strategies.strategies.qc27_tech_inverse_vol.live import live_config


def test_the_live_qc27_config_survives_a_json_round_trip():
    cfg = live_config()
    d = json.loads(json.dumps(to_parameters(cfg)))
    assert d["market_view"] == {"signal": "index_vs_ma", "window": 50, "action": "exit_only", "dwell": 1}
    assert d["rebalance_period"] == "D" and d["momentum_price_field"] == "close"
    assert from_parameters(QC27TechInverseVolConfig, d) == cfg


def test_defaults_round_trip_with_an_inert_view():
    cfg = QC27TechInverseVolConfig()
    d = to_parameters(cfg)
    assert d["market_view"]["signal"] == "none" and d["market_view"]["action"] is None
    back = from_parameters(QC27TechInverseVolConfig, json.loads(json.dumps(d)))
    assert back == cfg
    assert back.market_view.signal is MarketSignal.NONE and back.market_view.action is None


def test_an_undeclared_key_is_refused_by_name():
    d = to_parameters(QC27TechInverseVolConfig())
    d["portfolio_sizee"] = 12
    with pytest.raises(KeyError, match="does not declare.*portfolio_sizee"):
        from_parameters(QC27TechInverseVolConfig, d)


def test_a_nested_undeclared_key_is_refused_by_name():
    d = to_parameters(QC27TechInverseVolConfig())
    d["market_view"]["windw"] = 20
    with pytest.raises(KeyError, match="does not declare.*windw"):
        from_parameters(QC27TechInverseVolConfig, d)


def test_enum_values_are_rebuilt_as_enums_not_strings():
    d = {"signal": "index_vs_ma", "window": 20, "action": "liquidate", "dwell": 2}
    mv = from_parameters(MarketViewConfig, d)
    assert mv.signal is MarketSignal.INDEX_VS_MA and mv.action is MarketAction.LIQUIDATE
