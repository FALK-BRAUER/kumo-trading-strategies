"""Re-export shim — the module moved to `kumo_strategies.strategies.momentum_rotation.nautilus_intraday` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.momentum_rotation.nautilus_intraday`.
"""

from kumo_strategies.strategies.momentum_rotation.nautilus_intraday import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.momentum_rotation.nautilus_intraday as _target

__all__ = [
    "Bar",
    "BarType",
    "CandidateSource",
    "DEFAULT_ORDER_ID_TAG",
    "EXTERNAL_ID",
    "HeldSeedMixin",
    "InstrumentId",
    "IntradayMomentumRotation",
    "LONG",
    "MomentumRotationConfig",
    "OrderSide",
    "Quantity",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "STRATEGY_TAG",
    "Strategy",
    "StrategyConfig",
    "annotations",
    "apply_gates",
    "closing_quantity",
    "completes",
    "dataclass",
    "decide",
    "defaultdict",
    "deque",
    "evaluate_exits",
    "fill_side",
    "is_foreign",
    "needs_atr",
    "needs_highs",
    "pd",
    "protective_close",
    "reconstruct_trail",
    "refusal_reason",
    "score_intraday",
    "score_panel",
    "trailing_atr",
]


def __getattr__(name: str):
    return getattr(_target, name)
