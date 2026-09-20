"""Re-export shim — the module moved to `kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus`.
"""

from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus as _target

__all__ = [
    "Bar",
    "EXTERNAL_ID",
    "HeldSeedMixin",
    "InstrumentId",
    "LONG",
    "MarketAwareMixin",
    "OrderSide",
    "QC27RotationStrategy",
    "QC27TechInverseVolConfig",
    "Quantity",
    "RegistrationMixin",
    "SENT_BASIS",
    "SESSION_ALERT",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "Strategy",
    "StrategyConfig",
    "SymbolResolutionMixin",
    "TimeInForce",
    "annotations",
    "asyncio",
    "asyncio_CancelledError",
    "build_feature_panel",
    "call_record_terminal",
    "closing_quantity",
    "completes",
    "decide",
    "defaultdict",
    "deque",
    "fill_side",
    "is_foreign",
    "math",
    "pd",
    "protective_close",
    "rebalance_dates",
    "refusal_reason",
    "report_wrong_sided_positions",
    "session_outcome",
    "terminal_fields",
    "traceback",
    "warmup_bars_needed",
]


def __getattr__(name: str):
    return getattr(_target, name)
