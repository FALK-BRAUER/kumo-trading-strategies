"""Re-export shim — the module moved to `kumo_strategies.strategies.qc345_rotation.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.qc345_rotation.nautilus`.
"""

from kumo_strategies.strategies.qc345_rotation.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.qc345_rotation.nautilus as _target

__all__ = [
    "Bar",
    "CandidateSource",
    "DEFAULT_ORDER_ID_TAG",
    "EXTERNAL_ID",
    "HeldSeedMixin",
    "InstrumentId",
    "LONG",
    "MarketAwareMixin",
    "OrderSide",
    "QC345RotationConfig",
    "QC345RotationStrategy",
    "Quantity",
    "RegistrationMixin",
    "SENT_BASIS",
    "SESSION_ALERT",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "STRATEGY_TAG",
    "Strategy",
    "StrategyConfig",
    "SymbolResolutionMixin",
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
    "evaluate_exits",
    "fill_side",
    "is_foreign",
    "math",
    "needs_atr",
    "needs_highs",
    "pd",
    "protective_close",
    "rebalance_dates",
    "reconstruct_trail",
    "refusal_reason",
    "report_wrong_sided_positions",
    "session_outcome",
    "terminal_fields",
    "traceback",
    "trailing_atr",
    "warmup_bars_needed",
]


def __getattr__(name: str):
    return getattr(_target, name)
