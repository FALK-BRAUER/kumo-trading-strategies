"""Re-export shim — the module moved to `kumo_strategies.strategies.crsi_short.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.crsi_short.nautilus`.
"""

from kumo_strategies.strategies.crsi_short.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.crsi_short.nautilus as _target

__all__ = [
    "ADJUSTED",
    "Bar",
    "CrsiShortConfig",
    "CrsiShortStrategy",
    "EXTERNAL_ID",
    "HISTORY_REQUEST_PARAMS",
    "HeldSeedMixin",
    "InstrumentId",
    "MarketAwareMixin",
    "ORDER_PATH_COMPLETE",
    "OrderRequest",
    "PENDING_KINDS",
    "PendingOrder",
    "RegistrationMixin",
    "SENT_BASIS",
    "SESSION_ALERT",
    "SHORT",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "SessionBar",
    "SessionIntent",
    "ShortTrailState",
    "Strategy",
    "StrategyConfig",
    "SymbolResolutionMixin",
    "annotations",
    "asyncio",
    "asyncio_CancelledError",
    "build_feature_panel",
    "closing_order_side",
    "completes",
    "dataclass",
    "decide",
    "defaultdict",
    "deque",
    "evaluate_short_exits",
    "field",
    "fill_side",
    "is_foreign",
    "math",
    "open_state",
    "opening_leg",
    "pd",
    "protective_close",
    "replace",
    "replacement_is_owed",
    "replacement_leg",
    "report_wrong_sided_positions",
    "session_outcome",
    "traceback",
    "warmup_bars_needed",
]


def __getattr__(name: str):
    return getattr(_target, name)
