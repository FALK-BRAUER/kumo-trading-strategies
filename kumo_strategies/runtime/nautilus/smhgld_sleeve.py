"""Re-export shim — the module moved to `kumo_strategies.strategies.smhgld_sleeve.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.smhgld_sleeve.nautilus`.
"""

from kumo_strategies.strategies.smhgld_sleeve.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.smhgld_sleeve.nautilus as _target

__all__ = [
    "Bar",
    "EXECUTES_DELTAS",
    "EXTERNAL_ID",
    "HeldSeedMixin",
    "InstrumentId",
    "LONG",
    "MarketAwareMixin",
    "RegistrationMixin",
    "SENT_BASIS",
    "SESSION_ALERT",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "SlotRereadMixin",
    "SmhGldSleeveConfig",
    "SmhGldSleeveStrategy",
    "Strategy",
    "StrategyConfig",
    "SymbolResolutionMixin",
    "annotations",
    "asyncio",
    "asyncio_CancelledError",
    "bars_to_keep",
    "build_feature_panel",
    "completes",
    "defaultdict",
    "deque",
    "fill_side",
    "is_foreign",
    "live_config",
    "math",
    "pd",
    "protective_close",
    "rebalance_dates",
    "report_wrong_sided_positions",
    "require",
    "session_outcome",
    "traceback",
    "warmup_bars_needed",
    "weight_split",
]


def __getattr__(name: str):
    return getattr(_target, name)
