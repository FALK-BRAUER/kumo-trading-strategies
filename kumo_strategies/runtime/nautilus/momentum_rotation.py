"""Re-export shim — the module moved to `kumo_strategies.strategies.momentum_rotation.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.momentum_rotation.nautilus`.
"""

from kumo_strategies.strategies.momentum_rotation.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.momentum_rotation.nautilus as _target

__all__ = [
    "Bar",
    "CandidateSource",
    "DEFAULT_ORDER_ID_TAG",
    "EXECUTES_DELTAS",
    "EXTERNAL_ID",
    "HeldSeedMixin",
    "InstrumentId",
    "LONG",
    "MarketAwareMixin",
    "MomentumRotationConfig",
    "MomentumRotationStrategy",
    "OrderSide",
    "Quantity",
    "REGISTERED_ENVELOPES",
    "RegistrationMixin",
    "SENT_BASIS",
    "SESSION_ALERT",
    "SOURCE_REFRESH_SECS",
    "SOURCE_TIMER",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "STRATEGY_TAG",
    "Strategy",
    "StrategyConfig",
    "SymbolResolutionMixin",
    "annotations",
    "apply_gates",
    "asyncio",
    "asyncio_CancelledError",
    "call_record_terminal",
    "closing_quantity",
    "completes",
    "decide",
    "default_slots",
    "defaultdict",
    "deque",
    "elapsed_slots",
    "evaluate_exits",
    "fill_side",
    "is_foreign",
    "math",
    "needs_atr",
    "needs_highs",
    "needs_panel_stats",
    "next_slot_fire",
    "order_status_to_str",
    "panel_stats",
    "pd",
    "protective_close",
    "reconstruct_trail",
    "refusal_reason",
    "report_wrong_sided_positions",
    "require",
    "score_panel",
    "session_outcome",
    "terminal_fields",
    "traceback",
    "trailing_atr",
    "trailing_returns",
    "validate_slots",
]


def __getattr__(name: str):
    return getattr(_target, name)
