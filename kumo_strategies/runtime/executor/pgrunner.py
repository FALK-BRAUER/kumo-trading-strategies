"""Re-export shim — the module moved to `kumo_strategies.strategies.momentum_rotation.runner` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.momentum_rotation.runner`.
"""

from kumo_strategies.strategies.momentum_rotation.runner import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.momentum_rotation.runner as _target

__all__ = [
    "ADOPTED",
    "BUY",
    "Broker",
    "DECISION",
    "DEFAULT_SLOT",
    "DEFAULT_STRATEGY_ID",
    "DryRunBroker",
    "DuplicateDecision",
    "ERROR",
    "LIVE",
    "Lifecycle",
    "MomentumRotationConfig",
    "ORDER",
    "OrderRequest",
    "POOL",
    "PgJournal",
    "PgSessionRunner",
    "PgSymbolPool",
    "PositionState",
    "RISK",
    "Reduction",
    "RiskLimits",
    "SELL",
    "STATE",
    "SessionResult",
    "State",
    "TrailState",
    "already_attempted",
    "annotations",
    "apply_gates",
    "attempts_for",
    "claims_that_contradict_the_account",
    "daily_loss",
    "dataclass",
    "decide",
    "evaluate_exits",
    "field",
    "filter_entries_by_gap",
    "held_sessions_before",
    "math",
    "needs_atr",
    "needs_highs",
    "needs_panel_stats",
    "np",
    "over_claimed",
    "own_ceiling",
    "panel_stats",
    "pd",
    "reducible",
    "retire_claims",
    "score_panel",
    "select",
    "sizing_denominator",
    "submitted_ok",
    "trailing_atr",
    "trailing_returns",
    "unsupported_live_exits",
    "utcnow",
]


def __getattr__(name: str):
    return getattr(_target, name)
