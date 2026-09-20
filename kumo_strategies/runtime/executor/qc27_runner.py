"""Re-export shim — the module moved to `kumo_strategies.strategies.qc27_tech_inverse_vol.runner` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.qc27_tech_inverse_vol.runner`.
"""

from kumo_strategies.strategies.qc27_tech_inverse_vol.runner import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.qc27_tech_inverse_vol.runner as _target

__all__ = [
    "Broker",
    "DECISION",
    "DECISION_SLOT",
    "DryRunBroker",
    "DuplicateDecision",
    "ERROR",
    "Lifecycle",
    "OPEN_OFFSET_MINUTES",
    "ORDER",
    "OrderRequest",
    "PgJournal",
    "QC27SessionRunner",
    "QC27TechInverseVolConfig",
    "RISK",
    "RiskLimits",
    "SessionResult",
    "State",
    "annotations",
    "build_feature_panel",
    "daily_loss",
    "dataclass",
    "decide",
    "field",
    "math",
    "pd",
    "rebalance_dates",
]


def __getattr__(name: str):
    return getattr(_target, name)
