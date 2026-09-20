"""Re-export shim — the module moved to `kumo_strategies.strategies.bct.nautilus` (ks#211).

Everything for a strategy lives in `strategies/<name>/`; this path stays because cockpit imports
it. Every name below IS the object at the new path (`runtime/tests/test_old_import_paths_are_shims.py`
asserts identity, not equality), so `isinstance`, registries and `monkeypatch` through this path
land on the real one. Underscore names reach through `__getattr__`.

Do not add code here. New code goes in `kumo_strategies.strategies.bct.nautilus`.
"""

from kumo_strategies.strategies.bct.nautilus import *  # noqa: F401,F403 — a shim re-exports by design

import kumo_strategies.strategies.bct.nautilus as _target

__all__ = [
    "BCTRotationStrategy",
    "DECISION_SLOTS",
    "EXTERNAL_ID",
    "MomentumRotationStrategy",
    "STRATEGY_LABEL",
    "STRATEGY_NAME",
    "annotations",
]


def __getattr__(name: str):
    return getattr(_target, name)
