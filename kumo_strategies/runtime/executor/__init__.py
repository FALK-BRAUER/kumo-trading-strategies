"""Runtime for systematic strategies — Postgres-backed, one path only.

Spec: kumo-trading-platform docs/spec-systematic-strategy-runtime.md, issues #185-#193.

There is exactly ONE way to run the strategy. An earlier SQLite implementation existed alongside
this and was reachable from the CLI's `--live` flag, which meant the single path that could reach a
real broker was also the one without a shrink guard, without durable give-back state and without
Postgres idempotency. It has been deleted rather than deprecated.
"""

from kumo_strategies.runtime.executor.broker import (
    Broker, DryRunBroker, OrderRequest, OrderResult)
from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State, TransitionRejected
from kumo_strategies.runtime.executor.pgjobs import PgJobRunner
from kumo_strategies.runtime.executor.pgjournal import DuplicateDecision, PgJournal
from kumo_strategies.runtime.executor.pgpool import (
    EXCLUDE, PIN, PgSymbolPool, PoolEntry, ShrinkRejected, SourceHealth, UnknownSource,
    check_shrink, check_source_known)
from kumo_strategies.runtime.executor.runner import RiskLimits, SessionResult

__all__ = ["Broker", "DryRunBroker", "OrderRequest", "OrderResult",
           "Lifecycle", "State", "TransitionRejected", "PgJobRunner", "PgJournal",
           "DuplicateDecision", "EXCLUDE", "PIN", "PgSymbolPool", "PoolEntry", "ShrinkRejected",
           "UnknownSource", "check_shrink", "check_source_known",
           "SourceHealth", "PgSessionRunner", "RiskLimits",
           "SessionResult"]


def __getattr__(name: str):
    """`PgSessionRunner` is the MOMENTUM session runner and lives with its strategy
    (strategies/momentum_rotation/runner.py, ks#211). It is still reachable here because cockpit
    imported it from here — but lazily: the runner imports this package for `daily_loss`, so an eager
    re-export would cycle whenever the runner is imported before the package."""
    if name == "PgSessionRunner":
        from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
        return PgSessionRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
