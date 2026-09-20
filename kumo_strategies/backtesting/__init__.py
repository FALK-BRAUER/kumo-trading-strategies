"""Backtesting — verified evidence paths should stay usable without Nautilus installed."""

__all__: list[str] = []

# THE engine for the rotation family (MOMENTUM, BCTROT) since 2026-09-08: N decision slots a
# session from the config, live ranking semantics, next-bar fills, gap dead band applied.
# The former `runner_verified` is its DAILY path (#270): daily bars, next-open fill.
# Guarded: it imports `backtesting.instruments`, which needs Nautilus, and
# Nautilus is an optional extra — an unguarded import here would break EVERY `backtesting.*`
# import in an environment without it (codex review, PR #121).
try:
    from kumo_strategies.backtesting.runner_sessions import run_sessions
except ModuleNotFoundError as exc:
    if exc.name != "nautilus_trader":
        raise
else:
    __all__.append("run_sessions")
