# kumo_strategies

Core Python package for Kumo strategy contracts, decision engines, runtime adapters, and evaluation helpers. Keep imports lightweight at package level. Runtime-specific dependencies belong behind optional extras.

## Layout (ks#211: everything for a strategy in one folder)

| path | what it holds |
|---|---|
| `contracts/` | identity, decisions, orders, fills — the vocabulary every layer shares |
| `strategies/<name>/` | ONE strategy: pure engine (`config.py`, `engine.py`, …), Nautilus lane (`nautilus.py`), session runner where it has one (`runner.py`), `tests/`, `backtests/` |
| `strategies/_layout.py` | the one discovery point for lanes and runners; every cross-lane sweep in `runtime/tests/` uses it |
| `runtime/` | SHARED runtime only — `nautilus/` (adapter base, broker, contract, mixins), `executor/` (broker seam, lifecycle, Postgres store, journal, pool, sources, daily-loss halt), `calendar.py`, and the cross-lane conformance suite in `tests/` |
| `backtesting/` | the replay runner and engine-parity harnesses |
| `evaluation/` | ledgers and metrics |

Import paths from before the move — `kumo_strategies.runtime.nautilus.<strategy>` and
`kumo_strategies.runtime.executor.{pgrunner,qc27_runner,template_runner}` — still resolve: each is a
re-export shim of the module in `strategies/<name>/`, identity-asserted by
`runtime/tests/test_old_import_paths_are_shims.py`. New code imports the `strategies/` path.
