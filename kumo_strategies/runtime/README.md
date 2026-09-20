# runtime

Runtime ports and adapters SHARED between strategies live here: the Nautilus adapter base and
mixins (`nautilus/`), the executor — broker seam, lifecycle, Postgres store, journal, pool, sources —
(`executor/`), and the calendar. Keep strategy decisions independent from runtime APIs.

Nothing strategy-specific lives here (ks#211): a strategy's lane and session runner sit in
`strategies/<name>/`. `runtime/tests/test_runtime_holds_only_shared_code.py` fails a module here
whose name is a strategy's. The strategy-named files still present (`nautilus/crsi_short.py`,
`executor/pgrunner.py`, …) are re-export shims for the pre-move import paths cockpit uses.
