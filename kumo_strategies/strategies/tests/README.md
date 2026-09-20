# strategies/tests

Tests about the `strategies/` package AS A WHOLE — the folder layout, the discovery helper
(`_layout.py`) every conformance sweep depends on, and the pure/runtime boundary inside each folder.

Holds: cross-strategy structural tests. Does NOT hold: a single strategy's tests (→ `strategies/<name>/tests/`),
lane conformance contracts (→ `runtime/tests/`).
