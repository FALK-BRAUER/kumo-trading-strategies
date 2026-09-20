# tests/strategies/crsi_short

Deterministic tests for the CRSISHORT pure layer (#123): indicator arithmetic against hand-computed
values, the split-adjusted boundary, entry/limit/borrow decisions, and the two short exits.

Belongs here: tests that need no engine, no network and no fixtures beyond a small frame.
Does not belong here: runner or Nautilus adapter tests — those live under `tests/backtesting/` and
`tests/runtime/nautilus/`.
