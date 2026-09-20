# tests/runtime/nautilus

Contract tests for the Nautilus binding: identity, warmup gating, and the rule that book state
moves only on order events.

Deliberately does NOT construct a `BacktestEngine` — a second engine per interpreter aborts
natively, so engine-level tests must run in their own process (see cockpit `scripts/run-tests.sh`
and its issue #184).
