# tests/runtime

Tests for the Nautilus adapter layer — the code that turns decisions into orders.

Covers strategy identity, order submission, and the pure/runtime boundary (`strategies/` must never
import nautilus; a test fails if it does).

Does NOT hold: decision-rule tests (→ `tests/strategies`), or reporting maths (→ `tests/backtesting`).
