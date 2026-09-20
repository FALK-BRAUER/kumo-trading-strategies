# tests/runtime/executor

Tests for the executable runtime — pool, lifecycle, broker, session runner.

Every test here pins a **safety property**, not a happy path: a pin surviving a feed drop, an
exclude surviving a re-add, a failed source keeping its last good set, nothing automatic entering
TRADING, ARMED submitting nothing, a session deciding only once, and missing bars blocking rather
than silently narrowing the universe.

Add a test here whenever a way to lose money quietly is discovered.
