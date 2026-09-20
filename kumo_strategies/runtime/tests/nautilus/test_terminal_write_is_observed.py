"""A terminal write that fails must not vanish (kumo-trading-platform issue 549).

Measured by kumo-trading-platform against the venue on 2026-08-24:

    Alpaca FILL events : MRVL 2, LRCX 2, INTC 4, DELL 4, AMAT 1
    journal terminal   : MRVL 2, LRCX 2, INTC 4, DELL 3, AMAT 1

Four of five match exactly — so these rows are per-FILL, and their earlier "duplicate writes"
hypothesis was wrong. **DELL is one row short**, in the same session, same lane, same code path as
three that landed.

THE MECHANISM. All three adapters schedule the write as

    asyncio.run_coroutine_threadsafe(record(...), self._loop)

and DISCARD the returned Future. Any exception inside `record_terminal` — a DB timeout, a pool
exhaustion, a closed session — is captured in that Future and never looked at. Nothing logs, nothing
retries, and the row that exists to say what the venue actually did simply is not there.

kumo-trading-platform fixed a pool exhaustion the same night (#542: SQLAlchemy's default 5+10, "which is
exactly what exhausted"). A write failing under that pressure and being swallowed here is consistent
with one missing row out of four, though this test does not prove that was the cause — it closes the
path by which any such failure goes unseen.

A missing terminal row is the worse direction: `attempts_for` reads these to decide a
`client_order_id`, and a lost failure row means a retry regenerates an id already used.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Host(RegistrationMixin):
    EXTERNAL_ID = "TEST-001"
    LABEL = "test"

    def __init__(self):
        self.logged: list[str] = []
        self._logger = SimpleNamespace(
            warning=lambda m, *a, **k: self.logged.append(str(m)),
            error=lambda m, *a, **k: self.logged.append(str(m)),
            info=lambda *a, **k: None)

    @property
    def log(self):
        return self._logger

    @property
    def id(self):
        return "TEST-001"


def test_a_write_that_RAISES_is_reported_not_swallowed():
    """The whole point. Before this, the exception lived in a Future nobody read."""
    import asyncio

    async def _boom():
        raise RuntimeError("QueuePool limit of size 5 overflow 10 reached")

    loop = asyncio.new_event_loop()
    try:
        h = _Host()
        fut = h.fire_and_report(_boom(), loop, "terminal for DELL")
        loop.run_until_complete(asyncio.sleep(0))
        # `run_coroutine_threadsafe` needs the loop running; drive it until the callback fires.
        for _ in range(50):
            if fut.done():
                break
            loop.run_until_complete(asyncio.sleep(0.01))
    finally:
        loop.close()

    assert any("QueuePool" in m for m in h.logged), (
        f"a failed terminal write left no trace: {h.logged} — that is one fewer row than the venue "
        f"says happened, and nothing anywhere says so")
    assert any("terminal for DELL" in m for m in h.logged), "the log does not say WHAT was lost"


def test_a_write_that_SUCCEEDS_logs_nothing():
    """An alarm that fires on every fill buries the one that matters — the same reason warmup is not
    counted as a missed rebalance."""
    import asyncio

    async def _ok():
        return None

    loop = asyncio.new_event_loop()
    try:
        h = _Host()
        fut = h.fire_and_report(_ok(), loop, "terminal for AAA")
        for _ in range(50):
            if fut.done():
                break
            loop.run_until_complete(asyncio.sleep(0.01))
    finally:
        loop.close()
    assert h.logged == [], f"a successful write logged: {h.logged}"


def test_scheduling_itself_failing_is_also_reported():
    """A dead or closed loop must not raise into Nautilus's dispatch, and must not be silent either."""
    h = _Host()
    dead = SimpleNamespace()   # no call_soon_threadsafe — scheduling will raise

    async def _noop():
        return None

    assert h.fire_and_report(_noop(), dead, "terminal for BBB") is None
    assert any("BBB" in m for m in h.logged), f"scheduling failure was silent: {h.logged}"


def test_NO_adapter_schedules_a_terminal_write_it_never_looks_at():
    """Class-aimed, on the same discovery the other seam tests use.

    All three adapters had the discarded-Future form and it was invisible in each of them. A per-lane
    test would have been written for whichever adapter was in mind that day — and the defect was in
    every one. Asserted on the AST: a bare `asyncio.run_coroutine_threadsafe(...)` as a statement is
    a scheduled coroutine whose result nobody reads.
    """
    import ast

    from kumo_strategies.strategies import _layout

    offenders = []
    # Every file the adapter layer is made of — shared mixins and every lane — from the ONE
    # discovery point (strategies/_layout.py, ks#211).
    for path in _layout.nautilus_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            fn = node.value.func
            if isinstance(fn, ast.Attribute) and fn.attr == "run_coroutine_threadsafe":
                offenders.append(f"{_layout.rel(path)}:{node.lineno}")

    assert not offenders, (
        f"scheduled a coroutine and discarded its Future at {offenders} — any exception inside it is "
        f"captured there and never read. That is how a terminal row went missing while three "
        f"identical writes landed. Use `fire_and_report`.")


def test_nautilus_log_calls_do_not_use_python_logging_kwargs():
    """Nautilus loggers are not stdlib loggers.

    Live 2026-09-14: CRSISHORT caught an exception after writing its session rows and then called
    `self.log.error(..., exc_info=True)`. Nautilus accepts a single message string plus optional
    color, so the failure handler raised `TypeError: error() got an unexpected keyword argument
    'exc_info'`. The callback above then reported "session WAS NOT WRITTEN", masking the real
    exception behind a logger-contract mismatch.
    """
    import ast

    from kumo_strategies.strategies import _layout

    offenders = []
    allowed = {"color"}
    levels = {"debug", "info", "warning", "error", "critical"}
    # Every file the adapter layer is made of — shared mixins and every lane — from the ONE
    # discovery point (strategies/_layout.py, ks#211).
    for path in _layout.nautilus_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in levels:
                continue
            if not isinstance(node.func.value, ast.Attribute) or node.func.value.attr != "log":
                continue
            bad = [kw.arg for kw in node.keywords if kw.arg not in allowed]
            if bad:
                offenders.append(f"{_layout.rel(path)}:{node.lineno}:{','.join(bad)}")

    assert not offenders, (
        f"Nautilus log calls used stdlib logging kwargs at {offenders}. Format tracebacks into the "
        f"message string instead; the live logger accepts message plus optional color only.")
