"""No `_j(...)` call may pass `slot=`, because `_j` supplies it.

THE DEFECT THIS CAUGHT, found by kumo-trading-platform's dry run against a database clone hours before an
open, in code that had never executed a real session:

    pgrunner.py:289   return await self.journal.write(kind, summary,
                                                      slot=getattr(self, "_slot", DEFAULT_SLOT), **kw)
    pgrunner.py:849       slot=slot)          <- inside that **kw

    TypeError: PgJournal.write() got multiple values for keyword argument 'slot'

It is the DECISION ROW write, which this module's own comment calls the idempotency key: "If it does
not persist, a retry would decide again — so a failed write must stop the session BEFORE any order
goes out." The `except DuplicateDecision:` below it does not catch TypeError, so the session dies
there. No decision row, no orders, every strategy on every stack.

WHY A GUARD AND NOT JUST A FIX. `_j` exists because `slot` was forgotten at 31 of 33 call sites and
an optional defaulted argument WILL be forgotten again — that is the wrapper's own docstring. What it
did not anticipate is the opposite mistake: a caller REMEMBERING to pass it, into a wrapper that
already does. The wrapper converted a silent wrong label into a hard crash, which is better, but the
crash lands in production at 09:35.

Binds the AST, so it covers every call site rather than the one that was found.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from kumo_strategies.strategies.momentum_rotation import runner as pgrunner

_SRC = Path(inspect.getfile(pgrunner))


def _j_calls_passing_slot() -> list[int]:
    tree = ast.parse(_SRC.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "_j"):
            continue
        if any(k.arg == "slot" for k in node.keywords):
            out.append(node.lineno)
    return out


def test_there_are_j_calls_to_check():
    """A scan that finds no call sites is a scan that cannot fail."""
    tree = ast.parse(_SRC.read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_j"]
    assert len(calls) > 20, f"only {len(calls)} `_j` calls found — this guard is reading the wrong file"


def test_no_j_call_passes_slot():
    """The wrapper owns the slot. A caller that also supplies one is a `TypeError` at runtime."""
    offenders = _j_calls_passing_slot()
    assert not offenders, (
        f"pgrunner.py lines {offenders} pass `slot=` to `_j`, which supplies it itself — "
        f"`TypeError: write() got multiple values for keyword argument 'slot'`, on the decision-row "
        f"write, which kills the session before any order goes out")


def test_the_wrapper_REFUSES_a_slot_rather_than_colliding():
    """A self-explaining refusal instead of an opaque `multiple values` TypeError.

    The AST guard above stops it shipping; this makes the failure legible if it ever reaches a
    runtime the guard does not cover — cockpit calls `PgSessionRunner.run` from its own gateway, and
    a stack trace ending in `multiple values for keyword argument` sends the reader to the journal
    rather than to the caller.
    """
    import asyncio

    class _J:
        async def write(self, kind, summary, **kw):
            return 1

    runner = pgrunner.PgSessionRunner.__new__(pgrunner.PgSessionRunner)
    runner.journal = _J()
    runner._slot = "open+150m"

    with pytest.raises(ValueError, match="slot"):
        asyncio.run(pgrunner.PgSessionRunner._j(
            runner, "decision", "x", session="2026-08-24", slot="close-20m"))
