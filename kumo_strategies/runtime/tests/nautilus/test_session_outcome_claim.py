"""The session-outcome row must not claim venue acceptance it does not have (kumo-trading-platform issue 512).

On ibkr-paper-retired, 2026-08-24, this row read `TRADING: decided (submitted 8)` while `/orders` held
ZERO — all eight had been rejected asynchronously inside the IBKR exec client. **That row is what
made the other two IBKR bugs invisible.** Both repos read it and concluded staging was fine.

The count is `submitted += int(ok)` where `ok` is `OrderResult.ok`, and `NautilusBroker`'s own module
docstring already says what that means:

    submit != held  `submit()` reports whether the order was ACCEPTED for submission, never that it
                    filled.

So `ok` is LOCAL acceptance by Nautilus. `initialized` is not `accepted`, and `accepted` is not
`filled`. The broker was honest and the journal row was not.

Blocking the session on terminal outcomes is the wrong trade — the venue answer is async and waiting
for it inside `run()` costs more than it buys. The fix is to say only what is known AT WRITE TIME, and
to name the basis so an operator can recover the distinction rather than having to know it.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from kumo_strategies.runtime.nautilus.contract import session_outcome


def test_the_summary_does_not_say_SUBMITTED():
    """"submitted N" reads as "N orders are at the venue". Nothing here knows that."""
    line = session_outcome("TRADING", decided=True, blocked=None, sent=8)
    assert "submitted" not in line.lower(), (
        f"the row still claims submission: {line!r} — that is the exact word that made eight "
        f"venue-rejected orders read as eight working ones")


def test_the_summary_NAMES_the_basis_of_its_count():
    """An operator cannot recover local-vs-venue from a bare integer, and that distinction is the
    whole finding. It has to be in the row, not in someone's memory."""
    line = session_outcome("TRADING", decided=True, blocked=None, sent=8)
    assert "8" in line
    assert "nautilus" in line.lower(), f"the count's basis is not named: {line!r}"
    assert "venue" in line.lower(), f"the row does not say the venue answer is unknown: {line!r}"


def test_it_still_distinguishes_decided_from_NO_DECISION_and_carries_the_block_reason():
    """The row's original purpose, which must survive. "Session ran and declined" and "session never
    ran" looked identical once, and that was the single most expensive ambiguity here."""
    assert "NO DECISION" in session_outcome("HALTED", decided=False, blocked="halted", sent=0)
    assert "halted" in session_outcome("HALTED", decided=False, blocked="halted", sent=0)
    assert "decided" in session_outcome("TRADING", decided=True, blocked=None, sent=0)


def test_a_zero_count_does_not_read_as_a_failure():
    """Nothing sent is a normal outcome — a session that decided to hold sends nothing."""
    line = session_outcome("TRADING", decided=True, blocked=None, sent=0)
    assert "decided" in line and "0" in line


@pytest.mark.parametrize("mod", ["momentum_rotation.nautilus", "qc345_rotation.nautilus"])
def test_both_adapters_build_the_row_from_the_SAME_derivation(mod):
    """The string was written out twice, in two adapters, and they could drift — two derivations of
    one fact. `qc345`'s outcome row did not exist at all until recently, and that asymmetry is what
    left 2026-08-21 unexplainable from the journal. One function, called by both."""
    import importlib

    m = importlib.import_module(f"kumo_strategies.strategies.{mod}")
    # Asserted as the EXISTENCE OF A CALL NODE, not as membership in a set derived from source text.
    # `tests/test_source_assertions_parse.py` taints any name assigned from a `getsource` expression
    # and rejects `in` against it — correctly, since it cannot tell an AST-derived set from a string.
    # Expressing the structural claim directly is both what it wants and the clearer sentence.
    calls_it = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "session_outcome"
                   for n in ast.walk(ast.parse(inspect.getsource(m))))
    assert calls_it, (
        f"{mod} still formats its own outcome row — two copies of one sentence drift, and this is "
        f"the sentence an operator trusts")
