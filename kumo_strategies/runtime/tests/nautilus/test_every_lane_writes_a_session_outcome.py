"""Every lane that runs a session writes exactly one outcome row for it (kumo-trading-platform issue 587).

MEASURED ON AN ALPACA PAPER INSTANCE 2026-08-28 13:52 ET: `exec_action_log` rows for `strategy_id='QC345-003'`,
`kind='state'`, summary like `%session outcome%` — **zero, for the lane's entire life.**

THE MECHANISM WAS A GUESS, MADE THREE TIMES, RIGHT ONCE:

    qc345      getattr(self._runner, "journal", None)                     -> None, silent return
    momentum   getattr(self._runner, "journal", None)
               or getattr(self._jobs, "journal", None)                    -> found, on `_jobs`
    qc27       (no outcome row at all)

Cockpit's `SessionGateway` stores it privately as `_journal` (`momentum.py:47`), so the FIRST lookup
fails for everyone. Momentum only works because it happens to have a second thing to try and
`PgJobRunner` happens to expose the same journal publicly. That is an accident, not a design, and
QC345 had no such accident.

AND THE FAILURE WORE THE COSTUME OF A FEATURE. `_journal_session` documents "degrades to a no-op
with no runner attached (the backtest and standalone paths)" — which is true, and is also exactly
what a wrong attribute name looks like. A guard written for a real absence silently absorbed a real
defect, in the journal path whose entire purpose is making silence impossible.

WHY THIS FILE AIMS AT THE CLASS. The ticket named QC345. Fixing the named lane would have left QC27
writing nothing at all — a different cause, same invisibility, unreported only because that lane has
barely run. "Ran and declined" and "never ran" being indistinguishable is the single most expensive
ambiguity in this system, and it must be impossible for every lane, including the fifth one.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


def _lanes():
    """Every cockpit lane, discovered."""

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    out = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, Strategy)
                    and getattr(obj, "EXTERNAL_ID", None)):
                out.append(obj)
    return sorted(set(out), key=lambda c: c.__name__)


def _source_of(cls):
    """The class's source PLUS its ancestors', because inheriting the behaviour is having it.

    BCTROT defines no session path — it inherits Momentum's, outcome row included. Reading only the
    leaf would report the lane with TWO slots, where losing a session matters most, as writing
    nothing. `test_resolution_reaches_the_order_path` learned the same lesson about `__dict__`.
    """
    out = []
    for base in cls.__mro__:
        try:
            out.append(textwrap.dedent(inspect.getsource(base)))
        except (OSError, TypeError):
            continue          # a Cython base has no retrievable source and defines none of this
    return "\n".join(out)


def _functions_calling(cls, name):
    """Every function in the MRO that calls `name`, as parse trees.

    Returns the FUNCTIONS rather than a boolean so the caller can ask what else happens inside them
    — which is the difference between "the sentence is composed" and "the row is written".
    """
    out = []
    for base in cls.__mro__:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
        except (OSError, TypeError, SyntaxError):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
                   for n in ast.walk(fn)):
                out.append(fn)
    return out


def _runs_sessions(cls) -> bool:
    """Does this lane actually drive a session runner?

    The TEMPLATE holds `self._runner = session_runner` and never calls `.run()` — it has no session
    path at all, so demanding an outcome row from it would be demanding a row for an event that
    cannot happen. Asked structurally rather than by checking `IS_TEMPLATE`, so a real lane that
    stops running sessions is not quietly excused by a flag.
    """
    tree = ast.parse(_source_of(cls))
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "run"
               and isinstance(n.func.value, ast.Attribute) and "runner" in n.func.value.attr
               for n in ast.walk(tree))


# ==================================================================================================
# THE RESOLVER — one lookup, and the two absences told apart
# ==================================================================================================

class _Host(RegistrationMixin):
    id = "TEST-001"

    def __init__(self, **attrs):
        self.errors: list[str] = []
        self._log = SimpleNamespace(error=lambda m, *a, **k: self.errors.append(str(m)),
                                    warning=lambda *a, **k: None, info=lambda *a, **k: None)
        for k, v in attrs.items():
            setattr(self, k, v)

    @property
    def log(self):
        return self._log


def test_it_finds_a_journal_stored_PRIVATELY():
    """Cockpit's gateway stores `_journal`. That is theirs to name, and it is the name that was
    silently missed for QC345's entire life."""
    j = object()
    assert _Host(_runner=SimpleNamespace(_journal=j)).session_journal() is j


def test_it_finds_a_journal_stored_PUBLICLY():
    """`PgJobRunner` exposes `journal`. Both spellings are legitimate."""
    j = object()
    assert _Host(_runner=SimpleNamespace(journal=j)).session_journal() is j


def test_it_looks_on_the_JOBS_holder_too():
    """Momentum's accidental fallback, made deliberate — this is where its journal actually came
    from, and the reason it was the only lane that worked."""
    j = object()
    assert _Host(_runner=SimpleNamespace(), _jobs=SimpleNamespace(journal=j)).session_journal() is j


def test_NO_RUNNER_is_a_quiet_None():
    """The backtest and standalone paths, which must not raise and must not shout."""
    h = _Host()
    assert h.session_journal() is None
    assert h.errors == [], "a legitimate absence was reported as a defect"


def test_A_RUNNER_WITH_NO_JOURNAL_IS_LOUD():
    """THE DISTINCTION THAT IS THE WHOLE FIX. These two returned the same None, so a wrong attribute
    name was indistinguishable from a backtest — and that is how the rows went missing for a lane's
    entire life without anything noticing."""
    h = _Host(_runner=SimpleNamespace(nothing_useful=1))
    assert h.session_journal() is None
    assert h.errors, "a runner with no journal was as quiet as having no runner at all"
    assert "587" in h.errors[0] or "journal" in h.errors[0]


def test_the_wiring_complaint_is_said_ONCE():
    """This runs per session. A line every session would bury it under itself — the 115-false-alarms
    failure `missed_rebalances` already paid for."""
    h = _Host(_runner=SimpleNamespace())
    for _ in range(5):
        h.session_journal()
    assert len(h.errors) == 1


# ==================================================================================================
# EVERY LANE — the class, not the lane that was reported
# ==================================================================================================

@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_every_lane_writes_a_session_outcome_row(cls):
    """THE HEADLINE, aimed at the class. The ticket named QC345; QC27 wrote nothing at all, from a
    different cause, and fixing only the named lane would have left it.

    Bound to the parse tree: a grep for "session outcome" is satisfied by this file's own docstring,
    and by every comment explaining the defect.
    """
    if not _runs_sessions(cls):
        pytest.skip(f"{cls.__name__} never calls a session runner, so it has no session to report")

    # STRUCTURAL, not "some string somewhere says it". The first version of this assertion looked
    # for any string constant containing "session outcome" anywhere in the MRO — which every
    # docstring explaining this defect satisfies, including the ones in this very fix. Deleting
    # qc27's entire write left it green: a source-substring assertion wearing an AST costume.
    #
    # So: find the function that COMPOSES the outcome, and require the same function to hand it to a
    # journal. Composing it and dropping it is precisely what "recorded and never read" looks like.
    composing = _functions_calling(cls, "session_outcome")
    assert composing, (
        f"{cls.__name__} never calls `session_outcome`, so it writes no outcome row. 'ran and "
        f"declined' and 'never ran' are then indistinguishable in the only record an operator "
        f"reads — the most expensive ambiguity in this system (kumo-trading-platform issue 587).")
    # AND THE OUTCOME MUST REACH THE WRITER. "the composing function also calls a writer somewhere"
    # was still too loose: qc345's `_session_coro` journals its ERROR path too, so deleting only the
    # outcome row left that check green. Bind to the summary the query actually selects on —
    # `kind='state'` and `summary LIKE '%session outcome%'` — which is a literal part of the
    # f-string handed to the write, not a docstring anywhere near it.
    for fn in composing:
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("write", "_journal_session"):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                parts = [v.value for v in ast.walk(arg)
                         if isinstance(v, ast.Constant) and isinstance(v.value, str)]
                if any("session outcome" in v for v in parts):
                    return
    pytest.fail(
        f"{cls.__name__} composes a session outcome and never journals it under a summary the "
        f"query can find. The sentence exists and the row does not, which is the exact shape of "
        f"QC345's entire life — zero rows on paper, for the lane's whole existence.")


@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_no_lane_GUESSES_at_the_journal_attribute(cls):
    """The defect was three independent guesses at where the journal lives, one of which happened to
    be right. One resolver, or the next lane guesses too."""
    src = _source_of(cls)
    tree = ast.parse(src)
    guesses = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ("journal", "_journal")):
            guesses.append(ast.unparse(node))
    assert not guesses, (
        f"{cls.__name__} reaches for the journal by name: {guesses}. Cockpit stores it as "
        f"`_journal` and `PgJobRunner` as `journal`; a lane that guesses gets one of them and is "
        f"silent about the other. Use `self.session_journal()`.")


@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_the_outcome_row_uses_the_SHARED_sentence(cls):
    """One derivation. The sentence was written out twice and drifted once already — and it is what
    both repos read to decide whether a session did anything."""
    if not _runs_sessions(cls):
        pytest.skip(f"{cls.__name__} never calls a session runner")
    src = _source_of(cls)
    called = {n.func.id for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "session_outcome" in called, (
        f"{cls.__name__} composes its own outcome sentence instead of calling `session_outcome`. "
        f"Two derivations of one fact disagree.")


def test_the_sweep_covers_more_than_the_reported_lane():
    """VACUITY GUARD. If discovery returned only QC345 these assertions would pass while proving
    nothing about the class — which is exactly the failure of fixing the lane that was reported."""
    names = {c.__name__ for c in _lanes()}
    assert len(names) >= 3, f"only {names} discovered; this file claims to check every lane"
    assert any("QC27" in n or "TECHIVOL" in n for n in names), (
        "QC27 is not in the sweep, and it is the lane the ticket did NOT name")


def test_a_lane_that_runs_sessions_is_actually_FOUND():
    """VACUITY GUARD for the skip above. If `_runs_sessions` stopped recognising the call shape,
    every lane would skip and this file would report green while checking nothing."""
    running = [c.__name__ for c in _lanes() if _runs_sessions(c)]
    assert len(running) >= 3, (
        f"only {running} appear to run sessions; `_runs_sessions` has stopped matching how lanes "
        f"drive their runner, and every assertion in this file is skipping")
