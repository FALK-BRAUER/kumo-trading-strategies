"""THE SEAM THAT COST 2026-08-17 AND 2026-08-18.

`076035c` added a `slot=` kwarg to one adapter's `_runner.run(...)` call. Cockpit's gateway did not
accept it. Every session died on the call with `TypeError: run() got an unexpected keyword argument
'slot'`, NOTHING journalled it, and two trading days were lost with 41 and 75 journal rows each and a
clean-looking book.

Four `run()` signatures exist across two repos with nothing constraining them:

    PgSessionRunner        run(panel, session, jobs=None, slot=DEFAULT_SLOT)
    QC27SessionRunner      run(panel, session)
    TemplateSessionRunner  run(panel, session)
    cockpit SessionGateway run(panel, session, jobs=None, slot="open+5m")

DERIVED, NEVER LISTED. kumo-trading-platform's own conformance test carries a hand-maintained `CALL_SHAPES`
literal, so a new adapter passing a new kwarg is invisible until it fails live. This scans every
`_runner.run(...)` CALL SITE in `runtime/nautilus` by AST and binds each shape against every runner —
so adding a kwarg anywhere fails here, at the seam, rather than at 09:35 on a Monday.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from kumo_strategies.strategies import _layout


def _adapter_call_shapes():
    """Every `<something>.run(...)` on an injected runner in `runtime/nautilus`, as (file, line, kwargs)."""
    shapes = []
    for path in _layout.nautilus_sources():                  # ks#211: shared layer + every lane
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "run"
                    and isinstance(f.value, ast.Attribute) and "runner" in f.value.attr.lower()):
                continue
            shapes.append((_layout.rel(path), node.lineno,
                           len(node.args), tuple(sorted(k.arg for k in node.keywords if k.arg))))
    return shapes


def _runners():
    """Every session runner — anything with an async `run(panel, session, ...)` — in the shared
    executor or in a strategy's `runner.py` (ks#211)."""
    import importlib

    found = []
    modules = [importlib.import_module(f"kumo_strategies.runtime.executor.{p.stem}")
               for p in _layout.shared_executor_files()] + _layout.runner_modules()
    for mod in modules:
        for name, obj in vars(mod).items():
            if not inspect.isclass(obj) or obj.__module__ != mod.__name__:
                continue
            run = getattr(obj, "run", None)
            if run is None or not inspect.iscoroutinefunction(run):
                continue
            params = list(inspect.signature(run).parameters)
            if params[:3] == ["self", "panel", "session"]:
                found.append(obj)
    assert found, "no session runners discovered — this test no longer describes the package"
    return sorted(found, key=lambda c: c.__name__)


def test_adapter_call_sites_are_actually_found():
    """A scan that finds nothing passes everything. This is the guard against the guard."""
    shapes = _adapter_call_shapes()
    assert shapes, "no `_runner.run(...)` call sites found — the scan has stopped describing the code"


def test_every_runner_is_discovered():
    """Guards the discovery, DERIVED rather than listed.

    A hand-written floor covers the runners someone remembered. kumo-trading-platform's own
    `test_broker_equity_seam.py` was hardcoded to two strategies, so QC27 was wired months later, the
    file passed, and the live strategy was broken — the defect the guard existed to prevent, one level
    out. My sibling guard in `test_contract.py` had the same flaw: it named three strategies while five
    lanes existed.

    So this counts classes that define an async `run(self, panel, session, ...)` by PARSING the
    executor package rather than importing it. Discovery uses `inspect`; this uses `ast`. Two
    derivations of one fact, which is a detector.
    """
    import ast

    on_disk = set()
    for path in _layout.executor_sources():
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for m in node.body:
                if (isinstance(m, ast.AsyncFunctionDef) and m.name == "run"
                        and [a.arg for a in m.args.args][:3] == ["self", "panel", "session"]):
                    on_disk.add(node.name)

    discovered = {c.__name__ for c in _runners()}
    missing = on_disk - discovered
    assert not missing, (
        f"these define an async run(panel, session) on disk but discovery misses them: "
        f"{sorted(missing)} — the seam test is silently covering fewer runners than exist")


@pytest.mark.parametrize("cls", _runners(), ids=lambda c: c.__name__)
def test_every_runner_accepts_every_shape_an_adapter_calls_it_with(cls):
    """`Signature.bind` performs exactly the operation the call site performs, so it fails precisely
    where the live call failed: an unexpected keyword.

    Reintroduce a runner that does not accept `slot` and the momentum shape goes red here — with the
    same TypeError that killed 08-17 and 08-18, two days earlier and for free.
    """
    sig = inspect.signature(cls.run)
    sig = sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "self"])
    for fname, lineno, n_args, kwargs in _adapter_call_shapes():
        try:
            sig.bind(*["panel", "session"][:max(n_args, 2)], **{k: None for k in kwargs})
        except TypeError as exc:
            pytest.fail(
                f"{cls.__name__}.run cannot accept the shape used at {fname}:{lineno} "
                f"(args={n_args}, kwargs={list(kwargs)}): {exc}. This is the 2026-08-17/18 failure — "
                f"the session dies on the call and nothing journals it.")


# -- the same binding, applied to the DOUBLES -----------------------------------------------------

#: The whole package: doubles live beside the tests that use them, and those tests now sit under
#: `runtime/tests/` AND `strategies/<name>/tests/` (ks#211).
_TESTS = pathlib.Path(__file__).resolve().parents[3]


def _runner_doubles():
    """Every stand-in for a session runner anywhere in the suite, found by SHAPE.

    A double is anything defining `async def run(self, panel, session, ...)`. Discovered rather than
    listed for the usual reason, and this file is the one place that reason has already been paid
    for: a hand-maintained list covers the doubles someone remembered.
    """
    out = []
    for path in sorted(_TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, ast.AsyncFunctionDef) or fn.name != "run":
                    continue
                names = [a.arg for a in fn.args.args]
                if "panel" in names and "session" in names:
                    out.append((path.name, cls.name, fn.lineno, set(names),
                                {k.arg for k in fn.args.kwonlyargs}))
    return out


@pytest.mark.parametrize("case", _runner_doubles(),
                         ids=lambda c: f"{c[0]}::{c[1]}")
def test_every_runner_DOUBLE_accepts_what_an_adapter_can_pass(case):
    """A double narrower than the protocol cannot observe a caller starting to pass an argument.

    THIS IS THE GAP THAT LET THE SLOT DEFECT LIVE. Five doubles across two files declared
    `run(self, panel, session)`. When the adapters began passing `slot=`, each died with
    `TypeError: run() got an unexpected keyword argument 'slot'` INSIDE the adapter's own `except`,
    which journalled a failed session — so the tests read as "the runner was never called", which is
    indistinguishable from the session being correctly blocked.

    That is the 2026-08-17 production failure reproduced inside the test suite that exists to catch
    it. One of them even reported the identical message:

        rebalance 2026-03-02 failed: TypeError: _RaisingRunner.run() got an unexpected keyword
        argument 'slot'

    So the doubles are bound against the SAME call shapes as the real runners above. A double that
    agrees with the code cannot test the code; a double narrower than the protocol cannot either.
    """
    fname, cname, lineno, args, kwonly = case
    accepted = (args | kwonly) - {"self"}
    for src, line, _pos, kwargs in _adapter_call_shapes():
        missing = set(kwargs) - accepted
        assert not missing, (
            f"{fname}::{cname} (line {lineno}) does not accept {sorted(missing)}, which "
            f"{src}:{line} passes; this double dies with TypeError inside the adapter's except and "
            f"the test reads as 'never called'")


@pytest.mark.parametrize("shape", _adapter_call_shapes(),
                         ids=lambda s: f"{s[0]}:{s[1]}")
def test_every_adapter_NAMES_the_slot_it_is_deciding(shape):
    """A call that omits `slot=` gets the runner's module default, which is a lie about which
    decision this was.

    Measured live 2026-08-22: `TECHIVOL-005_SLOTS = ['open+315m']` in settings, cockpit honoured it
    for the OFFSET, the lane fired at open+315m, and every journal row said `open+150m`. Two lanes
    were affected — qc27 and qc345 — and only one of them had a test, so the other's omission
    survived a mutation bite of exactly this line.

    Derived from the call sites rather than written per lane, because "which lanes call a runner" is
    the list that goes stale. `exec_action_log.slot` is half of the `(strategy_id, session, slot)`
    unique index, so this is the idempotency key and not merely a label.
    """
    src, line, _pos, kwargs = shape
    assert "slot" in kwargs, (
        f"{src}:{line} calls run() without naming a slot, so every row of that session is filed "
        f"under the runner's module default")
