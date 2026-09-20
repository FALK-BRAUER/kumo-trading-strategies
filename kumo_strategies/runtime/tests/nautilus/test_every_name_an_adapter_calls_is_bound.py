"""A function an adapter CALLS must be bound in the module that calls it.

WHY THIS IS A TEST AND NOT A LINT RUN. `momentum_rotation.py:863` and `qc345_rotation.py:685` both
call `closing_quantity(...)` and `refusal_reason(...)`, and NEITHER MODULE IMPORTS EITHER NAME. Both
calls sit on the exit branch of `_close`, so the `NameError` only fires when a position is actually
being closed — which is the branch that matters and the branch no test reaches. Three sibling lanes
import the same two names correctly, so the code reads right at every call site; the import is
missing one level up, where nobody looks.

It survived a green suite because the exit path is exercised through the SESSION RUNNER in tests,
never through the adapter's own `_close`. That is the shape of "unit tests harden the surface that
is not failing": the tests are real, they pass, and they do not touch the line.

DELIBERATELY NARROW. It reports a called name that is bound NOWHERE in the module — not an import,
not an assignment, not a parameter, not a def, not a comprehension target, not a builtin. That
cannot be a false positive from ordinary local variables, and it is the exact defect above.
"""

from __future__ import annotations

import ast
import builtins

import pytest

from kumo_strategies.strategies import _layout

#: The shared adapter layer plus every lane — what `runtime/nautilus/*.py` used to be (ks#211).
MODULES = [p for p in _layout.nautilus_sources() if p.name != "__init__.py"]


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds, ANYWHERE — scope-insensitively, on purpose.

    A scope-sensitive walk would be a type checker. This only has to catch a name bound in NO scope
    at all, so collapsing the scopes cannot produce a false positive; it can only miss a
    use-before-bind, which is not what this is for.
    """
    out = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            out |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            args = getattr(n, "args", None)
            if args is not None:
                out |= {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
                out |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out |= set(n.names)
    return out


@pytest.mark.parametrize("path", MODULES, ids=_layout.rel)
def test_every_called_name_is_bound_somewhere_in_the_module(path):
    tree = ast.parse(path.read_text())
    bound = _bound_names(tree)
    called = {n.func.id: n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    unbound = {k: v for k, v in called.items() if k not in bound}
    assert not unbound, (
        f"{_layout.rel(path)} calls names it never binds: "
        + ", ".join(f"{k}() at line {v}" for k, v in sorted(unbound.items()))
        + ". A NameError on a branch a test does not reach reads as working code.")
