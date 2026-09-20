"""Every keyword cockpit passes must be one our constructors accept (kumo-trading-platform issue 622, #628).

THREE SEAM GAPS SHIPPED GREEN IN ONE DAY, 2026-08-28, each found by a crash-loop rather than a suite:

    _lane_symbols            defined on cockpit's momentum and never called, so that lane bypassed
                             the shared seam while qc27 and qc345 used it. Deleting the call left
                             304/304 green.
    supplies_trading_calendar  an edit anchored on a field that existed only on an unmerged branch,
                             so every replace silently no-op'd. The capability existed nowhere while
                             three lanes already passed `calendar=...`, which was therefore None.
    claimed_symbols          cockpit passed a keyword this repo never accepted. Third boot failure,
                             ~22h into an outage.

One root: A CHANGE THAT ANCHORS ON SOMETHING ABSENT DOES NOTHING, QUIETLY. Cockpit has wiring tests
in their direction (`test_qc27_wiring`, `test_qc345_wiring`) that read call sites by AST. There was
no equivalent here, which is why gaps in this direction arrive as crash-loops instead of red tests.

WHY `**kwargs` IS THE INTERESTING PART, and why a naive version of this test would have missed the
defect it was written for. `BCTRotationStrategy(*args, **kwargs)` accepts ANY keyword — so
"does the class accept it?" answers yes and the call proceeds. It then forwards to
`MomentumRotationStrategy`, which does not accept it, and THAT is where it raises:

    TypeError: MomentumRotationStrategy.__init__() got an unexpected keyword argument 'claimed_symbols'
      momentum.py:710
      bctrot_rotation.py:113

So acceptance is resolved across the whole MRO, not at the leaf. A `**kwargs` that forwards into a
signature without one is a funnel, not an escape hatch.

CROSS-REPO, so it SKIPS when cockpit's tree is not present — loudly, naming what it could not check.
A silent skip here would be the same defect as the ones it exists to catch.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib

import pytest


#: The constructors cockpit calls. Derived below rather than trusted — this is only the search key.
_ADAPTER_SUFFIX = "RotationStrategy"


def _cockpit_root() -> pathlib.Path | None:
    """Cockpit's tree, if this machine has it. Env var first so CI can point at a checkout."""
    env = os.environ.get("KUMO_COCKPIT_PATH")
    if env:
        p = pathlib.Path(env)
        return p if (p / "backend").is_dir() else None
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "kumo-trading-platform"
        if (candidate / "backend").is_dir():
            return candidate
    return None


def _our_adapters() -> dict[str, type]:
    """Every adapter this package exports, by name. Discovered, so a new lane is covered by existing."""

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    out = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ == m.__name__ and issubclass(obj, Strategy) and obj is not Strategy:
                out[name] = obj
    return out


def _accepted_keywords(cls) -> tuple[set[str], bool]:
    """Every keyword `cls` can actually be constructed with, resolved ACROSS THE MRO.

    Returns (names, terminates_in_varkw). The second is the escape hatch: if some class in the chain
    takes `**kwargs` AND nothing further down constrains it, an unknown keyword is genuinely legal.
    For these adapters it never is — BCTROT's `**kwargs` forwards into Momentum's fixed signature —
    which is exactly why the leaf signature is the wrong thing to read.
    """
    names: set[str] = set()
    chain_varkw = False
    for base in cls.__mro__:
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):
            # A Cython base (Nautilus's Strategy). Nothing readable, and nothing that accepts our
            # keywords either — it is the end of the chain, not a wildcard.
            continue
        for name, p in params.items():
            if name == "self":
                continue
            if p.kind is p.VAR_KEYWORD:
                chain_varkw = True
            elif p.kind is not p.VAR_POSITIONAL:
                names.add(name)
    return names, chain_varkw


def _adapter_names(tree, known: set[str]) -> set[str]:
    """Local names bound to one of our adapters in this module, via import."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "kumo_strategies.runtime.nautilus"):
            for a in node.names:
                if a.name in known:
                    out.add(a.asname or a.name)
    return out


def _sites_in_module(tree, bound: set[str]) -> list[tuple[str, int, list[str]]]:
    """(class, lineno, keywords) for constructions in one module, DIRECT and one hop of indirection.

    THE INDIRECT CASE IS THE ONE THAT MATTERS AND IT IS WHAT THIS FILE MISSED FIRST TIME. Cockpit's
    production lane is not built by name:

        momentum.py:636   def _build_rotation(strategy_cls, *, strategy_id, ...)
        momentum.py:720       strategy = strategy_cls(cfg=..., symbols=..., ...)
        momentum.py:589       _build_rotation(MomentumRotationStrategy, ...)
        momentum.py:623       _build_rotation(BCTRotationStrategy, ...)

    A walk that only matched `ast.Name` callees found three call sites, two of them in cockpit's own
    tests, and MISSED both production lanes — including the exact one whose `claimed_symbols=` broke
    staging. It passed a non-empty vacuity guard while checking nothing that mattered: non-empty is
    not complete, which is the same lesson as an empty dict being invisible to a detector that looks
    for populated ones.
    """
    sites = []
    # Direct: `MomentumRotationStrategy(...)`.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in bound):
            sites.append((node.func.id, node.lineno,
                          [k.arg for k in node.keywords if k.arg is not None]))

    # One hop: a helper handed an adapter CLASS, which then constructs through the parameter.
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    dispatch: dict[tuple[str, str], set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        fn = funcs.get(node.func.id)
        if fn is None:
            continue
        positional = [p.arg for p in fn.args.args]
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name) and arg.id in bound and i < len(positional):
                dispatch.setdefault((fn.name, positional[i]), set()).add(arg.id)
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in bound:
                dispatch.setdefault((fn.name, kw.arg), set()).add(kw.value.id)

    for (fname, param), classes in dispatch.items():
        for inner in ast.walk(funcs[fname]):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == param):
                kws = [k.arg for k in inner.keywords if k.arg is not None]
                for cls_name in sorted(classes):
                    sites.append((cls_name, inner.lineno, kws))
    return sites


def _call_sites(root: pathlib.Path):
    """(class, file, lineno, keywords) for every adapter construction in cockpit's tree.

    AST, never text: a grep for `MomentumRotationStrategy(` is satisfied by a docstring, and this
    package has already been bitten by a source-substring assertion a comment could satisfy.
    """
    known = set(_our_adapters())
    found, importers = [], {}
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        bound = _adapter_names(tree, known)
        if not bound:
            continue
        importers[str(path)] = bound
        for cls_name, lineno, kws in _sites_in_module(tree, bound):
            found.append((cls_name, str(path), lineno, kws))
    return found, importers


_ROOT = _cockpit_root()


@pytest.mark.skipif(_ROOT is None,
                    reason="kumo-trading-platform tree not found — set KUMO_COCKPIT_PATH to check the seam")
def test_the_sweep_actually_found_call_sites():
    """VACUITY GUARD, and it is the assertion this file most needs.

    If the AST walk finds nothing — cockpit renames its builders, moves them, or constructs through
    a variable rather than a bare name — every test below iterates an empty list and passes. That is
    the identical failure this file exists to catch: a check that quietly stopped checking.
    """
    sites, importers = _call_sites(_ROOT)
    assert sites, (
        f"no adapter construction found anywhere under {_ROOT}. Either cockpit no longer builds "
        f"these, or it constructs through a form this walk cannot see — and in both cases every "
        f"assertion in this file is now vacuous."
    )

    # NON-EMPTY IS NOT COMPLETE, and that distinction cost this file its whole point once already.
    # The first version matched only `ast.Name` callees, found three sites — two of them in
    # cockpit's own tests — and missed BOTH production lanes, including the one whose
    # `claimed_symbols=` broke staging. It passed a "found something" guard while checking nothing
    # that mattered.
    #
    # So the guard is COVERAGE, not presence: a module that imports one of our adapters must yield
    # at least one construction of it. Anything cockpit builds through a form this walk cannot
    # resolve fails here loudly instead of silently narrowing what is checked.
    # PRODUCTION MODULES ONLY. Cockpit's own tests legitimately import an adapter to INSPECT it —
    # isinstance checks, AST wiring assertions — without ever constructing one, and four of them do.
    # That is import-without-construction for a real reason, and failing on it would train whoever
    # hits this to widen the exemption rather than look. The lanes that must be covered are the ones
    # that build a node, and those are the non-test modules.
    def _only_inspects(path: str, name: str) -> bool:
        """Does this module reference `name` WITHOUT ever calling it?

        `decision_slots.py` reads `inspect.signature(CrsiShortStrategy.__init__)` to mirror the
        lane's own `decision_slots` default rather than declaring a second one (#858). That is an
        import for a real reason and there is no construction to check — but the exemption has to be
        the NARROW one, because this file's own history is a "found something" guard that checked
        nothing. So: exempt only when the name never appears in a CALL position anywhere in the
        module. A module that constructs it in a form the walk cannot follow still fails.
        """
        try:
            tree = ast.parse(pathlib.Path(path).read_text())
        except Exception:                                              # noqa: BLE001
            return False
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
                return False
        return True

    missing = []
    for path, bound in sorted(importers.items()):
        if pathlib.Path(path).name.startswith("test_"):
            continue
        built = {cls for cls, p, _, _ in sites if p == path}
        for name in sorted(bound - built):
            if _only_inspects(path, name):
                continue
            missing.append(f"{path} imports {name} but no construction of it was resolved")
    assert not missing, (
        "cockpit builds an adapter through a form this walk cannot follow, so those call sites are "
        "unchecked and this file is quietly narrower than it claims:\n  " + "\n  ".join(missing))


@pytest.mark.skipif(_ROOT is None,
                    reason="kumo-trading-platform tree not found — set KUMO_COCKPIT_PATH to check the seam")
def test_every_keyword_cockpit_passes_is_one_we_accept():
    """THE HEADLINE, and the defect it was written for.

    `BCTRotationStrategy(claimed_symbols=...)` is accepted by BCTROT's `**kwargs` and rejected by
    `MomentumRotationStrategy` two frames later, at `Trader.add_strategy` — so the node does not
    boot, and takes every other lane with it. Resolved across the MRO for that reason.
    """
    adapters = _our_adapters()
    sites, _ = _call_sites(_ROOT)
    problems = []
    for cls_name, path, lineno, kws in sites:
        cls = adapters.get(cls_name)
        if cls is None:
            continue                                  # covered by the sweep test above
        accepted, varkw = _accepted_keywords(cls)
        for kw in kws:
            if kw in accepted:
                continue
            problems.append(
                f"{path}:{lineno} passes `{kw}=` to {cls_name}, which does not accept it"
                + ("" if not varkw else
                   f" — its `**kwargs` forwards into a signature that has no `**kwargs`, so this "
                   f"raises on the FORWARD, not at the call"))
    assert not problems, (
        "cockpit passes keywords this package does not accept. Each of these is a TypeError at "
        "`Trader.add_strategy`, which stops the node booting and takes every other lane with it:\n  "
        + "\n  ".join(problems))


@pytest.mark.skipif(_ROOT is None,
                    reason="kumo-trading-platform tree not found — set KUMO_COCKPIT_PATH to check the seam")
def test_a_kwargs_forward_is_not_treated_as_a_wildcard():
    """PROPERTY OF THE CHECK ITSELF, asserted so a future simplification cannot quietly gut it.

    Reading only the leaf signature says BCTROT accepts anything, and a version of this test that
    did so would pass against the exact call that broke staging. The chain must terminate somewhere
    fixed for the check to mean anything.
    """
    adapters = _our_adapters()
    bctrot = next((c for n, c in adapters.items() if "BCT" in n), None)
    if bctrot is None:
        pytest.skip("no BCTROT adapter in this package")

    leaf = inspect.signature(bctrot.__init__).parameters
    assert any(p.kind is p.VAR_KEYWORD for p in leaf.values()), (
        "BCTROT no longer takes `**kwargs`, so this property is untested — check whether the "
        "forwarding hazard still exists before deleting it")

    accepted, varkw = _accepted_keywords(bctrot)
    assert "claimed_symbols" not in accepted, (
        "`claimed_symbols` is now accepted; if that was deliberate, update the note in this file — "
        "claims cannot be resolved lazily because reconciliation completes before `on_start`")
    assert "symbols" in accepted and "cfg" in accepted, (
        "the MRO walk no longer reaches Momentum's parameters, so it would accept anything and "
        "this whole file would pass vacuously")
