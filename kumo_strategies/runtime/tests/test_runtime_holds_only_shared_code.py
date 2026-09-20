"""`runtime/` is shared code. A strategy's own runtime lives in `strategies/<name>/` (ks#211).

The rule is by NAME, derived from the strategy folders that exist: a module under `runtime/` whose
file stem names a strategy fails, by name. The only files allowed to carry such a name are the
re-export shims that hold the pre-move import paths for cockpit — and a shim is allowed only while it
is structurally nothing but a shim: no class, no function but `__getattr__`, no import but the one
star-import from its target and the one module import of it. The day someone adds a helper to
`runtime/nautilus/crsi_short.py` because the name looked right, this fails.

Seen red first: a `def helper()` planted in a shim, and a planted `runtime/nautilus/smhgld_x.py`,
each failed by name.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from kumo_strategies import runtime
from kumo_strategies.strategies import _layout as L

RUNTIME = pathlib.Path(runtime.__file__).parent


def _strategy_tokens() -> set[str]:
    """`bct` from `bct`, `crsi` from `crsi_short`, `qc27` from `qc27_tech_inverse_vol`… The first
    component of each folder name is the strategy's own token; the rest (`rotation`, `sleeve`,
    `short`) are generic words that shared modules may legitimately use (`sides.py` speaks of
    shorts; `runner.py` of rotations)."""
    return {f.name.split("_")[0] for f in L.strategy_folders()}


def _names_a_strategy(stem: str, tokens: set[str]) -> bool:
    return any(part.startswith(t) for part in stem.split("_") for t in tokens)


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in RUNTIME.rglob("*.py")
                  if "tests" not in p.relative_to(RUNTIME).parts and "__pycache__" not in p.parts)


def _dotted(path: pathlib.Path) -> str:
    rel = path.relative_to(RUNTIME.parent.parent).with_suffix("")
    return ".".join(rel.parts)


def _is_pure_shim(path: pathlib.Path, target: str) -> list[str]:
    """Empty list if the file is nothing but a shim of `target`; else what disqualifies it."""
    tree = ast.parse(path.read_text())
    bad: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue                                      # the docstring
        if isinstance(node, ast.ImportFrom):
            if node.module == target and [a.name for a in node.names] == ["*"]:
                continue
            bad.append(f"import from {node.module}")
        elif isinstance(node, ast.Import):
            if [a.name for a in node.names] == [target]:
                continue
            bad.append(f"import {[a.name for a in node.names]}")
        elif isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["__all__"]:
            continue
        elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            continue
        else:
            bad.append(type(node).__name__ + (f" {node.name}" if hasattr(node, "name") else ""))
    return bad


def test_the_tokens_are_derived_not_listed():
    assert _strategy_tokens() == {"bct", "crsi", "momentum", "qc27", "qc345", "smhgld", "template"}


def test_the_scan_sees_the_shared_modules():
    """A scan over nothing passes over nothing."""
    stems = {p.stem for p in _modules()}
    assert {"adapter", "broker", "contract", "store", "runner", "daily_loss"} <= stems, sorted(stems)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(RUNTIME)))
def test_no_shared_module_names_a_strategy(path: pathlib.Path):
    dotted = _dotted(path)
    if dotted in L.OLD_IMPORT_PATHS:
        bad = _is_pure_shim(path, L.OLD_IMPORT_PATHS[dotted])
        assert not bad, f"{path.relative_to(RUNTIME)} is a shim for {L.OLD_IMPORT_PATHS[dotted]} but also holds {bad}"
        return
    assert not _names_a_strategy(path.stem, _strategy_tokens()), (
        f"runtime/{path.relative_to(RUNTIME)} names a strategy — it belongs under strategies/<name>/")


def test_every_shim_path_exists():
    """The table says where the shims are; a shim deleted by accident is a cockpit import gone."""
    for old in L.OLD_IMPORT_PATHS:
        path = RUNTIME.parent.parent / (old.replace(".", "/") + ".py")
        assert path.exists(), old
