"""The give-back trail has ONE implementation, and this is what keeps it that way (#110, #123).

Bound to the AST, not to a source substring. A grep-the-source test is satisfied by deleting a
comment and broken by writing one; this asserts that the long-side rule actually CALLS the shared
function and that neither side re-derives the arithmetic locally.
"""

from __future__ import annotations

import ast
import pathlib

SRC = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies" / "strategies"
LONG_SIDE = SRC / "momentum_rotation" / "exits.py"
SHORT_SIDE = SRC / "crsi_short" / "exits.py"


def _fn(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _calls(fn: ast.FunctionDef) -> set[str]:
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


#: Everything in `strategies/give_back.py`. Either side may use whichever of these it needs — what
#: is banned is arriving at the same numbers locally.
SHARED = {"gave_back", "give_back_trigger_px", "favourable_gain", "favourable_extreme"}


def test_both_sides_call_the_shared_give_back():
    """#110 and #123 arrived at this rule from opposite directions. If they are two implementations,
    "the same mechanism" is an unverifiable claim and a fix to one leaves the other wrong."""
    assert SHARED & _calls(_fn(LONG_SIDE, "evaluate_exits"))
    assert SHARED & _calls(_fn(SHORT_SIDE, "evaluate_short_exits"))


def test_neither_side_recomputes_the_ratio_locally():
    """The shape being banned is `peak_px / entry_px - 1` written out again next to the call —
    which is how a shared function ends up decorative while a stale local copy decides the trade."""
    for path, fname in ((LONG_SIDE, "evaluate_exits"), (SHORT_SIDE, "evaluate_short_exits")):
        for node in ast.walk(_fn(path, fname)):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert not {"entry_px"} & names, (
                f"{path.name}:{fname} divides by entry_px inline; call `give_back.gave_back`")
