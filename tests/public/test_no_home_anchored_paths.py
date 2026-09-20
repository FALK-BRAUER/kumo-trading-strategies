"""No tracked Python under `research/`, `kumo_strategies/` or `tests/` builds a path from a HOME directory (#211).

A home-anchored path — `Path.home() / "projects/…"`, `os.path.expanduser("~/projects/…")`,
`Path("~/…").expanduser()` — is a real identity with good manners: it opens the right file on the
machine it was written on and `FileNotFoundError`s everywhere else, and the export cannot rewrite it
into anything a stranger can run (the 2026-09-18 public review found 24 such files and 93 scripts
carrying a literal `${KUMO_DATA_ROOT}` string the rewrite had made of them). Data comes from
`kumo_strategies.data_root` and nowhere else.

BOUND TO THE AST, NOT TO SUBSTRINGS (a grep-the-source test is satisfied by deleting a comment and
broken by writing one). What is refused, per file:

  * a call `Path.home()` / `pathlib.Path.home()` — anywhere;
  * a call `expanduser(...)` whose receiver or first argument is a STRING LITERAL — `Path("~/x")
    .expanduser()`, `os.path.expanduser("~/x")`. `Path(raw).expanduser()` on an env-var VALUE is
    allowed: that is how the root itself is read;
  * a string constant that starts with `~/` — outside docstrings, which may describe the defect
    (`data_root.py` does).

The literal `/Users/…` shape is `bin/check-public-tree.sh`'s and is not repeated here (two tests
plant it on purpose to prove that check fires); this test closes the `Path.home()` and `~/` shapes
the shell check cannot see. The 2026-09-18 main was RED under it (24 files) before the port.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("research/*.py", "kumo_strategies/*.py", "tests/*.py")
#: The home marker, spelled so this file's own literal does not trip its own rule.
_HOME = "~" + "/"


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--", *SCOPES], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [ROOT / p for p in out]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _is_str(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def home_anchored(path: Path) -> list[str]:
    """Every home-anchored shape in one file, as `line: reason`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    docs = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "home" and (
                    (isinstance(f.value, ast.Name) and f.value.id == "Path")
                    or (isinstance(f.value, ast.Attribute) and f.value.attr == "Path")):
                found.append(f"{node.lineno}: Path.home()")
            if isinstance(f, ast.Attribute) and f.attr == "expanduser":
                recv = f.value
                # Path("~/x").expanduser()  — the receiver is a Path(...) call over a literal
                if isinstance(recv, ast.Call) and recv.args and _is_str(recv.args[0]):
                    found.append(f"{node.lineno}: Path({recv.args[0].value!r}).expanduser()")
                # os.path.expanduser("~/x")
                if node.args and _is_str(node.args[0]):
                    found.append(f"{node.lineno}: expanduser({node.args[0].value!r})")
        elif _is_str(node) and id(node) not in docs:
            v = node.value
            if v.startswith(_HOME):
                found.append(f"{node.lineno}: literal {v[:60]!r}")
    return found


@pytest.mark.parametrize("path", _tracked(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_tracked_python_builds_a_home_anchored_path(path: Path) -> None:
    found = home_anchored(path)
    assert not found, (
        f"{path.relative_to(ROOT)} builds a path from a home directory:\n  "
        + "\n  ".join(found)
        + "\nRead data through kumo_strategies.data_root.data_path(...) — see that module's table.")


def test_the_detector_sees_every_shape_it_claims_to(tmp_path: Path) -> None:
    """A detector that cannot fire is indistinguishable from a clean tree. Each refused shape, and
    the two allowed ones, checked on a planted file."""
    src = '''"""Docstring may say ~/projects/x — that is prose."""
import os
from pathlib import Path
A = Path.home() / "projects/x"
B = Path("~/projects/x").expanduser()
C = os.path.expanduser("~/projects/x")
D = "~/projects/x"
raw = os.environ.get("X")
OK1 = Path(raw).expanduser()
OK2 = os.path.expanduser(raw)
'''
    f = tmp_path / "planted.py"
    f.write_text(src)
    got = home_anchored(f)
    lines = sorted({int(g.split(":", 1)[0]) for g in got})
    assert lines == [4, 5, 6, 7], got            # A, B, C, D — and neither OK line
    reasons = {g.split(": ", 1)[1] for g in got}
    assert reasons == {
        "Path.home()",
        "Path('~/projects/x').expanduser()",
        "expanduser('~/projects/x')",
        "literal '~/projects/x'",
    }, got
