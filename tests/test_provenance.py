"""The container must be able to say what code it is running, without being asked to trust a label.

cockpit's paper image installs this package from a NAMED BUILD CONTEXT pointing at a local checkout,
so it ships whatever is in that working tree at build time. `KUMO_STRATEGIES_SHA` stamps
`rev-parse --short HEAD` plus a `-dirty` marker, which says the tree was dirty and never what was in
it. A dirty build is untraceable BY CONSTRUCTION: no commit exists to point at.

A content digest makes it traceable instead of merely flagged — it matches some commit's digest, or
it matches none and you diff against the nearest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kumo_strategies
from kumo_strategies.provenance import digest, digest_tree, files

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]



def _pkg() -> Path:
    return Path(kumo_strategies.__file__).resolve().parent


def test_the_digest_is_stable_across_calls():
    """A digest that moved between two calls in one process could never be compared to anything."""
    assert digest() == digest()


def test_the_digest_covers_the_whole_package():
    """A digest over zero files is a perfectly stable hash of NOTHING and would agree with any other
    empty install — the failure mode that looks most like success."""
    assert files() > 50, f"only {files()} files hashed; this is not the whole package"


def test_a_ONE_BYTE_change_moves_the_digest(tmp_path):
    """The property the whole mechanism rests on. Copies the package rather than editing it in place,
    so the test cannot corrupt the tree it is measuring."""
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)

    target = a / "provenance.py"
    target.write_bytes(target.read_bytes() + b"\n")
    assert digest_tree(a) != before, "an edited file did not change the digest"


def test_a_RENAME_moves_the_digest_even_with_identical_bytes(tmp_path):
    """Path goes into the hash as well as content. Hashing contents alone would call a renamed
    module the same package, and a rename is exactly how a strategy adapter silently stops being
    discovered."""
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)
    (a / "provenance.py").rename(a / "provenance_renamed.py")
    assert digest_tree(a) != before, "a rename with identical contents left the digest unchanged"


def test_IMPORTING_the_package_does_not_change_its_digest(tmp_path):
    """`__pycache__` is written on first import. Including it would make a container's digest change
    simply by having been USED, so two identical deployments would disagree and the mechanism would
    be worse than nothing."""
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)
    (a / "__pycache__").mkdir(exist_ok=True)
    (a / "__pycache__" / "fake.cpython-312.pyc").write_bytes(b"artifact")
    (a / "sub").mkdir(exist_ok=True)
    (a / "sub" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (a / "sub" / "__pycache__" / "x.py").write_bytes(b"# even a .py under __pycache__")
    assert digest_tree(a) == before, "build artifacts leaked into the identity of the source"


def test_two_copies_of_the_same_source_agree_from_DIFFERENT_paths(tmp_path):
    """Comparing a container against a working tree is the only use, and they never share a path."""
    a, b = tmp_path / "a", tmp_path / "deeply" / "elsewhere" / "b"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(_pkg(), b, ignore=shutil.ignore_patterns("__pycache__"))
    assert digest_tree(a) == digest_tree(b), "the digest depends on where the package is installed"


def test_version_is_NOT_offered_as_a_build_identity():
    """`__version__` has been "0.1.0" since the scaffold and answers nothing. It stays for packaging,
    but the module that could be mistaken for provenance must say so."""
    import kumo_strategies.provenance as prov

    assert kumo_strategies.__version__ == "0.1.0"
    assert "digest" in prov.__doc__ and "claim" in prov.__doc__


def test_a_change_in_a_NESTED_module_moves_the_digest(tmp_path):
    """The digest must cover the WHOLE tree, not just the modules beside it.

    Narrowing the walk from `rglob` to `glob` survived every other guard here: the top-level files
    still hashed, the count still reported eighty-two, and only two of them were actually covered.
    A digest over a fraction of the package agrees with any build that differs only in the rest of
    it — which is the whole package.

    Edits `strategies/momentum_rotation/runner.py` (the MOMENTUM session runner, ks#211), the deepest
    thing that matters, rather than a leaf chosen for convenience.
    """
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)

    nested = a / "strategies" / "momentum_rotation" / "runner.py"
    assert nested.exists(), "fixture drifted — the nested module this binds no longer exists"
    nested.write_bytes(nested.read_bytes() + b"\n")
    assert digest_tree(a) != before, "a change deep in the package did not move the digest"


def test_the_COUNT_and_the_DIGEST_describe_the_same_files(tmp_path):
    """Reported alongside each other, so they must not be able to diverge.

    `files()` saying eighty-two while the digest covers two is worse than either being wrong alone:
    the count is what makes the digest look trustworthy.
    """
    from kumo_strategies.provenance import _source_files

    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    assert len(_source_files(a)) == files(), (
        "the count and the digest walk the package differently")


def test_a_NON_python_file_moves_the_digest(tmp_path):
    """A wheel is not only its `.py` files, and kumo-trading-platform issue 486 makes this digest the FACT that a
    deploy compares against a SHA's CLAIM. Hashing `*.py` alone left a blind spot: a data file — a
    universe list, a JSON config, a pinned CSV — could differ between two builds while the digest
    called them identical.

    Found by probing the mechanism's limits rather than by a failure, because a provenance tool that
    is silently partial is worse than none: it converts "we do not know" into "we checked".
    """
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)
    (a / "universe.json").write_text('{"symbols": ["AAPL"]}')
    assert digest_tree(a) != before, "a shipped non-.py file did not affect the package identity"


def test_bytecode_STILL_does_not_move_the_digest(tmp_path):
    """Widening to all files must not widen to build artifacts. `__pycache__` is written on first
    import, so including it would make a container's digest change simply by having been USED."""
    a = tmp_path / "a"
    shutil.copytree(_pkg(), a, ignore=shutil.ignore_patterns("__pycache__"))
    before = digest_tree(a)
    (a / "__pycache__").mkdir(exist_ok=True)
    (a / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"stale bytecode")
    (a / "runtime" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (a / "runtime" / "__pycache__" / "y.cpython-312.pyc").write_bytes(b"more")
    assert digest_tree(a) == before, "bytecode leaked into the package identity"


def test_every_hashed_file_is_a_file_the_WHEEL_actually_SHIPS():
    """Identity and payload must be the same set, or the digest compares two different things.

    kumo-trading-platform issue 486 records STRATEGIES_DIGEST from the pinned ref's SOURCE and compares it against
    what the CONTAINER reports. `pip install` drops non-.py files unless they are declared as package
    data — measured at 97 files in the tree and 82 in the wheel — so the two numbers disagreed for a
    reason having nothing to do with which code was running, and every deploy would have failed.

    Two ways out and only one is honest: narrow the digest back to `*.py` and go blind to data files,
    or make the wheel carry what the digest counts. If a file is worth hashing it is worth shipping.

    This asserts the extensions the digest can see are the extensions `package-data` declares, so
    adding a `.yaml` universe to the package cannot silently reintroduce the split.
    """
    import tomllib

    root = Path(__file__).resolve().parents[1]
    declared = set(tomllib.loads((root / "pyproject.toml").read_text())
                   ["tool"]["setuptools"]["package-data"]["*"])
    suffixes = {f".py"} | {d.lstrip("*") for d in declared}

    stray = sorted({p.suffix for p in _src_only(_pkg().rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts} - suffixes)
    assert not stray, (
        f"the package ships {stray} files that `package-data` does not declare, so the wheel is a "
        f"SUBSET of the source and the digests cannot be compared")


def test_a_digest_of_NOTHING_is_refused(tmp_path):
    """`sha256()` of an empty walk is `e3b0c44…` — stable, plausible, and shared by every empty tree.

    Observed while wiring the kumo-trading-platform issue 486 round-trip end to end: a worktree that failed to
    materialise still produced a digest, and the deploy comparison read as a legitimate MISMATCH
    rather than as a broken measurement. A comparison that cannot tell "different code" from "no
    code" is the failure class this whole module exists to remove, so it must not live inside it.
    """
    import pytest as _pytest

    with _pytest.raises(ValueError, match="nothing"):
        digest_tree(tmp_path / "does-not-exist")

    (tmp_path / "empty").mkdir()
    with _pytest.raises(ValueError, match="nothing"):
        digest_tree(tmp_path / "empty")
