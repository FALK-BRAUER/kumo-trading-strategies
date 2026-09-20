"""The suite refuses to run against a different checkout's source (#161).

MEASURED, not supposed: 17 worktrees, one venv, whose `.pth` holds a single absolute path. Sixteen
of them import `kumo_strategies` from the MAIN checkout whatever branch they hold.

The case that bit was a MISSING module, which errors as soon as anything points at the branch. The
case this guard exists for is a MODIFIED one — tests from the branch, package from main — which does
not error and can pass either way round.

The last test here is the one that matters: it builds a REAL git worktree and runs a REAL pytest in
it, because a guard verified only against a synthetic path is a guard verified against the fixture
rather than the defect.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kumo_strategies._checkout import (
    _git_common_dir, checkout_conflict, explain, served_from)

ROOT = Path(__file__).resolve().parent.parent


def test_the_guard_is_QUIET_when_the_package_matches_the_checkout():
    """This suite's own run. If it ever fires here, the guard is broken, not the checkout."""
    assert checkout_conflict(ROOT) is None, checkout_conflict(ROOT)


def test_served_from_reports_a_CHECKOUT_ROOT_not_the_package_dir():
    served = served_from()
    assert (served / "kumo_strategies" / "__init__.py").exists(), served


def test_a_DIFFERENT_CHECKOUT_OF_THIS_REPO_IS_A_CONFLICT(tmp_path):
    """THE DEFECT ITSELF: rootdir is a worktree of this repo, the package is served from main.

    A REAL worktree, not a hand-built one. The first version wrote a `.git` file pointing at
    "elsewhere" and asserted a conflict — and that fixture stopped representing anything the moment
    the guard started asking git which repo a path belongs to, because it belongs to none. A
    synthetic fixture can only test the implementation it was written against; this one tests the
    situation all 16 worktrees are actually in.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    wt = tmp_path / "research-worktree"
    made = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                           str(wt), "HEAD"], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"could not create a worktree: {made.stderr.strip()}")
    try:
        conflict = checkout_conflict(wt)
        assert conflict is not None, "a sibling worktree was not reported as a conflict"
        expected, served = conflict
        assert expected == wt.resolve() and served == served_from()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


def test_a_WORKTREE_SHARES_ITS_REPOS_COMMON_GIT_DIR(tmp_path):
    """The property the whole guard now rests on: worktrees of one repo share a common git dir,
    separate clones do not. Asserted against a REAL worktree, because a worktree's `.git` is a FILE
    rather than a directory and an implementation that did not ask git would need to know that.

    THIS TEST REPLACES ONE THAT WAS DEAD. Its predecessor was named
    `test_a_GIT_FILE_counts_as_a_checkout_not_just_a_directory` and asserted only against the main
    clone — where `.git` is a directory, so `is_dir()` and `exists()` agree and the mutation between
    them killed nothing. It named the worktree case and never exercised it.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    wt = tmp_path / "wt"
    made = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                           str(wt), "HEAD"], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"could not create a worktree: {made.stderr.strip()}")
    try:
        assert (wt / ".git").is_file(), "a worktree's .git is expected to be a FILE"
        assert _git_common_dir(wt) == _git_common_dir(ROOT), (
            "a worktree does not share its repo's common git dir — the guard cannot tell a sibling "
            "branch from an unrelated project")
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


def test_A_DIFFERENT_REPO_IS_NOT_A_CONFLICT(tmp_path, monkeypatch):
    """FALSE POSITIVE, found by research-lab reviewing #162 and reproduced here.

    The first version fired for `research-lab` and `kumo-trading-platform` — entirely different repos — because
    it compared the two paths for inequality and asked only whether the SERVED side was a checkout,
    which it always is. A caller running from another project that merely depends on this package
    is not this defect, and a guard that fires on a legitimate case gets turned off.
    """
    other = tmp_path / "some-other-repo"
    (other / "kumo_strategies").mkdir(parents=True)
    assert subprocess.run(["git", "init", "-q", str(other)],
                          capture_output=True).returncode == 0
    assert _git_common_dir(other) != _git_common_dir(ROOT), "the fixture is not a separate repo"
    assert checkout_conflict(other) is None, (
        "a different repo was reported as a conflicting checkout of this one")


def test_A_ROOTDIR_THAT_IS_NOT_A_CHECKOUT_IS_NOT_A_CONFLICT(tmp_path):
    """FALSE POSITIVE, and the sharpest one because the module's own docstring already promised
    otherwise: "Returns None — deliberately quiet — when the package is installed somewhere that is
    not a checkout at all."

    It could not have been true. The early return tested SERVED, which is always a checkout in this
    setup, so the branch was unreachable and the rootdir's nature was never examined. A comment
    describing machinery that does not exist — the same shape this repo has found before.
    """
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert _git_common_dir(plain) is None, "the fixture is unexpectedly a checkout"
    assert checkout_conflict(plain) is None, "a non-checkout rootdir was reported as a conflict"


def test_GIT_COMMON_DIR_IS_RESOLVED_NOT_COMPARED_AS_A_STRING():
    """`--git-common-dir` answers relative (".git") at a repo root and absolute from a worktree.
    Comparing the raw output would make EVERY repo root look identical to every other, which turns
    the same-repo check back into the bug it was written to fix."""
    got = _git_common_dir(ROOT)
    assert got is not None and got.is_absolute(), got


def test_a_NON_CHECKOUT_INSTALL_IS_NOT_COMPLAINED_ABOUT(tmp_path, monkeypatch):
    """A genuine non-editable install lives in site-packages with no competing checkout. That is a
    legitimate deployment, not this defect, and a guard that fires on it would be turned off."""
    import kumo_strategies._checkout as mod
    monkeypatch.setattr(mod, "served_from", lambda: tmp_path / "site-packages-ish")
    assert mod.checkout_conflict(tmp_path / "somewhere-else") is None


def test_the_message_NAMES_BOTH_PATHS_AND_THE_FIX():
    """"Wrong package" without the paths sends the reader looking in the right file for a bug that
    is not there."""
    msg = explain(Path("/repo/branch"), Path("/repo/main"))
    assert "/repo/branch" in msg and "/repo/main" in msg
    assert "PYTHONPATH" in msg, "the message does not say how to fix it"
    assert "161" in msg


# -- the one that proves it against the real defect ----------------------------------------------------

def test_A_REAL_WORKTREE_RUNNING_A_REAL_PYTEST_TESTS_ITS_OWN_PACKAGE(tmp_path):
    """THE TEST THAT EARNS THE LAYOUT.

    The defect this guard was written for: a worktree running pytest with no PYTHONPATH imported
    main's package (the venv's editable install pointed at `<main>/src`) and tested the branch's
    tests against main's code — green, and about nothing. With the package at the repo ROOT the
    rootdir conftest puts the worktree first on sys.path, so a real `git worktree` running a real
    pytest with NO PYTHONPATH imports ITS OWN package. This test builds that worktree and asserts
    where `kumo_strategies` came from — the worktree, not this checkout — and that the guard, which
    still stands for scripts run from elsewhere, stays quiet because there is no conflict.

    THE CONFTEST IS COPIED IN rather than taken from the worktree's own tree: `git worktree add
    --detach HEAD` materialises the COMMITTED tree, so a conftest being edited does not exist there
    yet and the run would fail for the wrong reason.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    wt = tmp_path / "wt"
    made = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                           str(wt), "HEAD"], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"could not create a worktree: {made.stderr.strip()}")
    try:
        shutil.copy2(ROOT / "conftest.py", wt / "conftest.py")
        (wt / "tests" / "test_zz_where_from.py").write_text(
            "import kumo_strategies\n"
            "def test_where_from():\n"
            "    print('SERVED_FROM=' + kumo_strategies.__file__)\n")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}   # NO PYTHONPATH — the old defect's shape
        run = subprocess.run(
            [sys.executable, "-P", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
             "tests/test_zz_where_from.py"],
            cwd=wt, capture_output=True, text=True, env=env, timeout=300)
        out = run.stdout + run.stderr
        assert run.returncode == 0, f"the worktree run failed:\n{out[-2000:]}"
        served = next((l.split("=", 1)[1] for l in out.splitlines() if l.startswith("SERVED_FROM=")), "")
        assert served.startswith(str(wt.resolve())), (
            f"the worktree imported the package from {served!r}, not from itself — the layout no longer "
            "protects a branch from testing main's code")
        assert "different checkout" not in out.lower(), out[-2000:]
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


def test_THE_SAME_WORKTREE_RUN_PASSES_THE_GUARD_WITH_PYTHONPATH(tmp_path):
    """VERIFY BY DISAGREEMENT. The test above shows the guard firing; alone it cannot distinguish
    "detects the conflict" from "always fires". Same worktree, same pytest, PYTHONPATH set to the
    worktree's own src — the guard must go quiet.

    This is also the documented fix, so a green here is the fix being checked rather than asserted.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    wt = tmp_path / "wt2"
    made = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                           str(wt), "HEAD"], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"could not create a worktree: {made.stderr.strip()}")
    try:
        # ALL THREE COPIED IN, for the reason given in the test above: `worktree add --detach HEAD`
        # materialises the COMMITTED tree, so files still being written are absent there. Copying
        # only two of them is how the first version of this test failed on a MISSING FILE rather
        # than on the guard — and it then "failed" under every mutation, which reads exactly like
        # a well-earned test and is in fact a dead one.
        shutil.copy2(ROOT / "conftest.py", wt / "conftest.py")
        shutil.copy2(ROOT / "kumo_strategies" / "_checkout.py",
                     wt / "kumo_strategies" / "_checkout.py")
        shutil.copy2(ROOT / "tests" / "test_checkout_guard.py",
                     wt / "tests" / "test_checkout_guard.py")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "PYTHONPATH": str(wt)}
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-p", "no:cacheprovider", "tests/test_checkout_guard.py"],
            cwd=wt, capture_output=True, text=True, env=env, timeout=300)
        out = run.stdout + run.stderr
        # NOT VACUOUS: an absence-assertion passes for free if the run died for another reason, so
        # the run must have SUCCEEDED and actually collected this file.
        assert run.returncode == 0, f"the run failed for some other reason:\n{out[-2000:]}"
        assert "test_checkout_guard.py" in out, f"nothing was collected:\n{out[-2000:]}"
        assert "different checkout" not in out.lower(), (
            f"the guard fired even with PYTHONPATH pointing at the worktree:\n{out[-2000:]}")
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


# -- narrowing must not have cost coverage -------------------------------------------------------------

def test_NARROWING_THE_GUARD_DID_NOT_CREATE_FALSE_NEGATIVES(tmp_path):
    """THE CHECK THAT LICENSES THE SAME-REPO NARROWING, raised by research-lab reviewing #162.

    Killing false positives by tightening a condition is the classic way to lose the cases the guard
    exists for, and "it stopped firing on research-lab" and "it still fires everywhere it should" are two
    different claims — only the second one licenses merging. Measured against the real estate at the
    time: 16 non-main worktrees, 0 missed.

    That was a one-off check of a property nothing enforced, which is the shape this repo keeps
    finding dead. So it is enforced here instead, on worktrees built for the purpose — including the
    two shapes most likely to slip through a path-based check: one NESTED under another, and one
    whose path is a STRING PREFIX of the main checkout's.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    made = []
    shapes = {
        "plain": tmp_path / "plain",
        "nested": tmp_path / "plain" / "inner",
        "prefix-of-main": tmp_path / (ROOT.name + "-extra"),
    }
    try:
        for name, path in shapes.items():
            r = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach",
                                str(path), "HEAD"], capture_output=True, text=True)
            if r.returncode != 0:
                pytest.skip(f"could not create the {name} worktree: {r.stderr.strip()}")
            made.append(path)
        missed = [n for n, p in shapes.items() if checkout_conflict(p) is None]
        assert not missed, (
            f"the guard does NOT fire for these worktree shapes: {missed}. The same-repo narrowing "
            f"has cost coverage, which is worse than the false positives it removed.")
    finally:
        for path in reversed(made):
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(path)],
                           capture_output=True)


def test_EVERY_WORKTREE_GIT_KNOWS_ABOUT_IS_COVERED():
    """The same claim against the REAL estate rather than fixtures, so a shape nobody thought to
    build is still checked. Skips when only the main checkout exists, since there is then nothing to
    be a false negative about."""
    out = subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git worktree list unavailable")
    roots = [Path(l.split(" ", 1)[1]) for l in out.stdout.splitlines() if l.startswith("worktree ")]
    main = served_from()
    others = [r for r in roots if r.resolve() != main]
    if not others:
        pytest.skip("only the main checkout exists")
    missed = [str(r) for r in others if checkout_conflict(r) is None]
    assert not missed, f"the guard misses these real worktrees: {missed}"
