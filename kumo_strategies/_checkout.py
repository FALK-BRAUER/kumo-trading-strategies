"""Is the package being imported the one in the checkout we are running from? (#161)

THE MEASUREMENT THAT PROMPTED THIS: 17 worktrees, ONE venv, and its `.pth` holds one absolute path —
`<home>/projects/kumo-trading-strategies`. Sixteen of the seventeen have no venv of their own, so
every one of them imports `kumo_strategies` FROM THE MAIN CHECKOUT whatever branch it is holding.
`PYTHONPATH=<worktree>` is the only thing that points at the branch, and nothing required it.

    worktree under test : <home>/projects/research-worktree
    package served from : <home>/projects/kumo-trading-strategies
    SAME?               : False

THE CASE THAT BIT WAS THE HARMLESS ONE. On 2026-09-11 a commit importing a module the branch did not
contain was pushed after a run "verified" it — the run passed because the venv served the module from
main. A MISSING module fails loudly the moment anything points at the branch.

THE DANGEROUS CASE IS A MODIFIED ONE, and it does not error at all: a branch that changes
`kumo_strategies/**` and a test together runs the TEST from the branch and the PACKAGE from main.
That asserts new expectations against old code, or old expectations against code the branch already
changed — and either can pass. Every session working on package code in a worktree is exposed, which
is most of them.

WHAT THIS DOES NOT COMPLAIN ABOUT: a genuine non-editable install, where the package lives in
site-packages and there is no competing checkout; and a rootdir belonging to some OTHER repo
entirely, which is a caller asking a question about a project this package is merely a dependency
of. Neither is this defect, and a guard that fires on a legitimate case gets turned off.

The check fires only when the served package sits inside a DIFFERENT git checkout of THIS SAME
REPO — the one situation that silently substitutes one branch's source for another's.

"THIS SAME REPO" IS CHECKED, NOT ASSERTED, AND THE FIRST VERSION DID NOT CHECK IT. It compared the
two paths for inequality and asked only whether the SERVED side was a checkout — which it always is
here, so that branch could never fire and the rootdir's own nature was never examined at all. The
docstring claimed same-repo and non-checkout handling that the code did not contain. Measured by
research-lab against five rootdirs:

    kumo-trading-strategies (main)              quiet   correct
    research-worktree      FIRES   correct
    research-lab      (a different repo)    FIRES   WRONG
    kumo-trading-platform  (another one)         FIRES   WRONG
    /tmp          (not a checkout)      FIRES   WRONG

`git rev-parse --git-common-dir` is what separates them: worktrees of one repo SHARE a common git
dir, separate clones do not. It also answers "is this a checkout at all" in the same call, and it
handles a worktree's `.git` being a FILE rather than a directory without anyone having to know that.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["served_from", "checkout_conflict", "assert_matches_checkout"]


def served_from() -> Path:
    """The checkout root of the `kumo_strategies` actually imported, or its install directory."""
    import kumo_strategies
    pkg = Path(kumo_strategies.__file__).resolve().parent      # .../<checkout root>/kumo_strategies
    return pkg.parent                                          # .../<checkout root>


def _git_common_dir(root: Path) -> Path | None:
    """The repo a path belongs to, or None if it is not a checkout.

    Worktrees of one repo SHARE this; separate clones do not. That is the whole distinction between
    the hazard (one repo, two branches, one package) and a caller simply working in a different
    project — and asking git means a worktree's `.git` being a FILE rather than a DIRECTORY is git's
    problem rather than a special case here.

    `--git-common-dir` answers RELATIVE (".git") when the path is the repo root and ABSOLUTE from a
    worktree, so it is resolved against the path queried. Comparing the raw strings would make every
    repo root look identical to every other.
    """
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None                       # no git, or it would not run: not a claim either way
    if done.returncode != 0:
        return None                       # not a checkout
    out = Path(done.stdout.strip())
    return out.resolve() if out.is_absolute() else (root / out).resolve()


def checkout_conflict(rootdir: str | os.PathLike[str]) -> tuple[Path, Path] | None:
    """`(expected, served)` if the imported package comes from a different checkout, else None.

    Returns None — deliberately quiet — when either side is not a checkout (a real install, or a
    caller running from a temp directory) and when the two belong to DIFFERENT repos. See the module
    note: neither is this defect, and both fired in the first version.
    """
    expected = Path(rootdir).resolve()
    served = served_from()
    if served == expected:
        return None
    mine, theirs = _git_common_dir(expected), _git_common_dir(served)
    if mine is None or theirs is None:
        return None                       # one side is not a checkout — an install, or a temp dir
    if mine != theirs:
        return None                       # a different repo entirely, not two branches of ours
    return expected, served


def assert_matches_checkout(rootdir: str | os.PathLike[str]) -> None:
    """Raise unless the imported package belongs to `rootdir`.

    For research scripts, which `tests/` never imports and which the suite is therefore silent
    about — the gap this defect actually came through.
    """
    conflict = checkout_conflict(rootdir)
    if conflict is None:
        return
    expected, served = conflict
    raise RuntimeError(explain(expected, served))


def explain(expected: Path, served: Path) -> str:
    """Both paths named. "Wrong package" without them sends the reader looking in the right file."""
    return (
        f"kumo_strategies is being imported from a DIFFERENT CHECKOUT than the one under test.\n"
        f"    running from  : {expected}\n"
        f"    package from  : {served}\n"
        f"\n"
        f"The venv is installed editable against one absolute path, so every worktree imports the\n"
        f"package from that checkout whatever branch it holds. A test from this branch would be\n"
        f"asserting against {served.name}'s source, which can pass while proving nothing.\n"
        f"\n"
        f"Fix: run with PYTHONPATH={expected} (the only thing that points at this branch),\n"
        f"or give this worktree its own editable install. See #161."
    )
