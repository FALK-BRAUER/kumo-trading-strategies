"""What this INSTALLED package actually is, computed from its own bytes.

THE PROBLEM THIS ANSWERS. cockpit's `deploy/Dockerfile.backend` installs this package from a NAMED
BUILD CONTEXT pointing at a local checkout:

    COPY --from=strategies . /tmp/kumo-trading-strategies
    RUN pip install --no-cache-dir /tmp/kumo-trading-strategies

so the image ships whatever is in that working tree at build time -- not origin/main, not a pinned
SHA, not what anyone said they pushed. Two silent failure directions: a DIRTY tree ships uncommitted
code with no commit to trace it to, and a tree BEHIND origin ships older code than the push everyone
is looking at.

`KUMO_STRATEGIES_SHA` is stamped as `rev-parse --short HEAD` plus a `-dirty` marker. That catches the
second only if someone reads the label, and it cannot catch the first AT ALL: it says the tree was
dirty, never what was in it. On 2026-08-22 a stale cockpit build shipped the wrong commit with a build
log byte-identical to a correct one, and only the artifact stamp knew.

SO THIS MEASURES CONTENT RATHER THAN REPEATING A CLAIM. `digest()` hashes every `.py` file in the
installed package, so:

    the digest matches a commit's digest      the container is running that commit, provably
    the digest matches NO commit              the container is running something uncommitted, and
                                              you can find out WHAT by diffing against the nearest

That is the same standard that caught the stale build and that cockpit used for `release_for_exit`:
an image tag is a claim, a signature read inside the container is an observation. This makes the
observation available for the code itself.

    docker exec <api> python -c "from kumo_strategies.provenance import digest; print(digest())"

and compare against `digest_tree(Path('kumo_strategies'))` at any local ref. No git inside the
container, no build-time cooperation, nothing to keep in sync -- the bytes are the record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Directories whose contents are build artifacts rather than source. `__pycache__` in particular
#: MUST be excluded: it is populated on first import, so a container's digest would change simply by
#: having been used, and two identical deployments would disagree.
_IGNORED_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache"})


def _source_files(root: Path) -> list[Path]:
    """The files that constitute this package, enumerated ONCE.

    `digest_tree` and `files` must never be able to describe different sets. They could: a mutation
    narrowing the digest's walk from `rglob` to `glob` hashed only the top-level modules while the
    count still reported the whole tree, and every guard stayed green — a digest silently covering
    two files out of eighty-two, reported as eighty-two.
    """
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and not _IGNORED_DIRS & set(p.relative_to(root).parts))


def digest_tree(root: Path) -> str:
    """A stable content hash of every SHIPPED file under `root`.

    Every file, not only `*.py`: a wheel is not only its modules, and a data file — a universe list,
    a JSON config, a pinned CSV — could otherwise differ between two builds while the digest called
    them identical. A provenance tool that is silently partial is worse than none, because it turns
    "we do not know" into "we checked".

    Path-relative and sorted, so it does not depend on where the package is installed -- the same
    source yields the same digest in a container, in a venv and in a working tree, which is the whole
    point of comparing them.

    Both the path and the bytes go into the hash: a file RENAMED with identical contents is a
    different package, and hashing contents alone would call it the same.
    """
    paths = _source_files(root)
    if not paths:
        # A HASH OF NOTHING IS STILL A VALID-LOOKING HASH, and `e3b0c44298fc1c14` is what a missing
        # or empty directory produces -- identical for every empty tree, so two builds of nothing
        # would "agree". Observed while wiring the kumo-trading-platform issue 486 round-trip: a worktree that
        # failed to materialise still yielded a plausible digest, and the comparison read as a
        # legitimate MISMATCH rather than as a broken measurement.
        #
        # This is the same failure class the module exists to remove, so it must not appear inside
        # it: refuse rather than return a number nobody can distinguish from a real one.
        raise ValueError(
            f"no files under {root} — refusing to return a digest of nothing, which is a valid-"
            f"looking hash that every empty tree shares")
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def digest() -> str:
    """The content hash of THIS installed package, wherever it is running from."""
    return digest_tree(Path(__file__).resolve().parent)


def files() -> int:
    """How many files were hashed. A digest over ZERO files is a perfectly stable hash of
    nothing, and would silently agree with any other empty install -- so the count is reported
    alongside it rather than left implicit."""
    return len(_source_files(Path(__file__).resolve().parent))
