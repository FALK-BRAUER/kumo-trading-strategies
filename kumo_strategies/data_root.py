"""Where the data lives — one variable, no default, refusal by name.

Market data and research artefacts are not in this repository (they cannot be redistributed; the
private `research-data` repository holds them with a MANIFEST row per file). Anything here that
reads one takes the root from `KUMO_DATA_ROOT` and nowhere else — the way a secret is a keychain
NAME in this tree, never a value.

THERE IS NO DEFAULT, on purpose. A default path is a real identity with good manners: a reader
that quietly opened `<home>/projects/<repo>/…` on the machine it was written on would pass every test
there and read nothing everywhere else — and the shape this replaced did exactly that, returning
an EMPTY FRAME for a session whose file was absent (`runner_penny_gap.load_session`), so a missing
day and a flat day were one value. Absence is not permission: an unset root, or a file the root
does not hold, is a refusal that NAMES the variable or the path it wanted.

The layout under the root mirrors this repository's own: a file that used to be tracked at
`research/<study>/<name>` is served from `$KUMO_DATA_ROOT/research/kumo-trading-strategies/<study>/<name>`.
Mirroring keeps provenance readable in both trees and makes the migration of a research script
a one-line change — `data_path("research/<study>/<name>")`.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "KUMO_DATA_ROOT"
#: The subtree of the data repository that mirrors this repository's former `research/` files.
MIRROR = "research/kumo-trading-strategies"

#: THE OTHER THREE SUBTREES, and where each came from. Research scripts once opened these by a home
#: path (`~/projects/<repo>/…`), which is a real identity with good manners: right on one machine,
#: `FileNotFoundError` everywhere else, and the export cannot rewrite it into anything a stranger can
#: run. The public rewrite and this table name the SAME subtrees, so `$KUMO_DATA_ROOT/<subtree>/…`
#: is the path in both trees.
#:
#:   private machine layout                       served from
#:   ~/projects/ledger-tool/research/<ledger>/…      $KUMO_DATA_ROOT/ledger/…
#:   <KUMO_DATA_ROOT>/lab/…                   $KUMO_DATA_ROOT/lab/…
#:   <KUMO_DATA_ROOT>/lab/<other>/…                $KUMO_DATA_ROOT/lab/<other>/…
#:   <KUMO_DATA_ROOT>/legacy/…                $KUMO_DATA_ROOT/legacy/…
#:   ~/projects/kumo-trading-strategies/research/…        $KUMO_DATA_ROOT/research/kumo-trading-strategies/…  (MIRROR)
LEDGER = "ledger"
LAB = "lab"
LEGACY = "legacy"
SUBTREES = (LEDGER, LAB, LEGACY)


class DataRootUnset(RuntimeError):
    """`KUMO_DATA_ROOT` is not set. The message names the variable and what was wanted under it."""


class DataFileMissing(FileNotFoundError):
    """The root is set but does not hold the file. The message is the full path that was wanted."""


def data_root() -> Path:
    """The directory named by `KUMO_DATA_ROOT`, or `DataRootUnset`. Never a default."""
    raw = os.environ.get(ENV)
    if raw is None or raw.strip() == "":
        raise DataRootUnset(
            f"{ENV} is not set. It must name the checkout of the private data repository "
            f"(research-data); nothing in this tree reads data from anywhere else.")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise DataRootUnset(f"{ENV}={raw!r} is not a directory.")
    return root


def data_path(rel: str, *, must_exist: bool = True) -> Path:
    """`$KUMO_DATA_ROOT/<rel>`; refuses by name when the root is unset or the file is absent.

    `rel` is the path as this repository used to track it (`research/<study>/<name>`); the
    `research/` prefix is served from the mirror subtree. Any other prefix is used as given.
    """
    root = data_root()
    r = Path(rel)
    if r.is_absolute():
        raise ValueError(f"data_path takes a path relative to {ENV}, got {rel!r}")
    if r.parts and r.parts[0] == "research":
        r = Path(MIRROR, *r.parts[1:])
    p = root / r
    if must_exist and not p.exists():
        raise DataFileMissing(
            f"{p} — wanted for {rel!r}; {ENV}={root}. The data repository's MANIFEST.md lists "
            f"what it holds.")
    return p
