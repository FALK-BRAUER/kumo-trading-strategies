"""Every lane kumo-trading-platform DECLARES must have an adapter here (the cross-repo floor).

The failure this prevents is not subtle and has happened in the other direction already: cockpit's
registry allocates `order_id_tag` and maps `external_id -> internal id`, and a declared lane whose
adapter is missing or RENAMED does not degrade — cockpit cannot construct it, and whichever way that
surfaces, it surfaces at boot on a trading morning rather than here.

Both repos already police their own half. Nothing policed the JOIN. This repo's suite proves every
adapter satisfies the contract; cockpit's proves every registry entry is well-formed and tag-unique.
An adapter that this repo renamed and cockpit still declares passes both.

READ FROM `origin/main`, NEVER FROM THE SIBLING WORKING TREE. A checkout can be dirty, behind, or
sitting on another branch — on 2026-08-22 a cockpit worktree held `main` 119 commits back, `git
checkout main` failed silently, and a paper image was built from the stale tree. The published branch
is the only thing both repos can agree on.

WHAT THIS CANNOT DO, stated because a guard whose limits are unstated gets trusted past them: it
needs the cockpit checkout to exist. Without it the check SKIPS, and a skip is not a pass. It is a
local pre-flight, not a CI gate — the CI gate has to live where both repos are present, which is
cockpit's deploy.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

#: The cockpit checkout this test reads the registry from — a SIBLING repository, so its location
#: is the environment's to say, never a path baked in here (#211). Unset → the test skips and says
#: what it wanted; it must not silently pass, and it must not assume the author's layout.
COCKPIT_ENV = "KUMO_COCKPIT_TREE"
REGISTRY = "backend/api/strategy_registry.py"


def _cockpit() -> Path | None:
    """The checkout `KUMO_COCKPIT_TREE` names, or None — the skip below says what was wanted."""
    import os
    raw = os.environ.get(COCKPIT_ENV)
    return Path(raw).expanduser() if raw else None

#: Lanes cockpit declares and implements ITSELF, so no adapter is expected here.
#:
#: MANUAL is the operator's own book, run by cockpit's `UiFeedStrategy` — it holds `001`, has live
#: positions keyed to `MANUAL-001`, and predates this repo.
#:
#: AN EXEMPTION LEDGER, NOT A FILTER. Each entry is named and justified, so a NEW declared lane with
#: no adapter fails here loudly instead of being absorbed into a pattern. Adding a second entry has
#: to be a deliberate edit with a reason beside it.
COCKPIT_NATIVE = {"MANUAL": "cockpit's own UiFeedStrategy — the operator's manual book"}


def _declared_external_ids() -> set[str]:
    """`external_id=` on every `StrategyEntry` in cockpit's published registry, by AST.

    Parsed rather than imported: importing cockpit's module drags in its dependencies, and parsed
    rather than grepped because a substring match over source is satisfied by a comment — which this
    file has several of, discussing `external_id` in prose.
    """
    src = subprocess.run(["git", "show", f"origin/main:{REGISTRY}"], cwd=_cockpit(),
                         capture_output=True, text=True, check=True).stdout
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "external_id" and isinstance(kw.value, ast.Constant):
                out.add(kw.value.value)
    return out


def _shipped_external_ids() -> set[str]:
    from kumo_strategies.runtime.tests.nautilus.test_contract import _shipped_adapters

    return {c.EXTERNAL_ID for c in _shipped_adapters()}


def _cockpit_available() -> bool:
    root = _cockpit()
    if root is None or not (root / ".git").exists():
        return False
    return subprocess.run(["git", "rev-parse", "--verify", "origin/main"], cwd=root,
                          capture_output=True).returncode == 0


needs_cockpit = pytest.mark.skipif(
    not _cockpit_available(),
    reason=f"no cockpit checkout with origin/main at {COCKPIT_ENV}={_cockpit()} — the JOIN between "
           f"the two repos is NOT being checked in this run")


@needs_cockpit
def test_every_lane_cockpit_declares_has_an_adapter_here():
    """The floor. A declared lane with no adapter is a node that cannot construct what it registered.

    Derived on both sides — cockpit's registry is read from its published branch, this side is the
    same discovery every other conformance test uses. Neither list is written down anywhere.
    """
    declared = _declared_external_ids()
    assert declared, "parsed no external_ids from cockpit's registry — this guard is reading nothing"

    missing = sorted(declared - _shipped_external_ids() - set(COCKPIT_NATIVE))
    assert not missing, (
        f"kumo-trading-platform declares {missing} and no adapter here publishes that EXTERNAL_ID. Either the "
        f"adapter was renamed, or the lane was declared before it existed. Cockpit maps "
        f"external_id -> internal id at registration, so this surfaces at boot.")


@needs_cockpit
def test_the_exemptions_are_still_lanes_cockpit_actually_DECLARES():
    """An exemption for a lane that no longer exists is a hole nobody is watching.

    `COCKPIT_NATIVE` suppresses a failure. If cockpit drops MANUAL, the entry stops describing
    reality and starts describing nothing — and the next lane that needs exempting gets added beside
    a stale one rather than argued about.
    """
    stale = sorted(set(COCKPIT_NATIVE) - _declared_external_ids())
    assert not stale, f"exempted {stale}, which cockpit no longer declares — delete the exemption"


@needs_cockpit
def test_this_repo_may_ship_MORE_than_cockpit_declares():
    """The floor is one-directional, deliberately.

    `TEMPLATE` is the scaffold new lanes are copied from and must never be registered; a research
    adapter may exist here for a strategy cockpit has not adopted. Asserting equality would make
    every unadopted adapter a failure and would push people to delete work to make a test pass.
    """
    extra = _shipped_external_ids() - _declared_external_ids()
    assert "TEMPLATE" in extra or not extra, (
        "the scaffold stopped being unregistered, which would make it constructible as a lane")


def test_the_cross_repo_check_is_NOT_silently_skipped():
    """Runs even without cockpit, because the skip above is the interesting failure mode.

    A guard that skips is not a guard that passes, and the whole point of this file is a join nobody
    was checking. So the skip has to be visible as a skip: this asserts the marker exists and carries
    a reason naming what went unchecked, rather than the file quietly contributing nothing.
    """
    assert needs_cockpit.kwargs["reason"], "the skip must say what was not checked"
    if not _cockpit_available():
        pytest.skip(f"CROSS-REPO FLOOR NOT CHECKED — no cockpit checkout at {COCKPIT_ENV}={_cockpit()}")
