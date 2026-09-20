"""ONE backtest runner for every strategy (#270, Falk 2026-09-18: "no per strategy runner").

`runner_sessions` is THE runner. Every other `backtesting/runner*.py` is scheduled for folding and
deletion, in the ticket's order, and is listed here BY NAME so that (a) a fold deletes its line and
the test goes red if the file survives, and (b) a NEW per-strategy runner fails the suite the moment
it is added. When the list is empty this test is the guard the ticket asked for; until then it is the
countdown.

Bound to the FILES, not to a substring: a module is a runner when it is named `runner*` under
`backtesting/` — that is the convention every one of the eight followed.
"""

from __future__ import annotations

from pathlib import Path

BACKTESTING = Path(__file__).resolve().parents[1]   # this test sits beside the package it guards

THE_RUNNER = "runner_sessions"

#: Still to fold, in #270's order. Delete the line in the PR that deletes the file.
SCHEDULED_FOR_DELETION = (
)


def _runner_modules() -> set[str]:
    return {p.stem for p in BACKTESTING.glob("runner*.py")}


def test_the_one_runner_exists() -> None:
    assert THE_RUNNER in _runner_modules()


def test_no_runner_module_outside_the_one_and_the_scheduled() -> None:
    extra = _runner_modules() - {THE_RUNNER} - set(SCHEDULED_FOR_DELETION)
    assert not extra, (
        f"new runner module(s) under backtesting/: {sorted(extra)}. There is ONE runner "
        f"(`{THE_RUNNER}`); a per-strategy runner is the shape #270 removes. Fold the behaviour into "
        f"it or, for a Nautilus-engine replay, add a case to the engine-parity test.")


def test_every_scheduled_runner_still_exists_or_its_line_was_removed() -> None:
    """A fold that deletes the file must delete the line: a stale entry would let the NEXT fold's
    file slip past the previous assertion under a name that no longer means anything."""
    gone = set(SCHEDULED_FOR_DELETION) - _runner_modules()
    assert not gone, f"folded already, remove from SCHEDULED_FOR_DELETION: {sorted(gone)}"
