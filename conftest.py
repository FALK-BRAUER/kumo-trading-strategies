"""Refuse to run the suite against a different checkout's source (#161).

17 worktrees share ONE venv whose `.pth` names one absolute path, so 16 of them import
`kumo_strategies` from the MAIN checkout whatever branch they hold. A suite run in such a worktree
takes its TESTS from the branch and its PACKAGE from main.

The failure that prompted this was the harmless kind — a missing module, which errors as soon as
anything points at the branch. The kind this exists for is a MODIFIED module: new expectations
against old code, or old expectations against code the branch already changed. Neither errors.
Either can pass.

Reported rather than merely asserted: the served path is printed on every run, right or wrong, so it
is on the record instead of assumed.
"""

from __future__ import annotations

import pytest

from kumo_strategies._checkout import checkout_conflict, explain, served_from


def pytest_report_header(config) -> str:
    """One line, every run. An invariant nobody can see is one nobody notices breaking."""
    try:
        served = served_from()
    except Exception as exc:                                  # noqa: BLE001
        return f"kumo_strategies: could not resolve the served package ({exc})"
    root = config.rootpath
    mark = "OK" if served == root.resolve() else "DIFFERENT CHECKOUT"
    return f"kumo_strategies served from: {served}  [{mark}]"


def pytest_collection(session) -> None:
    """Fail the whole run, before collection, rather than per-test.

    A guard that fires once per test buries itself; and a suite testing the wrong source has nothing
    worth collecting. `pytest.exit` rather than an assertion so the message is the last thing on the
    terminal instead of the first of several hundred.
    """
    conflict = checkout_conflict(session.config.rootpath)
    if conflict is not None:
        pytest.exit(explain(*conflict), returncode=3)


def pytest_runtest_setup(item) -> None:
    """A data-dependent test skips with the PATH it wanted when `KUMO_DATA_ROOT` is unset (#211).

    One place, so the reason is the same sentence on every such test and the count of them is
    something `tests/public/test_data_skips_are_counted.py` can assert — a data test that quietly
    stopped running would otherwise look like a green one.
    """
    import os
    from kumo_strategies.data_root import ENV, MIRROR
    mark = item.get_closest_marker("needs_data")
    if mark is None:
        return
    rel = mark.args[0] if mark.args else "<unnamed>"
    if not os.environ.get(ENV):
        pytest.skip(f"{ENV} unset — wanted {rel} (served from $"
                    f"{ENV}/{MIRROR}/… for research/ paths)")
