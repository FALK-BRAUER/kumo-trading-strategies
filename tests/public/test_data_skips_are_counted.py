"""The number of data-dependent tests is a number, and this file holds it (#211).

A test marked `needs_data` skips by name when `KUMO_DATA_ROOT` is unset. A skip is a test that did
not run; if such a test is deleted, or its marker dropped, nothing else in the suite notices —
the green is the same. So the count is asserted here, against the collected session rather than
against the source, and changing it is a deliberate edit with the reason beside it.

EXPECTED is 0 today, and that is a measured fact, not a placeholder: with the 114 data files
physically removed the suite was byte-for-byte the same (2387 passed / 37 skipped / 17 xfailed,
2026-09-14). The mechanism exists so the FIRST data test cannot vanish, not because one exists.
"""

from __future__ import annotations

import os

import pytest

from kumo_strategies.data_root import ENV

#: Tests carrying `@pytest.mark.needs_data(...)`. Change it in the same commit as the test.
EXPECTED = 0


def _marked(session) -> list:
    return [i for i in session.items if i.get_closest_marker("needs_data") is not None]


def test_the_count_of_data_dependent_tests_is_exactly_the_declared_one(request) -> None:
    marked = _marked(request.session)
    names = sorted(i.nodeid for i in marked)
    assert len(marked) == EXPECTED, (
        f"{len(marked)} tests carry needs_data, EXPECTED says {EXPECTED}. Update EXPECTED in the "
        f"same commit — a data test appearing or vanishing must be a deliberate edit. {names}")


def test_every_marked_test_names_what_it_wants(request) -> None:
    # The marker's argument is the path the skip reason will name; a bare marker would skip with
    # "<unnamed>", which is a reason nobody can act on.
    bare = [i.nodeid for i in _marked(request.session)
            if not i.get_closest_marker("needs_data").args]
    assert bare == [], f"needs_data without the path it wants: {bare}"


def test_the_skip_mechanism_is_live(request, monkeypatch) -> None:
    # Drive the hook itself rather than trusting the marker table: with the root unset, a marked
    # item raises pytest's Skipped naming the variable and the path.
    from _pytest.outcomes import Skipped
    import conftest as root_conftest
    monkeypatch.delenv(ENV, raising=False)

    class _Mark:
        args = ("research/example-study/example.parquet",)

    class _Item:
        def get_closest_marker(self, name):
            return _Mark() if name == "needs_data" else None

    with pytest.raises(Skipped) as e:
        root_conftest.pytest_runtest_setup(_Item())
    assert ENV in str(e.value) and "research/example-study/example.parquet" in str(e.value)


def test_with_the_root_set_a_marked_test_is_not_skipped(monkeypatch, tmp_path) -> None:
    import conftest as root_conftest
    monkeypatch.setenv(ENV, str(tmp_path))

    class _Mark:
        args = ("research/x/y.csv",)

    class _Item:
        def get_closest_marker(self, name):
            return _Mark() if name == "needs_data" else None

    root_conftest.pytest_runtest_setup(_Item())   # no Skipped raised
