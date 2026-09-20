"""`KUMO_DATA_ROOT` is the only place data comes from, and its absence is a refusal by NAME (#211).

Each case names what the reader wanted, because a refusal that does not say what was missing is a
failure nobody can act on. There is deliberately no test for a default: there is none.
"""

from __future__ import annotations

import pytest

from kumo_strategies.data_root import (
    ENV, MIRROR, SUBTREES, DataFileMissing, DataRootUnset, data_path, data_root)


def test_unset_refuses_and_names_the_variable(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(DataRootUnset, match=ENV):
        data_root()


def test_empty_is_unset(monkeypatch) -> None:
    # `export KUMO_DATA_ROOT=` is the shape compose produces for an undeclared variable; an empty
    # string must not resolve to the current directory.
    monkeypatch.setenv(ENV, "")
    with pytest.raises(DataRootUnset, match=ENV):
        data_root()


def test_not_a_directory_refuses(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV, str(tmp_path / "absent"))
    with pytest.raises(DataRootUnset, match="not a directory"):
        data_root()


def test_missing_file_names_the_full_path_it_wanted(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(DataFileMissing) as e:
        data_path("research/momentum-003/daily.parquet")
    # The mirror subtree, the study, the file — and the variable that pointed there.
    assert str(tmp_path / MIRROR / "momentum-003" / "daily.parquet") in str(e.value)
    assert ENV in str(e.value)


def test_research_prefix_is_served_from_the_mirror(monkeypatch, tmp_path) -> None:
    f = tmp_path / MIRROR / "residual-gate" / "half_spreads.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    monkeypatch.setenv(ENV, str(tmp_path))
    assert data_path("research/residual-gate/half_spreads.json") == f


def test_other_prefixes_are_used_as_given(monkeypatch, tmp_path) -> None:
    f = tmp_path / "bars" / "alpaca" / "x.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"")
    monkeypatch.setenv(ENV, str(tmp_path))
    assert data_path("bars/alpaca/x.parquet") == f


def test_absolute_paths_are_refused(monkeypatch, tmp_path) -> None:
    # An absolute path would bypass the root entirely — the private-machine default in disguise.
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(ValueError, match="relative"):
        data_path(str(tmp_path / "anything.csv"))


@pytest.mark.parametrize("subtree", SUBTREES)
def test_the_external_subtrees_are_served_verbatim_under_the_root(monkeypatch, tmp_path, subtree) -> None:
    """`ledger/`, `lab/`, `legacy/` are the three places research scripts used to reach by a home
    path. They are NOT mirrored: the path under the root is the path the script names."""
    monkeypatch.setenv(ENV, str(tmp_path))
    want = tmp_path / subtree / "a" / "b.parquet"
    want.parent.mkdir(parents=True)
    want.write_bytes(b"")
    assert data_path(f"{subtree}/a/b.parquet") == want


@pytest.mark.parametrize("subtree", SUBTREES)
def test_an_unset_root_refuses_by_name_even_when_the_file_need_not_exist(monkeypatch, subtree) -> None:
    """The scripts resolve their inputs at import with `must_exist=False`; the refusal they get on a
    machine without the root must still say WHICH variable, not `FileNotFoundError: ~/…`."""
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(DataRootUnset, match=ENV):
        data_path(f"{subtree}/anything.csv", must_exist=False)
