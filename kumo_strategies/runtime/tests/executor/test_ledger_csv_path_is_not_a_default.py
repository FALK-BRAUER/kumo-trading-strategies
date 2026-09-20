"""`LedgerBookSource` has no default ledger path (#211).

It had one: an absolute path on the machine the source was written on. The deployed instances
build the source through `build(SourceSpec(kind="ledger_book", params={"max_open_days": 57}))`
— no `csv_path` — so every one of them read that default, and a public tree would ship a
reader that silently looks for a stranger's home directory. A default that is a real identity is
a wrong answer with good manners: nothing downstream can tell "configured" from "assumed".

The path comes from `KUMO_LEDGER_CSV` when the source is not told one explicitly, and its
absence is a `SourceError` that names the variable — the refresher records a source's error per
source, so the pool keeps its previous set and the health surface says why.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.executor.sources import LedgerBookSource
from kumo_strategies.runtime.executor.sources.base import SourceError, SourceSpec, build
from kumo_strategies.runtime.executor.sources.ledger import ENV_CSV


def test_no_path_and_no_env_refuses_by_name(monkeypatch) -> None:
    monkeypatch.delenv(ENV_CSV, raising=False)
    with pytest.raises(SourceError, match=ENV_CSV):
        LedgerBookSource().fetch()


def test_the_deployed_construction_shape_refuses_the_same_way(monkeypatch) -> None:
    # EXACTLY what the instances' pool specs carry: kind + max_open_days, nothing else.
    monkeypatch.delenv(ENV_CSV, raising=False)
    src = build(SourceSpec(name="ledger_book", kind="ledger_book", params={"max_open_days": 57}))
    with pytest.raises(SourceError, match=ENV_CSV):
        src.fetch()


def test_the_env_path_travels(monkeypatch, tmp_path) -> None:
    # A path that does not exist, so the refusal it produces NAMES the path the env supplied —
    # proving the variable reached the reader rather than some other default.
    wanted = tmp_path / "ledger-book.csv"
    monkeypatch.setenv(ENV_CSV, str(wanted))
    with pytest.raises(SourceError, match=str(wanted)):
        LedgerBookSource().fetch()


def test_an_explicit_path_wins_over_the_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_CSV, str(tmp_path / "from-env.csv"))
    explicit = tmp_path / "explicit.csv"
    with pytest.raises(SourceError, match=str(explicit)):
        LedgerBookSource(csv_path=str(explicit)).fetch()


def test_upstream_changed_at_is_none_rather_than_raising_when_unresolvable(monkeypatch) -> None:
    # Its contract: None, and fetch() reports the problem properly. A raise here would turn a
    # fetchable source into a silently skipped one.
    monkeypatch.delenv(ENV_CSV, raising=False)
    assert LedgerBookSource().upstream_changed_at() is None
