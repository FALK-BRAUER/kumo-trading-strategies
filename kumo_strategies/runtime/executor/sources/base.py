"""Source definitions — pluggable contributors to the symbol pool.

A source owns a set of symbols and knows how to refresh it. It never mutates the pool directly and
never issues add/remove deltas: `fetch()` returns the WHOLE set it currently believes in, and the
pool replaces that source's contribution wholesale. That is what makes a refresh idempotent and
means a source can never delete a symbol another source (or the operator) contributed.

A source that cannot fetch must RAISE. Returning an empty set would silently empty the pool, which
is far more dangerous than running on a stale one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RefreshResult:
    source: str
    symbols: dict[str, dict]          # symbol -> per-source metadata (rank, opened_at, score...)
    detail: str = ""

    def __len__(self) -> int:
        return len(self.symbols)


class SourceError(RuntimeError):
    """Raised when a source cannot produce a trustworthy set. Never swallow this into an empty set."""


@runtime_checkable
class Source(Protocol):
    name: str
    def fetch(self, asof: datetime | None = None) -> RefreshResult: ...


@dataclass
class SourceSpec:
    """How a source is configured and scheduled. Serialisable, so the UI can render and edit it."""

    name: str
    kind: str                                  # registry key
    enabled: bool = True
    every_minutes: int = 24 * 60               # nightly by default
    params: dict = field(default_factory=dict)
    last_run: datetime | None = None
    last_status: str = "never"
    last_detail: str = ""
    last_count: int = 0

    def due(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        now = now or datetime.now(timezone.utc)
        return now - self.last_run >= timedelta(minutes=self.every_minutes)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["last_run"] = self.last_run.isoformat() if self.last_run else None
        return d


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a source kind under its `kind` attribute."""
    _REGISTRY[cls.kind] = cls
    return cls


def build(spec: SourceSpec) -> Source:
    if spec.kind not in _REGISTRY:
        raise SourceError(f"unknown source kind {spec.kind!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[spec.kind](name=spec.name, **spec.params)


def kinds() -> list[str]:
    return sorted(_REGISTRY)
