"""A hand-maintained list as a source — the simplest contributor, and the baseline any smarter
source has to beat.

Distinct from the operator's PIN override: a pin is an exception to what the sources say, whereas
this is a source in its own right that happens to be curated by hand. Use this for a standing
watchlist; use a pin to force one name in against the feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kumo_strategies.runtime.executor.sources.base import RefreshResult, SourceError, register


@register
@dataclass
class StaticListSource:
    kind = "static_list"
    name: str = "watchlist"
    symbols: list[str] = field(default_factory=list)
    path: str | None = None          # optional newline/comma file, so it can be edited outside

    def fetch(self, asof: datetime | None = None) -> RefreshResult:
        syms = list(self.symbols)
        if self.path:
            p = Path(self.path)
            if not p.exists():
                raise SourceError(f"list file not found: {p}")
            raw = p.read_text().replace(",", "\n").split()
            syms += [s.strip().upper() for s in raw if s.strip()]
        uniq = {s.upper(): {"via": "static"} for s in syms if s}
        if not uniq:
            raise SourceError("static list is empty — refusing to contribute nothing")
        return RefreshResult(self.name, uniq, f"{len(uniq)} names")
