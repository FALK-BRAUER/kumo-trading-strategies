"""The symbol pool, on Postgres. Async over asyncpg, the same stack the cockpit uses.

Semantics are identical to the earlier local-file version and to the spec (#186):

    effective pool = union(source sets) + pins - excludes,  derived on read, never stored

A refresh REPLACES a source's whole set in one transaction, so an import is idempotent and a source
can never delete a symbol it does not own. A source that fails keeps its last good set — an empty
pool is far more dangerous than a stale one.

Pool membership governs BUYING only. `must_liquidate` is true only for an explicit operator
blacklist entry, never because a feed dropped a name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kumo_strategies.runtime.executor.store import (
    PoolOverride, PoolRefresh, PoolSource, pg_insert, utcnow)

PIN, EXCLUDE = "pin", "exclude"


@dataclass(frozen=True)
class SourceHealth:
    source: str
    refreshed_at: datetime
    status: str
    symbol_count: int
    detail: str | None = None

    def is_stale(self, max_age_hours: float = 36.0) -> bool:
        if self.status != "ok":
            return True
        ts = self.refreshed_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - ts > timedelta(hours=max_age_hours)


@dataclass(frozen=True)
class PoolEntry:
    symbol: str
    sources: tuple[str, ...]
    pinned: bool
    meta: dict

    @property
    def provenance(self) -> str:
        bits = [*self.sources] + (["pinned"] if self.pinned else [])
        return " + ".join(bits) if bits else "?"


class ShrinkRejected(RuntimeError):
    """A refresh that would drop an implausible share of a source's set."""


class UnknownSource(KeyError):
    """A refresh naming a source that does not exist, without asking to create one.

    Creating on first write is convenient and, on a surface anyone can POST to, wrong: a typo
    produces a source that is FRESH, REAL, AND FEEDS NOTHING — well-formed, plausible, and
    indistinguishable from health by anything downstream. Same shape as a default identity that is a
    live lane.
    """


def check_source_known(source: str, *, known: bool, create: bool) -> None:
    """Raise unless the source exists or creating one was asked for explicitly."""
    if known or create:
        return
    raise UnknownSource(
        f"{source!r} is not a known pool source. A refresh does not invent one: a mistyped name "
        f"yields a source that is fresh, real and feeds nothing, while the source it was meant to "
        f"refresh goes stale. Pass create=True to add it deliberately.")


def check_shrink(source: str, *, prev: int, now: int, max_shrink: float,
                 reason: str | None) -> str | None:
    """The message for a refusal, or None if the refresh may proceed.

    Pure, so the RULE is testable without a database — the write path needs Postgres and the
    decision does not, and the decision is where the defects live.

    THE GUARD EXISTS because a pool departure is acted on as the followed trader's SELL SIGNAL,
    worth 29 points of return, so a truncated feed liquidates. It catches the case an empty-check
    never would: a source returning 3 symbols of 93 SUCCEEDS, and the stale gate does not fire.

    THE TRAP IN IT, and the reason `reason` exists: the same floor blocks a followed trader
    GENUINELY GOING FLAT — the single most significant thing that source can ever say, and the one
    the pool exists to act on. Refusing it forever makes the event that matters most the one event
    that cannot happen.

    So the escape is a REASON, never a boolean: a loud, recorded override rather than a refusal. A
    rule with no escape gets bypassed under pressure, which removes the record and not the case.
    """
    if reason is not None and not reason.strip():
        raise ValueError(
            "an empty reason is not a reason — state why this source may lose its set, or let the "
            "guard refuse")
    if reason:
        return None
    if prev and now < prev * (1 - max_shrink):
        return (f"{source}: refresh returned {now} symbols, down from {prev} "
                f"(>{100*max_shrink:.0f}% shrink) — keeping the previous set. A pool departure is "
                f"acted on as a sell signal, so a truncated feed would liquidate. If this shrink is "
                f"real, say why: refresh_source(..., allow_shrink='<reason>').")
    return None



class PgSymbolPool:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession],
                 max_shrink: float = 0.5) -> None:
        self._sm = sessionmaker
        #: reject a refresh that loses more than this fraction of the source's previous set.
        #: A pool departure is treated as the followed trader's SELL SIGNAL and acted on, which is
        #: worth 29 points of return — so the feed must be trusted, and that trust needs a floor.
        #: A source that FAILS is caught by the stale gate; a source that SUCCEEDS while returning
        #: a truncated set is not, and is the case this catches.
        self._max_shrink = max_shrink

    # -- sources ---------------------------------------------------------------------------
    async def refresh_source(self, source: str, symbols: dict[str, dict] | set[str] | list[str],
                             *, status: str = "ok", detail: str | None = None,
                             create: bool = True, allow_shrink: str | None = None) -> int:
        """REPLACE a source's whole set. Returns the number of rows now held.

        THE ONLY WRITER to `exec_pool_source`, deliberately. Cockpit's `POST /pool/source/{name}`
        calls this rather than writing the table, because two writers with replace semantics do not
        race — the loser's rows vanish.

        `refreshed_at` is stamped HERE and is not a parameter. A caller-supplied timestamp is a
        caller-supplied lie the first time a clock drifts, and `source_health()` blocks a decision on
        staleness.

        `create` DEFAULTS TO TRUE, which is not the rule I would choose for a fresh system. The
        scheduled refreshers are running against a deployed stack; flipping the default would make
        any source lacking a health row start failing, and a stale source is a HARD BLOCK on
        deciding — an unrun refresher does not degrade a lane, it stops it deciding at all. So the
        NEW surface passes `create=False` and the running one is untouched. Tighten it once every
        live source is known to have a health row.
        """
        if status != "ok":
            raise ValueError("use mark_source_failed() — refresh_source must not store a bad set")
        meta = symbols if isinstance(symbols, dict) else {s: {} for s in symbols}
        now = utcnow()
        async with self._sm() as s:
            health = (await s.execute(select(PoolRefresh.symbol_count)
                                      .where(PoolRefresh.source == source))).first()
        check_source_known(source, known=health is not None, create=create)
        prev = (health[0] if health else 0) or 0
        refused = check_shrink(source, prev=prev, now=len(meta),
                               max_shrink=self._max_shrink, reason=allow_shrink)
        if refused:
            raise ShrinkRejected(refused)
        if allow_shrink and len(meta) < prev:
            # The override is RECORDED, not merely honoured. An unexplained empty set and one an
            # operator justified must not read the same afterwards.
            detail = f"shrink {prev}->{len(meta)} allowed: {allow_shrink}" + (
                f" | {detail}" if detail else "")
        async with self._sm() as s, s.begin():
            # whole-set replacement, in ONE transaction: no window where the pool is short.
            # Delete only what LEFT the set, then upsert the rest — a plain insert here raced two
            # concurrent refreshes of the same source into a duplicate-key error on every symbol
            # they had in common, rolling back the whole 93-row batch.
            if meta:
                await s.execute(delete(PoolSource).where(
                    PoolSource.source == source, PoolSource.symbol.notin_(list(meta.keys()))))
                stmt = pg_insert(PoolSource).values([
                    {"source": source, "symbol": k, "meta": v, "refreshed_at": now}
                    for k, v in meta.items()])
                await s.execute(stmt.on_conflict_do_update(
                    index_elements=[PoolSource.source, PoolSource.symbol],
                    set_={"meta": stmt.excluded.meta, "refreshed_at": stmt.excluded.refreshed_at}))
            else:
                await s.execute(delete(PoolSource).where(PoolSource.source == source))
            await s.execute(pg_insert(PoolRefresh).values(
                source=source, refreshed_at=now, status="ok", symbol_count=len(meta),
                detail=detail).on_conflict_do_update(
                index_elements=[PoolRefresh.source],
                set_={"refreshed_at": now, "status": "ok", "symbol_count": len(meta),
                      "detail": detail}))
        return len(meta)

    async def mark_source_failed(self, source: str, detail: str) -> None:
        """Record the failure, KEEP the previous set. The pool degrades; it does not empty."""
        async with self._sm() as s, s.begin():
            cur = (await s.execute(select(PoolRefresh.symbol_count)
                                   .where(PoolRefresh.source == source))).scalar()
            await s.execute(pg_insert(PoolRefresh).values(
                source=source, refreshed_at=utcnow(), status="failed",
                symbol_count=cur or 0, detail=detail).on_conflict_do_update(
                index_elements=[PoolRefresh.source],
                set_={"refreshed_at": utcnow(), "status": "failed", "detail": detail}))

    async def drop_source(self, source: str) -> None:
        """Forget a source entirely — its symbols and its health row."""
        async with self._sm() as s, s.begin():
            await s.execute(delete(PoolSource).where(PoolSource.source == source))
            await s.execute(delete(PoolRefresh).where(PoolRefresh.source == source))

    async def sources(self) -> list[SourceHealth]:
        async with self._sm() as s:
            rows = (await s.execute(select(PoolRefresh).order_by(PoolRefresh.source))).scalars().all()
        return [SourceHealth(r.source, r.refreshed_at, r.status, r.symbol_count, r.detail)
                for r in rows]

    # -- overrides -------------------------------------------------------------------------
    async def set_override(self, symbol: str, kind: str, *, reason: str = "",
                           by: str = "operator", expires_at: datetime | None = None) -> None:
        if kind not in (PIN, EXCLUDE):
            raise ValueError(f"kind must be {PIN!r} or {EXCLUDE!r}, got {kind!r}")
        other = EXCLUDE if kind == PIN else PIN
        async with self._sm() as s, s.begin():
            await s.execute(delete(PoolOverride).where(
                PoolOverride.symbol == symbol, PoolOverride.kind == other))
            await s.execute(pg_insert(PoolOverride).values(
                symbol=symbol, kind=kind, reason=reason, created_by=by,
                created_at=utcnow(), expires_at=expires_at).on_conflict_do_update(
                index_elements=[PoolOverride.symbol, PoolOverride.kind],
                set_={"reason": reason, "created_by": by, "created_at": utcnow(),
                      "expires_at": expires_at}))

    async def clear_override(self, symbol: str, kind: str) -> None:
        async with self._sm() as s, s.begin():
            await s.execute(delete(PoolOverride).where(
                PoolOverride.symbol == symbol, PoolOverride.kind == kind))

    async def overrides(self, kind: str | None = None) -> list[dict]:
        q = select(PoolOverride)
        if kind:
            q = q.where(PoolOverride.kind == kind)
        async with self._sm() as s:
            rows = (await s.execute(q.order_by(PoolOverride.symbol))).scalars().all()
        now = utcnow()
        return [{"symbol": r.symbol, "kind": r.kind, "reason": r.reason,
                 "created_by": r.created_by, "created_at": r.created_at,
                 "expires_at": r.expires_at}
                for r in rows
                if not r.expires_at or (r.expires_at.replace(tzinfo=timezone.utc)
                                        if r.expires_at.tzinfo is None else r.expires_at) > now]

    # -- the derived pool ------------------------------------------------------------------
    async def effective(self) -> dict[str, PoolEntry]:
        async with self._sm() as s:
            src = (await s.execute(select(PoolSource).order_by(PoolSource.source))).scalars().all()
            ovs = (await s.execute(select(PoolOverride))).scalars().all()
        by: dict[str, dict] = {}
        for r in src:
            e = by.setdefault(r.symbol, {"sources": [], "meta": {}})
            e["sources"].append(r.source)
            e["meta"].update(r.meta or {})
        now = utcnow()
        live = [o for o in ovs
                if not o.expires_at or (o.expires_at.replace(tzinfo=timezone.utc)
                                        if o.expires_at.tzinfo is None else o.expires_at) > now]
        pins = {o.symbol for o in live if o.kind == PIN}
        excl = {o.symbol for o in live if o.kind == EXCLUDE}
        for p in pins:
            by.setdefault(p, {"sources": [], "meta": {}})
        return {s2: PoolEntry(s2, tuple(v["sources"]), s2 in pins, v["meta"])
                for s2, v in sorted(by.items()) if s2 not in excl}

    async def symbols(self) -> set[str]:
        return set(await self.effective())

    async def is_buyable(self, symbol: str) -> bool:
        return symbol in await self.effective()

    async def must_liquidate(self, symbol: str) -> bool:
        """Only an explicit operator EXCLUDE forces a sale — never a source dropping a name."""
        return symbol in {o["symbol"] for o in await self.overrides(EXCLUDE)}
