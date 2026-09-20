"""Job runner on Postgres — keeps sources fresh, gates the session decision.

Source specs live in `exec_source_spec` rather than a JSON file, so the UI can list and edit what
feeds the pool and the config survives a container restart like everything else.

Two kinds of job, deliberately separate:
  SOURCE REFRESH   per source, on its own cadence, failure-isolated
  SESSION DECIDE   once per session, and only after every enabled source is fresh — a stale source
                   is a HARD BLOCK, because deciding on yesterday's pool silently is the failure
                   this whole design exists to prevent
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kumo_strategies.runtime.executor.pgjournal import ERROR, POOL, PgJournal
from kumo_strategies.runtime.executor.pgpool import PgSymbolPool
from kumo_strategies.runtime.executor.sources import SourceError, SourceSpec, build
from kumo_strategies.runtime.executor.store import SourceSpecRow, pg_insert, utcnow


_RETRY_AFTER_FAILURE = timedelta(minutes=2)
"""How soon to re-attempt a source that FAILED, regardless of its cadence. Short, because the runner
blocks the session on a stale source, so a failure that waits out a long cadence costs the day."""


def _due(row: SourceSpecRow, now: datetime | None = None,
         upstream: datetime | None = None) -> bool:
    """Due on cadence, OR whenever the upstream is newer than what we cached.

    Cadence alone answers "is our copy old?" but not "has the source moved?", and those come apart
    exactly when it matters. A ledger_book cached at 18:26 sat happily inside its 24h cadence while
    ledger-tool published a corrected file at 22:20 -- so the pool served a symbol that upstream had
    already fixed, and reported itself healthy doing it. Comparing our last refresh against the
    upstream's own published timestamp closes that: newer upstream means our cache is behind,
    whatever the clock says.
    """
    if not row.enabled:
        return False
    if row.last_run is None:
        return True
    # A FAILED refresh still stamps last_run, so the cadence clock restarts on failure and the
    # source is not retried for a full period. A transient fault — an unreadable file, a 500, a
    # mount that was not there yet — therefore blocks the whole day, and the runner treats a failed
    # source as a HARD BLOCK on deciding. That is exactly what happened: a missing mount failed the
    # refresh at 13:05, the 30-minute cadence pushed the retry to 13:35, and the session fires at
    # 13:35. Retry a failure quickly instead; success returns to the normal cadence.
    if (row.last_status or "") == "failed":
        lr = row.last_run.replace(tzinfo=timezone.utc) if row.last_run.tzinfo is None else row.last_run
        return (now or utcnow()) - lr >= _RETRY_AFTER_FAILURE
    lr = row.last_run.replace(tzinfo=timezone.utc) if row.last_run.tzinfo is None else row.last_run
    if upstream is not None:
        up = upstream.replace(tzinfo=timezone.utc) if upstream.tzinfo is None else upstream
        if up > lr:
            return True
    now = now or utcnow()
    return now - lr >= timedelta(minutes=row.every_minutes)


@dataclass
class PgJobRunner:
    sessionmaker: async_sessionmaker[AsyncSession]
    pool: PgSymbolPool
    journal: PgJournal

    async def add(self, spec: SourceSpec) -> None:
        async with self.sessionmaker() as s, s.begin():
            await s.execute(pg_insert(SourceSpecRow).values(
                name=spec.name, kind=spec.kind, enabled=spec.enabled,
                every_minutes=spec.every_minutes, params=spec.params
            ).on_conflict_do_update(index_elements=[SourceSpecRow.name], set_={
                "kind": spec.kind, "enabled": spec.enabled,
                "every_minutes": spec.every_minutes, "params": spec.params}))

    async def remove(self, name: str) -> None:
        """Remove the spec AND everything it contributed.

        Deleting only the spec left the source's symbols buyable with nothing refreshing them, and
        left an orphaned health row that would eventually go stale and block every session with no
        way to clear it from the UI.
        """
        await self.pool.drop_source(name)
        async with self.sessionmaker() as s, s.begin():
            await s.execute(delete(SourceSpecRow).where(SourceSpecRow.name == name))

    async def specs(self) -> list[SourceSpecRow]:
        async with self.sessionmaker() as s:
            return list((await s.execute(
                select(SourceSpecRow).order_by(SourceSpecRow.name))).scalars().all())

    async def refresh(self, name: str, *, session: str, force: bool = False) -> dict:
        async with self.sessionmaker() as s:
            row = (await s.execute(select(SourceSpecRow)
                                   .where(SourceSpecRow.name == name))).scalars().first()
        if row is None:
            return {"source": name, "status": "unknown"}
        src = build(SourceSpec(name=row.name, kind=row.kind, params=dict(row.params or {})))
        # Asking the source when it last changed must never be able to block a refresh. A source
        # that cannot answer falls back to cadence; one that raises is a source that needs fetching
        # anyway, and fetch() will report the real error.
        upstream = None
        try:
            probe = getattr(src, "upstream_changed_at", None)
            upstream = probe() if probe is not None else None
        except Exception:                                          # noqa: BLE001
            upstream = None
        if not force and not _due(row, upstream=upstream):
            return {"source": name, "status": "not_due",
                    "last_run": row.last_run.isoformat() if row.last_run else None,
                    "upstream_changed_at": upstream.isoformat() if upstream else None,
                    "last_count": row.last_count}
        status, detail, count = "ok", "", 0
        try:
            res = src.fetch()
            detail = res.detail
            # `detail` GOES ON THE HEALTH ROW, not only in the journal. It was computed here and
            # passed only to `journal.write`, so `exec_pool_source.detail` stayed NULL for every
            # source this refresher maintains — and that blank is what cost 2026-08-26. `/pool`
            # showed alpaca-paper `detail=None, age 0h` beside staging `detail="seeded from instance
            # config", age 57h`, with ZERO POSTs in either access log. Two repos independently
            # concluded staging was "missing a pusher", because the one field that could have said
            # what maintains a source was empty on the instance where one was working.
            count = await self.pool.refresh_source(name, res.symbols, detail=detail)
            await self.journal.write(POOL, f"{name}: {count} symbols — {detail}", session=session,
                                     detail={"source": name, "count": count})
            out = {"source": name, "status": "ok", "count": count, "detail": detail}
        except (SourceError, Exception) as e:                     # noqa: BLE001
            # keep the last good set — a stale pool beats an empty one
            await self.pool.mark_source_failed(name, str(e))
            status, detail = "failed", str(e)
            await self.journal.write(ERROR, f"{name} refresh failed: {e}", session=session,
                                     detail={"source": name})
            out = {"source": name, "status": "failed", "detail": detail}
        async with self.sessionmaker() as s, s.begin():
            vals = {"last_run": utcnow(), "last_status": status, "last_detail": detail[:2000]}
            # keep the last SUCCESSFUL count on failure — zeroing it hides how big the set was
            if status == "ok":
                vals["last_count"] = count
            await s.execute(SourceSpecRow.__table__.update()
                            .where(SourceSpecRow.name == name).values(**vals))
        return out

    async def refresh_due(self, *, session: str, force: bool = False) -> list[dict]:
        return [await self.refresh(r.name, session=session, force=force)
                for r in await self.specs() if r.enabled]

    async def status(self) -> list[dict]:
        health = {h.source: h for h in await self.pool.sources()}
        out = []
        for r in await self.specs():
            h = health.get(r.name)
            out.append({"name": r.name, "kind": r.kind, "enabled": r.enabled,
                        "every_minutes": r.every_minutes, "params": r.params,
                        "last_run": r.last_run.isoformat() if r.last_run else None,
                        "last_status": r.last_status, "last_detail": r.last_detail,
                        "last_count": r.last_count,
                        "symbols_in_pool": h.symbol_count if h else 0,
                        "pool_status": h.status if h else "never",
                        "stale": h.is_stale() if h else True, "due": _due(r)})
        return out
