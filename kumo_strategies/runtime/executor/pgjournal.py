"""The action log, on Postgres. Append-only; never updated.

Spec #188. Orders answer WHAT happened; this answers WHY, months later. A decision row carries the
ranking, the gates, and a reason per symbol, so "why did it buy PARR on 4 August" is answerable
without re-running anything.

A write failure must never break the trading path — `write()` returns None rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kumo_strategies.runtime.executor.store import DEFAULT_SLOT, ActionLog, utcnow

class DuplicateDecision(RuntimeError):
    """A second decision for a session the database already has one for."""


DECISION, ORDER, FILL, STATE, POOL, RISK, ERROR = (
    "decision", "order", "fill", "state", "pool", "risk", "error")

# The runtime identity, in ONE place. It used to be a literal default here AND a literal default on
# PgSessionRunner, and when the tag moved 001 -> 002 only one of them changed: the audit trail was
# written under MOMENTUM-001 while position claims and Nautilus positions were under MOMENTUM-002.
# Self-consistent on each side, so nothing failed — the log simply described a strategy that, by id,
# held nothing.
#: NO DEFAULT LANE. This was "MOMENTUM-002" — a real, live, position-holding strategy. The three
#: defaults (here, `PgSessionRunner`, `OrderRequest`) were guarded by a test asserting they AGREED,
#: and on 2026-08-22 they agreed perfectly while `pgrunner`'s BUY path passed none of them: every
#: entry from BCTROT-004 and QC345-003 was built claiming to be MOMENTUM-002. Three defaults in
#: perfect agreement about the wrong lane, with the guard green throughout.
#:
#: Agreeing was never the property that mattered. Kept as an empty sentinel so "no lane named" is
#: representable and distinguishable from a lane.
DEFAULT_STRATEGY_ID = ""


@dataclass
class PgJournal:
    sessionmaker: async_sessionmaker[AsyncSession]
    strategy_id: str
    """REQUIRED. Every audit row carries it; a defaulted one files another lane's history under
    MOMENTUM-002, well-formed and undetectable."""

    async def write(self, kind: str, summary: str, *, session: str, detail: dict | None = None,
                    symbol: str | None = None, correlation: str | None = None,
                    slot: str | None = None) -> int | None:
        try:
            async with self.sessionmaker() as s, s.begin():
                row = ActionLog(ts=utcnow(), strategy_id=self.strategy_id, session=session,
                                kind=kind, symbol=symbol, summary=summary,
                                detail=detail or {}, correlation=correlation,
                                slot=slot or DEFAULT_SLOT)
                s.add(row)
                await s.flush()
                return int(row.id)
        except IntegrityError:
            # the one-decision-per-(session, slot) unique index fired: another run already decided
            # this slot. Re-raise so the caller stops rather than submitting a duplicate book.
            raise DuplicateDecision(
                f"{self.strategy_id} already decided {session}/{slot or DEFAULT_SLOT}") from None
        except Exception:                                   # noqa: BLE001 — deliberate
            # any other journal failure must not break the trading path
            return None

    async def decided_this_session(self, session: str, slot: str | None = None) -> bool:
        """The idempotency key: a restart mid-session must not decide the same SLOT twice (#189, #29).

        Slot-scoped, so an intraday schedule can decide at midday having already decided at the
        open, while a retry of the midday decision is still refused. `slot=None` means the current
        single slot, which keeps every existing caller behaving exactly as before.
        """
        async with self.sessionmaker() as s:
            r = (await s.execute(select(ActionLog.id).where(
                ActionLog.strategy_id == self.strategy_id, ActionLog.session == session,
                ActionLog.slot == (slot or DEFAULT_SLOT),
                ActionLog.kind == DECISION).limit(1))).first()
        return r is not None

    async def tail(self, n: int = 50, kind: str | None = None,
                   symbol: str | None = None) -> list[dict]:
        q = select(ActionLog).where(ActionLog.strategy_id == self.strategy_id)
        if kind:
            q = q.where(ActionLog.kind == kind)
        if symbol:
            q = q.where(ActionLog.symbol == symbol)
        async with self.sessionmaker() as s:
            rows = (await s.execute(q.order_by(ActionLog.id.desc()).limit(n))).scalars().all()
        return [{"id": r.id, "ts": r.ts.isoformat(), "strategy_id": r.strategy_id,
                 "session": r.session, "kind": r.kind, "symbol": r.symbol,
                 "summary": r.summary, "detail": r.detail, "correlation": r.correlation}
                for r in rows]

    async def explain(self, session: str, slot: str | None = None) -> dict | None:
        """The decision row for a SESSION AND SLOT — the basis for every entry and exit that slot made.

        Unfiltered by slot until #54: a strategy with more than one decision per day (BCTROT-004's
        `open+150m`/`close-20m`) got whichever slot decided most recently, regardless of which one a
        caller actually meant -- silently reading a DIFFERENT slot's book as this slot's basis for
        resuming. `slot=None` keeps every single-slot caller (MOMENTUM-002) behaving exactly as
        before, matching `decided_this_session`'s own default.
        """
        async with self.sessionmaker() as s:
            r = (await s.execute(select(ActionLog).where(
                ActionLog.strategy_id == self.strategy_id, ActionLog.session == session,
                ActionLog.slot == (slot or DEFAULT_SLOT),
                ActionLog.kind == DECISION).order_by(ActionLog.id.desc()).limit(1))).scalars().first()
        if not r:
            return None
        return {"id": r.id, "ts": r.ts.isoformat(), "strategy_id": r.strategy_id,
                "session": r.session, "kind": r.kind, "symbol": r.symbol,
                "summary": r.summary, "detail": r.detail, "correlation": r.correlation}
