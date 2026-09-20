"""Postgres persistence for the executor — pool, overrides, source health, action log.

WHY POSTGRES AND NOT SQLITE: the storage architecture already decided this. App data lives in
Postgres (watchlists, command ledger, trade-cycle envelope, manager/event tables all do), Nautilus
owns trade state, bars stay in Parquet. A second store would fork that decision, and merging into
the cockpit later would mean porting anyway.

It is also simply the right tool here. An earlier SQLite pass cost two bugs that do not exist in a
pooled client/server database: connections are thread-affine, and two concurrent reads on one
connection raise "bad parameter or other API misuse". An ASGI threadpool serving three parallel API
calls hits both immediately.

Schema matches kumo-trading-platform `docs/spec-systematic-strategy-runtime.md` (#186, #188) so the tables
port unchanged. Tables are namespaced `exec_` to sit alongside the cockpit's own without collision.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text, func, select, text)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import asyncio
import contextvars
import logging
import weakref

from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_DEFAULT_URL = "postgresql+asyncpg://kumo:kumo@localhost:5432/kumo"

_log = logging.getLogger(__name__)


def database_url() -> str:
    """`KUMO_DATABASE_URL`, same variable the cockpit uses, so one env configures both."""
    return os.environ.get("KUMO_DATABASE_URL", _DEFAULT_URL)


class Base(DeclarativeBase):
    pass


class PoolSource(Base):
    """One row per (source, symbol). A refresh REPLACES a source's whole set — see pool.py."""
    __tablename__ = "exec_pool_source"
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class PoolRefresh(Base):
    """Per-source health, so staleness is observable rather than inferred from an empty pool."""
    __tablename__ = "exec_pool_refresh"
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    status: Mapped[str] = mapped_column(String(16))
    symbol_count: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class PoolOverride(Base):
    """Operator whitelist / blacklist. Sits ABOVE every source; a source can never remove one."""
    __tablename__ = "exec_pool_override"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8), primary_key=True)      # pin | exclude
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), default="operator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


DEFAULT_SLOT = "open+5m"
"""The single slot MOMENTUM-002 runs today. Also the server default, so rows written before #29
carry the value they implicitly had rather than a NULL that would weaken the unique index."""


class ActionLog(Base):
    """Append-only. Never updated. A decision row carries the whole basis for that session."""
    __tablename__ = "exec_action_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    strategy_id: Mapped[str] = mapped_column(String(32))
    session: Mapped[str] = mapped_column(String(10))
    kind: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    correlation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    slot: Mapped[str] = mapped_column(String(32), nullable=False, server_default=DEFAULT_SLOT)
    """WHICH decision of the session this is (#29). One slot a day today; several under an
    intraday schedule.

    NOT NULL with a server default rather than nullable, and that is the load-bearing choice.
    Postgres treats NULLs as DISTINCT in a unique index, so a nullable column would let unlimited
    NULL-slot decisions coexist for one session — silently removing the guarantee this index exists
    to provide, at exactly the moment the schema looks like it is being strengthened. A default of
    the current live slot keeps a single-slot config exactly as strict as it is today."""

    __table_args__ = (
        Index("ix_exec_log_session", "strategy_id", "session"),
        Index("ix_exec_log_kind", "strategy_id", "kind", "ts"),
        # ONE decision per (strategy, session, SLOT), enforced by the database. A check-then-write in
        # application code races: two concurrent session runs both read "no decision yet" before
        # either commits, and both submit orders. A partial unique index makes the second insert
        # fail instead, which is the only place that can be decided atomically.
        #
        # `slot` was added for #29. With one slot configured this is identical in strength to the
        # old (strategy_id, session) index; with several it permits exactly one decision per slot,
        # which is what an intraday schedule needs and what a retry must still be refused by.
        Index("uq_exec_one_decision_per_session", "strategy_id", "session", "slot",
              unique=True, postgresql_where=text("kind = 'decision'")),
    )


class StrategyState(Base):
    """Lifecycle state, durable across restarts — a restart must not silently resume TRADING."""
    __tablename__ = "exec_strategy_state"
    strategy_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    state: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class PositionState(Base):
    """Entry and running peak per open position — the give-back trail's memory.

    This MUST be durable. The API builds a fresh session runner per request, so holding the trail
    in memory meant `entry` and `peak` were re-seeded from today's price on every run: a position
    that had run +40% and was handing it back looked brand new, and the exit never fired. The
    give-back rule was effectively dead in the live runtime while passing every backtest, because
    the backtest keeps one runner for the whole loop.

    It also records QUANTITY, and that is not cosmetic. Ownership used to be symbol-only, so a
    claim on AAPL meant the whole account's AAPL was treated as this strategy's: 10 ours beside a
    manual 90 exited as SELL 100. Quantity makes the claim say how much, so an exit can never
    reach further than what we actually opened.
    """
    __tablename__ = "exec_position_state"
    strategy_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    qty: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    entry: Mapped[float] = mapped_column(Float)
    peak: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Trail PROVENANCE (#197 B1). `peak` alone cannot say whether it was observed or invented, and
    # the old seeding path wrote `entry, peak = (today's price, today's price)` for any position it
    # had no row for — asserting a peak that never happened. `quality` lets the shared evaluator skip
    # peak-relative rules on a position whose peak is unknown instead of firing on fiction.
    #   live           opened while this runner was watching; both values observed
    #   reconstructed  entry from the real fill, peak rebuilt from bars covering the whole hold
    #   adopted        predates any usable record — PEAK IS NOT KNOWN
    # `sessions_*` exist because max_hold_days and stall_days count SESSIONS, not calendar days, and
    # a restart must not reset either.
    sessions_held: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    sessions_since_high: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    quality: Mapped[str] = mapped_column(String(16), default="adopted", server_default="adopted")


class SourceSpecRow(Base):
    """Source configuration + schedule, so the UI can list and edit what feeds the pool."""
    __tablename__ = "exec_source_spec"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    every_minutes: Mapped[int] = mapped_column(Integer, default=1440)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status: Mapped[str] = mapped_column(String(16), default="never")
    last_detail: Mapped[str] = mapped_column(Text, default="")
    last_count: Mapped[int] = mapped_column(Integer, default=0)


def make_engine(url: str | None = None, *, cross_loop: bool = False):
    """pool_pre_ping recycles dead connections so a Postgres restart does not wedge the pool.

    `cross_loop=True` uses NullPool. A pooled AsyncEngine holds asyncpg connections bound to the
    event loop that created them, and SQLAlchemy's async docs are explicit that such an engine must
    not be shared across loops. The cockpit builds this engine during synchronous node startup --
    inside a short-lived `asyncio.run()` -- and then every later query runs on the TradingNode's
    own loop. The pooled connections belong to a loop that no longer exists, so the first real
    database call fails, and it fails at the worst moment: the first session, before any durable
    decision row exists to explain it.

    NullPool opens per use, so nothing is carried between loops. That costs a connect per query,
    which for a once-a-day strategy plus a 60s refresh is not a cost worth optimising against a
    startup crash.
    """
    kw = {"poolclass": NullPool} if cross_loop else {"pool_size": 5, "max_overflow": 5}
    return create_async_engine(url or database_url(), pool_pre_ping=True, **kw)


def make_sessionmaker(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine) -> None:
    """Dev/standalone convenience. In the cockpit these come from Alembic like every other table."""
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["Base", "PoolSource", "PoolRefresh", "PoolOverride", "ActionLog", "StrategyState",
           "PositionState", "SourceSpecRow", "database_url", "make_engine", "make_sessionmaker", "create_all",
           "utcnow", "pg_insert", "select",
           "write_claim", "record_claim", "drop_claim", "claim_upsert", "drop_claim_stmt", "trail_upsert_stmt",
           "canonical_symbol", "ClaimWriteStalled", "ClaimWriteReentered"]

# -- the ownership claim, as ONE writer (kumo-trading-platform issue 540) -------------------------------------------
def canonical_symbol(symbol) -> str:
    """The row key's spelling of a symbol — and therefore the LOCK key's. `claim_upsert` and
    `drop_claim_stmt` have always written `strip().upper()`; a lock keyed on anything else would
    serialise `gmab` and `GMAB` as two keys while they write one row (codex, #124 scope review)."""
    return str(symbol or "").strip().upper()


def claim_upsert(strategy_id: str, symbol: str, qty: float, entry_px: float, *,
                 only_if_absent: bool = False):
    """The statement that records "this lane holds this much of this symbol", and nothing else.

    WHY THIS EXISTS. `exec_position_state` had rows only for the two lanes that run a trailing stop.
    QC345-003 and TECHIVOL-005 hold real positions and never wrote one, so they contributed nothing
    to `pgrunner._foreign_claims` -- and `own_ceiling(acct, mine, other)` narrows sizing by exactly
    that term. Measured on alpaca-paper 2026-08-25: DELL is QC345 7 + TECHIVOL 2, neither claiming, so
    any lane that DOES claim DELL computes `own_ceiling(acct, mine, 0)` = `mine`. The attribution
    term VANISHES precisely when there are two other holders, which is the "two claimants, no
    attribution" case `own_ceiling`'s docstring names as the one it exists for. The guard is not
    weakened -- it is silently skipped, and from the outside that is identical to being sole claimant.

    ONE IMPLEMENTATION, TWO CALLERS. `qc27_runner` is here; QC345's gateway is kumo-trading-platform's and
    calls this rather than writing its own. Two implementations of one claim is the shape that
    produced the defect.

    THE CONFLICT BRANCH MOVES `qty` ALONE, and that is the load-bearing part. `entry`, `peak`,
    `quality` and the `sessions_*` counters are the give-back trail's memory; `_save_state` owns
    them, and its docstring records what resetting them costs -- `max_hold_days` and `stall_days`
    count SESSIONS and "would silently reset on restart". A claim write that touched them would
    reintroduce that from a second writer, on a row the first writer owns. So the first write
    establishes the row and every later one moves the quantity.

    `only_if_absent=True` is ADOPTION's shape (cockpit's `claims_backfill`): insert if no row, and do
    NOTHING if one exists — never move a quantity a terminal sync may just have written. It is not a
    substitute for the lock: a DELETE landing between backfill's read and this write leaves no row, and
    this would happily re-insert; only `write_claim`'s `build` callable, re-reading under the lock,
    closes that (#124).

    `quality="adopted"` on insert, never a computed peak. Its own comment: the old seeding path wrote
    `entry, peak = (today's price, today's price)` for a position it had no row for -- "asserting a
    peak that never happened". A rotation lane knows its real fill, so `entry` is TRUE; it has no
    peak history, so `peak = entry` under `adopted`, which `TrailState.peak_is_trustworthy` already
    reads as PEAK IS NOT KNOWN.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("claim needs a symbol")
    # `(strategy_id, symbol)` is the primary key, so `DELL` and `dell` would be two claims on ONE
    # holding -- and `_foreign_claims` SUMS by symbol, making the account appear to owe more than it
    # holds. Two callers in two repos make the casing genuinely likely.
    q = float(qty or 0.0)
    # ZERO IS THE INVALID VALUE, NOT NEGATIVE (#173). The guard's purpose is unchanged — a row
    # claiming NOTHING still tells `_foreign_claims` this lane owns the symbol — but `q > 0` also
    # refused the one quantity a short lane needs to write. Claims are SIGNED now: a negative is a
    # real claim on a real position and `reducible` reads the sign to know the lane covers by
    # BUYING.
    if q == 0:
        raise ValueError(
            f"claim qty cannot be zero, got {qty!r} — a row claiming nothing still tells "
            f"`_foreign_claims` this lane owns the symbol. Use drop_claim() to release it.")
    e = float(entry_px) if entry_px is not None else float("nan")
    if not math.isfinite(e) or e <= 0:
        raise ValueError(
            f"claim needs a real entry price, got {entry_px!r} — a nan survives every comparison "
            f"and every `or` default, and would poison `entry` for whatever reads it as a fill.")
    now = utcnow()
    stmt = pg_insert(PositionState).values(
        strategy_id=strategy_id, symbol=sym, qty=q,
        entry=e, peak=e, quality="adopted",
        sessions_held=0, sessions_since_high=0, updated_at=now,
    )
    if only_if_absent:
        return stmt.on_conflict_do_nothing(index_elements=[PositionState.strategy_id, PositionState.symbol])
    return stmt.on_conflict_do_update(
        index_elements=[PositionState.strategy_id, PositionState.symbol],
        set_={"qty": q, "updated_at": now},
    )


def trail_upsert_stmt(strategy_id: str, symbol: str, *, entry: float, peak: float, quality: str,
                      sessions_held: int, sessions_since_high: int, qty: float | None):
    """`PgSessionRunner._save_state`'s statement, built HERE so every `PositionState` write is built
    beside the lock that serialises it (#124). Persists the WHOLE trail — `_save_state`'s own
    docstring: writing only `peak` "was survivable while give-back was the one rule; `max_hold_days`
    and `stall_days` count sessions and would silently reset on restart". `qty=None` leaves the
    quantity alone on an existing row."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    sym = canonical_symbol(symbol)
    now = utcnow()
    update = {"peak": peak, "sessions_held": sessions_held, "sessions_since_high": sessions_since_high,
              "quality": quality, "updated_at": now}
    if qty is not None:
        update["qty"] = qty
    return pg_insert(PositionState).values(
        strategy_id=strategy_id, symbol=sym, entry=entry, peak=peak,
        sessions_held=sessions_held, sessions_since_high=sessions_since_high,
        quality=quality, qty=qty or 0.0, updated_at=now,
    ).on_conflict_do_update(
        index_elements=[PositionState.strategy_id, PositionState.symbol], set_=update)


def drop_claim_stmt(strategy_id: str, symbol: str):
    """Release the claim on a FULL exit. Both keys, always — a `WHERE` missing either would clear
    another lane's claim, or every lane's claim on the symbol."""
    from sqlalchemy import delete as _delete

    sym = canonical_symbol(symbol)
    return _delete(PositionState).where(
        PositionState.strategy_id == strategy_id, PositionState.symbol == sym)


# -- THE writer (#124 / kumo-trading-platform issue 845) ----------------------------------------------------------
#
# Measured on an Alpaca paper instance 2026-09-09 19:40:19Z: three fills for one SELL 59 GMAB, 22 ms apart, became three
# concurrent transactions writing absolute quantities (24, 20, drop). The DELETE from the third
# committed before the INSERT from the first and the ledger ended on a resurrected row — qty 24,
# `adopted`, `sessions_held` 0 — for a position the broker had already closed. Against a real Postgres
# with the pool cockpit runs (NullPool) that is 13% of three-fill exits and 91% of six-fill ones
# (`scripts/probe_claim_write_order.py`).
#
# TWO LOCKS, BECAUSE EACH ALONE IS WRONG. An advisory lock serialises by ARRIVAL at Postgres, and with
# one connection per transaction arrival order is connect-latency order, not event order. An
# in-process lock orders callers on the one node loop (every strategy's `_loop` is that loop, and
# `run_coroutine_threadsafe` enqueues FIFO) but cannot see a second process, a migration, or psql.
# So: the in-process lock gives ORDER, the transaction-scoped advisory lock gives ATOMICITY, and both
# live in the one function every writer calls.

#: Two int4 keys — `hashtext(strategy_id), hashtext(symbol)` — no delimiter ambiguity, and a
#: collision only over-serialises two unrelated keys; the DML still names the exact row.
_KEY_LOCK_SQL = text("SELECT pg_advisory_xact_lock(hashtext(:sid), hashtext(:sym))")

#: A same-key write waits behind a slow Postgres for at most this long, then fails LOUDLY. A silent
#: timeout that let a waiter proceed would be the race wearing a timeout's clothes.
CLAIM_WRITE_TIMEOUT_S = 30.0

#: After the body times out, how long the UNWIND (SQLAlchemy's rollback on the same connection) is
#: given before the in-process lock is released anyway. A Postgres that is slow rather than dead can
#: block the rollback indefinitely (step-9 review); without this bound the key wedged for good.
UNWIND_GRACE_S = 5.0

#: loop -> {(strategy_id, canonical symbol): Lock}. Weak on the loop so a finished loop (tests,
#: `asyncio.run` at boot) takes its locks with it; weak on the lock so an idle key holds nothing —
#: whoever is waiting on a lock is what keeps it alive (codex, #124 scope review).
_KEY_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, weakref.WeakValueDictionary]" = (
    weakref.WeakKeyDictionary())

#: Keys this task is currently writing — a nested `write_claim` for one of them from inside `build`
#: could only wait for itself. Tasks copy the context at creation, so the transaction task sees it.
_IN_WRITE: contextvars.ContextVar[frozenset] = contextvars.ContextVar("kumo_claim_write_keys",
                                                                        default=frozenset())


class ClaimWriteStalled(RuntimeError):
    """A claim write waited longer than `CLAIM_WRITE_TIMEOUT_S`. Raised, never swallowed here — the
    caller's NEVER-RAISES handler names it, which is the point."""


class ClaimWriteReentered(RuntimeError):
    """`write_claim` called for a key from inside that key's own `build` — refused at once, by name,
    rather than discovered as a 30 s stall."""


def _key_lock(strategy_id: str, symbol: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    per_loop = _KEY_LOCKS.get(loop)
    if per_loop is None:
        per_loop = _KEY_LOCKS[loop] = weakref.WeakValueDictionary()
    key = (str(strategy_id), canonical_symbol(symbol))
    lock = per_loop.get(key)
    if lock is None:
        lock = asyncio.Lock()
        per_loop[key] = lock
    return lock


async def _transaction(journal, strategy_id: str, sym: str, stmts, build) -> None:
    async with journal.sessionmaker() as s, s.begin():
        # Bound INTO the statement, not passed beside it: the statement then describes the key
        # it locks on its own, which is what a journal double can inspect.
        await s.execute(_KEY_LOCK_SQL.bindparams(sid=str(strategy_id), sym=sym))
        for st in stmts:
            await s.execute(st)
        more = await build(s) if build is not None else None
        for st in (more or ()):
            await s.execute(st)


async def write_claim(journal, strategy_id: str, symbol: str, *stmts, build=None) -> None:
    """Execute `stmts` — and whatever `build(session)` returns — for ONE `(strategy_id, symbol)`,
    serialised in call order against every other writer of that key.

    `build` is awaited INSIDE the locked transaction with the session, so a caller that must READ
    before it decides what to write (adoption: "is there a row now?") reads a truth no concurrent
    writer can change under it. It returns statements to execute, or nothing. It must not call
    `write_claim` for the SAME key — that is refused as `ClaimWriteReentered`.

    THE TRANSACTION RUNS AS ITS OWN TASK (step-9 review). `asyncio.timeout` around the body spends
    its one cancel on the body; the unwind — rollback on the same connection — then runs unbounded,
    and against a Postgres that is slow rather than dead that wedged the key permanently while the
    docstring promised "later writers proceed". Now: wait for the task up to `CLAIM_WRITE_TIMEOUT_S`;
    on expiry cancel it and wait `UNWIND_GRACE_S` for the unwind; then release the in-process lock
    REGARDLESS and say whether the task ended or was ABANDONED.

    WHAT KEEPS THE ABANDONMENT PATH ORDERED IS THE ADVISORY LOCK, NOT A ROLLBACK (#128). An abandoned
    task holds its NullPool connection — and with it `pg_advisory_xact_lock` — until its transaction
    ends, so later writers of the key queue behind it INSIDE Postgres and cannot overtake it. And a
    task that ended `cancelled()` was not necessarily rolled back: a cancel arriving during COMMIT
    cancels something already at the server, and asyncpg cannot un-commit it. Neither message claims
    a rollback it cannot know about.

    AN EXTERNAL CANCEL CANCELS THE TRANSACTION (#128). `asyncio.wait` does not cancel what it waits
    on, so a `CancelledError` here would otherwise release the in-process lock and leave the
    transaction running detached — free to commit after a later writer of the key. It is cancelled
    and awaited for `UNWIND_GRACE_S` before the cancel is re-raised.

    A `TimeoutError` raised INSIDE the transaction (asyncpg's ETIMEDOUT is one, and on 3.13 it is the
    same class the timeout machinery raises) propagates as ITSELF: the task either finished — its
    exception is re-raised unchanged — or it did not, and only then is it a stall.

    `journal` is anything carrying a `sessionmaker` — `PgJournal` does.
    """
    sym = canonical_symbol(symbol)
    key = (str(strategy_id), sym)
    if key in _IN_WRITE.get():
        raise ClaimWriteReentered(f"{strategy_id}/{sym}: write_claim called from inside its own build")
    lock = _key_lock(strategy_id, sym)
    acquired = False
    try:
        # ACQUIRE INSIDE THE SAME try THAT RELEASES (codex, implementation review): with the acquire
        # outside it, a cancellation delivered between the lock being granted and the `try` being
        # entered would leak the key. Here `acquired` flips with no await between the grant and the
        # flag, so the `finally` always knows whether it holds the lock.
        try:
            async with asyncio.timeout(CLAIM_WRITE_TIMEOUT_S):
                await lock.acquire()
                acquired = True
        except TimeoutError as exc:
            raise ClaimWriteStalled(
                f"{strategy_id}/{sym}: claim write waited {CLAIM_WRITE_TIMEOUT_S:.0f}s "
                f"behind another writer of the key and gave up — ordering NOT applied") from exc
        token = _IN_WRITE.set(_IN_WRITE.get() | {key})
        try:
            task = asyncio.create_task(_transaction(journal, strategy_id, sym, stmts, build))
            try:
                done, _ = await asyncio.wait({task}, timeout=CLAIM_WRITE_TIMEOUT_S)
            except asyncio.CancelledError:
                # AN EXTERNAL CANCEL MUST UNWIND THE TRANSACTION, NOT DETACH IT (#128). `asyncio.wait`
                # does not cancel what it waits on: without this the caller unwinds, the `finally`s
                # release the in-process key lock, and the transaction task runs on. Cancelled while
                # still in the CONNECT window — before `pg_advisory_xact_lock`, where nothing orders
                # it — it then takes the lock after a later writer released it and its INSERT lands on
                # top of that writer's DELETE. That is #124's resurrected row. `on_stop()` cancels the
                # in-flight session task by design (operator pause mid-session), so this is a
                # production path: `momentum_rotation.py:334`, `qc27_rotation.py:240`,
                # `qc345_rotation.py:278`.
                task.cancel()
                await asyncio.wait({task}, timeout=UNWIND_GRACE_S)
                raise
            if task in done:
                task.result()                       # errors from inside propagate UNCHANGED
                return
            task.cancel()
            unwound, _ = await asyncio.wait({task}, timeout=UNWIND_GRACE_S)
            if task in unwound:
                # NEVER SAY "rolled back" FROM TASK COMPLETION (#128). A cancel arriving during COMMIT
                # is a cancel of something already at the server: asyncpg's cancel does not un-commit,
                # and the task still ends `cancelled()`. Ended-cancelled means the transaction is over,
                # not that it was undone.
                outcome = ("unwound; it was rolled back UNLESS the cancel arrived mid-COMMIT, in which "
                           "case the write may have landed — disposition unknown from here")
            else:
                outcome = (f"rollback ABANDONED after {UNWIND_GRACE_S:.0f}s — its connection HOLDS its "
                           f"advisory lock until its transaction ends, so later writers of the key "
                           f"queue behind it inside Postgres (that, not a rollback, is what keeps the "
                           f"abandonment path ordered)")
                # An abandoned transaction that later fails with a real DB error must not vanish
                # (#128): retrieving the exception silences asyncio's warning, so LOG it.
                task.add_done_callback(
                    lambda t: None if t.cancelled() else (
                        _log.error("%s/%s: abandoned claim-write transaction failed after the stall: %r",
                                   strategy_id, sym, t.exception())
                        if t.exception() is not None else None))
            raise ClaimWriteStalled(
                f"{strategy_id}/{sym}: Postgres did not finish the claim write within "
                f"{CLAIM_WRITE_TIMEOUT_S:.0f}s — {outcome}; the in-process lock is released and later "
                f"writers for the key proceed")
        finally:
            _IN_WRITE.reset(token)
    finally:
        if acquired:
            lock.release()


#: WHETHER THIS BUILD'S CLAIMS CAN REPRESENT A SHORT (#173).
#:
#: A CAPABILITY, NOT A REVISION, because cockpit's gateway must gate on what is INSTALLED rather
#: than on what a pin says — an editable install serves one checkout to every worktree, so a
#: revision check can pass while the code running is a different one entirely (#162).
#:
#: True means `sync_claim_to` writes SIGNED quantities and `reducible` can size a short lane's
#: cover. While it was False, cockpit refused every CRSISHORT entry with `claims_unsigned`, and
#: that refusal was right: an unclaimed short is a silent subtraction from every long lane's
#: residue — the same freeze, arrived at by omission.
CLAIMS_REPRESENT_SHORT = True


async def record_claim(journal, strategy_id: str, symbol: str, qty: float, entry_px: float, *,
                       only_if_absent: bool = False) -> None:
    """Idempotent upsert of this lane's claim, in call order per key. See `claim_upsert` for what it
    deliberately does not touch."""
    await write_claim(journal, strategy_id, symbol,
                      claim_upsert(strategy_id, symbol, qty, entry_px, only_if_absent=only_if_absent))


async def sync_claim_to(journal, strategy_id: str, symbol: str, qty, px, *,
                        write=None) -> bool:
    """Bring this lane's claim to what the BOOK says. One implementation, every runner (#133, #829).

    Returns True when a statement was written.

    The claim follows the venue's terminal answer, not submit-time acceptance. A local accept only
    says Nautilus has the order; the venue may still deny or reject it. A denial once reverted
    nothing and MOMENTUM-002 carried a 260-share LAND claim on a position that never existed
    (kumo-trading-platform issue 829); a protective stop that filled left TECHIVOL-005 claiming 97 TOST against a
    book of 0 for a whole session (#133).

    ZERO IS FLAT, AND NOTHING ELSE IS. `qty <= 0` is the tempting test and it is wrong the moment a
    lane holds the other side: CRSISHORT's positions are NEGATIVE, so that test reads a live 57-share
    short as flat and releases the claim on a real position — this method's own defect, mirrored.
    The claim records MAGNITUDE, and only an exactly flat book drops it.

    A PARTIAL MOVES THE QUANTITY AND MUST NOT DROP IT. Dropping on a partial makes the lane stop
    claiming shares it still holds, which loosens every OTHER lane's ceiling onto our position — the
    mirror of the breach rather than a fix for it.

    A MISSING PRICE DOES NOT BECOME A FABRICATED ONE. `claim_upsert` moves `qty` alone on conflict
    and reads `entry_px` only on INSERT, so with no price and no existing row a write would invent an
    entry every later reader treats as observed. Nothing downstream can tell a fabricated entry from
    a real one, so the row is left alone and the caller is told. Dropping needs no price and still
    happens: a reconciled or synthesised terminal event may carry none, and that is exactly the case
    where the breach would otherwise stand.
    """
    writer = write if write is not None else write_claim
    if qty is None:
        return False
    # `int()` TRUNCATES, so a fractional remainder below one share reads as flat. Every lane here
    # trades whole shares — `closing_quantity` returns an int and the adapters submit
    # `Quantity.from_int` — so this cannot arise today, and it is guarded explicitly rather than
    # left to that being true forever: a fractional-share venue would otherwise DELETE the claim on
    # a live position, which is this function's own defect in the direction it least expects.
    # WHOLE SHARES, ASSERTED. `int()` TRUNCATES: 0.4 becomes 0 and DELETES the claim on a live
    # position, and 1.5 becomes 1 and silently under-claims half a share. Every lane here trades
    # whole shares — `closing_quantity` returns an int and the adapters submit `Quantity.from_int` —
    # so a fractional quantity means a CALLER computed one wrongly, and truncating it here would
    # bury that in a number no reader can question.
    f = float(qty)
    if f != int(f):
        raise ValueError(
            f"claim quantity {qty!r} for {symbol} is not a whole number of shares. This package "
            f"trades whole shares, so a fraction means a caller computed one wrongly — truncating "
            f"would {'DELETE the claim on a live position' if abs(f) < 1 else 'silently under-claim'}.")
    q = int(qty)
    if q == 0:
        await writer(journal, strategy_id, symbol, drop_claim_stmt(strategy_id, symbol))
        return True
    if px is None:
        return False
    # SIGNED, NOT `abs(q)` (#173). The magnitude form cannot represent a short, and
    # `reducible(acct, mine, other)` needs the sign to know which way a lane unwinds: a short lane
    # covers by BUYING. It also made a short lane's claim SUBTRACT from every other lane's residue
    # as though it were a competing long — MOMENTUM holding +30 with CRSISHORT short 10 on an
    # account of +20 could sell 10 of the 30 shares it actually held, and the symbol then failed
    # `if q > 0` in `pgrunner.run` and vanished from `held_qty` entirely, so give-back, stall,
    # forced exits and LIQUIDATING all skipped it in silence.
    #
    # THE MIGRATION IS FREE AND ITS WINDOW IS CLOSING. For a long lane signed and magnitude are the
    # same number, so every row written before today is already correct and nothing has to be
    # rewritten. The moment an `abs()` records a SHORT, the table holds two conventions and no row
    # says which it is — which is why this lands before `ORDER_PATH_COMPLETE` flips rather than
    # before the lane trades.
    await writer(journal, strategy_id, symbol,
                 claim_upsert(strategy_id, symbol, q, float(px)))
    return True


async def drop_claim(journal, strategy_id: str, symbol: str) -> None:
    """Release the claim, in call order per key. Call on a FULL exit only — a partial exit should
    `record_claim` the remaining quantity instead, or the lane stops claiming shares it still holds."""
    await write_claim(journal, strategy_id, symbol, drop_claim_stmt(strategy_id, symbol))
