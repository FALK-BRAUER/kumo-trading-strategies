"""issue 124 / kumo-trading-platform issue 845 — every write to `exec_position_state` for one key takes the
key's Postgres advisory transaction lock FIRST, inside the transaction that writes.

Measured on an Alpaca paper instance, BCTROT-004 GMAB, 2026-09-09 19:40:19Z: three fills 22 ms apart produced three
concurrent `sync_claim` transactions (24, 20, 0). The DELETE from the third committed before the
INSERT ... ON CONFLICT from the first, and the ledger ended on a resurrected row — `qty=24
quality=adopted sessions_held=0 opened_at` moved — for a position the broker had already closed.

WHY THE LOCK IS HERE AND NOT IN COCKPIT (codex, platform issue 845 scope review). A cockpit-side asyncio
lock would cover the two gateways and miss `PgSessionRunner._save_state/_drop_state`, which write the
same keys on the same loop at decision and reconcile time; it is loop-affine; and it cannot span two
engine processes. The boundary is the table, and the only thing every writer shares is the
transaction it writes in. `pg_advisory_xact_lock` is scoped to that transaction: released on commit or
rollback, nothing to leak on a swallowed exception, and Postgres queues waiters in arrival order — so
callers reaching it in event order commit in event order.

THE ENUMERATION IS ENFORCED AT THE STATEMENT. Listing every caller is what keeps being wrong, so the
static guard refuses any `PositionState` DML built outside `store.py`, and the behavioural tests drive
`store`'s helpers and `pgrunner`'s two writers through a journal double that records transaction
boundaries.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import kumo_strategies
from sqlalchemy.sql.dml import Delete, Insert
from sqlalchemy.sql.elements import TextClause

from kumo_strategies.runtime.executor.store import claim_upsert, drop_claim, record_claim

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]


LANE, SYM = "BCTROT-004", "GMAB"


class _Journal:
    """Records every statement, grouped by the transaction (`begin()` block) it ran in."""

    def __init__(self, *, fail_on=None):
        self.transactions: list[list] = []
        self.strategy_id = LANE
        self.fail_on = fail_on          # a statement TYPE whose execution raises inside the transaction

    def sessionmaker(self):
        journal = self

        class _S:
            def __init__(self):
                self.stmts = []

            @asynccontextmanager
            async def begin(self):
                # BEGIN IS MARKED (codex, step 3): a lock executed on the session BEFORE `begin()` is a
                # session-scoped lock in production and would otherwise look transaction-scoped here.
                self.stmts.append("BEGIN")
                yield
                journal.transactions.append(list(self.stmts))

            async def execute(self, stmt, params=None):
                self.stmts.append(stmt)
                if journal.fail_on is not None and isinstance(stmt, journal.fail_on):
                    raise ConnectionError("postgres went away mid-transaction")
                return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

        @asynccontextmanager
        async def _cm():
            yield _S()

        return _cm()


def _is_key_lock(stmt, strategy_id: str, symbol: str) -> bool:
    """A `pg_advisory_xact_lock` TextClause whose bound values name BOTH halves of the key."""
    if not isinstance(stmt, TextClause) or "pg_advisory_xact_lock" not in str(stmt):
        return False
    bound = " ".join(str(v) for v in stmt.compile().params.values())
    return strategy_id in bound and symbol.upper() in bound


def _one_transaction_lock_then_dml(journal: _Journal, dml_type, strategy_id=LANE, symbol=SYM):
    assert len(journal.transactions) == 1, journal.transactions
    stmts = journal.transactions[0]
    assert stmts[0] == "BEGIN", f"a statement ran on the session BEFORE the transaction: {stmts[0]!r}"
    assert any(isinstance(s, dml_type) for s in stmts), f"no {dml_type.__name__} was executed — vacuous"
    assert _is_key_lock(stmts[1], strategy_id, symbol), (
        f"the transaction's FIRST statement is {type(stmts[1]).__name__}, not the key's advisory lock; "
        f"a write that takes the lock after its DML, or not at all, races every other writer of the key")


# -- the store's two helpers ----------------------------------------------------------------------------
def test_record_claim_takes_the_key_lock_FIRST_in_the_transaction_that_upserts():
    j = _Journal()
    asyncio.run(record_claim(j, LANE, SYM, 24, 33.14))
    _one_transaction_lock_then_dml(j, Insert)


def test_drop_claim_takes_the_key_lock_FIRST_in_the_transaction_that_deletes():
    j = _Journal()
    asyncio.run(drop_claim(j, LANE, SYM))
    _one_transaction_lock_then_dml(j, Delete)


def test_the_lock_key_is_case_normalised_like_the_row_key_on_EVERY_writer():
    """`claim_upsert`/`drop_claim_stmt` upper-case the symbol so `gmab` and `GMAB` are ONE row; the
    lock must agree on every path, or two casings would serialise as two keys while writing one row."""
    from kumo_strategies.strategies.momentum_rotation.exits import TrailState

    for call in (lambda j: record_claim(j, LANE, " gmab ", 1, 1.0),
                 lambda j: drop_claim(j, LANE, "gmab"),
                 lambda j: _runner(j)._save_state("gmab", TrailState(entry_px=1.0, peak_px=1.0), qty=1),
                 lambda j: _runner(j)._drop_state(" gmab")):
        j = _Journal()
        asyncio.run(call(j))
        assert _is_key_lock(j.transactions[0][1], LANE, "GMAB"), j.transactions[0][1]


def test_pgrunner_save_state_with_NO_quantity_still_locks_and_moves_the_trail_alone():
    """`qty=None` is the peak-only session update; it must be serialised like a quantity write, and
    its conflict branch must not touch `qty`."""
    from kumo_strategies.strategies.momentum_rotation.exits import TrailState

    j = _Journal()
    asyncio.run(_runner(j)._save_state(SYM, TrailState(entry_px=33.37, peak_px=35.0), qty=None))
    _one_transaction_lock_then_dml(j, Insert)
    ins = next(s for s in j.transactions[0] if isinstance(s, Insert))
    moved = {(c if isinstance(c, str) else c.name) for c, _ in ins._post_values_clause.update_values_to_set}
    assert "qty" not in moved and "peak" in moved, moved


# -- pgrunner's two writers, the ones a cockpit-side lock could not reach --------------------------------
def _runner(journal):
    from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.strategies.momentum_rotation import MomentumRotationConfig

    return PgSessionRunner(pool=None, journal=journal, lifecycle=Lifecycle(State.TRADING, "armed"),
                           cfg=MomentumRotationConfig(), broker=None, strategy_id=LANE)


def test_pgrunner_save_state_takes_the_key_lock_FIRST():
    from kumo_strategies.strategies.momentum_rotation.exits import TrailState

    j = _Journal()
    asyncio.run(_runner(j)._save_state(SYM, TrailState(entry_px=33.37, peak_px=34.32), qty=59))
    _one_transaction_lock_then_dml(j, Insert)


def test_pgrunner_drop_state_takes_the_key_lock_FIRST():
    j = _Journal()
    asyncio.run(_runner(j)._drop_state(SYM))
    _one_transaction_lock_then_dml(j, Delete)


# -- backfill's read-then-write gap: adoption must not overwrite a claim written in between -----------------
def test_claim_upsert_can_insert_ONLY_IF_ABSENT_for_adoption():
    """`claims_backfill` reads a lane's claims, then adopts what is unclaimed. A terminal sync landing
    between the read and the write must win: the adoption inserts only if the row is still absent."""
    from sqlalchemy.dialects.postgresql.dml import OnConflictDoNothing

    stmt = claim_upsert(LANE, SYM, qty=7, entry_px=120.5, only_if_absent=True)
    assert isinstance(stmt._post_values_clause, OnConflictDoNothing)
    default = claim_upsert(LANE, SYM, qty=7, entry_px=120.5)
    assert not isinstance(default._post_values_clause, OnConflictDoNothing), "the default must still move qty"


# -- the enumeration, enforced at the statement -----------------------------------------------------------
def test_PositionState_DML_is_built_ONLY_in_store_py():
    """Every `insert/pg_insert/delete(PositionState)` in the package must live in `store.py`, where the
    lock is taken beside it. A writer built anywhere else is a writer the behavioural tests above
    cannot see — which is how `pgrunner` became the writer a cockpit-side lock would have missed."""
    root = pathlib.Path(kumo_strategies.__file__).parent
    offenders = []
    for f in _src_only(sorted(root.rglob("*.py"))):
        if f.name == "store.py":
            continue
        src = f.read_text()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Call):
                fn = n.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                first = n.args[0] if n.args else None
                # insert(PositionState) / pg_insert(...) / delete(...) / sa.insert(...) / update(...)
                if name in ("insert", "pg_insert", "delete", "update") and isinstance(first, ast.Name) \
                        and first.id == "PositionState":
                    offenders.append(f"{f.relative_to(root)}:{n.lineno} {name}(PositionState)")
                # PositionState.__table__.insert() and friends
                if isinstance(fn, ast.Attribute) and "PositionState" in ast.dump(fn) and name in (
                        "insert", "delete", "update"):
                    offenders.append(f"{f.relative_to(root)}:{n.lineno} PositionState.__table__.{name}()")
                # session.add(PositionState(...))
                if name == "add" and isinstance(first, ast.Call) and getattr(first.func, "id", None) == "PositionState":
                    offenders.append(f"{f.relative_to(root)}:{n.lineno} session.add(PositionState(...))")
            # raw SQL naming the table in a writing verb
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "exec_position_state" in n.value \
                    and any(v in n.value.upper() for v in ("INSERT", "UPDATE", "DELETE")):
                offenders.append(f"{f.relative_to(root)}:{n.lineno} raw SQL")
    assert offenders == [], offenders


# -- ORDER, not only atomicity (codex, platform issue 845 scope v2): applied order == call order per key ----------
class _Ledger:
    """A Postgres that commits in whatever order the test releases, and models the advisory lock as a
    per-key wait until the holder's transaction ends. The in-process lock is production's own code
    and is NOT simulated here — that is the whole point of the test."""

    def __init__(self):
        self.applied: list[tuple[str, float | None]] = []
        self._pending: list[asyncio.Event] = []
        self._advisory: dict[str, asyncio.Lock] = {}
        self._arrivals: list[asyncio.Event] = []
        self.strategy_id = LANE
        self.fail_on = None
        self.fail_with = None           # the exception `fail_on` raises; default ConnectionError
        self.passthrough = False        # no arrival/commit gates — for tests that are not about ordering

    def sessionmaker(self):
        ledger = self

        class _S:
            def __init__(self):
                self.stmts, self.held = [], []

            @asynccontextmanager
            async def begin(self):
                try:
                    yield
                    if not ledger.passthrough:
                        gate = asyncio.Event()
                        ledger._pending.append(gate)
                        await gate.wait()
                    for st in self.stmts:
                        ledger._apply(st)
                finally:
                    for lock in self.held:
                        lock.release()

            async def execute(self, stmt, params=None):
                if ledger.fail_on is not None and isinstance(stmt, ledger.fail_on):
                    raise ledger.fail_with or ConnectionError("postgres went away mid-transaction")
                if isinstance(stmt, TextClause) and "pg_advisory_xact_lock" in str(stmt):
                    # ARRIVAL IS NOT CALL ORDER. With one connection per transaction (NullPool, what
                    # cockpit runs) the lock request reaches Postgres after a connect whose latency
                    # varies per transaction; measured 13% inverted on a real server. The double
                    # holds every in-flight arrival and lets the LAST one in first — so the advisory
                    # lock alone orders nothing here, exactly as it orders nothing live, and only the
                    # in-process lock (production's own code) can make this test pass.
                    if not ledger.passthrough:
                        arrive = asyncio.Event()
                        ledger._arrivals.append(arrive)
                        await arrive.wait()
                    key = " ".join(str(v) for v in stmt.compile().params.values())
                    lock = ledger._advisory.setdefault(key, asyncio.Lock())
                    await lock.acquire()
                    self.held.append(lock)
                    return None
                self.stmts.append(stmt)
                return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

        @asynccontextmanager
        async def _cm():
            yield _S()

        return _cm()

    async def release_adversarially(self, n: int) -> None:
        """Until `n` commits have applied: let the LAST-arrived lock request in, then the LAST-begun
        commit out. Inverts whatever is concurrently in flight at BOTH seams; a serialised writer
        only ever has one thing in flight and passes straight through."""
        released = 0
        while released < n:
            while not self._pending and not self._arrivals:
                await asyncio.sleep(0)
            if self._arrivals:
                self._arrivals.pop().set()
                await asyncio.sleep(0)
                continue
            self._pending.pop().set()
            released += 1
            await asyncio.sleep(0)

    def _apply(self, stmt) -> None:
        if isinstance(stmt, Delete):
            self.applied.append(("delete", None))
        elif isinstance(stmt, Insert):
            self.applied.append(("upsert", float(stmt.compile().params["qty"])))


def test_the_ledger_double_CAN_invert_unserialised_writes():
    """Fixture property: three bare transactions released last-begun-first apply in reverse."""
    async def run():
        ledger = _Ledger()

        async def bare(kind, qty=None):
            async with ledger.sessionmaker() as s, s.begin():
                await s.execute(claim_upsert(LANE, SYM, qty, 1.0) if kind == "upsert"
                                else __import__("kumo_strategies.runtime.executor.store", fromlist=["x"]).drop_claim_stmt(LANE, SYM))

        rel = asyncio.create_task(ledger.release_adversarially(3))
        await asyncio.wait_for(asyncio.gather(bare("upsert", 24), bare("upsert", 20), bare("delete")), 5)
        await asyncio.wait_for(rel, 5)
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [("delete", None), ("upsert", 20.0), ("upsert", 24.0)], ledger.applied


def test_three_concurrent_writes_for_one_key_apply_in_CALL_order_whatever_order_postgres_commits_in():
    """(24, 20, 0) — the three GMAB fills. Whatever the double does with commit order, the writer must
    have made the DELETE the last thing applied. This is the property the advisory lock alone cannot
    give (it serialises by arrival), and the property a cockpit-side lock could not give pgrunner."""
    async def run():
        ledger = _Ledger()
        rel = asyncio.create_task(ledger.release_adversarially(3))
        await asyncio.wait_for(asyncio.gather(
            record_claim(ledger, LANE, SYM, 24, 33.14),
            record_claim(ledger, LANE, SYM, 20, 33.14),
            drop_claim(ledger, LANE, SYM),
        ), 5)
        await asyncio.wait_for(rel, 5)
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [("upsert", 24.0), ("upsert", 20.0), ("delete", None)], ledger.applied


def test_pgrunner_and_the_store_helpers_are_ordered_TOGETHER_for_one_key():
    """The writer a cockpit-side lock would have missed: a decision-time `_save_state(59)` scheduled
    before a fill's `drop_claim` must land before it, not after it."""
    from kumo_strategies.strategies.momentum_rotation.exits import TrailState

    async def run():
        ledger = _Ledger()
        r = _runner(ledger)
        rel = asyncio.create_task(ledger.release_adversarially(2))
        await asyncio.wait_for(asyncio.gather(
            r._save_state(SYM, TrailState(entry_px=33.37, peak_px=34.32), qty=59),
            drop_claim(ledger, LANE, SYM),
        ), 5)
        await asyncio.wait_for(rel, 5)
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [("upsert", 59.0), ("delete", None)], ledger.applied


def test_a_transaction_that_RAISES_does_not_wedge_a_writer_ALREADY_WAITING_on_the_key():
    """The failure happens INSIDE the transaction, with both locks held, while a second writer is
    already queued behind it (codex, step 3: a failure before any lock proves nothing; and a leaked
    lock with NO waiter is collected by the weak registry, so a sequential test proves nothing
    either). The waiter must still land, within the bound."""
    async def run():
        ledger = _Ledger()
        ledger.fail_on = Insert                        # the first write's DML raises mid-transaction
        first = asyncio.create_task(record_claim(ledger, LANE, SYM, 24, 33.14))
        while not ledger._arrivals:                    # it holds the in-process lock, awaiting arrival
            await asyncio.sleep(0)
        second = asyncio.create_task(drop_claim(ledger, LANE, SYM))
        await asyncio.sleep(0)                         # the second is now queued on the in-process lock
        ledger._arrivals.pop().set()                   # the first proceeds to its DML and raises
        try:
            await asyncio.wait_for(first, 5)
        except ConnectionError:
            pass
        ledger.fail_on = None
        rel = asyncio.create_task(ledger.release_adversarially(1))
        await asyncio.wait_for(second, 5)
        await asyncio.wait_for(rel, 5)
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [("delete", None)]


# -- step-9 review of ebca3d3: three ways the writer could wedge or lie ------------------------------------
class _HangingRollback(_Ledger):
    """A Postgres that is slow rather than dead: the transaction's unwind (rollback) never returns.
    `asyncio.timeout` spends its one cancel on the body; SQLAlchemy's `__aexit__` then awaits
    rollback on the same stuck connection, and nothing bounded that."""

    def sessionmaker(self):
        ledger = self

        class _S:
            def __init__(self):
                self.stmts = []

            @asynccontextmanager
            async def begin(self):
                try:
                    yield
                    await asyncio.Event().wait()                 # the body never completes
                finally:
                    ledger.rollbacks_started += 1
                    await asyncio.Event().wait()                 # ...and neither does the rollback

            async def execute(self, stmt, params=None):
                self.stmts.append(stmt)

        @asynccontextmanager
        async def _cm():
            yield _S()

        return _cm()


def test_a_rollback_that_HANGS_does_not_wedge_the_key_forever(monkeypatch):
    """Review: `store.py` said "rolled back, later writers for the key proceed" about a path nothing
    bounded. With the unwind hung, the in-process lock must still be released after a bounded grace,
    the stall must say the rollback was ABANDONED, and the next writer of the key must proceed."""
    import kumo_strategies.runtime.executor.store as store

    monkeypatch.setattr(store, "CLAIM_WRITE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(store, "UNWIND_GRACE_S", 0.2)

    async def run():
        ledger = _HangingRollback()
        ledger.rollbacks_started = 0
        with __import__("pytest").raises(store.ClaimWriteStalled) as e:
            await asyncio.wait_for(record_claim(ledger, LANE, SYM, 24, 33.14), 3)
        assert "ABANDONED" in str(e.value), str(e.value)
        assert ledger.rollbacks_started == 1
        assert not store._key_lock(LANE, SYM).locked(), "the key is wedged: the in-process lock was never released"
        good = _Ledger()
        rel = asyncio.create_task(good.release_adversarially(1))
        await asyncio.wait_for(drop_claim(good, LANE, SYM), 3)      # a later writer proceeds
        await asyncio.wait_for(rel, 3)
        return good

    assert asyncio.run(run()).applied == [("delete", None)]


def test_a_driver_TimeoutError_inside_the_transaction_is_NOT_relabelled_as_a_stall():
    """On 3.13 `TimeoutError is asyncio.TimeoutError` and it is OSError-family: asyncpg's ETIMEDOUT
    arrives as the same class the timeout context raises. It must propagate as itself — a message
    saying "waited 30 s" about 30 s that did not elapse is a lie in the log."""
    import kumo_strategies.runtime.executor.store as store

    async def run():
        ledger = _Ledger()
        ledger.passthrough = True
        ledger.fail_on = Insert
        ledger.fail_with = TimeoutError("[Errno 60] ETIMEDOUT")
        try:
            await asyncio.wait_for(record_claim(ledger, LANE, SYM, 24, 33.14), 3)
        except store.ClaimWriteStalled as e:
            raise AssertionError(f"a driver timeout was relabelled as a stall: {e}")
        except TimeoutError as e:
            return str(e)
        raise AssertionError("no exception at all")

    assert "ETIMEDOUT" in asyncio.run(run())


def test_re_entering_write_claim_for_the_SAME_key_from_build_raises_at_once(monkeypatch):
    """`build` runs under the key's lock, so a nested `write_claim` for the same key can only wait
    for itself. It must be refused immediately and by name — not discovered as a 30 s stall."""
    import kumo_strategies.runtime.executor.store as store

    monkeypatch.setattr(store, "CLAIM_WRITE_TIMEOUT_S", 2.0)

    async def run():
        ledger = _Ledger()
        ledger.passthrough = True
        ledger.hold_commits = False

        async def nested(_s):
            await store.write_claim(ledger, LANE, SYM, drop_claim_stmt_for(LANE, SYM))
            return []

        t0 = asyncio.get_running_loop().time()
        with __import__("pytest").raises(store.ClaimWriteReentered):
            await asyncio.wait_for(store.write_claim(ledger, LANE, SYM, build=nested), 5)
        return asyncio.get_running_loop().time() - t0

    assert asyncio.run(run()) < 1.0, "the re-entry was discovered as a stall, not refused"


def test_a_nested_write_for_a_DIFFERENT_key_from_build_is_allowed():
    import kumo_strategies.runtime.executor.store as store

    async def run():
        ledger = _Ledger()
        ledger.passthrough = True

        async def nested(_s):
            await store.write_claim(ledger, LANE, "AEM", drop_claim_stmt_for(LANE, "AEM"))
            return []

        await asyncio.wait_for(store.write_claim(ledger, LANE, SYM, build=nested), 3)
        return ledger

    assert asyncio.run(run()).applied == [("delete", None)]


def drop_claim_stmt_for(sid, sym):
    from kumo_strategies.runtime.executor.store import drop_claim_stmt
    return drop_claim_stmt(sid, sym)
