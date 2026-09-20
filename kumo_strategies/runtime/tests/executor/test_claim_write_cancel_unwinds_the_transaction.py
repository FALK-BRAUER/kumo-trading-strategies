"""issue 128 — an EXTERNAL cancel of a `write_claim` caller must not leave the transaction
running behind it.

#126 moved the transaction into its own task and awaited it with `asyncio.wait({task}, timeout)`.
`asyncio.wait` does not cancel what it waits on: a `CancelledError` delivered at that await unwinds
the caller, runs both `finally`s, RELEASES the in-process key lock — and leaves the transaction task
running detached. Before #126 the body ran inline under `asyncio.timeout`, so an external cancel
propagated into the transaction and rolled it back. That protection was lost.

WHY IT MATTERS, AND WHERE. `on_stop()` cancels the in-flight session task by design — an operator
pausing mid-session — at `momentum_rotation.py:334`, `qc27_rotation.py:240`, `qc345_rotation.py:278`.
On process shutdown the leaked task dies with the loop and no later writer runs; a PAUSE followed by
continued activity is the exposure.

THE WINDOW IS THE CONNECT, NOT THE LOCK. Once the task holds `pg_advisory_xact_lock` a later writer
of the key queues behind it inside Postgres and ordering holds. Cancelled BEFORE that first
statement — inside `sessionmaker()`, the connect-latency window `store.py`'s own module comment
describes — the leaked task takes the lock AFTER the later writer released it, and its INSERT lands
on top of the later DELETE. That is #124's resurrected row, reproduced through the new code path:
ledger order `['W2-DELETE', 'W1-qty24']`.

The double parks in that window on purpose. Its fixture property is asserted FIRST in every test
here: a harness that could not apply a parked INSERT after a completed DELETE could not express the
bug, and its silence would prove nothing.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from kumo_strategies.runtime.executor.store import (claim_upsert, drop_claim, drop_claim_stmt,
                                                    record_claim)
from kumo_strategies.runtime.tests.executor.test_claim_writes_are_serialised import LANE, SYM, _Ledger

#: How long the leaked task is given to do its damage before the ledger is read. A fix that unwinds
#: the transaction has nothing to show after it; the unfixed code commits well inside this.
GRACE_TICKS = 200


class _ConnectGatedLedger(_Ledger):
    """`_Ledger`, plus a gate in the CONNECT window — inside `sessionmaker()`, before `begin()`, before
    the advisory lock, before any statement. `hold_connects` parks whoever connects next; releasing
    the gate lets that transaction run to completion at a moment the test chooses.

    Commit/arrival gating from `_Ledger` is off (`passthrough`): the ordering under test here comes
    from the connect window alone, not from an adversarial commit order.
    """

    def __init__(self):
        super().__init__()
        self.passthrough = True
        self.hold_connects = True
        self.parked: list[asyncio.Event] = []
        self.txn_tasks: list[asyncio.Task] = []

    def sessionmaker(self):
        inner = super().sessionmaker()
        ledger = self

        @asynccontextmanager
        async def _cm():
            if ledger.hold_connects:
                gate = asyncio.Event()
                ledger.parked.append(gate)
                # The task that reaches `sessionmaker()` IS the transaction task production created.
                # Recording it here needs no monkeypatching of `create_task` and cannot drift.
                ledger.txn_tasks.append(asyncio.current_task())
                await gate.wait()
            async with inner as s:
                yield s

        return _cm()

    async def until_parked(self, n: int = 1) -> None:
        for _ in range(1000):
            if len(self.parked) >= n:
                return
            await asyncio.sleep(0)
        raise AssertionError(f"nothing reached the connect window (parked={len(self.parked)})")

    def release_connects(self) -> None:
        for gate in self.parked:
            gate.set()

    @staticmethod
    async def grace(ticks: int = GRACE_TICKS) -> None:
        """Let every runnable task drain. A leaked transaction commits inside this."""
        for _ in range(ticks):
            await asyncio.sleep(0)


async def _bare(ledger, kind: str, qty=None) -> None:
    """One transaction with no `write_claim` around it — the harness's own control."""
    async with ledger.sessionmaker() as s, s.begin():
        await s.execute(claim_upsert(LANE, SYM, qty, 33.14) if kind == "upsert"
                        else drop_claim_stmt(LANE, SYM))


# -- fixture property: the harness CAN express #124's resurrection through the connect window ----------
def test_the_connect_gated_double_CAN_apply_a_parked_INSERT_after_a_completed_DELETE():
    """Without this, an empty or ordered `applied` in the tests below would prove nothing — it could
    mean the double never applies a parked write at all. Two bare transactions, no `write_claim`:
    W1 parks in the connect window, W2 runs to completion, W1 is released. The INSERT lands LAST."""
    async def run():
        ledger = _ConnectGatedLedger()
        w1 = asyncio.create_task(_bare(ledger, "upsert", 24))
        await ledger.until_parked()
        ledger.hold_connects = False
        await asyncio.wait_for(_bare(ledger, "delete"), 5)
        ledger.release_connects()
        await asyncio.wait_for(w1, 5)
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [("delete", None), ("upsert", 24.0)], ledger.applied


# -- (a) the cancel must unwind the transaction, not detach it -----------------------------------------
def test_the_double_commits_W1_when_NOBODY_cancels_it():
    """The other half of the fixture property, on the real `record_claim` path: with no cancel the
    parked transaction commits its INSERT. So `applied == []` in the next test is the CANCEL."""
    async def run():
        ledger = _ConnectGatedLedger()
        caller = asyncio.create_task(record_claim(ledger, LANE, SYM, 24, 33.14))
        await ledger.until_parked()
        ledger.release_connects()
        await asyncio.wait_for(caller, 5)
        return ledger

    assert asyncio.run(run()).applied == [("upsert", 24.0)]


def test_an_external_cancel_in_the_connect_window_does_NOT_let_the_transaction_commit_later():
    """`on_stop()` cancelling the session task, with `write_claim` awaiting its transaction task. The
    caller must raise `CancelledError` — and the transaction must be CANCELLED with it, so that
    opening the connect window afterwards produces nothing at all."""
    async def run():
        ledger = _ConnectGatedLedger()
        caller = asyncio.create_task(record_claim(ledger, LANE, SYM, 24, 33.14))
        await ledger.until_parked()

        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        assert ledger.txn_tasks, "the transaction task was never observed — vacuous"
        txn = ledger.txn_tasks[0]
        await ledger.grace()
        assert txn.done(), "the transaction task outlived the cancelled caller — it is DETACHED"
        assert txn.cancelled(), f"the transaction ended {txn!r}, not cancelled"

        ledger.release_connects()       # a detached task would connect, lock, insert and commit here
        await ledger.grace()
        return ledger

    ledger = asyncio.run(run())
    assert ledger.applied == [], f"the cancelled write committed anyway: {ledger.applied}"


# -- (b) ORDER: the cancelled W1 must not land on top of a completed W2 --------------------------------
def test_a_cancelled_W1_cannot_apply_AFTER_a_later_W2_completed():
    """#124's resurrection, through #126's path. W1 (`record_claim` qty 24) is cancelled while its
    transaction is still in the connect window; W2 (`drop_claim`) then runs to completion; then the
    loop is allowed to drain. The measured failure is `['W2-DELETE', 'W1-qty24']` — a qty-24 row for
    a position the broker had already closed."""
    async def run():
        ledger = _ConnectGatedLedger()
        w1 = asyncio.create_task(record_claim(ledger, LANE, SYM, 24, 33.14))
        await ledger.until_parked()

        w1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w1

        ledger.hold_connects = False                            # W2 connects freely
        await asyncio.wait_for(drop_claim(ledger, LANE, SYM), 5)
        assert ("delete", None) in ledger.applied, "W2 never applied — the fixture cannot show the bug"

        ledger.release_connects()                               # now let W1's leak, if any, proceed
        await ledger.grace()
        return ledger

    ledger = asyncio.run(run())
    ops = ledger.applied
    assert ops.index(("delete", None)) == len(ops) - 1, (
        f"a cancelled writer's INSERT landed after a later writer's DELETE — #124 resurrected: {ops}")
    assert ops == [("delete", None)], ops
