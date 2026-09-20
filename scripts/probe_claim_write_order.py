"""Measure the claim-write race against a REAL Postgres (issue 124 / kumo-trading-platform issue 845).

For each round: seed one row (59, live), then schedule N concurrent writers for the same key in event
order — record_claim(q1), record_claim(q2), ..., drop_claim() — exactly as N terminal events would, and
read the row back. A round is WRONG when the row survives (the DELETE was not the last write applied)
or, for a partial-exit sequence, when qty != the last event's qty.

Run with the engine pool the cockpit uses (NullPool, `cross_loop=True`) and with the default 5+5 pool.
The number that goes in the PR is the wrong-round fraction before and after the fix.

    docker run --rm -d --name kumo-probe-pg -e POSTGRES_USER=kumo -e POSTGRES_PASSWORD=kumo \
        -e POSTGRES_DB=kumo -p 55432:5432 postgres:16-alpine
    PYTHONPATH=src python scripts/probe_claim_write_order.py --rounds 200 --writers 3
"""
from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import delete, select, text

from kumo_strategies.runtime.executor.store import (PositionState, create_all, drop_claim, make_engine,
                                                    make_sessionmaker, record_claim)

URL = "postgresql+asyncpg://kumo:kumo@127.0.0.1:55432/kumo"
LANE, SYM = "BCTROT-004", "GMAB"


class _Journal:
    def __init__(self, sm):
        self._sm = sm
        self.strategy_id = LANE

    def sessionmaker(self):
        return self._sm()


async def _seed(sm):
    async with sm() as s, s.begin():
        await s.execute(delete(PositionState))
        await s.execute(text(
            "INSERT INTO exec_position_state (strategy_id, symbol, qty, entry, peak, quality, "
            "sessions_held, sessions_since_high, opened_at, updated_at) "
            "VALUES (:sid, :sym, 59, 33.37, 34.32, 'live', 6, 1, now(), now())"),
            {"sid": LANE, "sym": SYM})


async def _read(sm):
    async with sm() as s:
        return (await s.execute(select(PositionState).where(
            PositionState.strategy_id == LANE, PositionState.symbol == SYM))).scalars().first()


async def _round(journal, sm, writers: int, full_exit: bool) -> tuple[bool, str]:
    await _seed(sm)
    qtys = [59 - (59 * (i + 1)) // writers for i in range(writers)]      # 39, 19, 0 for 3 writers
    if not full_exit:
        qtys = [max(q, 1) for q in qtys]                                # partial: last event holds >0
    loop = asyncio.get_running_loop()
    futs = []
    for q in qtys:                       # scheduled in EVENT order, the way fire_and_report does
        coro = drop_claim(journal, LANE, SYM) if q == 0 else record_claim(journal, LANE, SYM, q, 33.14)
        futs.append(asyncio.run_coroutine_threadsafe(coro, loop))
    await asyncio.gather(*(asyncio.wrap_future(f) for f in futs))
    row = await _read(sm)
    if full_exit:
        return (row is None), ("flat" if row is None else f"row survived qty={row.qty} {row.quality}")
    ok = row is not None and float(row.qty) == qtys[-1] and row.quality == "live"
    return ok, ("ok" if ok else f"qty={getattr(row, 'qty', None)} quality={getattr(row, 'quality', None)}")


async def main(rounds: int, writers: int, cross_loop: bool) -> None:
    engine = make_engine(URL, cross_loop=cross_loop)
    await create_all(engine)
    sm = make_sessionmaker(engine)
    journal = _Journal(sm)
    for label, full in (("FULL EXIT  (last event = drop)", True), ("PARTIAL    (last event = qty>0)", False)):
        wrong, reasons, t0 = 0, {}, time.perf_counter()
        for _ in range(rounds):
            ok, why = await _round(journal, sm, writers, full)
            if not ok:
                wrong += 1
                reasons[why] = reasons.get(why, 0) + 1
        dt = time.perf_counter() - t0
        print(f"{label}  pool={'NullPool' if cross_loop else '5+5'}  writers={writers}  rounds={rounds}  "
              f"WRONG={wrong} ({100 * wrong / rounds:.1f}%)  {dt:.1f}s")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    {n:4d}  {why}")
    await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--writers", type=int, default=3)
    ap.add_argument("--pool", choices=["null", "pooled"], default="null")
    ap.add_argument("--variant", choices=["writer", "advisory-only"], default="writer",
                    help="advisory-only disables the in-process FIFO lock: the number that must NOT be zero")
    a = ap.parse_args()
    if a.variant == "advisory-only":
        import kumo_strategies.runtime.executor.store as _store
        _store._key_lock = lambda sid, sym: asyncio.Lock()      # a fresh lock per call orders nothing
    asyncio.run(main(a.rounds, a.writers, cross_loop=(a.pool == "null")))
