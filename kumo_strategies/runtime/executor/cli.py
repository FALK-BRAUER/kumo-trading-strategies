"""CLI for the runtime. Postgres-backed, same store and same code path as the API.

There is no `--live` broker flag here. Order submission happens through the API, behind the
lifecycle and the confirm gate; a CLI that could reach a real broker while bypassing those was the
single most dangerous thing in this package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date

from kumo_strategies.runtime.executor.pgjobs import PgJobRunner
from kumo_strategies.runtime.executor.pgjournal import PgJournal
from kumo_strategies.runtime.executor.pgpool import EXCLUDE, PIN, PgSymbolPool
from kumo_strategies.runtime.executor.sources import SourceSpec, kinds
from kumo_strategies.runtime.executor.store import (
    StrategyState, create_all, make_engine, make_sessionmaker, select)


async def _run(a: argparse.Namespace) -> int:
    eng = make_engine(a.db)
    await create_all(eng)
    sm = make_sessionmaker(eng)
    pool, jrn = PgSymbolPool(sm), PgJournal(sm)
    jobs = PgJobRunner(sm, pool, jrn)
    try:
        if a.cmd == "status":
            async with sm() as s:
                st = (await s.execute(select(StrategyState))).scalars().first()
            print(f"state    {st.state if st else 'DISABLED'}   ({st.reason if st else 'initial'})")
            print(f"pool     {len(await pool.symbols())} symbols")
            for h in await jobs.status():
                flag = "STALE" if h["stale"] else "ok"
                print(f"  {h['name']:16} {h['symbols_in_pool']:>4}  {h['last_status']:8} {flag}")
            for o in await pool.overrides():
                print(f"  {o['kind']:8} {o['symbol']:6} {o['reason'] or ''}")

        elif a.cmd == "pool":
            if a.action == "show":
                eff = await pool.effective()
                print(f"{len(eff)} symbols")
                for s2, e in eff.items():
                    print(f"  {s2:8} {e.provenance}")
            elif a.action in (PIN, EXCLUDE):
                await pool.set_override(a.value.upper(), a.action, reason=a.reason)
                await jrn.write("pool", f"{a.action} {a.value.upper()}: {a.reason}",
                                session=str(date.today()), symbol=a.value.upper())
                print(f"{a.action} {a.value.upper()}")
            elif a.action == "clear":
                await pool.clear_override(a.value.upper(), a.kind)
                print(f"cleared {a.kind} on {a.value.upper()}")

        elif a.cmd == "sources":
            if a.action == "list":
                print("kinds:", ", ".join(kinds()))
                for h in await jobs.status():
                    print(f"  {h['name']:16} {h['kind']:14} every {h['every_minutes']}m  "
                          f"{h['last_status']}  {h['symbols_in_pool']} names")
            elif a.action == "add":
                await jobs.add(SourceSpec(name=a.value, kind=a.kind_,
                                          every_minutes=a.every,
                                          params=json.loads(a.params or "{}")))
                print(f"added {a.value}")
            elif a.action == "refresh":
                print(await jobs.refresh(a.value, session=str(date.today()), force=True))
            elif a.action == "remove":
                await jobs.remove(a.value)
                print(f"removed {a.value}")

        elif a.cmd == "log":
            if a.explain:
                # `explain` filters by slot now (#54) -- a multi-slot strategy (BCTROT-004:
                # open+150m/close-20m) has no decision under DEFAULT_SLOT at all, so an operator who
                # does not pass --slot gets "no decision" even though one exists. --slot defaults to
                # None, which explain() itself resolves to DEFAULT_SLOT, so a single-slot strategy
                # (MOMENTUM-002) is unaffected without it.
                d = await jrn.explain(a.explain, slot=a.slot)
                print(json.dumps(d, indent=1, default=str) if d else "no decision for that session/slot")
            else:
                for e in reversed(await jrn.tail(a.n, kind=a.kind)):
                    print(f"{e['ts'][:19]}  {e['kind']:9} {(e['symbol'] or ''):6} {e['summary']}")
        return 0
    finally:
        await eng.dispose()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kumo-executor")
    ap.add_argument("--db", default=None, help="override KUMO_DATABASE_URL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pp = sub.add_parser("pool")
    pp.add_argument("action", choices=["show", PIN, EXCLUDE, "clear"])
    pp.add_argument("value", nargs="?")
    pp.add_argument("--reason", default="")
    pp.add_argument("--kind", default=PIN, choices=[PIN, EXCLUDE])
    ps = sub.add_parser("sources")
    ps.add_argument("action", choices=["list", "add", "refresh", "remove"])
    ps.add_argument("value", nargs="?")
    ps.add_argument("--kind_", default="static_list", help="source kind for `add`")
    ps.add_argument("--every", type=int, default=1440)
    ps.add_argument("--params", default=None, help="JSON params for `add`")
    pl = sub.add_parser("log")
    pl.add_argument("-n", type=int, default=25)
    pl.add_argument("--kind")
    pl.add_argument("--explain")
    pl.add_argument("--slot", default=None, help="slot to explain, for a multi-slot strategy (#54)")
    return asyncio.run(_run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
