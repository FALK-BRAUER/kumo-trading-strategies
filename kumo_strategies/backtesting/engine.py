"""THE ONE RUNNER — a session engine that executes every strategy family through the same loop.

    from kumo_strategies.backtesting.engine import SessionEngine, Order

`docs/runner-architecture.md` §3: the engine owns the clock (sessions and the slots inside them),
the Book and Venue ports, order execution, the mark-to-market curve and the Report. A strategy
FAMILY (`families/`) owns only policy — how it scores, when it decides, what it exits, how it sizes
— and expresses every trade as an `Order` the engine executes. The venue never decides and the
family never touches cash: the same split the live executor has between `PgSessionRunner` and
`NautilusBroker`.

WHY ORDERS, NOT FILLS. A family that called `venue.buy` itself would own the arithmetic again — the
shape that produced nine runners. Handing the engine a priced `Order` keeps one execution path for
an entry, an exit, a rebalance slice, a stop-out and a write-off, and makes the sequence a family
asks for (sells before buys, exits before entries) visible in one list.

EVERY NUMBER THIS ENGINE PRODUCES REPRODUCES THE RUNNER IT REPLACED TO THE LAST DIGIT. That is the
gate for each family folded in (ks#270), and the reason the loop below is shaped the way it is: an
adapter hook per point where the old loops differed, in the order they ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from kumo_strategies.backtesting.report import Report, round_trips
from kumo_strategies.backtesting.result import BacktestResult
from kumo_strategies.backtesting.sim_venue import Book, SimVenue


@dataclass
class Order:
    """One priced instruction for the venue. `kind` says which book operation goes with it:

        exit        the whole position leaves (Book.close) and is sold
        trim        a slice leaves (Book.trim) and is sold — never to zero
        entry       bought and opened (Book.open, trail armed at the fill)
        add         bought onto an open position (Book.add, trail untouched)
        set         bought/sold TO a quantity with the family's own basis bookkeeping
                    (the monthly family: `held[sym] = want`)
        liquidate   written off at the last mark, no cost (a delisting)
        short       a short opened: signed shares in `tags["shares"]` (the short family)
        cover       a short covered at `px` from `tags["mark"]`, P&L to the marked equity
    """

    sym: str
    side: str                      # "BUY" | "SELL"
    qty: int
    px: float
    ts: pd.Timestamp
    kind: str
    at: tuple[int, int] | None = None
    tags: dict = field(default_factory=dict)
    label: str = ""                # liquidate: what kind of write-off


class Family(Protocol):
    """What a strategy family gives the engine. Every hook may return `()`; the engine executes
    whatever it gets in the order given, so the family states its own sell-before-buy sequence."""

    sessions: list          # the sessions the engine walks, in order
    starting_cash: float
    n_trials: int
    benchmarks: dict

    def skip(self, d) -> bool: ...
    def slots(self, d) -> list:                              # [(slot_name, when)] — daily: one
        ...
    def session_open(self, d) -> list[Order]: ...            # before any slot (write-offs, pending stops)
    def before_decision(self, d, slot) -> None: ...          # exits evaluated, state advanced
    def decide(self, d, slot) -> bool: ...                   # False: nothing to do this slot
    def exit_orders(self, d, slot) -> list[Order]: ...
    def entry_orders(self, d, slot) -> list[Order]: ...      # sized against the book AFTER the exits
    def session_close(self, d) -> list[Order]: ...           # after the slots (sweeps)
    def after_close(self, d) -> None: ...                    # stop triggers on the close
    def marks(self, d) -> dict[str, float]: ...              # close marks for the curve
    def curve_row(self, d, book_value: float) -> bool: ...   # False: the family drops the row
    def decisions(self) -> list[dict]: ...
    def diagnostics(self) -> dict: ...


class SessionEngine:
    def __init__(self, book: Book, venue: SimVenue) -> None:
        self.book, self.venue = book, venue

    # -- execution: one path for every kind of order --------------------------------------------

    def execute(self, orders) -> None:
        book, venue = self.book, self.venue
        for o in orders:
            if o.kind == "exit":
                qty = book.close(o.sym)
                venue.sell(book, o.sym, qty, o.px, o.ts, at=o.at, **o.tags)
            elif o.kind == "trim":
                book.trim(o.sym, o.qty)
                venue.sell(book, o.sym, o.qty, o.px, o.ts, at=o.at, **o.tags)
            elif o.kind == "entry":
                venue.buy(book, o.sym, o.qty, o.px, o.ts, at=o.at, **o.tags)
                book.open(o.sym, o.qty, o.px)
            elif o.kind == "add":
                venue.buy(book, o.sym, o.qty, o.px, o.ts, at=o.at, **o.tags)
                book.add(o.sym, o.qty)
            elif o.kind == "set":
                # The family has already moved `held` to the target; the venue only transacts.
                if o.side == "SELL":
                    venue.sell(book, o.sym, o.qty, o.px, o.ts, at=o.at, **o.tags)
                else:
                    venue.buy(book, o.sym, o.qty, o.px, o.ts, at=o.at, **o.tags)
            elif o.kind == "liquidate":
                venue.liquidate(book, o.sym, o.qty, o.px, o.ts, o.label)
            elif o.kind == "short":
                # the short side: `tags["shares"]` is the signed (negative) quantity
                venue.short(book, o.sym, o.tags["shares"], o.px, o.ts)
            elif o.kind == "cover":
                venue.cover(book, o.sym, o.tags["shares"], o.px, o.tags["mark"], o.ts)
            else:
                raise ValueError(f"unknown order kind {o.kind!r}")

    # -- the loop ---------------------------------------------------------------------------------

    def run(self, fam: Family) -> BacktestResult:
        book = self.book
        curve: list[dict] = []
        for d in fam.sessions:
            if fam.skip(d):
                continue
            self.execute(fam.session_open(d))
            for slot in fam.slots(d):
                fam.before_decision(d, slot)
                if not fam.decide(d, slot):
                    continue
                self.execute(fam.exit_orders(d, slot))
                self.execute(fam.entry_orders(d, slot))
            self.execute(fam.session_close(d))
            marks = fam.marks(d)
            mtm = sum(q * marks.get(s, np.nan) for s, q in book.held.items())
            value = book.cash + (mtm if book.held else 0.0)
            if fam.curve_row(d, value):
                curve.append({"date": pd.Timestamp(d), "equity": value})
            fam.after_close(d)

        f = pd.DataFrame(book.fills)
        if not f.empty and "ts" in f.columns:
            f["ts"] = pd.to_datetime(f["ts"])
        eq = pd.DataFrame(curve).set_index("date")["equity"] if curve else pd.Series(dtype=float)
        rep = Report(equity=eq, trades=round_trips(f) if len(f) else pd.DataFrame(),
                     fills=f if len(f) else pd.DataFrame(
                         columns=["ts", "symbol", "side", "qty", "price", "cost", "venue"]),
                     starting_cash=fam.starting_cash, n_trials=fam.n_trials,
                     benchmarks=fam.benchmarks or {})
        return BacktestResult(report=rep, decisions=pd.DataFrame(fam.decisions()),
                              diagnostics=fam.diagnostics())
