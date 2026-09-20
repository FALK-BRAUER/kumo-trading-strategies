"""The replay side of the Venue and Book ports (`docs/runner-architecture.md`, #270 step 2).

`run_sessions` used to carry the fill arithmetic inline, four times: sell (cost, cash, book, fill
row), buy with the cash rule, rebalance trims and adds — once per clock. Each copy was a place where
the two clocks could drift and a fifth runner would have copied it again. This module is where a
simulated venue and an in-memory book live; the runner keeps the DECISION of what to trade and at
which price (the fill RULE — next bar, opening print, resting limit), and hands the venue the order.

WHAT A PORT METHOD IS FOR. Live, `budget_allows` and `owns` are cockpit's budget gate and
`exit_ownership.classify_exit`, sitting behind `NautilusBroker`. Here they are the same two
questions asked of the replay book, so a budget refusal or an ownership refusal is a measurable
event in history rather than a live-only surprise. They are deliberately the ONLY two gates the
venue applies: sizing, bands and caps stay in the runner, where live's `_submit` has them too.

ARITHMETIC IS BYTE-IDENTICAL to the inline code it replaced; the gate for this extraction was three
recorded numbers unchanged (ledger-book-daily-2024-2026 baseline and bctrot, the cadence reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.strategies.momentum_rotation.exits import TrailState


@dataclass
class Book:
    """The replay Book port: cash, shares held, the trail per open name, and every fill."""

    cash: float
    held: dict[str, int] = field(default_factory=dict)
    state: dict[str, TrailState] = field(default_factory=dict)
    fills: list[dict] = field(default_factory=list)

    def open(self, sym: str, qty: int, px: float) -> None:
        """An ENTRY: shares added, the trail armed at the fill (peak = entry, not in profit)."""
        self.held[sym] = self.held.get(sym, 0) + qty
        self.state[sym] = TrailState(entry_px=px, peak_px=px)

    def close(self, sym: str) -> int:
        """An EXIT: the whole position leaves the book with its trail. Returns the shares sold."""
        qty = self.held.pop(sym)
        self.state.pop(sym, None)
        return qty

    def add(self, sym: str, qty: int) -> None:
        """A rebalance ADD: shares only — the trail is untouched, the position was already open."""
        self.held[sym] += qty

    def trim(self, sym: str, qty: int) -> None:
        """A rebalance TRIM: shares only. Never to zero — that is an exit, and the caller
        guarantees it (`min(-dq, held - 1)`). The trail is untouched: the position stays open and
        its peak must carry, or give-back re-arms from today's price."""
        self.held[sym] -= qty

    def marked(self, marks: dict[str, float]) -> float:
        """Cash plus the book at `marks`; NaN when a held name has no mark."""
        return self.cash + sum(q * marks.get(s, float("nan")) for s, q in self.held.items())


class SimVenue:
    """The replay Venue port: charges the measured cost at the fill instant and moves the book.

    The fill PRICE is the runner's (it is the fill rule); the venue takes an order that is already
    priced and applies the two gates and the arithmetic.
    """

    def __init__(self, cost_model: CostModel, defs: dict) -> None:
        self._cm = cost_model
        self._defs = defs

    # -- port methods (docs/runner-architecture.md §3) -------------------------------------------

    def budget_allows(self, book: Book, sym: str, qty: int, px: float, ts: pd.Timestamp,
                      at: tuple[int, int] | None = None) -> bool:
        """Never spend cash the book does not have — the replay budget gate. Live: cockpit's."""
        return qty * px + self.charge(sym, qty * px, ts, at) <= book.cash

    def owns(self, book: Book, sym: str, qty: int) -> bool:
        """A SELL may only reduce what this book holds — the replay ownership gate. Live:
        `exit_ownership.classify_exit`, side-aware; the replay book is long-only today."""
        return 0 < qty <= book.held.get(sym, 0)

    def affordable(self, book: Book, sym: str, ts: pd.Timestamp) -> float:
        """The notional the book can buy of `sym` at this instant, cost included."""
        return book.cash / (1.0 + self._cm.bps(sym, ts.hour, ts.minute) / 1e4)

    def charge(self, sym: str, notional: float, ts: pd.Timestamp,
               at: tuple[int, int] | None = None) -> float:
        """The measured cost at the fill INSTANT. `at` names the (hour, minute) the price column
        represents when it is not the row's timestamp — the monthly family stamps every fill 09:35
        and prices it at a caller-named instant (the slot-price arm); the two travel together or a
        fill gets priced at a spread it never paid."""
        h, m = at if at is not None else (ts.hour, ts.minute)
        return self._cm.charge(sym, notional, h, m)

    # -- fills -----------------------------------------------------------------------------------

    def sell(self, book: Book, sym: str, qty: int, px: float, ts: pd.Timestamp,
             at: tuple[int, int] | None = None, **tags) -> float:
        """Sell `qty` at `px`: cost charged, cash credited, fill recorded. Returns the cost. The
        caller has already moved the shares off the book (`close` or `trim`), because WHICH shares
        leave — the whole position or a slice — is the runner's decision, not the venue's."""
        cost = self.charge(sym, qty * px, ts, at)
        book.cash += qty * px - cost
        book.fills.append({"ts": ts, "symbol": sym, "side": "SELL", "qty": qty, "price": px,
                           "cost": cost, "venue": self._defs[sym].id.venue.value, **tags})
        return cost

    def buy(self, book: Book, sym: str, qty: int, px: float, ts: pd.Timestamp,
            at: tuple[int, int] | None = None, **tags) -> float:
        """Buy `qty` at `px`: cost charged, cash debited, fill recorded. Returns the cost. The
        caller has applied `budget_allows` (or shrunk the quantity to what is affordable) and
        moves the shares on afterwards (`open` or `add`), for the same reason as `sell`."""
        cost = self.charge(sym, qty * px, ts, at)
        book.cash -= qty * px + cost
        book.fills.append({"ts": ts, "symbol": sym, "side": "BUY", "qty": qty, "price": px,
                           "cost": cost, "venue": self._defs[sym].id.venue.value, **tags})
        return cost

    def liquidate(self, book: Book, sym: str, qty: int, px: float, ts: pd.Timestamp,
                  label: str) -> None:
        """A position the tape no longer prices, closed at the LAST MARK with no cost — a delisting
        or a name that fell out of the panel. Not a venue fill: `label` says which kind, so the
        round-trip table can tell a trade from a write-off. The shares are already off the book."""
        book.cash += qty * px
        book.fills.append({"ts": ts, "symbol": sym, "side": "SELL", "qty": qty, "price": px,
                           "cost": 0.0, "venue": label})


class LabelDefs:
    """`defs` for a family that never built Nautilus instruments: the venue label per symbol
    (`defs[sym].id.venue.value`), constant or from a per-symbol table."""

    class _Id:
        def __init__(self, label: str) -> None:
            self.venue = type("V", (), {"value": label})()

    class _Def:
        def __init__(self, label: str) -> None:
            self.id = LabelDefs._Id(label)

    def __init__(self, default: str, per_symbol: dict[str, str] | None = None) -> None:
        self._default, self._per = default, per_symbol or {}

    def __getitem__(self, sym: str) -> LabelDefs._Def:
        return LabelDefs._Def(self._per.get(sym, self._default))
