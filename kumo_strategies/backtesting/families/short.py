"""The SHORT family for the one runner (`engine.SessionEngine`): CRSISHORT — daily bars, a limit
that RESTS for one session and fills against that session's range, the short side, borrow carry,
and a ledger that reconciles itself every run.

A faithful decomposition of `runner_crsi_short.run_crsi_short` at the engine's hooks (the session
sequence is the lab's, and the order is the semantics):

    1. mark every open position to the OPEN
    2. cover what the exits say to cover, at the price the RULE defines
    3. otherwise mark to the CLOSE and charge the borrow carry
    4. drop a resting order whose name left the entry universe overnight
    5. fill the resting orders the session's HIGH reached, at `max(limit, open)`

THE RECONCILIATION ASSERTION IS THE POINT. #123 was produced by four engines that disagreed with
each other by 20–40 points and every disagreement was a defect found by two numbers disagreeing. So
this family checks, when the engine asks for its diagnostics at the end of the run, that

    equity change  ==  booked P&L + open P&L - costs

and raises if it does not.

THE BOOK'S CASH IS THE MARKED EQUITY here: a short book has no "cash minus positions" reading that
survives a mark, so the venue's short-side methods (`ShortVenue.short` / `.cover`) move the marked
equity by the P&L of each event and the family carries the marks. The gate for this fold is the
frozen acceptance panel's numbers reproduced to the cent (+80.75 % / 231 trades, #123).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.engine import Order, SessionEngine
from kumo_strategies.backtesting.sim_venue import Book, LabelDefs, SimVenue
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import apply_gates, build_feature_panel, decide
from kumo_strategies.strategies.crsi_short.exits import SessionBar, evaluate_short_exits, open_state

NODATA = "nodata"
#: Dollars. The ledger is exact arithmetic on floats, so the only tolerance needed is rounding
#: noise; anything larger is a defect, not a modelling choice.
RECONCILE_TOLERANCE = 1.0
#: Daily bars carry no time of day; a cost model configured with `time_of_day=True` would apply the
#: intraday profile to a price that has no hour, so the multiplier is pinned to midday.
MIDDAY = (12, 0)


class ShortVenue(SimVenue):
    """The replay venue's SHORT side: the same cost model, the P&L arithmetic of a position that
    is negative shares. `book.cash` is the marked equity (see the module docstring)."""

    def short(self, book: Book, sym: str, shares: int, px: float, ts, **tags) -> float:
        """Open a short of `shares` (negative) at `px`: the cost is charged; the position is
        marked at the fill so no P&L moves yet. Records a SELL fill."""
        cost = self.charge(sym, abs(shares * px), ts, MIDDAY)
        book.cash -= cost
        book.fills.append({"ts": ts, "symbol": sym, "side": "SELL", "qty": abs(shares), "price": px,
                           "cost": cost, **tags})
        return cost

    def cover(self, book: Book, sym: str, shares: int, px: float, mark: float, ts, **tags) -> float:
        """Cover `shares` (negative) at `px` from a position last marked at `mark`: the P&L from the
        mark to the fill moves the equity, the cost is charged. Records a BUY fill."""
        cost = self.charge(sym, abs(shares * px), ts, MIDDAY)
        book.cash += shares * (px - mark)
        book.cash -= cost
        book.fills.append({"ts": ts, "symbol": sym, "side": "BUY", "qty": abs(shares), "price": px,
                           "cost": cost, **tags})
        return cost


class CrsiShort:
    def __init__(self, bars: pd.DataFrame | None, *, cfg: CrsiShortConfig, cost_model: CostModel,
                 adjustment: str | None, panel: pd.DataFrame | None, start, end,
                 starting_cash: float, borrow_annual: float, borrow: dict[str, float | None] | None,
                 n_trials: int, benchmarks: dict[str, pd.Series] | None,
                 entry_fill: Callable[[str, pd.Timestamp, float, pd.Series], float | None] | None) -> None:
        if (bars is None) == (panel is None):
            raise ValueError("pass exactly one of `bars` (raw daily rows) or `panel` (a built panel)")
        if panel is None:
            if adjustment is None:
                raise ValueError(
                    "`adjustment` is required when building from raw bars: CRSISHORT is only defined "
                    "on split-adjusted prices (#123).")
            panel = build_feature_panel(bars, cfg, adjustment=adjustment)
        else:
            panel = apply_gates(panel, cfg)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        self.sessions = [d for d in sorted(panel["date"].unique()) if lo <= d <= hi]
        if not self.sessions:
            raise ValueError(f"no sessions in the panel between {lo.date()} and {hi.date()}")
        self.by_session = {d: g.set_index("ticker") for d, g in panel.groupby("date", sort=False)}
        self.cfg, self.borrow_annual, self.borrow, self.entry_fill = cfg, borrow_annual, borrow, entry_fill
        self.starting_cash, self.n_trials, self.benchmarks = starting_cash, n_trials, benchmarks
        self.book = Book(starting_cash)                    # cash IS the marked equity
        self.venue = ShortVenue(cost_model, LabelDefs("SIM"))
        self.engine = SessionEngine(self.book, self.venue)
        self.pos: dict[str, dict] = {}                      #: ticker -> {trail, shares, mark}
        self.booked = self.costs = self.carry_total = 0.0
        self.covers: list[dict] = []
        self._decisions: list[dict] = []
        self.refusals: list[dict] = []

    # -- helpers ------------------------------------------------------------------------------------

    def _cover(self, d, sym: str, px: float, kind: str, reason: str) -> Order:
        """Close `sym` at `px`: the venue moves the equity and charges; this books the trade."""
        p = self.pos.pop(sym)
        o = Order(sym, "BUY", abs(p["shares"]), px, d, "cover",
                  tags={"shares": p["shares"], "mark": p["mark"]})
        self.engine.execute([o])
        fee = self.book.fills[-1]["cost"]
        self.costs += fee
        pnl = p["shares"] * (px - p["trail"].entry_px)
        self.booked += pnl
        self.covers.append(dict(date=d, ticker=sym, kind=kind, price=px, reason=reason,
                                pnl=pnl, sessions_held=p["trail"].sessions_held))
        return o

    # -- engine hooks -------------------------------------------------------------------------------

    def skip(self, d) -> bool:
        return False

    def slots(self, d) -> list:
        return [("session", None)]

    def session_open(self, d) -> list[Order]:
        return []

    def before_decision(self, d, slot) -> None:
        frame = self.by_session[d]
        # THE DECISION IS TAKEN BEFORE THE SESSION OPENS, so it sees the book as it stood at the
        # last close — today's covers have not happened yet. `frame` is dated to the session the
        # order RESTS in and carries the prior close's ConnorsRSI and limit.
        held_at_open = set(self.pos)
        self._dec = decide(frame.reset_index(), self.cfg, held_at_open, borrow=self.borrow)
        manageable = frame[frame["manageable"]]
        self._entry_pool = frame[frame["eligible"]]
        # `hold_through` decides only whether the ENTRY floor also closes positions. The manageable
        # floor closes them either way — below it there is no price to manage against.
        self._managed = manageable if self.cfg.hold_through else manageable[manageable["eligible"]]
        self._frame = frame

    def decide(self, d, slot) -> bool:
        return True

    def exit_orders(self, d, slot) -> list[Order]:
        """Steps 1–3: no-data covers at the last mark, mark to the open, the rule's covers, mark
        the rest to the close and charge the carry. Executed as built (each cover reads the mark
        the previous step set)."""
        pos, managed, book = self.pos, self._managed, self.book
        for sym in [s for s in pos if s not in managed.index]:
            self._cover(d, sym, pos[sym]["mark"], NODATA, "no manageable row this session")
        session_bars = {sym: SessionBar(open=float(managed.loc[sym, "open"]),
                                        high=float(managed.loc[sym, "high"]),
                                        low=float(managed.loc[sym, "low"]),
                                        close=float(managed.loc[sym, "close"]))
                        for sym in pos}
        for sym, bar in session_bars.items():           # mark to the open before anything acts
            p = pos[sym]
            book.cash += p["shares"] * (bar.open - p["mark"])
            p["mark"] = bar.open
        plan = evaluate_short_exits(self.cfg, session_bars, {s: p["trail"] for s, p in pos.items()})
        for sym, why in plan.exits.items():
            self._cover(d, sym, plan.exit_px[sym], plan.kind[sym], why)
        for sym, p in pos.items():
            bar = session_bars[sym]
            book.cash += p["shares"] * (bar.close - p["mark"])
            p["mark"] = bar.close
            p["trail"] = plan.state[sym]
            carry = abs(p["shares"] * bar.close) * self.borrow_annual / 252.0
            book.cash -= carry
            self.costs += carry
            self.carry_total += carry
        return []

    def entry_orders(self, d, slot) -> list[Order]:
        """Steps 4–5: the resting orders — cancelled overnight when the name left the universe,
        filled at `max(limit, open)` when the session's HIGH reached the limit."""
        frame, dec, entry_pool, cfg, book = self._frame, self._dec, self._entry_pool, self.cfg, self.book
        for sym in frame.index[frame["signalled_yesterday"] & ~frame["eligible"]]:
            self.refusals.append(dict(date=d, ticker=sym, reason="left the entry universe overnight"))
        for sym in dec.enter:
            row = entry_pool.loc[sym]
            limit = dec.limits[sym]
            if float(row["high"]) < limit:
                continue                                # never traded up to the limit
            # `max(limit, open)`: a session that GAPS above the limit fills at the open.
            fill = max(limit, float(row["open"]))
            # `entry_fill` ASKS A CALLER where this entry actually transacts (the slot arm).
            if self.entry_fill is not None:
                got = self.entry_fill(sym, d, limit, row)
                if got is None:
                    self.refusals.append(dict(date=d, ticker=sym, reason="no fill at the modelled slot"))
                    continue
                fill = float(got)
            shares = -int(cfg.slot_notional(book.cash) / fill)
            if not shares:
                self.refusals.append(dict(date=d, ticker=sym, reason="slot too small for one share"))
                continue
            self.engine.execute([Order(sym, "SELL", abs(shares), fill, d, "short", tags={"shares": shares})])
            self.costs += book.fills[-1]["cost"]
            self.pos[sym] = dict(shares=shares, mark=fill,
                                 trail=open_state(fill, SessionBar(open=float(row["open"]),
                                                                   high=float(row["high"]),
                                                                   low=float(row["low"]),
                                                                   close=float(row["close"])),
                                                  session=d.date()))
        for sym, why in dec.refused.items():
            self.refusals.append(dict(date=d, ticker=sym, reason=why))
        self._decisions.append(dict(date=d, equity=book.cash, held=len(self.pos),
                                    resting=len(dec.enter), signalled=int(frame["signal"].sum()),
                                    entered=",".join(dec.enter), refused=len(dec.refused)))
        return []

    def session_close(self, d) -> list[Order]:
        return []

    def after_close(self, d) -> None:
        return None

    def marks(self, d) -> dict[str, float]:
        return {}          # the equity is already marked; the book holds no long shares

    def curve_row(self, d, value: float) -> bool:
        return True

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        """The assertion this family exists for, at the end of the run."""
        from kumo_strategies.backtesting.families import (
            short as _self,  # the module's tolerance, patchable
        )
        equity, pos = self.book.cash, self.pos
        open_pnl = sum(p["shares"] * (p["mark"] - p["trail"].entry_px) for p in pos.values())
        residual = (equity - self.starting_cash) - (self.booked + open_pnl - self.costs)
        if abs(residual) > _self.RECONCILE_TOLERANCE:
            raise AssertionError(
                f"equity does not reconcile with the trade ledger: residual {residual:,.2f} "
                f"(equity change {equity - self.starting_cash:,.2f}, booked {self.booked:,.2f}, "
                f"open {open_pnl:,.2f}, costs {self.costs:,.2f}). One of the two is wrong and the run is "
                "void — this is the check that found four separate engine defects in the lab.")
        return {"covers": pd.DataFrame(self.covers), "borrow_paid": self.carry_total,
                "refused": pd.DataFrame(self.refusals)}
