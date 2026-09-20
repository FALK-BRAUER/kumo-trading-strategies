"""The ROTATION family for the one runner (`engine.SessionEngine`): momentum rotation and its BCT
variation, on two clocks.

    IntradayRotation   N decisions a session at the config's slots; a fill is the next bar after
                       the slot (or the opening print under `execution.fill == "moo"`). The live
                       lane's shape.
    DailyRotation      one decision a session on the close, filled at the next session's opening
                       print — the former `runner_verified`, where every MOMENTUM number before
                       2026-09-08 and the `ledger-book-daily-2024-2026` backtest set came from.

Both are faithful decompositions of the loops they replace (`runner_sessions.py` at #270 step 2):
the same statements in the same order, split at the engine's hooks. The gate for this fold is
three recorded numbers unchanged to the last digit — ledger-book-daily-2024-2026 baseline 13.6297 % / bctrot
8.8249 % on the daily clock, the cadence reference 9.2976 % on the intraday clock.

WHAT THIS FAMILY OWNS: the score, the exit evaluation, the decision, the gap dead band, the market
view's rule 3, sizing (slot budget, inverse-vol scale, `max_weight`, rebalance and trim bands) and
the fill RULE (which price, at which instant). What it does not own: cash, the fill arithmetic, the
curve, the report — those are the engine's and the venue's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from kumo_strategies.backtesting.costs import CostModel
from kumo_strategies.backtesting.engine import Order, SessionEngine
from kumo_strategies.backtesting.instruments import build_all
from kumo_strategies.backtesting.sim_venue import Book, SimVenue
from kumo_strategies.strategies.market_view import MarketSignal, MarketViewConfig, market_state
from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
from kumo_strategies.strategies.momentum_rotation.engine import (
    Decision,
    apply_gates,
    decide,
    filter_entries_by_gap,
    gap_declines_entry,
    needs_panel_stats,
    panel_stats,
    score_panel,
    sizing_denominator,
    trailing_atr,
    trailing_returns,
)
from kumo_strategies.strategies.momentum_rotation.exits import (
    evaluate_exits,
    needs_atr,
    needs_highs,
)
from kumo_strategies.strategies.momentum_rotation.slots import resolve, validate


class IntradayRotation:
    """The intraday clock. Built once per run; the engine calls the hooks per session and slot."""

    def __init__(self, bars: pd.DataFrame, source: CandidateSource, *, cfg: MomentumRotationConfig,
                 instruments_path: str | Path, cost_model: CostModel,
                 instrument_type: dict[str, str] | None, starting_cash: float, deployed: float,
                 compound: bool, n_trials: int, trade_from: pd.Timestamp | None,
                 benchmarks: dict[str, pd.Series] | None, market_view: MarketViewConfig | None) -> None:
        slots = tuple(cfg.execution.decision_slots or ())
        validate(slots)
        # MARKET-ON-OPEN, from the CONFIG (`execution.fill == "moo"`), never from an argument. The
        # order is submitted before the open, so every input to it must be knowable before the open:
        # the ranking already is (prior completed session), and the exit trail is priced off the PRIOR
        # CLOSE rather than the live quote — which is the one thing MOO gives up, the overnight-gap
        # exit `pgrunner._trail_exits` exists to catch. Fills at the session's opening print, for the
        # first slot only; any later slot is an ordinary market order at its instant.
        #
        # STRUCTURALLY IMPOSSIBLE WITH THE GAP DEAD BAND, and refused rather than silently ignored:
        # the band reads today's open / prior close, which does not exist until the open has printed.
        self.moo = cfg.execution.fill == "moo"
        if self.moo and cfg.execution.min_abs_gap_pct is not None:
            raise ValueError(
                "execution.fill='moo' with min_abs_gap_pct set: a market-on-open order cannot evaluate "
                "an overnight-gap rule, because the gap is not known until the open has printed.")

        defs = build_all(instruments_path)
        tradable = set(defs)
        b = bars.copy()
        b["ts"] = pd.to_datetime(b["ts"])
        b["date"] = pd.to_datetime(b["date"])
        b = b[b.ticker.isin(tradable)].sort_values(["ticker", "ts"])
        if b.empty:
            raise ValueError("no bars for any tradable symbol")
        self.b = b
        self.sessions = sorted(b["date"].unique())
        # Session bounds from the BARS, not a calendar: the tape is what the venue printed, so a half
        # day is short here without anyone having to know it is a half day.
        self.bounds = b.groupby("date")["ts"].agg(["min", "max"])
        self.per_session: dict[pd.Timestamp, list[tuple[str, pd.Timestamp]]] = {}
        for d in self.sessions:
            o, last_bar = self.bounds.loc[d, "min"], self.bounds.loc[d, "max"]
            close = last_bar + pd.Timedelta(minutes=5)     # the last bar's close IS the session close
            got = resolve(slots, o.to_pydatetime(), close.to_pydatetime())
            if got:
                self.per_session[d] = [(name, pd.Timestamp(w)) for name, w in got]
        self.prior_session = {d: p for p, d in zip(self.sessions, self.sessions[1:])}

        # THE LANE'S OWN MARKET (#873, #147). Equal-weight over the pool this lane ranks, not a
        # platform index. INERT BY DEFAULT: `signal=NONE` yields RISK_ON and every existing result
        # stands byte for byte.
        self.mv = market_view if market_view is not None else cfg.market_view
        self.market_panel = None
        if self.mv.signal is not MarketSignal.NONE:
            self.market_panel = (b.sort_values("ts")
                                 .pivot_table(index="date", columns="ticker", values="close")
                                 .sort_index())
        self.market_states: dict[pd.Timestamp, object] = {}

        # COMPLETED daily bars. This is the ONLY panel that ranks, at every slot, because it is the only
        # one live has: today's session cannot appear in its own ranking.
        daily = (b.sort_values("ts").groupby(["ticker", "date"], sort=False)
                 .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                      close=("close", "last"), volume=("volume", "sum")).reset_index())
        self.daily = daily
        self.scored = score_panel(apply_gates(daily, cfg, instrument_type), cfg)
        # Today's OPEN and the prior session's CLOSE — the two halves of the overnight gap.
        self.session_open_px = daily.set_index(["ticker", "date"])["open"]
        self.daily_close_px = daily.set_index(["ticker", "date"])["close"]
        nxt = b.assign(next_open=b.groupby("ticker")["open"].shift(-1),
                       next_ts=b.groupby("ticker")["ts"].shift(-1))
        self.fill_at = nxt.set_index(["ticker", "ts"])[["next_open", "next_ts"]]
        self.last_close = b.set_index(["ticker", "ts"])["close"]
        self.ret_pivot = None
        if needs_panel_stats(cfg):
            self.ret_pivot = daily.pivot_table(index="date", columns="ticker",
                                               values="close").sort_index().pct_change()

        self.cfg, self.source = cfg, source
        self.starting_cash, self.deployed, self.compound = starting_cash, deployed, compound
        self.n_trials, self.benchmarks, self.trade_from = n_trials, benchmarks, trade_from
        self.book = Book(starting_cash)
        self.venue = SimVenue(cost_model, defs)
        self.engine = SessionEngine(self.book, self.venue)
        self._decisions: list[dict] = []
        #: Names this session has already traded, reset each session. Counts the live re-entry path
        #: (module docstring of runner_sessions) rather than preventing it.
        self.reentries = 0
        self._traded_today: set[str] = set()
        # per-slot scratch, set by `before_decision` and read by the order hooks
        self._at_now: dict = {}
        self._dec: Decision | None = None
        self._exits: tuple = ()
        self._enters: tuple = ()
        self._moo_now = False

    # -- engine hooks -------------------------------------------------------------------------------

    def skip(self, d) -> bool:
        return self.trade_from is not None and d < pd.Timestamp(self.trade_from)

    def slots(self, d) -> list:
        self._traded_today = set()
        prior = self.prior_session.get(d)
        if prior is None:
            return []                    # no completed session to rank on; live would not decide
        day = self.scored[self.scored["date"] == prior]
        if day.empty:
            return []
        # Present the prior session's ranking AS today's decision. The scores are unchanged; only
        # the label moves, so `decide` dates the Decision to the session being traded.
        self._day, self._prior = day.assign(date=d), prior
        return [(idx, name, when) for idx, (name, when) in enumerate(self.per_session.get(d, []))]

    def session_open(self, d) -> list[Order]:
        return []

    def before_decision(self, d, slot) -> None:
        slot_idx, _slot_name, when = slot
        held, state, cfg, b = self.book.held, self.book.state, self.cfg, self.b
        self._moo_now = moo_now = self.moo and slot_idx == 0
        # --- exits first, on prices AS OF this instant ---------------------------------
        # Under MOO the instant is BEFORE the open, so the only honest price is the prior
        # session's close. Otherwise, the last bar closed at or before `when`.
        at_now = {}
        for sym in held:
            try:
                at_now[sym] = (float(self.daily_close_px.loc[(sym, self._prior)]) if moo_now
                               else float(self.last_close.loc[(sym, when)]))
            except KeyError:
                at_now[sym] = None      # no bar at this instant: keep last known, never a gap
        hi_now = None
        if needs_highs(cfg.exits):
            upto = b[(b.date == d) & (b.ts <= when)]
            h = upto.groupby("ticker")["high"].max().to_dict()
            hi_now = {s: h[s] for s in held if h.get(s) is not None}
        # Strictly BEFORE today: today's range is still forming, and pricing a stop off it
        # would shrink the distance toward the open.
        atr_now = trailing_atr(self.daily[self.daily.date < d]) if needs_atr(cfg.exits) else None
        plan = evaluate_exits(cfg.exits, at_now,
                              {s: st for s, st in state.items() if s in held},
                              atr=atr_now, highs=hi_now)
        state.update(plan.state)
        self._at_now, self._forced = at_now, set(plan.exits)

    def decide(self, d, slot) -> bool:
        _slot_idx, slot_name, when = slot
        held, cfg, day, prior = self.book.held, self.cfg, self._day, self._prior
        corr = vol = None
        if self.ret_pivot is not None:
            corr, vol = panel_stats(self.ret_pivot, cfg.portfolio.corr_window,
                                    sorted(set(held) | set(day["ticker"])), as_of=prior)
        dec = decide(day, self.source, cfg, set(held), corr=corr, vol=vol)
        exits = tuple(sorted(set(dec.exit) | self._forced))
        enters = tuple(x for x in dec.enter if x not in exits)

        # RULE 3, THROUGH THE SHARED FUNCTION so this driver cannot implement half of it (#170):
        # LIQUIDATE exits the whole book and enters nothing; EXIT_ONLY enters nothing.
        view = (market_state(self.market_panel, self.mv, asof=d) if self.market_panel is not None else None)
        if view is not None:
            self.market_states[d] = view
            if view.liquidates:
                exits = tuple(sorted(set(exits) | set(held)))
                enters = ()
            elif view.blocks_entries:
                enters = ()

        # --- the overnight gap dead band, the LAST look before the order goes out -------
        gap_declined: tuple[str, ...] = ()
        if cfg.execution.min_abs_gap_pct is not None:
            opens = {s: float(self.session_open_px.loc[(s, d)])
                     for s in enters if (s, d) in self.session_open_px.index}
            closes = {s: float(self.daily_close_px.loc[(s, prior)])
                      for s in enters if (s, prior) in self.daily_close_px.index}
            enters, gap_declined = filter_entries_by_gap(enters, opens, closes, cfg)

        # How often the correlation cap BINDS: the same decision with the cap removed, and the names
        # it would have entered that the capped one did not.
        corr_rejected = 0
        if cfg.portfolio.max_correlation is not None and corr is not None:
            free = decide(day, self.source, cfg, set(held), corr=None, vol=vol)
            corr_rejected = len(set(free.enter) - set(dec.enter))
        self._decisions.append({"date": d, "slot": slot_name, "at": when,
                                "hold": ",".join(dec.hold), "enter": ",".join(enters),
                                "exit": ",".join(exits), "corr_rejected": corr_rejected,
                                "gap_declined": ",".join(gap_declined),
                                "reentry": ",".join(sorted(set(enters) & self._traded_today)),
                                "market_state": None if view is None else view.state,
                                "market_reasons": None if view is None else "; ".join(view.reasons),
                                "scores": {k: round(float(v), 6) for k, v in dec.scores.items()}})
        self.reentries += len(set(enters) & self._traded_today)
        self._dec, self._exits, self._enters = dec, exits, enters
        return True

    def _fill(self, sym, d, when):
        if self._moo_now:
            # The session's opening print — what a market-on-open order receives.
            try:
                p = float(self.session_open_px.loc[(sym, d)])
            except KeyError:
                return None, None
            return (p, pd.Timestamp(self.bounds.loc[d, "min"])) if np.isfinite(p) else (None, None)
        try:
            row = self.fill_at.loc[(sym, when)]
        except KeyError:
            return None, None
        p, ts = row["next_open"], row["next_ts"]
        if p is None or not np.isfinite(p) or pd.isna(ts):
            return None, None
        return float(p), pd.Timestamp(ts)

    def exit_orders(self, d, slot) -> list[Order]:
        _, slot_name, when = slot
        held = self.book.held
        out = []
        for sym in self._exits:
            if sym not in held:
                continue
            p, ts = self._fill(sym, d, when)
            if p is None:
                continue
            out.append(Order(sym, "SELL", held[sym], p, ts, "exit", tags={"slot": slot_name}))
            self._traded_today.add(sym)
        return out

    def entry_orders(self, d, slot) -> list[Order]:
        """Sized against the book AFTER the exits — the engine executed them before calling this.
        The orders are executed one by one as they are built, because each buy reads the cash the
        previous one left (`venue.affordable`); the list returned is what was already done."""
        _, slot_name, when = slot
        book, venue, cfg = self.book, self.venue, self.cfg
        held, at_now, dec, exits, enters = book.held, self._at_now, self._dec, self._exits, self._enters
        starting_cash, deployed, compound = self.starting_cash, self.deployed, self.compound
        done: list[Order] = []

        def _do(o: Order) -> None:
            # executed here (the next order's sizing reads the book), recorded for the engine
            self.engine.execute([o])
            done.append(o)

        # --- TARGET-BASED REBALANCING (2026-09-08): every held name is traded TO ITS
        # TARGET each decision when it has drifted outside `rebalance_band`; sells run first so the
        # freed cash funds the buys in the same decision.
        rb = cfg.portfolio.rebalance_band
        if rb is not None and held:
            mtm_now = sum(q * (at_now.get(s2) or 0.0) for s2, q in held.items())
            base = (book.cash + mtm_now) if compound else starting_cash
            book_after = (set(held) - set(exits)) | set(enters)
            tgt_each = base * deployed / sizing_denominator(len(book_after), cfg.portfolio.n_hold, cfg)
            if cfg.portfolio.max_weight:
                tgt_each = min(tgt_each, cfg.portfolio.max_weight * base)
            deltas = []
            for s2 in sorted(set(held) - set(exits)):
                p2, ts2 = self._fill(s2, d, when)
                if p2 is None:
                    continue
                have_q = held.get(s2, 0)
                tgt_q = int(tgt_each / p2)
                if abs(have_q - tgt_q) * p2 <= tgt_each * rb:
                    continue                              # inside the band
                deltas.append((s2, tgt_q - have_q, p2, ts2))
            for s2, dq, p2, ts2 in sorted(deltas, key=lambda x: x[1]):   # SELLS FIRST
                if dq < 0:
                    q_sell = min(-dq, held[s2] - 1)               # never to zero: that is an exit
                    if q_sell < 1:
                        continue
                    _do(Order(s2, "SELL", q_sell, p2, ts2, "trim", tags={"slot": slot_name, "rebalance": True}))
                elif dq > 0:
                    dq = int(min(dq * p2, venue.affordable(book, s2, ts2)) / p2)
                    if dq < 1:
                        continue
                    _do(Order(s2, "BUY", dq, p2, ts2, "add", tags={"slot": slot_name, "rebalance": True}))

        if enters:
            mtm_now = sum(q * (at_now.get(s2) or 0.0) for s2, q in held.items())
            base = (book.cash + mtm_now) if compound else starting_cash
            # The SHARED denominator, sized on the book AFTER the gap filter, exactly as pgrunner
            # computes `book_after`.
            book_after = (set(held) - set(exits)) | set(enters)
            slot_budget = base * deployed / sizing_denominator(
                len(book_after), cfg.portfolio.n_hold, cfg)
            # Inverse-vol scale per entering name, normalised against the WEIGHT SET.
            scale_by: dict[str, float] = {}
            if dec.weights:
                scale_by = {s2: dec.weights[s2] * len(dec.weights)
                            for s2 in enters if s2 in dec.weights}
                if cfg.portfolio.inverse_vol_entry_relative and scale_by:
                    mean = sum(scale_by.values()) / len(scale_by)
                    if mean > 0:
                        scale_by = {s2: x / mean for s2, x in scale_by.items()}
            for sym in enters:
                p, ts = self._fill(sym, d, when)
                if p is None:
                    continue
                scale = scale_by.get(sym, 1.0)
                # SIZE TO THE CASH ACTUALLY AVAILABLE, spread included — do not skip.
                affordable = venue.affordable(book, sym, ts)
                # `max_weight` ENFORCED AT SIZING (FINDINGS-size-to-book.md).
                cap = cfg.portfolio.max_weight * base if cfg.portfolio.max_weight else np.inf
                qty = int(min(slot_budget * scale, cap, affordable) / p)
                if qty < 1:
                    continue
                if not venue.budget_allows(book, sym, qty, p, ts):
                    continue                      # never spend cash we do not have
                _do(Order(sym, "BUY", qty, p, ts, "entry", tags={"slot": slot_name}))
                self._traded_today.add(sym)
        return []          # everything above was executed as it was sized

    def session_close(self, d) -> list[Order]:
        return []

    def after_close(self, d) -> None:
        return None

    def marks(self, d) -> dict[str, float]:
        # Mark to market at the session close, so the curve is daily and comparable.
        eod = {}
        for sym in self.book.held:
            if (sym, d) in self.daily_close_px.index:
                eod[sym] = float(self.daily_close_px.loc[(sym, d)])
        return eod

    def curve_row(self, d, value: float) -> bool:
        return (not self.book.held) or np.isfinite(value)

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        return {}


class DailyRotation:
    """THE DAILY CLOCK — one decision a session on the close, filled at the next session's open.
    `runner_verified.run` moved in whole (#270 fold 2), decomposed at the engine's hooks.

    `entry_filter(symbol, date, fill_price, signal_close)` is a RESEARCH hook: a sweep's way of
    asking a question at the fill. Not a deployment knob; the intraday clock refuses it."""

    def __init__(self, panel: pd.DataFrame, source: CandidateSource, *, cfg: MomentumRotationConfig,
                 instruments_path: str | Path, cost_model: CostModel,
                 instrument_type: dict[str, str] | None, starting_cash: float, deployed: float,
                 compound: bool, n_trials: int, trade_from: pd.Timestamp | None,
                 benchmarks: dict[str, pd.Series] | None,
                 entry_filter: Callable[[str, pd.Timestamp, float, float], bool] | None) -> None:
        px = panel.sort_values(["ticker", "date"]).copy()
        px["date"] = pd.to_datetime(px["date"])
        self.scored = score_panel(apply_gates(px, cfg, instrument_type), cfg)
        defs = build_all(instruments_path)
        self.tradable = set(defs)
        nxt = px.groupby("ticker")[["open", "date"]].shift(-1).rename(
            columns={"open": "next_open", "date": "next_date"})
        px = px.join(nxt)
        # ON DAILY BARS "next_open" IS THE NEXT SESSION'S OPENING PRINT — the only fill a daily bar
        # can express. `"close"` is the signal session's own close — research only.
        self.fill_px = px.set_index(["ticker", "date"])["close" if cfg.execution.fill == "close" else "next_open"].to_dict()
        self.close_px = px.set_index(["ticker", "date"])["close"].to_dict()
        self.high_px = px.set_index(["ticker", "date"])["high"].to_dict()
        self.px = px
        self.gap_armed = cfg.execution.min_abs_gap_pct is not None
        # Trailing correlations and volatilities, derived FROM THE CONFIG, so `max_correlation` and
        # `inverse_vol_sizing` can be exercised at all.
        self.window = cfg.portfolio.corr_window if needs_panel_stats(cfg) else None
        self.ret_pivot = trailing_returns(px) if self.window else None
        self.sessions = sorted(self.scored["date"].unique())
        self.cfg, self.source, self.entry_filter = cfg, source, entry_filter
        self.starting_cash, self.deployed, self.compound = starting_cash, deployed, compound
        self.n_trials, self.benchmarks, self.trade_from = n_trials, benchmarks, trade_from
        self.book = Book(starting_cash)
        self.venue = SimVenue(cost_model, defs)
        self.engine = SessionEngine(self.book, self.venue)
        self._decisions: list[dict] = []
        self._dec: Decision | None = None
        self._ts = None

    def skip(self, d) -> bool:
        return self.trade_from is not None and d < pd.Timestamp(self.trade_from)   # warmup: scored, not traded

    def slots(self, d) -> list:
        day = self.scored[self.scored["date"] == d]
        # a name with no Alpaca instrument cannot be traded live, so it must not trade here either
        day = day[day.ticker.isin(self.tradable)]
        if day.empty:
            return []
        self._day = day
        return [("close", None)]

    def session_open(self, d) -> list[Order]:
        return []

    def before_decision(self, d, slot) -> None:
        held, state, cfg, px = self.book.held, self.book.state, self.cfg, self.px
        # --- capital-efficiency exits, before the ranking decides anything ---------------
        # ONE implementation, shared with the live runner (#197 P2).
        plan = evaluate_exits(
            cfg.exits,
            {sym: self.close_px.get((sym, d)) for sym in held},
            {sym: st for sym, st in state.items() if sym in held},
            atr=trailing_atr(px[px.date <= d]) if needs_atr(cfg.exits) else None,
            highs=({sym: self.high_px.get((sym, d)) for sym in held}
                   if needs_highs(cfg.exits) else None),
        )
        state.update(plan.state)          # the evaluator returns advanced state; persist it
        self._forced = set(plan.exits)

    def decide(self, d, slot) -> bool:
        held, cfg, day, forced = self.book.held, self.cfg, self._day, self._forced
        corr = vol = None
        if self.ret_pivot is not None:
            corr, vol = panel_stats(self.ret_pivot, self.window,
                                    sorted(set(held) | set(day["ticker"])), as_of=d)
        dec = decide(day, self.source, cfg, set(held), corr=corr, vol=vol)
        if forced:
            # Weights renormalised over the SURVIVING book, or every entry is silently undersized.
            surviving = tuple(x for x in dec.hold if x not in forced)
            kept = {k: v for k, v in dec.weights.items() if k in set(surviving)}
            tot = sum(kept.values())
            dec = Decision(date=dec.date, hold=surviving,
                           enter=dec.enter, exit=tuple(sorted(set(dec.exit) | forced)),
                           scores=dec.scores,
                           weights={k: v / tot for k, v in kept.items()} if tot > 0 else {})
        self._decisions.append({"date": d, "hold": ",".join(dec.hold),
                                "enter": ",".join(dec.enter), "exit": ",".join(dec.exit)})
        self._ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=35)
        self._dec = dec
        return True

    def exit_orders(self, d, slot) -> list[Order]:
        held, ts = self.book.held, self._ts
        out = []
        for sym in self._dec.exit:
            p = self.fill_px.get((sym, d))
            if not p or not np.isfinite(p) or sym not in held:
                continue
            out.append(Order(sym, "SELL", held[sym], p, ts, "exit"))
        return out

    def entry_orders(self, d, slot) -> list[Order]:
        book, venue, cfg, dec, ts = self.book, self.venue, self.cfg, self._dec, self._ts
        held, close_px, fill_px = book.held, self.close_px, self.fill_px
        starting_cash, deployed, compound = self.starting_cash, self.deployed, self.compound
        if not dec.enter:
            return []
        done: list[Order] = []

        def _do(o: Order) -> None:
            self.engine.execute([o])
            done.append(o)

        # Size on CURRENT equity, not starting cash; SIZE TO THE BOOK THAT EXISTS.
        mtm_now = sum(q * close_px.get((s2, d), np.nan) for s2, q in held.items())
        base = (book.cash + mtm_now) if (compound and np.isfinite(mtm_now)) else starting_cash
        slot_budget = base * deployed / sizing_denominator(
            len(dec.hold), cfg.portfolio.n_hold, cfg)
        # `dec.weights` consumed WHENEVER PRESENT, as a multiple of the equal-weight slot.
        scale = {}
        if dec.weights and dec.hold:
            scale = {s2: dec.weights[s2] * len(dec.weights)
                     for s2 in dec.enter if s2 in dec.weights}
            if cfg.portfolio.inverse_vol_entry_relative and scale:
                mean = sum(scale.values()) / len(scale)
                if mean > 0:
                    scale = {s2: x / mean for s2, x in scale.items()}
        # TARGET-BASED SIZING: every held name traded to its target; sells before buys.
        rb = cfg.portfolio.rebalance_band
        if rb is not None:
            targets = {s2: slot_budget * scale.get(s2, 1.0) for s2 in dec.hold}
            deltas = []
            for s2 in sorted(dec.hold):
                px_now = fill_px.get((s2, d))
                if not px_now or not np.isfinite(px_now):
                    continue
                tgt_q = int(targets[s2] / px_now)
                have_q = held.get(s2, 0)
                if have_q < 1:
                    continue          # a genuine entry; the loop below handles it
                if abs(have_q - tgt_q) * px_now <= targets[s2] * rb:
                    continue          # inside the band, leave it alone
                deltas.append((s2, tgt_q - have_q, px_now))
            for s2, dq, px_now in sorted(deltas, key=lambda x: x[1]):   # SELLS FIRST
                if dq < 0:
                    q_sell = min(-dq, held.get(s2, 0) - 1)   # never to zero: that is an exit
                    if q_sell < 1:
                        continue
                    _do(Order(s2, "SELL", q_sell, px_now, ts, "trim"))
                elif dq > 0:
                    if not venue.budget_allows(book, s2, dq, px_now, ts):
                        c = venue.charge(s2, dq * px_now, ts)
                        dq = int((book.cash - c) / px_now)
                        if dq < 1:
                            continue
                    _do(Order(s2, "BUY", dq, px_now, ts, "add"))
        # TRIM BACK TOWARD TARGET before funding entries; sells first.
        band = cfg.portfolio.trim_band
        if band is not None:
            for sym in sorted(set(dec.hold) - set(dec.enter)):
                q_held = held.get(sym, 0)
                px_now = fill_px.get((sym, d))
                if q_held < 1 or not px_now or not np.isfinite(px_now):
                    continue
                target = slot_budget * scale.get(sym, 1.0)
                if q_held * px_now <= target * (1.0 + band):
                    continue
                excess = int(q_held - target / px_now)
                if excess < 1 or excess >= q_held:
                    continue          # never trim a position out of existence — that is an exit
                _do(Order(sym, "SELL", excess, px_now, ts, "trim"))
        for sym in dec.enter:
            budget = slot_budget * scale.get(sym, 1.0)
            p = fill_px.get((sym, d))
            if not p or not np.isfinite(p):
                continue
            # Last look at the FILL, not at the signal: the research hook, then the CONFIGURED
            # overnight-gap rule through the shared predicate.
            if self.entry_filter is not None or self.gap_armed:
                sig_close = close_px.get((sym, d))
                if not sig_close or not np.isfinite(sig_close):
                    continue
                if self.entry_filter is not None and not self.entry_filter(sym, d, float(p), float(sig_close)):
                    continue
                if gap_declines_entry(float(p), float(sig_close), cfg):
                    continue
            qty = int(budget / p)
            if qty < 1:
                continue
            if not venue.budget_allows(book, sym, qty, p, ts):
                continue                      # never spend cash we do not have
            _do(Order(sym, "BUY", qty, p, ts, "entry"))
        return []

    def session_close(self, d) -> list[Order]:
        return []

    def after_close(self, d) -> None:
        return None

    def marks(self, d) -> dict[str, float]:
        return {s: self.close_px.get((s, d), np.nan) for s in self.book.held}

    def curve_row(self, d, value: float) -> bool:
        return np.isfinite(value)

    def decisions(self) -> list[dict]:
        return self._decisions

    def diagnostics(self) -> dict:
        return {}
