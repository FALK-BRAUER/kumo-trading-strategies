"""Intraday momentum rotation — 5-minute bars, decisions and stops inside the session.

The daily-bar version cannot express this strategy. It enters and exits at daily opens, so nothing
can intervene between one close and the next: no stop, no flat-by-close, no reaction to a name
rolling over at 11am. The -30% drawdown measured at x=3 accumulated with the strategy unable to act.

What changes here:
  cadence   decisions at a fixed intraday grid (default every 30 min) rather than once per session
  stops     a live ATR stop checked on EVERY bar, so a position can be cut mid-session
  exit      optional flat-by-close, to test holding overnight against not

What does NOT change: the ranking. `engine.decide()` is the same pure function the daily version
calls, so the two are comparable and there is still one implementation.

Performance matters at this cadence. The daily version rebuilt a pandas DataFrame from every
symbol's deque on each decision — ~400 times. At 5-minute cadence that would be tens of thousands of
rebuilds over a larger window. State is therefore accumulated incrementally: one rolling daily bar
per symbol per session, updated in place, and the score panel is built only at decision points.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.contract import protective_close

from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd
from kumo_strategies.runtime.nautilus.sides import LONG
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.runtime.nautilus.sides import (
    LONG, closing_quantity, refusal_reason)

from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
from kumo_strategies.strategies.momentum_rotation.engine import (
    apply_gates, decide, score_intraday, score_panel, trailing_atr)
from kumo_strategies.strategies.momentum_rotation.exits import (
    evaluate_exits, needs_atr, needs_highs, reconstruct_trail)

STRATEGY_NAME = "MOMENTUM"
EXTERNAL_ID = "MOMENTUM_INTRADAY"
STRATEGY_LABEL = "BCT momentum rotation (intraday overlay)"

#: Cockpit allocates the `order_id_tag` — it is the only place that sees every strategy in ONE
#: trader and it is what calls `Trader.add_strategy`. This default exists so an existing deployment
#: keeps the id it already has; under NETTING the position id is `{instrument}-{strategy_id}`, so a
#: tag that moves after anything has traded orphans real positions. Callers SHOULD pass it.
DEFAULT_ORDER_ID_TAG = "001"
STRATEGY_TAG = DEFAULT_ORDER_ID_TAG  # deprecated alias; cockpit owns this


@dataclass
class _Day:
    """A session's bar, accumulated as 5-minute bars arrive rather than rebuilt."""
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def update(self, bar: Bar) -> None:
        self.high = max(self.high, bar.high.as_double())
        self.low = min(self.low, bar.low.as_double())
        self.close = bar.close.as_double()
        self.volume += bar.volume.as_double()


class IntradayMomentumRotation(HeldSeedMixin, Strategy):
    """Rotation on 5-minute bars, with live stops and an optional flat-by-close."""

    #: DECLARED EXEMPTION from the subscribe-implies-request rule (#200), so that
    #: "this lane does not need history" and "this lane was forgotten" cannot look alike.
    NO_HISTORY_REQUEST = (
        "requests no bar history: an INTRADAY lane warms on interval bars within a single session, so its `_need` is minutes rather than sessions. It is also deployed nowhere. Revisit before it is ever registered — the #200 reasoning applies the moment its interval is long enough that `_need` spans a session boundary.")

    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG

    def __init__(
        self,
        cfg: MomentumRotationConfig,
        source: CandidateSource,
        instrument_ids: list[InstrumentId],
        bar_type_suffix: str = "-5-MINUTE-LAST-EXTERNAL",
        instrument_type: dict[str, str] | None = None,
        equity_per_position: float = 10_000.0,
        decide_every_mins: int = 30,
        stop_atr_mult: float | None = 2.0,
        flat_by_close: bool = False,
        close_time: str = "15:55",
    ) -> None:
        super().__init__(config=StrategyConfig(strategy_id=STRATEGY_NAME, order_id_tag=STRATEGY_TAG))
        self._cfg = cfg
        self._source = source
        self._iids = list(instrument_ids)
        self._suffix = bar_type_suffix
        self._itype = instrument_type or {}
        self._equity = equity_per_position
        # total capital the book may deploy; weights divide THIS, so equal weights
        # reproduce equity_per_position exactly and the change is inert by default.
        self._deployed = equity_per_position * cfg.portfolio.n_hold
        self._every = decide_every_mins
        self._stop_mult = stop_atr_mult
        self._flat_by_close = flat_by_close
        self._close_time = close_time

        need = max(cfg.score.vol_window, cfg.score.lookback) + cfg.gates.corporate_action_window
        self._need = need
        self._hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=need + 5))  # closed sessions
        self._today: dict[str, _Day] = {}
        # 5-minute bars, for the INTRADAY atr. self._hist holds completed SESSIONS, so an atr
        # taken from it prices a daily range (~6.1% of price) — 12x too wide to ever fire inside
        # a position held for tens of minutes. The stop has to be built from the bars the
        # strategy actually trades on.
        intra_need = cfg.intraday.vol_window_bars + cfg.intraday.lookback_bars + 5
        self._intra: dict[str, deque] = defaultdict(lambda: deque(maxlen=max(64, intra_need)))
        self._session: pd.Timestamp | None = None

        self._held: set[str] = set()
        self._pending: dict[str, str] = {}
        self._stops: dict[str, float] = {}
        self._entry: dict[str, float] = {}
        self._sizes: dict[str, int] = {}    # filled qty, to value committed capital
        self._last_slot: tuple | None = None
        self._last_ts: pd.Timestamp | None = None
        self._due_ts: pd.Timestamp | None = None

    # -- lifecycle ---------------------------------------------------------------------------
    def on_start(self) -> None:
        # A RESTART IS THE NORMAL CASE, and `_held` is built empty. Until it is seeded, every
        # position predating this boot is invisible to `protective_close`, so the venue's answer
        # to a protective stop-out goes unrecorded and the claim outlives the position (#194).
        self.seed_held_from_positions()
        for iid in self._iids:
            self.subscribe_bars(BarType.from_str(f"{iid}{self._suffix}"))
        self.log.info(f"intraday rotation on {len(self._iids)} instruments, "
                      f"decide every {self._every}m, stop {self._stop_mult}xATR, "
                      f"flat_by_close={self._flat_by_close}")

    # -- data --------------------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        ts = pd.Timestamp(bar.ts_event, unit="ns")
        day = ts.normalize()

        # on_bar fires once PER SYMBOL per timestamp, so a decision taken the moment a slot's
        # first bar lands ranks a panel where only that one symbol has reported. Harmless mid-
        # session (the others still carry the bar before), fatal at 09:30 where _roll_session has
        # just cleared _today and the universe is literally one name. So a slot only becomes DUE
        # here and fires below, once the clock advances and every symbol's bar for it is in.
        if self._due_ts is not None and self._last_ts is not None and ts > self._last_ts:
            due, self._due_ts = self._due_ts, None
            self._decide(due)             # belongs to the old session -> before any roll
        self._last_ts = ts

        if self._session is None:
            self._session = day
        elif day > self._session:
            self._roll_session()          # previous session complete -> push to history
            self._session = day

        d = self._today.get(sym)
        if d is None or d.date != day:
            self._today[sym] = _Day(day, bar.open.as_double(), bar.high.as_double(),
                                    bar.low.as_double(), bar.close.as_double(),
                                    bar.volume.as_double())
        else:
            d.update(bar)

        self._intra[sym].append((ts, bar.high.as_double(), bar.low.as_double(),
                                 bar.close.as_double()))
        # stops are checked on EVERY bar — the entire reason for going intraday
        self._check_stop(sym, bar)

        hm = ts.strftime("%H:%M")
        if self._flat_by_close and hm >= self._close_time:
            self._flatten()
            return
        # minutes since the session opened, NOT minute-of-hour: `ts.minute % every` silently
        # collapses every cadence above 60 into "on the hour" (195 behaved exactly like 60).
        #
        # LIVE HAZARD: this reads ts as EXCHANGE wall clock. The backtest parquet carries naive ET
        # (09:30-15:55) passed through as raw nanoseconds, so Nautilus stores it as UTC and reading
        # it back yields ET again - self-consistent only by luck. Real Alpaca bars are genuinely
        # UTC, where both this arithmetic and the "15:55" compare below land on the wrong time of
        # day. Convert to America/New_York explicitly before wiring this strategy to a live feed.
        mins = (ts.hour - 9) * 60 + (ts.minute - 30)
        if mins >= 0 and mins % self._every == 0:
            slot = (day, ts.hour, ts.minute)
            if slot != self._last_slot:
                self._last_slot = slot
                self._due_ts = ts

    def _roll_session(self) -> None:
        for sym, d in self._today.items():
            self._hist[sym].append(d)
        self._today = {}

    @property
    def warm(self) -> bool:
        ready = sum(1 for v in self._hist.values() if len(v) >= self._need)
        return ready >= self._cfg.portfolio.n_hold + self._cfg.portfolio.buffer

    # -- stops -------------------------------------------------------------------------------
    def _check_stop(self, sym: str, bar: Bar) -> None:
        stop = self._stops.get(sym)
        if stop is None or sym not in self._held or sym in self._pending:
            return
        if bar.low.as_double() <= stop:
            self._close(sym, reason="stop")

    def _atr(self, sym: str, window: int = 14) -> float | None:
        """True range over the last `window` FIVE-MINUTE bars — the horizon the stop must match."""
        h = list(self._intra.get(sym, []))[-(window + 1):]
        if len(h) < max(5, window // 2):
            return None
        trs = [max(hi - lo, abs(hi - pc), abs(lo - pc))
               for (_, _, _, pc), (_, hi, lo, _) in zip(h, h[1:])]
        return sum(trs) / len(trs) if trs else None

    # -- decision ----------------------------------------------------------------------------
    def _decide(self, ts: pd.Timestamp) -> None:
        if not self.warm or self._pending:
            return
        rows = []
        for sym, hist in self._hist.items():
            for d in hist:
                rows.append((sym, d.date, d.open, d.high, d.low, d.close, d.volume))
            cur = self._today.get(sym)
            if cur is not None:      # the session so far counts as today's bar
                rows.append((sym, cur.date, cur.open, cur.high, cur.low, cur.close, cur.volume))
        if not rows:
            return
        panel = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        scored = score_panel(apply_gates(panel, self._cfg, self._itype), self._cfg)
        today = scored[scored.date == scored.date.max()]
        if today.empty:
            return
        corr = vol = None
        if self._cfg.portfolio.max_correlation is not None or self._cfg.portfolio.inverse_vol_sizing:
            w = self._cfg.portfolio.corr_window
            closes = {sym: [d.close for d in list(hist)[-(w + 1):]]
                      for sym, hist in self._hist.items() if len(hist) >= max(20, w // 2)}
            if closes:
                n = min(len(v) for v in closes.values())
                if n >= 20:
                    rets = pd.DataFrame({k: v[-n:] for k, v in closes.items()}).pct_change().dropna(how="all")
                    if len(rets) >= 10:
                        corr, vol = rets.corr(), rets.std()

        overlay = None
        if self._cfg.intraday.weight:
            ib = [(sym, bts, c) for sym, ring in self._intra.items() for (bts, _, _, c) in ring]
            if ib:
                overlay = score_intraday(
                    pd.DataFrame(ib, columns=["ticker", "ts", "close"]), self._cfg)
        d = decide(today, self._source, self._cfg, set(self._held), overlay=overlay,
                   corr=corr, vol=vol)

        # ExitConfig reached nothing here either (#26). This adapter substituted `stop_atr_mult` and
        # `flat_by_close`, which are CONSTRUCTOR arguments rather than config fields — so the
        # configured exit rules were silently inert while two unswept, unversionable ones ran in
        # their place. Both now coexist: the ATR stop is an intra-session mechanism, the ExitConfig
        # rules are session-level, and they answer different questions.
        forced = self._trail_exits()
        exits = tuple(sorted(set(d.exit) | forced))
        for sym in exits:
            self._close(sym, reason="rotate" if sym in d.exit else "trail")
        for sym in d.enter:
            if sym in exits:
                continue
            self._open(sym, today, d.weights.get(sym, 1.0 / self._cfg.portfolio.n_hold))

    def _trail_exits(self) -> set[str]:
        """Session-level exits, from a trail rebuilt out of completed sessions.

        Reconstructed rather than remembered, for the same reason as the daily adapter: #197 B1 was
        a peak that did not survive a restart. `self._hist` holds completed sessions, which is
        exactly the granularity `ExitConfig` counts in — `sessions_held` and `sessions_since_high`
        are session counts, not bar counts.
        """
        if not self._held:
            return set()
        states, prices = {}, {}
        for pos in self.cache.positions_open(strategy_id=self.id):
            sym = pos.instrument_id.symbol.value
            if sym not in self._held:
                continue
            hist = list(self._hist.get(sym, ()))
            if not hist:
                continue
            opened = pd.Timestamp(pos.ts_opened, unit="ns").normalize()
            held = [h for h in hist if pd.Timestamp(h.date).normalize() >= opened]
            covers = pd.Timestamp(hist[0].date).normalize() <= opened
            states[sym] = reconstruct_trail(float(pos.avg_px_open),
                                            [h.close for h in held], covers_entry=covers)
            prices[sym] = hist[-1].close
        if not states:
            return set()
        # ATR and highs from the same session history the trail was rebuilt from (#222); without
        # them an ATR-scaled rule raised here. `trailing_atr` wants a frame of ticker/date/high/low/
        # close, which `_hist` rows carry.
        atr = highs = None
        if needs_atr(self._cfg.exits) or needs_highs(self._cfg.exits):
            frame = pd.DataFrame([
                {"ticker": s2, "date": pd.Timestamp(h.date).normalize(), "high": float(h.high),
                 "low": float(h.low), "close": float(h.close)}
                for s2 in states for h in self._hist.get(s2, ())])
            if needs_atr(self._cfg.exits):
                atr = trailing_atr(frame)
            if needs_highs(self._cfg.exits):
                highs = {s2: float(self._hist[s2][-1].high) for s2 in states}
        return set(evaluate_exits(self._cfg.exits, prices, states, atr=atr, highs=highs).exits)

    # -- orders ------------------------------------------------------------------------------
    def _iid_for(self, symbol: str) -> InstrumentId | None:
        return next((i for i in self._iids if i.symbol.value == symbol), None)

    def _open(self, symbol: str, today: pd.DataFrame, weight: float) -> None:
        iid = self._iid_for(symbol)
        if iid is None:
            return
        px = float(today.loc[today.ticker == symbol, "close"].iloc[0])
        # Weights are computed across the WHOLE book, but only new entries are sized here — names
        # already held keep the size they were opened at. Multiplying the full deployment by a
        # fresh weight therefore double-counts capital already committed, and a low-vol name with a
        # large 1/sigma weight drove the account negative. Size against what is actually free.
        committed = sum(self._entry.get(s, 0.0) * self._sizes.get(s, 0)
                        for s in self._held if s != symbol)
        free = max(0.0, self._deployed - committed)
        qty = int(min(self._deployed * weight, free) / px)
        if qty < 1:
            return
        self._pending[symbol] = "enter"
        self._entry[symbol] = px
        atr = self._atr(symbol)
        if self._stop_mult and atr:
            self._stops[symbol] = px - self._stop_mult * atr
        self._sizes[symbol] = qty
        self.submit_order(self.order_factory.market(iid, OrderSide.BUY, Quantity.from_int(qty)))

    def _close(self, symbol: str, reason: str = "") -> None:
        iid = self._iid_for(symbol)
        if iid is None:
            return
        pos = next((p for p in self.cache.positions_open(strategy_id=self.id)
                    if p.instrument_id == iid), None)
        if pos is None:
            self._held.discard(symbol)
            self._stops.pop(symbol, None)
            self._sizes.pop(symbol, None)
            return
        # LONG ONLY, ASSERTED (#88). Research adapter, same exit route, same rule.
        qty = closing_quantity(pos.quantity, side=LONG)
        if qty is None:
            why = refusal_reason(pos.quantity, side=LONG)
            if why:
                self.log.error(f"{self.id}: {symbol} — {why}")
            self._held.discard(symbol)
            return
        self._pending[symbol] = "exit"
        self.submit_order(self.order_factory.market(iid, OrderSide.SELL, Quantity.from_int(qty)))

    def _flatten(self) -> None:
        for sym in list(self._held):
            if sym not in self._pending:
                self._close(sym, reason="eod")

    # -- order events are the only thing that moves book state --------------------------------
    def on_order_filled(self, event) -> None:
        # NOT EVERY EVENT THAT ARRIVES IS OURS (kumo-trading-platform issue 748). Nautilus routes order events by
        # the ORDER's strategy_id, so once cockpit stamps each protective stop with the lane whose
        # shares it covers, this lane receives `PROT-*` fills and cancels it never placed. Popping
        # `_pending` by SYMBOL alone read a protective SELL as this lane's entry completing.
        # A FOREIGN FILL CAN STILL HAVE CLOSED US (#133). `is_foreign` answers "did someone else
        # place this"; it does not answer "did our position change". A protective stop stamped with
        # this lane fills and the position is gone, whoever placed the order — so that case is
        # handled BEFORE the guard below, which exists for the other half (kumo-trading-platform issue 748): a
        # foreign fill must never mark a symbol held out of nothing.
        if protective_close(self, event):
            return
        if is_foreign(event):
            return
        sym = event.instrument_id.symbol.value
        intent = self._pending.get(sym)
        if not completes(intent, event, side=self.POSITION_SIDE):
            # NOT SILENT. `is_foreign` already returned above, so reaching here means an event that
            # is OURS could not be matched to what we are waiting for — an unreadable side, or a
            # side that cannot complete the intent. A soundless refusal on our own event is
            # indistinguishable from a clean poll, which is the shape two detectors here already
            # shipped with.
            self.log.warning(
                f"{self.id}: {event.instrument_id} fill does not complete pending intent "
                f"{intent!r} (side={fill_side(event)!r}); book state NOT moved"
            )
            return
        self._pending.pop(sym, None)
        if intent == "enter":
            self._held.add(sym)
        elif intent == "exit":
            self._held.discard(sym)
            self._stops.pop(sym, None)
            self._entry.pop(sym, None)

    def on_order_rejected(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)

    def on_order_denied(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)

    def on_order_canceled(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)


    def on_order_expired(self, event) -> None:
        """A DAY order unfilled at the close EXPIRES at the venue, and that is a terminal answer.

        THE FOURTH STATE, AND IT WAS SILENT (#165). Terminal rows came from filled / denied /
        rejected / canceled; every lane sends MARKET DAY orders, so an expiry is reachable by
        construction — a partial fill late in the session, or a market order that never completes.
        Nothing recorded it, so the journal read "still in flight" FOREVER: `attempts_for` saw no
        failure, `_resume` treated the symbol as accepted-but-unanswered and deliberately left it
        alone, and cockpit's readback — one terminal row per order — could not see the order end.

        Worse than a wrong row, which is why it is its own ticket: a wrong row can be argued with.

        `ok=False`, because an expiry is not a fill. Recording it as success would suppress the
        symbol's attempt count exactly as a fill does, and a name that never filled would look done
        for the session.
        """
        if is_foreign(event):
            return
        self._forget(event)
        self._record_terminal(event, ok=False, detail="expired at the close, unfilled")
    def _forget(self, event) -> None:
        sym = getattr(getattr(event, "instrument_id", None), "symbol", None)
        if sym is not None:
            self._pending.pop(sym.value, None)
