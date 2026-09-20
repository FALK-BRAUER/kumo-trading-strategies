"""TEMPLATE Nautilus adapter — copy this to wire a new strategy into the engine.

IT CONFORMS ON DAY ONE, and that is the point. `tests/runtime/nautilus/test_contract.py` DISCOVERS
every adapter in this package and applies the full deployment contract to each, so this file is
checked by the same machinery the live lanes are. Copy it and the copy inherits every rule; diverge
and the suite says so before the strategy ever sees a market.

WHY A TEMPLATE AT ALL. kumo-trading-platform measured 19 decided slots since 2026-07-31 and 10 of the 17 live
ones bad — EIGHT OF THE TEN on 08-19..08-21, the three days three hand-built lanes were added.
MOMENTUM, the oldest, is the only one that trades reliably. Five strategies were each built from
whichever predecessor someone happened to read, producing four `run()` signatures, three journal
schemas, two ownership conventions, two names for equity and a contract class defined twice. None of
that was chosen.

EVERY COMMENT BELOW MARKED `!!` IS A DEFECT THAT REACHED PRODUCTION. Leave them in the copy until the
line they guard has been deliberately considered.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.slot_reread import SlotRereadMixin
from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.sides import LONG

import math

from collections import defaultdict, deque

import asyncio
import traceback
from asyncio import CancelledError as asyncio_CancelledError

import pandas as pd
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kumo_strategies.runtime.nautilus.contract import (
    report_wrong_sided_positions, protective_close, protective_close, SENT_BASIS, RegistrationMixin, session_outcome)
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin, bars_to_keep
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin
from kumo_strategies.strategies.template import TemplateConfig, rebalance_dates

#: The operator-facing lane name. NOT the research id, and NOT carrying a tag suffix — cockpit
#: allocates the `order_id_tag` and maps `EXTERNAL_ID` to it. A name like "BCTROT-003" here is what
#: once made tag 003 look claimed from this side and free from cockpit's.
STRATEGY_NAME = "TEMPLATE"
EXTERNAL_ID = "TEMPLATE"
STRATEGY_LABEL = "Template — copy me to start a new lane"

#: !! ONE ALERT PER LANE, named. `override=True` reuses it, so there is only ever one pending — and
#: the NAME is what `on_stop` cancels. A lane that invents its own scheme leaves a timer running
#: after the operator stopped it.
SESSION_ALERT = "template_session_decide"

#: !! A live Nautilus Bar carries OHLCV and nothing else.
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


def warmup_bars_needed(cfg: TemplateConfig) -> int:
    """!! MAX, never SUM. Every window is a TRAILING window ending at the same bar, so they overlap.
    QC345 summed them and reported 295 bars where 254 was right."""
    return max(cfg.lookback_sessions + 2, cfg.warmup_sessions)


class TemplateRotationStrategy(MarketAwareMixin, SlotRereadMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin, Strategy):
    EXTERNAL_ID = EXTERNAL_ID
    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG
    LABEL = STRATEGY_LABEL
    #: Conforms like a lane so the suite can check it; must never be registered or traded.
    IS_TEMPLATE = True

    def __init__(
        self,
        cfg: TemplateConfig,
        instrument_ids: list[InstrumentId] | None = None,
        *,
        # !! SYMBOLS, not ids (kumo-trading-platform issue 622). An `InstrumentId` demands a VENUE at construction,
        # and the venue is a runtime fact only a connected adapter can produce — so a lane that takes
        # ids can only be built by asking a vendor offline, which is what stopped an IBKR-only node
        # booting at all. Ids still work so a pinned caller keeps working; new lanes take symbols.
        symbols: list[str] | None = None,
        calendar: object | None = None,
        decision_slots: tuple[str, ...] = ("open+5m",),
        # !! NAMED IN THE SIGNATURE, not swallowed by `**kwargs` (kumo-trading-platform issue 514). Cockpit passes
        # this only to adapters whose signature ACCEPTS it, because an unknown kwarg raises at build
        # and takes every other lane down — so forwarding it invisibly means the operator's schedule
        # edits silently never arrive. BCTROT shipped exactly that, with a green suite.
        read_slots: object | None = None,
        # !! WITHOUT THIS NO LANE BUILT FROM THIS TEMPLATE CAN WARM INSIDE A SESSION. `on_start`
        # subscribed and never requested, and `subscribe_bars` does not backfill — its Nautilus
        # signature carries no start, no limit and no lookback. A TEMPLATE THAT OMITS THE HARD PART
        # TEACHES THAT THE HARD PART IS OPTIONAL, and it did: `crsi_short.py` names this file as
        # where its own omission came from, and `smhgld_sleeve` inherited the same gap and could not
        # decide on its first live session (#200). Named in the signature for the #514 reason.
        history_days: int | None = None,
        order_id_tag: str,
        strategy_name: str = STRATEGY_NAME,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        session_runner: object | None = None,
        # !! THE BUS TOPIC IS AN ARGUMENT, NOT AN IMPORT. It is cockpit's to name, and this
        # package keeps no dependency on cockpit. Hardcoding it works until cockpit renames
        # the topic, and then fails the way #174 did: silently, forever, with no error.
        account_topic: str = "broker.account",
    ) -> None:
        # !! `order_id_tag` is REQUIRED and has NO DEFAULT. Cockpit allocates it; a default that
        # happens to agree is not an allocation. A duplicate does not degrade — Nautilus raises at
        # `Trader.add_strategy` and the node does not boot, taking every other lane down with it.
        super().__init__(config=StrategyConfig(strategy_id=strategy_name,
                                               order_id_tag=order_id_tag))
        # !! Refuse a price field the live feed can NEVER supply, at CONSTRUCTION. Unguarded this does
        # not fail at startup — it fails at the first rebalance, weeks later, in front of nobody.
        if cfg.price_field not in _BAR_FIELDS:
            raise ValueError(
                f"price_field={cfg.price_field!r} cannot be supplied by a live Nautilus bar, which "
                f"carries only {sorted(_BAR_FIELDS)}. Set price_field='close' for the live config.")
        self._cfg = cfg
        self._init_symbols(instrument_ids, symbols)
        self._calendar = calendar
        self._slots = tuple(decision_slots)
        self._read_slots = read_slots
        self._history_days = history_days
        self._armed_session = None
        self._armed_slot = None
        self._loop = None
        self._session_task = None
        self._suffix = bar_type_suffix
        self._need = warmup_bars_needed(cfg)
        # !! ENOUGH BARS FOR THE DECLARED VIEW, not only for the warmup (#212): a view whose window
        # exceeds the warmup reads UNKNOWN forever behind a `_need + 5` cap. Derived, never a literal.
        self._bars: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=bars_to_keep(self._need, self._cfg)))
        # !! Book state moves on ORDER EVENTS only, never because submit was called. A rejected entry
        # marked as held becomes a phantom the strategy then refuses to re-enter.
        self._held: set[str] = set()
        self._pending: dict[str, str] = {}
        # !! The runner is the safety model. Without one, `_try_decide` would decide and submit here,
        # bypassing lifecycle, journal, idempotency, risk and budget entirely.
        self._runner = session_runner
        self._account_topic = account_topic
        self._broker_account: dict | None = None

    # -- lifecycle -------------------------------------------------------------------------------
    # !! THE SESSION PATH IS THE POINT OF THIS FILE (#98). It was missing, and the omission is what
    # this template exists to prevent: three lanes each read a different predecessor and produced
    # three answers — momentum found its journal by an accidental fallback, qc345 looked for it under
    # a name cockpit does not use and wrote NOTHING for its entire life, qc27 wrote no outcome row at
    # all (kumo-trading-platform issue 587). A template that omits the hard part teaches that the hard part is
    # optional.

    def on_start(self) -> None:
        # A RESTART IS THE NORMAL CASE, and `_held` is built empty. Until it is seeded, every
        # position predating this boot is invisible to `protective_close`, so the venue's answer
        # to a protective stop-out goes unrecorded and the claim outlives the position (#194).
        self.seed_held_from_positions()
        # !! FIRST, before anything reads `_iids`. `self.cache` is a read-only Cython attribute on
        # Actor, so the guard has to be at the CALL SITE — merely evaluating it raises on a host
        # without one.
        self._resolve_symbols_if_needed()
        # !! A LANE THAT REQUESTS NO HISTORY CANNOT BE WARM INSIDE A SESSION, and says so ONCE.
        if not self._history_days:
            self.log.error(
                f"{self.id}: no history_days, so NO bar history will be requested. This lane needs "
                f"{self._need} sessions per name before it can decide and a live daily feed "
                f"delivers one bar per name per session — it will publish nothing for roughly "
                f"{self._need} sessions. Pass history_days (>= {self._need} sessions of days).")
        for iid in self._iids:
            # !! The instrument DEFINITION, not just its bars — `cache.instrument()` returning None
            # is a hard refusal at submit. Asked only when the cache does not already hold it: every
            # needless request spends an IBKR historical allowance four lanes share.
            self.request_instrument_if_missing(iid)
            bt = self._bar_type(iid)
            if self._history_days:
                self.request_bars(bt, self.clock.utc_now()
                                  - pd.Timedelta(days=self._history_days))
            self.subscribe_bars(bt)
        # !! CAPTURE THE LOOP HERE. `LiveClock` fires timer callbacks from its OWN thread, where
        # `asyncio.get_running_loop()` raises — so a session coroutine scheduled from the alert has
        # nowhere to go unless the loop was captured while still on it.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        # !! SUBSCRIBE THE HANDLER YOU DEFINED. `_on_broker_account` below is dead weight without
        # this line, and its absence is invisible: `broker_equity()` simply returns None forever and
        # nothing errors. THIS TEMPLATE OMITTED IT AND CRSISHORT INHERITED THE OMISSION (#174) —
        # the second defect copied from this file, after the `request_bars` warmup it also lacked.
        # A template that omits the hard part teaches that the hard part is optional.
        self.msgbus.subscribe(self._account_topic, self._on_broker_account)
        # !! ARM WITHOUT BLOCKING AND WITHOUT TAKING THE NODE DOWN. A calendar that cannot answer
        # used to raise out of `on_start`, and Nautilus re-raised it from `Trader.START` — so one
        # dead socket stopped every other lane that was queued behind it.
        self.begin_arming()

    def on_stop(self) -> None:
        # !! CANCEL BOTH: the arming retry AND the alert. `cancel_arming` also clears the re-arm
        # timer, which is a clock timer rather than a task and outlives `on_stop` otherwise.
        self.cancel_arming()
        if SESSION_ALERT in self.clock.timer_names:
            self.clock.cancel_timer(SESSION_ALERT)

    def _arm(self, after) -> None:
        """!! Arm the NEXT decision. Re-read the slots first so an operator edit takes effect without
        a redeploy — a schedule captured at construction is a setting that silently does nothing."""
        from kumo_strategies.runtime.calendar import next_slot_fire

        self._reread_slots()
        session, slot, fire_at = next_slot_fire(self._calendar, after, self._slots)
        self._armed_session, self._armed_slot = pd.Timestamp(session), slot
        self.clock.set_time_alert(SESSION_ALERT, fire_at, self._on_session_alert, override=True)

    def _on_session_alert(self, event=None) -> None:
        fired, self._armed_session = self._armed_session, None
        fired_slot, self._armed_slot = self._armed_slot, None
        # !! RE-ARM FIRST, and through the GUARDED path. Everything below may raise, and a lane that
        # has stopped scheduling looks exactly like one that decided to hold. `rearm_after_alert`
        # survives a calendar that cannot see far enough yet — a venue calendar knows a rolling
        # window, so refusing is normal and must not end the schedule.
        self.rearm_after_alert(self.clock.utc_now())
        if fired is None or not self.warm:
            return
        panel = self.panel()
        if not self._is_rebalance(fired, panel):
            return
        # !! FIRE AND REPORT. A bare `run_coroutine_threadsafe` discards the Future, so anything
        # raised inside the coroutine is captured there and never read — nothing logged, nothing
        # retried, and the write simply did not happen.
        if self._loop is not None:
            self._session_task = self.fire_and_report(
                self._session_coro(fired, panel, fired_slot), self._loop, "session")

    async def _session_coro(self, session, panel, slot) -> None:
        """!! THE RUNNER OWNS SAFETY. Lifecycle, journal, idempotency, risk and budget all live in
        the cockpit-side gateway; deciding here directly bypasses every one of them."""
        try:
            # A POSITION ON THE SIDE THIS LANE CANNOT MANAGE IS REPORTED EVERY SESSION (#66).
            # The exit path already REFUSES one, but only when the lane is trying to exit that
            # name — WHD sat in the book for twelve hours because nobody was. Before the
            # decision, because a book we do not understand is context for what follows.
            report_wrong_sided_positions(self, session=str(session.date()))
            if self._runner is None:
                return
            result = await self._runner.run(panel, str(session.date()), slot=slot)
            submitted = getattr(result, "submitted", 0)
            outcome = session_outcome(
                getattr(result, "state", "UNKNOWN"),
                decided=bool(getattr(result, "decided", False)),
                blocked=getattr(result, "blocked", None), sent=submitted)
            self.log.info(f"{self.id} {session.date()}: {outcome}")
            # !! ONE OUTCOME ROW PER SESSION, decided or not (kumo-trading-platform issue 587). Without it "ran and
            # declined" and "never ran" are indistinguishable in the only record an operator reads —
            # the single most expensive ambiguity in this system. `session_journal()` finds the
            # journal wherever the caller keeps it; reaching for the attribute by name is how one
            # lane wrote nothing for its whole life.
            journal = self.session_journal()
            if journal is not None:
                await journal.write(
                    "state", f"session outcome — {outcome}", session=str(session.date()), slot=slot,
                    detail={"state": getattr(result, "state", "UNKNOWN"),
                            "decided": bool(getattr(result, "decided", False)),
                            "blocked": getattr(result, "blocked", None),
                            "submitted": submitted, "submitted_basis": SENT_BASIS,
                            "entered": list(getattr(result, "entered", ()) or ()),
                            "exited": list(getattr(result, "exited", ()) or ())})
        # !! CancelledError BEFORE Exception, always. On 3.8+ it is a BaseException, but a bare
        # `except Exception` ordering mistake here swallows a shutdown and the task never ends.
        except asyncio_CancelledError:
            raise
        except Exception as exc:                                       # noqa: BLE001
            # !! TRACEBACK, ALWAYS. Logging only `type(exc).__name__: exc` produced "TypeError:
            # 'NoneType' object is not callable" with no file, no line and no frame — unlocatable.
            # Nautilus loggers accept one message string, not Python logging kwargs.
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.log.error(f"{self.id}: session {session.date()} failed: "
                           f"{type(exc).__name__}: {exc}\n{trace}")

    def _bar_type(self, iid: InstrumentId):
        from nautilus_trader.model.data import BarType

        return BarType.from_str(f"{iid}{self._suffix}")

    # -- data ------------------------------------------------------------------------------------
    def on_historical_data(self, data) -> None:
        """A REQUESTED batch arrives HERE, not on `on_bar` — and `_bars` is filled by `_ingest`.

        `request_bars` and `subscribe_bars` deliver on DIFFERENT handlers. ks#200 wired the request
        and not the receipt, so on ibkr-paper the batch arrived, the strategy logged receiving it, and
        `_bars` stayed empty:

            18:24:24  RequestBars(SMH.XNAS-1-DAY, start 2026-08-12)
            18:24:24  SMH.XNAS: Number of bars retrieved in batch: 21
            18:24:24  SMHGLD: Received <Bar[20]> data for SMH.XNAS-1-DAY
            18:40:00  SMHGLD-007: rung fired and this lane CANNOT DECIDE —
                      bars GLD 0/3 SHORT, SMH 0/3 SHORT

        Twenty bars per leg delivered, zero counted. The four lanes that warm all implement this;
        the two that could not warm did not. Wiring one end is half a fix — third instance today.

        `isinstance(data, Bar)` because this handler receives every requested data TYPE, and
        `_ingest` reads bar attributes: an instrument definition or a tick arriving here would raise
        into Nautilus's dispatch rather than being ignored.
        """
        if isinstance(data, Bar):
            self._ingest(data)

    def on_bar(self, bar: Bar) -> None:
        self._ingest(bar)

    def _ingest(self, bar) -> None:
        sym = str(bar.bar_type.instrument_id.symbol)
        row = {"date": pd.Timestamp(bar.ts_event, tz="UTC").normalize().tz_localize(None),
               "ticker": sym, "open": float(bar.open), "high": float(bar.high),
               "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume)}
        dq = self._bars[sym]
        # !! Alpaca REPUBLISHES the current session's daily bar as it updates. Appending every arrival
        # put ~103 copies of one date per symbol after thirteen hours, which gave the trailing window
        # zero variance, a NaN score, and a strategy that sat inert.
        if dq and dq[-1]["date"] == row["date"]:
            dq[-1] = row
        else:
            dq.append(row)

    # -- decision --------------------------------------------------------------------------------
    def _is_rebalance(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """!! Delegates to the PURE cadence function. A second adapter-local notion of 'first session
        of the month' is a second derivation of a fact the backtest already derives."""
        if panel.empty:
            return False
        return pd.Timestamp(session) in set(rebalance_dates(panel["date"],
                                                           self._cfg.rebalance_period))

    @property
    def warm(self) -> bool:
        """!! A cross-sectional strategy warms as a UNIVERSE."""
        ready = sum(1 for v in self._bars.values() if len(v) >= self._need)
        return ready >= self._cfg.portfolio_size

    def panel(self) -> pd.DataFrame:
        rows = [r for dq in self._bars.values() for r in dq]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    # -- book state, from ORDER EVENTS only --------------------------------------------------------
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
        sym = str(event.instrument_id.symbol)
        intent = self._pending.get(sym)
        # The `else` branch below ADDS to `_held` for any non-exit fill, so an unreadable or
        # unexpected side would mark a symbol held out of nothing.
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
        if intent == "exit":
            self._held.discard(sym)
        else:
            self._held.add(sym)

    def on_order_rejected(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)

    def on_order_denied(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)

    def on_order_canceled(self, event) -> None:
        # !! A cancel is a terminal venue answer and a FOREIGN one is ROUTINE: cockpit cancels and
        # re-arms protective stops on a 60s reconciler tick. Without the `is_foreign` guard,
        # `_forget` clears this lane's in-flight marker for an order that is still working.
        if is_foreign(event):
            return
        self._forget(event)

    def on_order_expired(self, event) -> None:
        # !! THE FOURTH STATE, AND THE ONE A NEW ADAPTER FORGETS (#165). Every lane sends MARKET
        # DAY orders, and a DAY order unfilled at the close EXPIRES at the venue. Handle it or the
        # journal reads "still in flight" forever: nothing counts it as a failure, nothing retries
        # it, and a per-order readback never sees the order end.
        #
        # THIS TEMPLATE IS WHAT GETS COPIED, so a gap here is a gap generator. The four live
        # rotation adapters were missing this handler; they were found by a scan that enumerates
        # the package rather than a list, and so was this file.
        if is_foreign(event):
            return
        self._forget(event)

    def _forget(self, event) -> None:
        """!! A denied EXIT must leave the position HELD. Forgetting the intent and the holding makes
        a real position read as foreign, and nothing ever exits it."""
        sym = str(event.instrument_id.symbol)
        self._pending.pop(sym, None)

    def _on_broker_account(self, snapshot: dict) -> None:
        self._broker_account = snapshot

    def broker_equity(self) -> float | None:
        """!! A METHOD, not a @property. `NautilusBroker.equity()` is
        `return self.strategy.broker_equity()` and assumes a method; as a property it evaluates first
        and the `()` lands on its result, so a legitimate None becomes `None()` and raises.

        !! The key is `equity`. Cockpit reads Alpaca's `portfolio_value` and REPUBLISHES it under the
        name `equity`; there is no `portfolio_value` key on the message. Reading that name returns
        None every time and every entry sizes to zero shares."""
        if not self._broker_account:
            return None
        v = self._broker_account.get("equity")
        if v is None:
            return None
        # NON-FINITE IS NOT A NUMBER. A nan passes `is not None` and `float()` happily, then
        # disarms every comparison downstream -- the daily-loss halt included, which fails by
        # never halting. None is already a legal return here and callers handle it.
        out = float(v)
        return out if math.isfinite(out) else None
