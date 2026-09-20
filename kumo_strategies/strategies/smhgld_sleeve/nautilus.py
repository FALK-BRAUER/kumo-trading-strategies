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
    report_wrong_sided_positions, protective_close, _sync_claim_to_book, SENT_BASIS,
    RegistrationMixin, session_outcome)
from kumo_strategies.runtime.nautilus.capabilities import EXECUTES_DELTAS, require
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin, bars_to_keep
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin
from kumo_strategies.strategies.smhgld_sleeve.engine import build_feature_panel
from kumo_strategies.strategies.smhgld_sleeve import (
    SmhGldSleeveConfig, live_config, rebalance_dates, weight_split)

#: The operator-facing lane name. NOT the research id, and NOT carrying a tag suffix — cockpit
#: allocates the `order_id_tag` and maps `EXTERNAL_ID` to it. A name like "BCTROT-003" here is what
#: once made tag 003 look claimed from this side and free from cockpit's.
STRATEGY_NAME = "SMHGLD"
EXTERNAL_ID = "SMHGLD"
# DERIVED, NOT TYPED (#219, #186). This read "a fixed 48/52 sleeve" as a literal while the config
# said 0.32 — and would have agreed by coincidence after the flip. The split is one function on the
# config; a label that restates it in its own words is the copy nobody checks.
STRATEGY_LABEL = (f"SMHGLD — a fixed {weight_split(live_config())} sleeve of semis and gold, "
                  f"rebalanced on drift")

#: !! ONE ALERT PER LANE, named. `override=True` reuses it, so there is only ever one pending — and
#: the NAME is what `on_stop` cancels. A lane that invents its own scheme leaves a timer running
#: after the operator stopped it.
SESSION_ALERT = "template_session_decide"

#: !! A live Nautilus Bar carries OHLCV and nothing else.
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


def warmup_bars_needed(cfg: SmhGldSleeveConfig) -> int:
    """!! MAX, never SUM. Every window is a TRAILING window ending at the same bar, so they overlap.
    QC345 summed them and reported 295 bars where 254 was right.

    This lane has no trailing feature at all -- it needs a prior close and proof both feeds print.
    Padding that number is not caution: every bar of warmup is a session held in cash, and 20
    unjustified bars cost 2.1 points of CAGR against the research before it was measured.

    `trend_sessions + 2` because the moving average is itself computed on a SHIFTED series: one bar
    for the shift, one so the first comparison has a defined average on both sides."""
    return max(cfg.warmup_sessions, 2)


class SmhGldSleeveStrategy(MarketAwareMixin, SlotRereadMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin,
                           Strategy):
    EXTERNAL_ID = EXTERNAL_ID
    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG
    LABEL = STRATEGY_LABEL
    #: Conforms like a lane so the suite can check it; must never be registered or traded.

    def __init__(
        self,
        cfg: SmhGldSleeveConfig,
        instrument_ids: list[InstrumentId] | None = None,
        *,
        # !! SYMBOLS, not ids (kumo-trading-platform issue 622). An `InstrumentId` demands a VENUE at construction,
        # and the venue is a runtime fact only a connected adapter can produce — so a lane that takes
        # ids can only be built by asking a vendor offline, which is what stopped an IBKR-only node
        # booting at all. Ids still work so a pinned caller keeps working; new lanes take symbols.
        symbols: list[str] | None = None,
        calendar: object | None = None,
        # !! `close-20m`, RULED, and the value is load-bearing (platform issue 965). This lane fills
        # MARKET-ON-CLOSE and a MOC has a CUTOFF — IB ~15:50 ET, Alpaca 15:45 ET, both DOCUMENTED
        # rather than measured. 15:40 clears them by +10m and +5m.
        #
        # `close-10m` was the first ruling and is 15:50: EXACTLY IB's cutoff and five minutes PAST
        # Alpaca's. Zero margin, and a rejection at 15:50 leaves no session to retry in — the same
        # mistake as `open-5m` against the OPG cutoff, at the other end of the day.
        #
        # It is also the slot BCTROT already runs, so the scheduler path is production-exercised
        # rather than new. `close-30m` was rejected: ten more minutes of margin costs twenty more
        # minutes of drift between the decision prices and the close, and on a 0.25pp band that
        # drift is the thing being traded away.
        decision_slots: tuple[str, ...] = ("close-20m",),
        # !! NAMED IN THE SIGNATURE, not swallowed by `**kwargs` (kumo-trading-platform issue 514). Cockpit passes
        # this only to adapters whose signature ACCEPTS it, because an unknown kwarg raises at build
        # and takes every other lane down — so forwarding it invisibly means the operator's schedule
        # edits silently never arrive. BCTROT shipped exactly that, with a green suite.
        read_slots: object | None = None,
        # !! WITHOUT THIS THE LANE CAN NEVER BE WARM INSIDE A SESSION. `on_start` subscribed to bars
        # and never requested any, and `subscribe_bars` does not backfill — read off the installed
        # Nautilus, not assumed: its signature is (bar_type, client_id, update_catalog, params),
        # with no start, no limit and no lookback of any kind. `request_bars` is the ONLY path to
        # history. On a 1-DAY feed the lane therefore accrued one bar per leg per SESSION against a
        # `_need` of `max(warmup_sessions, 2)`, so every rung returned at `not self.warm` writing
        # nothing, for days.
        #
        # NAMED IN THE SIGNATURE for the same #514 reason as `read_slots` directly above: cockpit
        # passes this only to adapters whose signature ACCEPTS it, so a lane that does not name it
        # silently never receives it. crsi_short, momentum, qc27 and qc345 all name it; this lane
        # and `template_rotation` did not, which is exactly why they are the two that cannot warm.
        history_days: int | None = None,
        order_id_tag: str,
        strategy_name: str = STRATEGY_NAME,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        session_runner: object | None = None,
        # !! THE ESCAPE HATCH IS NAMED AND BELONGS TO THE CALLER, exactly as CRSISHORT's does. A lane
        # that computes and publishes without trading is a legitimate thing to build; a lane that
        # trades nothing because its runner cannot execute its decisions is not.
        shadow_only: bool = False,
        # !! THE BUS TOPIC THE ACCOUNT SNAPSHOT ARRIVES ON (#208). Cockpit's to name, so an argument
        # rather than an import — this package keeps no dependency on cockpit. NAMED, like every
        # sibling (`crsi_short.py`, `momentum_rotation.py`), for the #514 reason `read_slots` and
        # `history_days` above give: cockpit passes a kwarg only to a signature that accepts it.
        # Cockpit passes none today and publishes on this default (`api/bus_topics.py`).
        account_topic: str = "broker.account",
    ) -> None:
        # !! `order_id_tag` is REQUIRED and has NO DEFAULT. Cockpit allocates it; a default that
        # happens to agree is not an allocation. A duplicate does not degrade — Nautilus raises at
        # `Trader.add_strategy` and the node does not boot, taking every other lane down with it.
        super().__init__(config=StrategyConfig(strategy_id=strategy_name,
                                               order_id_tag=order_id_tag))
        # !! THIS LANE DECIDES IN TARGETS AND DELTAS, AND EVERY RUNNER WE HAVE CONSUMES enter/exit.
        # Against one of those this lane registers, warms, decides every session, journals every
        # session — and executes NOTHING, while every surface reads healthy. Lab's own sentence:
        # "A runner consuming only `enter` and `exit` executes NOTHING here while the journal shows
        # a lane deciding every session."
        #
        # Refused at CONSTRUCTION, the `ORDER_PATH_COMPLETE` pattern: a lane that cannot act does not
        # exist. That is what stopped CRSISHORT being armed by accident.
        self._shadow_only = bool(shadow_only)
        if not shadow_only:
            require(
                session_runner, EXECUTES_DELTAS, lane=STRATEGY_NAME,
                needs=("This lane enters once and never exits: `order_plan()` emits "
                       "`target_qty - held_qty`, so an entry is a delta from zero and an exit is a "
                       "target of zero. It never produces `enter` or `exit` lists, and a runner "
                       "that consumes those will execute nothing while journalling a decision "
                       "every session (#177, platform issue 953)."))
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
        #: Rungs that fired and found nothing tradeable. Readable in-process, because
        #: `self.log` is Nautilus's logger and nothing can read it back.
        self.missed_sessions: list[pd.Timestamp] = []
        # !! ENOUGH BARS FOR THE VIEW, NOT ONLY FOR THE WARMUP (#212). `_need` is 3 here, and a
        # deque of `_need + 5` held 8 bars per leg against a 50-session average that needs 51 —
        # the declared view would have read UNKNOWN forever, on any history_days. Derived from the
        # config by `bars_to_keep`, never a literal.
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
        # !! A LANE THAT REQUESTS NO HISTORY CANNOT BE WARM INSIDE A SESSION, and says so ONCE,
        # loudly, rather than returning silently at every rung for days. This lane subscribed and
        # never requested: `subscribe_bars` is a FORWARD subscription with no lookback, so on a
        # 1-DAY feed it accrued one bar per leg per session against `_need` sessions.
        if not self._history_days:
            self.log.error(
                f"{self.id}: no history_days, so NO bar history will be requested. This lane needs "
                f"{self._need} sessions on BOTH legs {list(self._cfg.universe)} before it can "
                f"decide at all, and a live daily feed delivers one bar per leg per session — it "
                f"will publish nothing for roughly {self._need} sessions and every rung will return "
                f"silently. Pass history_days (>= {self._need} trading sessions of calendar days).")
        for iid in self._iids:
            # !! The instrument DEFINITION, not just its bars — `cache.instrument()` returning None
            # is a hard refusal at submit. Asked only when the cache does not already hold it: every
            # needless request spends an IBKR historical allowance four lanes share.
            self.request_instrument_if_missing(iid)
            bt = self._bar_type(iid)
            if self._history_days:
                # HISTORY BEFORE THE SUBSCRIPTION, the same order crsi_short.py:454 and
                # momentum_rotation.py:379 use.
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
        # !! THE HANDLER WAS DEFINED AND NEVER SUBSCRIBED (#208) — #174's sibling, one lane over.
        # `_on_broker_account` existed from this lane's first commit and nothing ever called it, so
        # `broker_equity()` returned None forever: not a sizing defect (this lane sizes from its
        # sleeve) but a preflight probe that could never pass, which is a probe nobody reads. The
        # class-level test #174 left behind listed its lanes by hand and this one was not on the
        # list; it discovers them now. Same call, same position as `crsi_short.py:505`.
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
        if fired is None:
            return
        if not self.warm:
            # !! A RUNG THAT FIRED AND COULD NOT DECIDE MUST SAY SO. This returned silently, so a
            # lane that fired six rungs and a lane that never fired at all were indistinguishable on
            # every surface — the single most expensive ambiguity in this system, and the reason the
            # missing history request survived a whole session. The reason names WHICH leg and the
            # exact counts, because "not warm" is not a diagnosis.
            self.log.warning(f"{self.id}: rung {fired} fired and this lane CANNOT DECIDE — "
                             f"{self._warmth()}. No decision was made and none was recorded.")
            return
        panel = self.panel()
        # !! THE SESSION'S OWN BAR DOES NOT EXIST WHILE THE SESSION IS OPEN (#207). Checked against
        # the REAL bars, before any placeholder row is added, or the guard measures its own invention.
        if self._refuse_a_stale_feed(fired, panel):
            return
        panel = self._with_a_row_for(fired, panel)
        if not self._is_rebalance(fired, panel):
            return
        # !! THE ADAPTER OWNS THE FEATURIZE AND THE DAY-SLICE, as every sibling does
        # (crsi_short.py:633, qc27_rotation.py:484, qc345_rotation.py:580). Handing `panel()` on
        # to a caller that must know to featurize is what killed every rung on 2026-09-11 with
        # `KeyError: 'eligible'`, and it is the asymmetry that made the mistake available.
        day = self._day_for(fired, panel)
        if day is None:
            return
        # !! FIRE AND REPORT. A bare `run_coroutine_threadsafe` discards the Future, so anything
        # raised inside the coroutine is captured there and never read — nothing logged, nothing
        # retried, and the write simply did not happen.
        if self._loop is not None:
            self._session_task = self.fire_and_report(
                self._session_coro(fired, day, fired_slot), self._loop, "session")

    async def _session_coro(self, session, day, slot) -> None:
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
            result = await self._runner.run(day, str(session.date()), slot=slot)
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
    def _with_a_row_for(self, session: pd.Timestamp, panel: pd.DataFrame) -> pd.DataFrame:
        """Ensure the panel carries a row for `session` per leg, so a decision can be taken WHILE
        that session is still open (#207).

        A DAILY BAR CANNOT EXIST AT ITS OWN OPEN — `momentum_rotation.py:537` says the same thing.
        IB serves a session's daily bar once that session COMPLETES; probed directly on 2026-09-12
        with the market shut, the newest bar was 20260911. So at a Monday 16:25Z rung the newest bar
        is Friday's, `== session` finds nothing, and the lane holds EVERY RUNG FOREVER. Measured:
        SMHGLD-007 was warm on 20 bars a leg and still held at 19:40Z, because 2026-09-11 was simply
        not in its panel.

        A PLACEHOLDER ROW RATHER THAN A DIFFERENT SLICE, and the difference is a whole day of
        staleness. The obvious repair is to slice to the newest session strictly before the decision
        (`panel.date < due`, momentum_rotation.py:523) — but `build_feature_panel` sets
        `asof_close = shift(1)` and `decide` reads prices from `asof_close` ALONE, so slicing to the
        PRIOR row yields the close BEFORE prior: two sessions stale, silently. The design assumes
        `decide` runs on the session's OWN row, whose `asof_close` is the last completed close.

        `close=NaN` INVENTS NO PRICE. The row asserts "deciding for session D, no bar for D yet",
        which is literally true, and `shift(1)` then makes `asof_close` the last COMPLETED close —
        exactly what the engine was designed to receive. `eligible` is satisfied because it tests
        `asof_close.notna()`, not `close`.

        IT MUST ALSO PRECEDE `_is_rebalance`, which reads the panel's dates: intraday the session is
        absent there too, so the lane returns before it ever reaches the slice. Adding the row does
        not disturb a non-daily cadence — `rebalance_dates` keeps the FIRST date per bucket, and a
        session that opens a new bucket genuinely is that bucket's first.
        """
        if panel.empty:
            return panel
        session = pd.Timestamp(session).normalize()
        dates = pd.to_datetime(panel["date"]).dt.normalize()
        legs = [leg for leg in self._cfg.universe
                if not ((panel["ticker"] == leg) & (dates == session)).any()]
        if not legs:
            return panel                       # the venue has served it; nothing to stand in for
        rows = [{"date": session, "ticker": leg, "open": float("nan"), "high": float("nan"),
                 "low": float("nan"), "close": float("nan"), "volume": 0.0} for leg in legs]
        return pd.concat([panel, pd.DataFrame(rows)], ignore_index=True)

    def _refuse_a_stale_feed(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """True when the newest REAL bar is too old to decide on. Refuses loudly and counts it.

        REQUIRED BY `_with_a_row_for`, NOT MERELY PRUDENT ALONGSIDE IT. Before #207 a stale feed
        produced an empty day-slice and the lane held with a named reason — loud, and safe. The
        placeholder row means the slice ALWAYS finds rows, so that signal is GONE, and without this
        the lane would decide on whatever the newest real bar happens to be, silently. That is the
        failure class #205 removed, re-entering through the fix for it. The two must ship together.

        MEASURED AGAINST THE REAL BARS ONLY, and this method runs BEFORE the placeholder is added:
        a guard that counted its own invented row would measure a gap of zero, always, and report
        health it manufactured. Same shape as `claims are circular` — a check must not read a value
        the checked thing produced.
        """
        if panel.empty:
            return False                       # nothing to be stale ABOUT; warmth already refused
        session = pd.Timestamp(session).normalize()
        newest = pd.to_datetime(panel["date"]).max()
        if pd.isna(newest):
            return False
        stale = (session - pd.Timestamp(newest).normalize()).days
        if stale <= self._cfg.max_stale_days:
            return False
        self.log.error(
            f"{self.id}: newest bar {pd.Timestamp(newest).date()} is {stale}d before session "
            f"{session.date()}, past max_stale_days={self._cfg.max_stale_days} — refusing to decide "
            f"on stale data. A gap this size is a data outage, not a holiday, and deciding would "
            f"trade prices this lane cannot see. The sleeve holds what it has.")
        self.missed_sessions.append(session)
        return True

    def _day_for(self, session: pd.Timestamp, panel: pd.DataFrame):
        """The FEATURIZED rows for THIS session, or None with a named hold.

        SLICED BY THE SESSION'S OWN DATE, never by `date.max()`, and the difference is the whole
        point. `decide()` takes `.iloc[0]` per leg, so on a frame holding the full window it silently
        decides on the OLDEST session — a plausible trade at month-old prices, with nothing saying
        so. `max()` looks like the safe version of the same slice and is not: on a stale feed it
        decides on whatever row happens to be newest, equally silently.

        `== session` turns that case into an EMPTY frame, and an empty frame is a HOLD WITH A REASON.
        That is what the three siblings do — `qc345_rotation.py:581`, `qc27_rotation.py:485`,
        `momentum_rotation.py:561` — and `qc345_rotation.py:593` is the hold this mirrors. The only
        `.max()` in any of them finds the PRIOR completed session, which is a different question.

        NOT A STALENESS CHECK. A hold here is not counted as a missed session the way
        `momentum_rotation.py:529` counts one past `max_stale_days`. Filed separately; with this
        slice the staleness case is already a refusal rather than a wrong trade.
        """
        try:
            featurized = build_feature_panel(panel, self._cfg)
        except Exception as exc:                                       # noqa: BLE001
            self.log.error(f"{self.id} {session.date()}: could not build the feature panel "
                           f"({type(exc).__name__}: {exc}) — holding")
            return None
        day = featurized.loc[featurized["date"] == pd.Timestamp(session)]
        if day.empty:
            newest = featurized["date"].max() if not featurized.empty else None
            self.log.warning(
                f"{self.id}: no feature rows for {session.date()} — holding. Newest row is "
                f"{newest.date() if newest is not None and pd.notna(newest) else 'none'}. The "
                f"decision is NOT taken on the newest available row: deciding on a stale session "
                f"would be a trade at prices this lane cannot see.")
            return None
        return day

    def _is_rebalance(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """!! Delegates to the PURE cadence function. A second adapter-local notion of 'first session
        of the month' is a second derivation of a fact the backtest already derives."""
        if panel.empty:
            return False
        return pd.Timestamp(session) in set(rebalance_dates(panel["date"],
                                                           self._cfg.rebalance_period))

    @property
    def warm(self) -> bool:
        """!! A cross-sectional strategy warms as a UNIVERSE — and this one is a two-asset
        switch, so BOTH legs must be warm, not `portfolio_size` of them. Warm on SMH alone and the
        lane can decide `defensive` with no GLD history to rotate into; warm on GLD alone and the
        gate it switches on does not exist. `all`, not a count."""
        want = set(self._cfg.universe)
        return all(len(self._bars.get(s, ())) >= self._need for s in want)

    def _warmth(self) -> str:
        """Per-leg bar counts against the requirement. Beside `warm` so the two read one source.

        A COUNT PER LEG, not a boolean. `warm` is `all(...)` over both legs, so a False says nothing
        about WHICH leg is short or by how much — and an operator at 3am needs the number, not the
        verdict. This is what turns six silent rungs into one readable line.
        """
        parts = []
        for sym in sorted(set(self._cfg.universe)):
            have = len(self._bars.get(sym, ()))
            parts.append(f"{sym} {have}/{self._need}" + ("" if have >= self._need else " SHORT"))
        return "bars " + ", ".join(parts)

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
        qty = self._claim_qty_after_event(sym, event, reason="fill")
        if qty is None:
            return
        self._pending.pop(sym, None)
        if qty == 0:
            self._held.discard(sym)
        else:
            self._held.add(sym)
        _sync_claim_to_book(self, sym, event, qty)

    #: The time-in-force that makes a rejection a MISSED REBALANCE rather than an ordinary refusal.
    _CLOSING_TIF = "AT_THE_CLOSE"

    def on_order_rejected(self, event) -> None:
        if is_foreign(event):
            return
        self._name_a_refused_close(event)
        self._forget(event)
        sym = str(event.instrument_id.symbol)
        qty = self._claim_qty_after_event(sym, event, reason="rejection")
        if qty is not None:
            _sync_claim_to_book(self, sym, event, qty)

    def _name_a_refused_close(self, event) -> None:
        """A REFUSED MOC AND A LANE THAT DECIDED NOTHING MUST BE DIFFERENT READINGS (platform issue 965).

        The decision slot clears both documented MOC cutoffs on a NORMAL session. On a half day the
        slot moves with the close — `close-20m` becomes 12:40 — and whether either venue shifts its
        cutoff by the same three hours is UNKNOWN. Roughly nine sessions a year, all low-volume, and
        at a 0.25pp band the next session catches the drift, so the cost is bounded and accepted.

        Missing it SILENTLY is not. Without this, a refused closing order is forgotten like any
        other and the lane shows a normal session while the sleeve sits unrebalanced.

        AN UNREADABLE TIME-IN-FORCE IS NOT A MOC REJECTION. Guessing would put a half-day incident
        in the log on an ordinary bad afternoon, and an alarm that cries wolf on a normal Tuesday is
        how the real one gets ignored. Three states, again.
        """
        sym = str(event.instrument_id.symbol)
        tif = getattr(getattr(event, "time_in_force", None), "name", None)
        reason = getattr(event, "reason", "") or "no reason given"
        if tif != self._CLOSING_TIF:
            self.log.warning(
                f"{self.id}: {sym} order rejected by the venue — {reason}")
            return
        self.log.error(
            f"{self.id}: {sym} MARKET-ON-CLOSE order REFUSED — {reason}. THE REBALANCE DID NOT "
            f"HAPPEN and this leg is unrebalanced until the next decision. A MOC has a cutoff (IB "
            f"~15:50 ET, Alpaca 15:45 ET) and this lane decides at close-20m, which clears both on "
            f"a normal session — so suspect a HALF DAY, where the slot moves with the close and the "
            f"venue cutoff may not (platform issue 965).")

    def on_order_denied(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)
        sym = str(event.instrument_id.symbol)
        qty = self._claim_qty_after_event(sym, event, reason="denial")
        if qty is not None:
            _sync_claim_to_book(self, sym, event, qty)

    def _forget(self, event) -> None:
        """!! A denied EXIT must leave the position HELD. Forgetting the intent and the holding makes
        a real position read as foreign, and nothing ever exits it."""
        sym = str(event.instrument_id.symbol)
        self._pending.pop(sym, None)

    def _claim_qty_after_event(self, sym: str, event, *, reason: str) -> int | None:
        """Read the lane's actual cache quantity after a terminal event.

        Claims follow the book, not the order plan. A local submit can be accepted while the venue
        later rejects the order, or while a restart replays only one filled leg; the cache quantity
        is the fact the ledger must mirror.
        """
        try:
            return int(sum(
                int(p.signed_qty)
                for p in self.cache.positions_open(
                    instrument_id=event.instrument_id,
                    strategy_id=self.id,
                )
            ))
        except Exception as exc:  # noqa: BLE001 — do not raise inside Nautilus event dispatch
            self.log.error(
                f"{self.id}: {sym} claim sync skipped after {reason}; could not read "
                f"the lane cache quantity ({type(exc).__name__}: {exc})"
            )
            return None

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
