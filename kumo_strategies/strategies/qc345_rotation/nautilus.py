"""Nautilus binding for the QC #345 monthly rotation. The SAME decision code runs here and in backtest.

This class contains no strategy logic. It collects bars, asks `engine.decide()` what to hold, and turns
the answer into orders. Every rule lives in the pure layer, so there is nothing to reconcile between
research and production — the reason the split exists at all.

Modelled on `momentum_rotation.py`, which carries the scars this file inherits rather than re-earns:
warmup refusal, order-event-driven book state, a calendar alert rather than a bar-rollover trigger, and
a captured event loop because `LiveClock` fires callbacks off its own thread.

Identity: `StrategyConfig(strategy_id="QC345", order_id_tag="003")` -> `QC345-003`. Set it this way and
NOT via `change_id()`, or Nautilus rewrites the tag at registration and desyncs client_order_id
generation from the position id.

The tag is 003 because `order_id_tag` must be UNIQUE across every strategy in one trader — it is the
suffix of the StrategyId and of every generated client order id. MANUAL holds 001 and has live positions
keyed to `MANUAL-001`; MOMENTUM holds 002. Registering this one as 001 does not degrade, it stops the
node booting with "order_id_tag conflict for '001'".

Monthly cadence
---------------
The alert fires per SESSION, exactly as MOMENTUM's does, and the monthly rule is applied on top by
asking the pure `rebalance_dates()` whether the due session is a month's first. That is deliberate: a
second, adapter-local notion of "is it rebalance day" would be a second derivation of a fact the
backtest already derives, and two derivations of one fact disagree — the failure this codebase has hit
repeatedly. Reusing the function means a change to the cadence rule moves research and production
together or not at all.

Firing per session and skipping non-rebalance days also keeps the strategy responsive to a mid-month
restart: it re-arms every session, so it can never be more than one session away from noticing the next
rebalance, whereas a month-long timer armed once would be blind until it expired.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin

import math

import asyncio
import traceback
from asyncio import CancelledError as asyncio_CancelledError
from collections import defaultdict, deque

import pandas as pd
from kumo_strategies.runtime.nautilus.sides import (
    LONG, closing_quantity, refusal_reason)
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies.momentum_rotation.engine import trailing_atr
from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.runtime.nautilus.contract import (
    report_wrong_sided_positions, protective_close, SENT_BASIS, RegistrationMixin, session_outcome,
    call_record_terminal, terminal_fields)
from kumo_strategies.strategies.momentum_rotation.exits import (
    evaluate_exits,
    needs_atr,
    needs_highs,
    reconstruct_trail,
)
from kumo_strategies.strategies.qc345_rotation import (
    QC345RotationConfig,
    build_feature_panel,
    decide,
    rebalance_dates,
)

STRATEGY_NAME = "QC345"
EXTERNAL_ID = "QC345"
STRATEGY_LABEL = "QC345 monthly top-down rotation"

#: Cockpit allocates the `order_id_tag` — it is the only place that sees every strategy in ONE
#: trader and it is what calls `Trader.add_strategy`. This default exists so an existing deployment
#: keeps the id it already has; under NETTING the position id is `{instrument}-{strategy_id}`, so a
#: tag that moves after anything has traded orphans real positions. Callers SHOULD pass it.
DEFAULT_ORDER_ID_TAG = "003"
STRATEGY_TAG = DEFAULT_ORDER_ID_TAG  # deprecated alias; cockpit owns this
SESSION_ALERT = "qc345_session_decide"


def _exit_rules_configured(cfg: QC345RotationConfig) -> bool:
    return any(value is not None for value in cfg.exits.__dict__.values())


def warmup_bars_needed(cfg: QC345RotationConfig) -> int:
    """Bars per symbol before any decision is trustworthy.

    Momentum is `close.shift(1) / close.shift(lookback + 1)`, so it needs `lookback + 2` observations
    before it is non-NaN — the +2 is the two shifts, not a safety margin.

    MAX, not sum. Every window here is a TRAILING window ending at the same bar, so they overlap: the
    41 sessions the split guard inspects are the most recent 41 of the 254 momentum already needs. An
    earlier version of this function summed them and reported 295, overstating the requirement by two
    months. Measured against `build_feature_panel` on a synthetic panel, the first bar at which any name
    becomes eligible is 254 — exactly `max(lookback + 2, corporate_action_window + 1)`.

    Liquidity and realized-vol windows are shorter in every shipped config but are included anyway,
    because a config is a thing an operator can change and a warmup that silently under-counts is how a
    universe of one gets ranked.

    Deciding while short of this does not fail loudly — `eligible` simply goes empty or near-empty and
    the strategy "chooses" to hold almost nothing. That is why warmup is a refusal with a log line, not
    a filter.
    """
    return max(
        cfg.lookback_sessions + 2,
        cfg.corporate_action_window + 1,
        cfg.liquidity_window + 1,
        cfg.realized_vol_window + 1,
    )


#: A live Nautilus `Bar` carries OHLCV and NOTHING ELSE. Added 2026-08-22 after the deployment
#: conformance suite found QC345 was the only lane with a price-field setting and no guard on it —
#: and that its own DEFAULT (`close_split_dividend`) is a value no bar can supply, so the default
#: config cannot run live at all. It works today only because cockpit overrides to "close".
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


class QC345RotationStrategy(MarketAwareMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin, Strategy):
    EXTERNAL_ID = 'QC345'
    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG
    LABEL = 'QC345 monthly top-down rotation'

    """Monthly top-down rotation. Logic lives in the pure engine, not here."""

    def __init__(
        self,
        cfg: QC345RotationConfig,
        source: CandidateSource,
        instrument_ids: list[InstrumentId] | None = None,
        *,
        # KEYWORD-ONLY, DELIBERATELY. Inserted as a positional parameter this silently shifts
        # every argument after it: a caller passing `(cfg, source, ids, suffix)` would bind the
        # suffix to `symbols`, and `_init_symbols` would then see both and refuse at BUILD — which
        # takes every other lane in the trader down with it (the #377 shape). Two repos share this
        # signature across a version pin, so a positional insert is not a change either side can
        # make safely.
        symbols: list[str] | None = None,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        equity_per_position: float = 1_000.0,
        history_days: int | None = None,
        calendar: object | None = None,
        open_offset_minutes: int = 5,
        read_open_offset: object | None = None,
        session_runner: object | None = None,
        account_topic: str = "broker.account",
        rebalance_override_topic: str = "qc345.rebalance_override",
        external_order_claims: list[InstrumentId] | None = None,
        order_id_tag: str = DEFAULT_ORDER_ID_TAG,
    ) -> None:
        # `external_order_claims` are EXCLUSIVE across the node: `register_external_order_claims` raises
        # InvalidConfiguration if two strategies claim one instrument, during `Trader.add_strategy`. So an
        # over-broad claim here does not degrade, it stops the node booting — and QC345 shares its
        # universe with MOMENTUM by construction. The caller decides, because only it knows what the
        # other strategies hold.
        super().__init__(config=StrategyConfig(
            strategy_id=STRATEGY_NAME, order_id_tag=order_id_tag,
            external_order_claims=list(external_order_claims) if external_order_claims else None))
        # Kept for `claimed_instruments` (kumo-trading-platform issue 39): cockpit needs to read back what
        # this strategy claims, and Nautilus does not expose it once configured.
        self._claims = list(external_order_claims or [])
        # REFUSE AT CONSTRUCTION a price field the live feed can NEVER produce. Unguarded this does
        # not fail at startup — it fails at the FIRST REBALANCE, weeks later, in front of nobody,
        # which is the shape of this strategy's own `KeyError: 'eligible'` on its first live session.
        # A node that will not boot, in front of whoever is deploying, is strictly better.
        if cfg.momentum_price_field not in _BAR_FIELDS:
            raise ValueError(
                f"momentum_price_field={cfg.momentum_price_field!r} cannot be supplied by a live "
                f"Nautilus bar, which carries only {sorted(_BAR_FIELDS)}. The strategy would run "
                f"until its first rebalance and then raise. Set momentum_price_field='close' for the "
                f"live config — QC345 pairs it with a corporate_action_window to neutralise splits.")
        self._cfg = cfg
        self._source = source
        # SYMBOLS OR IDS, EXACTLY ONE (kumo-trading-platform issue 622). See `symbol_resolution`.
        self._init_symbols(instrument_ids, symbols)
        self._suffix = bar_type_suffix
        self._equity = equity_per_position
        # Live only. A backtest catalog is preloaded and pushed through the subscription, so there is
        # nothing to request and no data client to request it from.
        self._history_days = history_days

        self._need = warmup_bars_needed(cfg)
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._need + 5))

        # Book state, driven by ORDER EVENTS — never by the fact that submit was called. A submit that
        # is rejected, denied or partially filled leaves this untouched, which is the point.
        self._held: set[str] = set()
        self._pending: dict[str, str] = {}          # symbol -> "enter" | "exit"

        if calendar is None:
            # `runtime.calendar`, NOT `runtime.executor.calendar` — the latter's package __init__ imports
            # SQLAlchemy, which has no business in a backtest.
            from kumo_strategies.runtime.calendar import build_calendar
            calendar = build_calendar()
        self._calendar = calendar
        self._open_offset = open_offset_minutes
        self._read_open_offset = read_open_offset
        self._armed_session: pd.Timestamp | None = None   # session the pending alert belongs to
        self._due: pd.Timestamp | None = None             # rebalance session we are trading now
        self.missed_rebalances: list[pd.Timestamp] = []   # rebalance days that found no tradeable data
        self.skipped_sessions = 0                         # non-rebalance sessions, for observability

        # Operator override: treat a session as a rebalance even though `rebalance_dates()` (the
        # monthly rule) says it is not. Only ADDS days, never removes -- there is no live case for
        # un-forcing a rebalance that has not fired yet, and a strategy that finds itself needing to
        # take one back has a bigger problem than this set. `_is_rebalance` stays a single derivation
        # of "is this a rebalance" -- this is a second, DISJOINT set of dates to also treat as one,
        # not a second rule that could disagree with `rebalance_dates()` about a date it already
        # covers.
        #
        # Seeded from `cfg.forced_rebalance_dates`, not a separate constructor arg -- `cfg` is a
        # settings-domain dataclass cockpit already persists and rebuilds on every process start, so
        # reading it here means the set is restored BEFORE the first `_arm()` on every restart. A
        # msgbus-only override (below) is fire-and-forget: nothing replays it, so a container
        # recreate between the operator's publish and this strategy's next arm loses it silently --
        # exactly the trap that cost BCTROT its 2026-08-19 session (a schedule resolved once at
        # on_start() and never rechecked). The topic stays live for edits that should not require a
        # restart; the config field is what survives one.
        self._forced_rebalance: set[pd.Timestamp] = {
            pd.Timestamp(d).normalize() for d in cfg.forced_rebalance_dates}
        self._rebalance_override_topic = rebalance_override_topic

        self._runner = session_runner
        self._session_task = None
        # Captured in on_start. `LiveClock` fires timer callbacks from its OWN thread, so
        # `asyncio.get_running_loop()` inside a callback raises RuntimeError — and treating that as
        # "not a live node" is how a strategy once started, subscribed, logged that it was watching,
        # and then silently never ran a session.
        self._loop = None
        self._account_topic = account_topic
        self._broker_account: dict | None = None

    # -- lifecycle -------------------------------------------------------------------------------
    def on_start(self) -> None:
        # A RESTART IS THE NORMAL CASE, and `_held` is built empty. Until it is seeded, every
        # position predating this boot is invisible to `protective_close`, so the venue's answer
        # to a protective stop-out goes unrecorded and the claim outlives the position (#194).
        self.seed_held_from_positions()
        self._resolve_symbols_if_needed()
        for iid in self._iids:
            bt = self._bar_type(iid)
            if self._history_days:
                # The instrument DEFINITION, not just its bars. `cache.instrument()` returning None is a
                # hard refusal at submit — every entry priced and journalled, then rejected with "no
                # instrument definition cached". Bars alone do not populate it.
                # ALREADY CACHED IS THE COMMON CASE since #622 — the adapter pushed every
                # instrument into the Cache during connect, and symbol resolution read them from
                # there. Asking again costs a request against an IBKR budget of ~60 per 10 minutes
                # that four lanes share, and the overflow comes back as an empty array with no
                # error (kumo-trading-platform issue 617). The definition is still requested when it is absent.
                self.request_instrument_if_missing(iid)
                start = self.clock.utc_now() - pd.Timedelta(days=self._history_days)
                self.request_bars(bt, start)
            self.subscribe_bars(bt)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None      # backtest: no loop, and none needed
        self.msgbus.subscribe(self._account_topic, self._on_broker_account)
        self.msgbus.subscribe(self._rebalance_override_topic, self._on_rebalance_override)
        # Deferred, and OFF the loop — `_arm` reaches the calendar over a blocking socket. See
        # `RegistrationMixin.begin_arming`.
        self.begin_arming()
        self.log.info(
            f"{STRATEGY_NAME}-{STRATEGY_TAG} watching {len(self._iids)} instruments, warmup needs "
            f"{self._need} bars, next session {self._first_session_note()}, "
            f"portfolio_size={self._cfg.portfolio_size}"
        )

    def on_stop(self) -> None:
        self.cancel_arming()
        if SESSION_ALERT in self.clock.timer_names:
            self.clock.cancel_timer(SESSION_ALERT)
        # Cancelling the alert is not enough. A session already in flight is a raw asyncio task, not
        # something Nautilus manages, so it outlives stop() and keeps running against the lifecycle it
        # read when it STARTED — an operator who pauses mid-session would watch it submit anyway.
        self._due = None
        if self._session_task is not None and not self._session_task.done():
            self._session_task.cancel()
        self._session_task = None

    # -- session clock ---------------------------------------------------------------------------
    #: The slot the session currently being decided FIRED at. A class-level default because a
    #: restart mid-flight can reach the decide path before any alert has fired in this process --
    #: the same reason `qc27_runner._slot` carries one. `_on_session_alert` overwrites it per
    #: session, before re-arming.
    _fired_slot: str | None = None

    @property
    def _slot_name(self) -> str:
        """The slot name DERIVED from the offset this lane fires at — one source, not two.

        The journal column and the alert time were two independent facts: cockpit resolved
        `*_SLOTS` into `open_offset_minutes` and passed that, while the runner filed every row under
        its own module constant. Measured live 2026-08-22 — settings said `open+315m`, the lane
        fired at open+315m, and the journal said `open+150m` for every row.

        Deriving the name from the offset makes them one derivation, so they cannot disagree. The
        inverse of what cockpit already did to get the offset, which is why it round-trips.
        """
        return f"open+{int(self._open_offset)}m"

    #: Optional `() -> int | None`, re-read at every `_arm()`. THE SETTINGS KNOB WAS DEAD WITHOUT IT
    #: (kumo-trading-platform issue 514): cockpit resolved `*_SLOTS` once at build and passed a VALUE, this captured
    #: it, and `set_time_alert(..., override=True)` re-armed from the boot-time copy forever. An
    #: operator could edit the slot, see it accepted, and watch the lane fire at the old time until
    #: the next redeploy. Same dead-knob class as `qc27.ALLOCATED_EQUITY`.
    #:
    #: A CALLABLE beside the captured value, exactly as `qc27_runner.read_state` sits beside the
    #: frozen `lifecycle` value object -- the same defect and the same shape of fix.
    #:
    #: MUST BE SYNC AND CHEAP. `_arm` runs on the Nautilus clock callback; a DB round-trip here
    #: blocks the event loop. Cockpit passes a cached read.
    #:
    #: `None` is today's behaviour byte-for-byte, which is what every backtest and test passes.

    def _reread_open_offset(self) -> None:
        """Refresh `_open_offset` from settings, or keep what we have.

        NEVER LETS A SETTINGS READ STOP THE LANE SCHEDULING. A lane that stopped arming looks exactly
        like one that decided to hold -- the reason `_on_session_alert` re-arms first. So a raising
        reader, a None, or a non-positive value all leave the current offset in place and log; they
        do not raise out of `_arm` and they do not adopt a value that would move the fire time to
        somewhere unresolvable.
        """
        if self._read_open_offset is None:
            return
        try:
            got = self._read_open_offset()
        except Exception as exc:                                       # noqa: BLE001
            self.log.warning(f"{self.id}: could not re-read the decision slot ({exc}) — "
                             f"keeping open+{int(self._open_offset)}m")
            return
        if got is None:
            return
        try:
            offset = int(got)
        except (TypeError, ValueError):
            self.log.warning(f"{self.id}: settings gave a non-numeric decision offset {got!r} — "
                             f"keeping open+{int(self._open_offset)}m")
            return
        if offset < 0:
            self.log.warning(f"{self.id}: settings gave a negative decision offset {offset} — "
                             f"keeping open+{int(self._open_offset)}m")
            return
        if offset != self._open_offset:
            self.log.info(f"{self.id}: decision slot moved open+{int(self._open_offset)}m -> "
                          f"open+{offset}m by settings")
            self._open_offset = offset

    def _arm(self, after) -> None:
        """Arm the next SESSION alert. Monthly-ness is decided when it fires, not when it is armed.

        Arming per session rather than per month is what makes a mid-month restart safe: the strategy is
        never more than one session away from noticing the next rebalance. A month-long alert armed once
        would be blind until it expired, and a restart would lose it entirely.
        """
        self._reread_open_offset()
        session, fire_at = self._calendar.next_fire(after, self._open_offset)
        self._armed_session = pd.Timestamp(session)
        self.clock.set_time_alert(SESSION_ALERT, fire_at, self._on_session_alert, override=True)

    def _on_session_alert(self, event=None) -> None:
        fired, self._armed_session = self._armed_session, None
        # SNAPSHOT THE SLOT THIS SESSION FIRED AT, BEFORE RE-ARMING. `_slot_name` is derived from
        # `self._open_offset`, and `_arm` re-reads that offset from settings (kumo-trading-platform issue 514). The
        # re-arm happens FIRST (see below), and the decision below runs AFTER it -- so reading the
        # live property at decide time would file this session under the slot armed NEXT.
        #
        # That is ba37ef9 reintroduced by the fix for #514, and worse than the original: the original
        # was a constant mismatch, this one appears only on the session after an operator edits a
        # setting, so it correlates with the edit and looks like the edit working.
        # `momentum_rotation` already does exactly this with `_armed_slot` -> `_due_slot`.
        self._fired_slot = self._slot_name
        # Re-arm FIRST. Anything below may raise, and a strategy that has stopped scheduling looks
        # exactly like a strategy that decided to hold — silent, and wrong for as long as nobody looks.
        # GUARDED RE-ARM (kumo-trading-platform issue 628). Still FIRST, for the reason above — but `_arm` can now
        # refuse, because a venue calendar knows only a rolling window. An unguarded raise here would
        # abort this callback: the session that just fired would never be marked due, no decision
        # would be taken for it, and no future alert would be set. Silent, permanent, mid-session.
        self.rearm_after_alert(self.clock.utc_now())
        if self._due is not None:
            self.log.warning(
                f"{STRATEGY_NAME}: rebalance {self._due.date()} closed without tradeable data — "
                "no decision was made"
            )
            self.missed_rebalances.append(self._due)
        self._due = fired
        self._try_decide()

    def _is_rebalance(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """Is `session` a rebalance day, per the SAME rule the backtest uses?

        Delegates to the pure `rebalance_dates()` rather than re-deriving "first session of the month"
        here. Two derivations of one fact disagree, and the cadence rule is exactly the sort of thing
        that gets tuned in research and forgotten in production.
        """
        if pd.Timestamp(session) in self._forced_rebalance:
            return True
        if panel.empty:
            return False
        return pd.Timestamp(session) in set(rebalance_dates(panel["date"]))

    def _try_decide(self) -> None:
        """Decide for `_due` if this is a rebalance day and the panel has caught up.

        Called by the alert AND by every bar: the alert says WHEN TO LOOK, the bars say WHETHER WE MAY
        DECIDE. A 1-DAY bar for session D is emitted at D's close, so a late feed must defer to `on_bar`
        rather than be polled — bar arrival IS the event being waited on.
        """
        if self._due is None or self._pending or not self.warm:
            return                      # orders in flight or still warming — retry on the next bar
        panel = self._panel()
        if panel.empty:
            return
        if self._due not in set(panel["date"]):
            return                      # this session has not reached the panel yet; keep waiting
        prior = panel.loc[panel["date"] < self._due, "date"].max()
        if pd.isna(prior):
            return                      # no completed session yet; keep waiting
        session, self._due = self._due, None
        if not self._is_rebalance(session, panel):
            self.skipped_sessions += 1
            self._exit_only_for(session)
            self._record_skipped(session)
            return
        if self._runner is not None:
            self._run_session(session, panel.loc[panel["date"] <= session].copy())
        else:
            self._decide_for(session, panel)

    @property
    def warm(self) -> bool:
        """Enough history on enough symbols to rank them against each other.

        A cross-sectional strategy warms as a UNIVERSE, not per symbol: ranking three warm names out of
        fifty is not an early answer, it is a different and wrong strategy. So this is all-or-nothing on
        the count of symbols that have cleared `_need`.
        """
        ready = sum(1 for sym in self._bars if len(self._bars[sym]) >= self._need)
        return ready >= max(self._cfg.portfolio_size, 1)

    def _panel(self) -> pd.DataFrame:
        rows = [row for sym in self._bars for row in self._bars[sym]]
        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def _run_session(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        """Hand the session to an async runner on the node's own loop when one is attached."""
        if self._session_task is not None and not self._session_task.done():
            self.log.warning(
                f"{STRATEGY_NAME}: rebalance {session.date()} skipped — the previous one is still running"
            )
            self.missed_rebalances.append(session)
            return
        if self._loop is None:
            self.log.error(
                f"{STRATEGY_NAME}: no event loop captured — refusing to run rebalance {session.date()} "
                "outside the live runtime"
            )
            self.missed_rebalances.append(session)
            return
        self._session_task = asyncio.run_coroutine_threadsafe(
            self._session_coro(session, panel),
            self._loop,
        )

    async def _session_coro(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        day = str(session.date())
        try:
            # A POSITION ON THE SIDE THIS LANE CANNOT MANAGE IS REPORTED EVERY SESSION (#66).
            # The exit path already REFUSES one, but only when the lane is trying to exit that
            # name — WHD sat in the book for twelve hours because nobody was. Before the
            # decision, because a book we do not understand is context for what follows.
            report_wrong_sided_positions(self, session=str(session.date()))
            result = await self._runner.run(panel, day, slot=self._fired_slot or self._slot_name)
            submitted = getattr(result, "submitted", 0)
            blocked = getattr(result, "blocked", None)
            decided = bool(getattr(result, "decided", False))
            state = getattr(result, "state", "UNKNOWN")
            outcome = session_outcome(state, decided=decided, blocked=blocked, sent=submitted)
            self.log.info(f"{STRATEGY_NAME} {session.date()}: {outcome}")
            # DURABLE OUTCOME, every session, decided or not. Without it "session ran and declined"
            # and "session never ran" are indistinguishable in the record an operator reads —
            # MOMENTUM's adapter has written this row since `_record` was added and QC345's never
            # has. That asymmetry is what left 2026-08-21 unexplainable from the journal.
            await self._journal_session(
                "state", f"session outcome — {outcome}", day,
                {"state": state, "decided": decided, "blocked": blocked, "submitted": submitted,
                 # See momentum_rotation: the integer keeps its name, this names its basis.
                 "submitted_basis": SENT_BASIS,
                 "entered": list(getattr(result, "entered", ()) or ()),
                 "exited": list(getattr(result, "exited", ()) or ())})
        except asyncio_CancelledError:
            raise
        # THE HANDLER BELOW IS DELIBERATELY COMPACT; the reasoning is here so that the traceback stays
        # adjacent to the log call it belongs to.
        #
        # exc_info, ALWAYS (2026-08-21). This logged only `type(exc).__name__: exc`, so QC345's second
        # live rebalance failure arrived as "TypeError: 'NoneType' object is not callable" with no
        # file, no line and no frame — unlocatable. The strategy has gone live twice and booked
        # nothing both times; the first was diagnosable only because the exception type happened to
        # name the missing key.
        #
        # AND A DURABLE ROW, not just a log line. On 2026-08-21 the gateway journalled its decision
        # and then raised mid-submit, so the journal read `['decision']` and nothing more: five
        # entries decided, no order, no refusal, no error — the same record a strategy leaves when it
        # decides and legitimately declines to act. A container log is not a record: it is not
        # queryable, and kumo-trading-platform issue 378 established that recreating a container destroys the prior
        # one's logs.
        #
        # LOG FIRST, THEN JOURNAL. If the database is what failed, the journal write cannot work
        # either, so the log line must already be out before anything else is attempted.
        except Exception as exc:                                       # noqa: BLE001
            self.missed_rebalances.append(session)
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.log.error(
                f"{STRATEGY_NAME}: rebalance {session.date()} failed: "
                f"{type(exc).__name__}: {exc}\n{trace}")
            await self._journal_session(
                "error", f"rebalance {day} failed: {type(exc).__name__}: {exc}", day,
                {"exception": type(exc).__name__, "traceback": trace[-4000:]})

    async def _journal_session(self, kind: str, summary: str, day: str, detail: dict) -> None:
        """Best-effort write to the runner's journal. Never raises into the session task.

        BEST EFFORT BY NECESSITY: the failures most worth recording include "the database was
        unreachable", and in that case this cannot work either. What it must never do is turn a
        handled failure into an unhandled one — a journal that raises here would kill the task and
        lose the log line's context along with it.

        Degrades to a no-op with no runner attached (the backtest and standalone paths), which must
        not raise for lack of somewhere to journal to.
        """
        # SHARED RESOLVER (kumo-trading-platform issue 587). This was `getattr(self._runner, "journal", None)`,
        # and cockpit's gateway stores it as `_journal` — so it was always None and this method
        # returned silently for the lane's entire life. Zero outcome rows for QC345-003 on paper.
        journal = self.session_journal()
        if journal is None:
            return
        # THE SLOT IS THE GATEWAY'S, not this adapter's. QC345's adapter has no slot concept at all
        # — the gateway owns `_slot` and uses it as the idempotency key — so inventing a literal here
        # would file these rows under a slot the decision row does not share, and the two halves of
        # one session would not join. Read it if it is there; otherwise let the journal apply its own
        # default rather than assert a value this layer does not know.
        kw = {}
        slot = getattr(self._runner, "_slot", None)
        if slot:
            kw["slot"] = slot
        try:
            await journal.write(kind, summary, session=day, detail=detail, **kw)
        except Exception as exc:                                       # noqa: BLE001
            self.log.error(
                f"{STRATEGY_NAME}: could not journal the {kind} above: {exc}\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def _next_rebalance_on_or_after(self, session: pd.Timestamp) -> pd.Timestamp:
        """The earliest date a rebalance can fire after `session`: the first calendar day of the next
        month (the first SESSION on or after it is the rebalance, per `rebalance_dates`), or a forced
        date if one is nearer. A label for the record, not a second derivation of the cadence."""
        month_first = (pd.Timestamp(session).normalize() + pd.offsets.MonthBegin(1))
        forced = [d for d in self._forced_rebalance if d > pd.Timestamp(session)]
        return min([month_first, *forced])

    def _record_skipped(self, session: pd.Timestamp) -> None:
        """One `state` row per skipped session — "skipped" is an outcome, and it gets the row every
        outcome gets (kumo-trading-platform issue 1099).

        This branch used to increment a counter and write NOTHING: no log, no journal row. On paper
        the lane read as dead for sixteen sessions, and that silence hid that protection had stopped
        out its whole book. A monthly lane that is quiet by design must SAY it is quiet, every
        session, with the date it will next act — or an operator cannot tell "waiting for October"
        from "stopped working in September".
        """
        nxt = self._next_rebalance_on_or_after(session)
        summary = (f"skipped: not a rebalance session (next {nxt.date()}); "
                   f"held {len(self._held)}")
        self.log.info(f"{STRATEGY_NAME} {session.date()}: {summary}")
        if self._loop is None:
            return                          # backtest / standalone: nowhere to journal to
        self.fire_and_report(
            self._journal_session(
                "state", summary, str(session.date()),
                # THE SAME SHAPE AS EVERY OTHER SESSION-OUTCOME ROW (kumo-trading-platform issue 1098): a reader
                # filtering on decided/blocked sees a skip as an undecided session with a named
                # reason, and `skip` says which of the three gates it was — here, by construction,
                # the lane was warm and the panel held the session; it was not a rebalance day.
                {"state": "SKIPPED", "decided": False, "blocked": "not a rebalance session",
                 "skip": {"warm": True, "panel_has_due": True, "is_rebalance": False},
                 "held": len(self._held), "held_symbols": sorted(self._held),
                 "next_rebalance_on_or_after": str(nxt.date()),
                 "skipped_sessions": self.skipped_sessions}),
            self._loop, f"skipped-session row for {session.date()}")

    def _exit_only_for(self, session: pd.Timestamp) -> None:
        forced = self._trail_exits(session)
        if not forced:
            return
        self.log.info(f"{STRATEGY_NAME} {session.date()}: forced exits {sorted(forced)}")
        for symbol in sorted(forced):
            self._close(symbol)

    def _decide_for(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        """Run the PURE decision for `session` and turn it into orders."""
        forced = self._trail_exits(session)
        scored, diag = build_feature_panel(panel, self._cfg)
        day = scored.loc[scored["date"] == session].copy()
        if forced:
            day = day.loc[~day["ticker"].isin(forced)].copy()
        if day.empty:
            if forced:
                self.log.info(
                    f"{STRATEGY_NAME} {session.date()}: no rankable rows — exiting only "
                    f"{sorted(forced)}"
                )
                for symbol in sorted(forced):
                    self._close(symbol)
                return
            self.log.warning(f"{STRATEGY_NAME}: no feature rows for {session.date()} — holding")
            return
        universe = self._source.eligible(session) - set(forced)
        dec = decide(day, self._cfg, set(self._held), universe=universe)
        exits = tuple(sorted(set(dec.exit) | set(forced)))
        enters = tuple(symbol for symbol in dec.enter if symbol not in exits)
        self.log.info(
            f"{STRATEGY_NAME} {session.date()}: universe={sorted(universe)} hold={list(dec.hold)} "
            f"enter={list(enters)} "
            f"exit={list(exits)} (split-blocked {diag.split_blocked_name_months}, "
            f"mania-blocked {diag.mania_blocked_name_months})"
        )
        today = panel.loc[panel["date"] == session].set_index("ticker")
        for symbol in exits:
            self._close(symbol)
        for symbol in enters:
            self._open(symbol, today)

    def _trail_exits(self, session: pd.Timestamp) -> set[str]:
        """Session-level exits, from a trail rebuilt out of completed sessions rather than memory."""
        if not self._held or not _exit_rules_configured(self._cfg):
            return set()

        prior = self._panel()
        prior = prior.loc[prior["date"] < session].copy()
        if prior.empty:
            return set()

        atr = trailing_atr(prior) if needs_atr(self._cfg.exits) else None
        highs = (
            prior.sort_values("date").groupby("ticker")["high"].last().astype(float).to_dict()
            if needs_highs(self._cfg.exits)
            else None
        )

        states: dict[str, object] = {}
        prices: dict[str, float] = {}
        for pos in self.cache.positions_open(strategy_id=self.id):
            sym = pos.instrument_id.symbol.value
            if sym not in self._held:
                continue
            hist = prior.loc[prior["ticker"] == sym].sort_values("date")
            if hist.empty:
                continue
            opened = pd.Timestamp(pos.ts_opened, unit="ns").normalize()
            held = hist.loc[hist["date"] >= opened]
            if held.empty:
                continue
            covers = pd.Timestamp(hist.iloc[0]["date"]).normalize() <= opened
            states[sym] = reconstruct_trail(
                float(pos.avg_px_open),
                held["close"].astype(float).tolist(),
                covers_entry=covers,
            )
            prices[sym] = float(held.iloc[-1]["close"])
            if highs is not None:
                highs[sym] = float(held.iloc[-1]["high"])
        if not states:
            return set()
        plan = evaluate_exits(self._cfg.exits, prices, states, atr=atr, highs=highs)
        return set(plan.exits)

    # -- orders ----------------------------------------------------------------------------------
    def _bar_type(self, iid: InstrumentId):
        from nautilus_trader.model.data import BarType

        return BarType.from_str(f"{iid}{self._suffix}")

    def _iid_for(self, symbol: str) -> InstrumentId | None:
        for iid in self._iids:
            if iid.symbol.value == symbol:
                return iid
        return None

    def _open(self, symbol: str, today: pd.DataFrame) -> None:
        iid = self._iid_for(symbol)
        if iid is None or symbol not in today.index:
            return
        instrument = self.cache.instrument(iid)
        if instrument is None:
            # A hard refusal at submit, so refuse here where it can be said plainly.
            self.log.error(f"{STRATEGY_NAME}: no instrument definition for {symbol} — not entering")
            return
        price = float(today.loc[symbol, "close"])
        if price <= 0:
            return
        qty = int(self._equity / price)
        if qty < 1:
            return
        order = self.order_factory.market(
            instrument_id=iid, order_side=OrderSide.BUY, quantity=Quantity.from_int(qty),
        )
        self._pending[symbol] = "enter"
        self.submit_order(order)

    def _close(self, symbol: str) -> None:
        iid = self._iid_for(symbol)
        if iid is None:
            return
        position = None
        for pos in self.cache.positions_open(strategy_id=self.id):
            if str(pos.instrument_id) == str(iid):
                position = pos
                break
        if position is None or position.quantity == 0:
            self._held.discard(symbol)
            return
        # LONG ONLY, ASSERTED (#88). This was `abs(position.quantity)` with an unconditional SELL —
        # on a SHORT that doubles the position and reports success. `abs()` turns "I do not
        # understand this position" into a confident order in the wrong direction.
        qty = closing_quantity(position.quantity, side=LONG)
        if qty is None:
            why = refusal_reason(position.quantity, side=LONG)
            if why:
                self.log.error(f"{self.id}: {symbol} — {why}")
            self._held.discard(symbol)
            return
        order = self.order_factory.market(
            instrument_id=iid, order_side=OrderSide.SELL,
            quantity=Quantity.from_int(qty),
        )
        self._pending[symbol] = "exit"
        self.submit_order(order)

    # -- order events ----------------------------------------------------------------------------
    # Book state moves HERE and nowhere else. Submit is a request; a fill is a fact.
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
        symbol = str(event.instrument_id.symbol)
        intent = self._pending.get(symbol)
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
        self._pending.pop(symbol, None)
        if intent == "enter":
            self._held.add(symbol)
        elif intent == "exit":
            self._held.discard(symbol)
        self._record_terminal(event, ok=True, detail="filled")
        self._try_decide()

    def on_order_denied(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)
        self._record_terminal(event, ok=False, detail=getattr(event, "reason", "") or "denied")

    def on_order_rejected(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)
        self._record_terminal(event, ok=False, detail=getattr(event, "reason", "") or "rejected")

    def on_order_canceled(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event)
        self._record_terminal(event, ok=False, detail="canceled")


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
    def _record_terminal(self, event, *, ok: bool, detail: str) -> None:
        """Forward the venue's actual answer to the runner's journal (kumo-trading-platform issue 383).

        The submit-time row records Nautilus ACCEPTING the order, not the venue filling it. Without
        this, the journal says an order went out and never says what came back -- and the retry
        counter that reads `phase == "terminal" and not ok` can never advance, so a symbol refused
        by the venue is silently treated as done.

        MEASURED INERT ON THE LIVE PATH, 2026-08-20: MOMENTUM has called this from all four handlers
        for weeks, and that session produced 22 `OrderFilled` events and ZERO terminal rows --
        because cockpit's `SessionGateway` never defined `record_terminal`, so the `getattr` below
        returned None on every fill. Wiring the caller is half a fix; the receiving end has to land
        in the same change or this is decoration.

        Best-effort by design, and deliberately so: it runs in a live event handler outside the
        session's own control flow, so a missing method or a dead loop degrades to the previous
        behaviour rather than raising inside Nautilus's dispatch. It closes an observability gap; it
        is not load-bearing for safety the way the intent-before-submit ordering is.
        """
        if self._runner is None or self._loop is None:
            return
        record = getattr(self._runner, "record_terminal", None)
        if record is None:
            return
        # The session comes from the order's own tags -- `NautilusBroker.submit` stamps every order
        # `session:{session}` (broker.py:85) -- rather than a parallel in-process map, so it survives
        # a restart between submit and fill exactly as the order does, via Nautilus's durable cache.
        order = self.cache.order(event.client_order_id)
        session = next((t.split(":", 1)[1] for t in (order.tags or ()) if t.startswith("session:")),
                       None) if order is not None else None
        if session is None:
            return          # an order this adapter did not tag -- not ours to record
        symbol = str(event.instrument_id.symbol)
        # OBSERVED, not fire-and-forget — see `RegistrationMixin.fire_and_report`.
        self.fire_and_report(
            call_record_terminal(record, session, symbol, ok, detail, log=self.log.warning,
                                 **terminal_fields(self, event)),
            self._loop, f"terminal row for {symbol}")
        # THE CLAIM FOLLOWS THE VENUE'S ANSWER (kumo-trading-platform issue 829) — same re-sync as
        # `MomentumRotationStrategy._record_terminal`: cockpit's `_claim` records on ACCEPT, so a
        # denial or rejection must bring the claim back to what the lane's cache position says.
        sync = getattr(self._runner, "sync_claim", None)
        if sync is not None:
            try:                    # never raise into Nautilus's dispatch — see momentum_rotation
                qty = int(sum(int(p.signed_qty) for p in self.cache.positions_open(
                    instrument_id=event.instrument_id, strategy_id=self.id)))
                px = float(event.last_px) if ok and getattr(event, "last_px", None) is not None else None
            except Exception as exc:                                       # noqa: BLE001
                self.log.error(f"{self.id}: claim sync for {symbol} NOT attempted — "
                               f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]}")
                return
            self.fire_and_report(sync(symbol, qty, px), self._loop, f"claim sync for {symbol}")

    def _forget(self, event) -> None:
        """A request that died changes nothing about what is held — only that nothing is in flight."""
        self._pending.pop(str(event.instrument_id.symbol), None)
        self._try_decide()

    # -- data ------------------------------------------------------------------------------------
    def on_historical_data(self, data) -> None:
        for bar in data if isinstance(data, list) else [data]:
            if isinstance(bar, Bar):
                self._ingest(bar)

    def on_bar(self, bar: Bar) -> None:
        self._ingest(bar)
        self._try_decide()

    def _ingest(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id.symbol)
        row = {
            "ticker": symbol,
            "date": pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        bars = self._bars[symbol]
        if bars and pd.Timestamp(bars[-1]["date"]).normalize() == row["date"]:
            bars[-1] = row
        else:
            bars.append(row)

    def _on_broker_account(self, snapshot: dict) -> None:
        self._broker_account = snapshot

    def _on_rebalance_override(self, payload: dict) -> None:
        """Operator has said: treat `payload["date"]` as a rebalance day even though it is not the
        first session of its month. Additive and idempotent -- publishing the same date twice, or a
        date that was already a real rebalance day, changes nothing.

        Every branch below only logs and returns, never raises. This runs inside Nautilus's own
        msgbus dispatch, not this strategy's own call stack: an uncaught exception here is not the
        same failure as one in `_try_decide`, and depending on how the bus dispatches it a raise can
        leave the SUBSCRIPTION dead rather than just this one message -- which then looks exactly
        like "no override was ever sent," the same silent-failure shape as the BCTROT scheduling bug.

        Logged at INFO with the full set every time a date is accepted, not just on change: a manual
        override on a live-money strategy must be visible in the same log an operator already reads
        for decisions. Also journalled (RISK kind) when a session runner is attached, so the record
        survives a log-losing container recreate -- an INFO line alone does not (see kumo-trading-platform's
        #378: recreation destroys the prior container's logs)."""
        date = payload.get("date") if isinstance(payload, dict) else payload
        if not date:
            self.log.error(f"{STRATEGY_NAME}: rebalance override received with no date — ignored")
            return
        try:
            ts = pd.Timestamp(date).normalize()
        except (ValueError, TypeError) as exc:
            self.log.error(
                f"{STRATEGY_NAME}: rebalance override date {date!r} unparseable: {exc} — ignored")
            return
        self._forced_rebalance.add(ts)
        forced_sorted = sorted(d.date() for d in self._forced_rebalance)
        self.log.info(
            f"{STRATEGY_NAME}: operator forced {ts.date()} as a rebalance day "
            f"(forced set now: {forced_sorted})"
        )
        if self._runner is not None and self._loop is not None:
            # "risk" as a literal, not `pgjournal.RISK` -- that module's package pulls in SQLAlchemy,
            # which this file (and its backtest use) deliberately stays free of, the same reason
            # `calendar` is imported from `runtime.calendar` and not `runtime.executor.calendar`.
            # OBSERVED, not fire-and-forget. This row is the AUDIT TRAIL for an operator action —
            # the only record that a date was forced rather than reached by the monthly rule. It is
            # what answers "why did this lane decide on a day that is not a month-start", which was
            # asked for real on 2026-08-25. A discarded Future meant a failed write left the
            # operator's action with no trace at all.
            self.fire_and_report(
                self._runner.journal.write(
                    "risk", f"operator forced {ts.date()} as a QC345 rebalance day",
                    session=str(ts.date()),
                    detail={"forced_rebalance_dates": [str(d) for d in forced_sorted]},
                ),
                self._loop,
                f"forced-rebalance audit row for {ts.date()}",
            )

    def broker_equity(self) -> float | None:
        """Net liquidation from the broker's own snapshot.

        Nautilus `AccountState` models CASH ONLY and cockpit's Alpaca client sets `AccountBalance.total`
        to the cash figure, so equity for any risk limit has to come off the bus.

        A METHOD, NOT A PROPERTY, and that is the whole of kumo-trading-platform's 2026-08-21 QC345 outage.
        `NautilusBroker.equity()` is `return self.strategy.broker_equity()` — it assumes a method, and
        MomentumRotationStrategy defines one. As a `@property` this evaluated FIRST and the `()` landed
        on the result; returning None here is legitimate, so `None()` raised
        `TypeError: 'NoneType' object is not callable` and killed the rebalance mid-submit, after the
        decision row was already written. The journal showed five entries decided and no order, no
        refusal, no error.

        It was latent for as long as an account frame always arrived. kumo-trading-platform issue 382 stopped
        publishing cash under the name equity when equity could not be derived — correct, the old value
        understated equity by 25% on the channel the daily-loss halt trusts — and that removed the wrong
        value which had been masking this.

        THE KEY IS `equity`, NOT `portfolio_value`. Cockpit reads Alpaca's `portfolio_value` and
        REPUBLISHES it under the name `equity` (exec_client.py:427); the message on `broker.account`
        carries equity/cash/buying_power/multiplier/long_market_value/last_equity/ts and NO
        `portfolio_value` key at all. Reading that name returned None every time, so sizing produced
        "sizing yielded 0 shares" for every entry against a $103,428 account — QC345 refused five
        names at 18:30Z on 2026-08-21 and TECHIVOL-005 failed its first live slot the same way.

        This was the SECOND cause of one symptom. Making `broker_equity` a method fixed the morning's
        TypeError and was correct, but it moved the failure one step later rather than removing it,
        because nothing forced the two ends of this topic to agree about a key name. Found by
        kumo-trading-platform aiming the scan at the class rather than at QC345.

        NOT fixable by cockpit adding a `portfolio_value` alias: two names for one number is the
        defect, and an alias preserves it while hiding it.

        RETURNING None STAYS LEGAL. Cockpit's `_equity_per_position` is `self._broker.equity() or 0.0`
        and already handles it; it simply never got there. Returning a plausible 0.0 from here instead
        would disarm the daily-loss halt, which anchors on this number and fails by never halting.
        """
        if not self._broker_account:
            return None
        value = self._broker_account.get("equity")
        if value is None:
            return None
        # NON-FINITE IS NOT A NUMBER -- see the note in `momentum_rotation.broker_equity`. A nan
        # passes `is not None`, survives `float()`, and then makes every downstream comparison
        # False, so the daily-loss halt fails by never halting. None is already legal here.
        out = float(value)
        return out if math.isfinite(out) else None
