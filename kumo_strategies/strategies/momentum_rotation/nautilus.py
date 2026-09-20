"""Nautilus binding for momentum rotation. The SAME decision code runs here and in backtest.

This class contains no strategy logic. It collects bars, asks `engine.decide()` what to hold, and
turns the answer into orders. Every rule lives in the pure layer, so there is nothing to reconcile
between research and production — which is the whole point of the split.

Three things differ between backtest and live even with identical code, and each is handled here
rather than assumed away (issue #11):

  warmup     backtest catalogs are preloaded; live history arrives asynchronously. The strategy
             refuses to trade until it holds enough bars, and reports that state explicitly.
  order life backtest fills are simulated; live orders get rejected, partially filled, and halted.
             Position state is driven by ORDER EVENTS, never by the fact that submit was called.
  clock      event time vs wall clock. Both run off `clock.set_time_alert`, so the trigger that
             fires in production is the trigger the backtest measured. See "The session clock".

Identity: StrategyConfig(strategy_id="MOMENTUM", order_id_tag="002") -> `MOMENTUM-002`, matching
cockpit's `api/strategy_ids.py`. Set it this way and not via change_id(), or Nautilus rewrites the
tag at registration and desyncs client_order_id generation from the position id.

The tag is 002, not 001, because Nautilus requires `order_id_tag` to be unique across every strategy
in one trader — it is the suffix of the StrategyId and of generated client order ids. MANUAL already
holds 001 and has live positions keyed to `MANUAL-001`, so it cannot move. The cockpit convention of
giving every strategy tag 001 works only while exactly one is registered; adding a second one fails
at `add_strategy` with "order_id_tag conflict for '001'". Changed here rather than there because
MOMENTUM has never traded and MANUAL has.

The session clock
-----------------
A calendar-derived ONE-SHOT alert, re-armed after every firing. Not `set_timer(interval)`: the gap
between sessions is not constant — weekends, holidays, and the 13:00 half-day close all move it.

This replaced a bar-rollover trigger ("decide when a bar carries a new date"), which could not
express the live rule at all. A 1-DAY bar for session D is emitted at D's close, so in live trading
the rollover fires when D+1's bar lands — a full day after the open we intended to trade. Replaying
a catalog hid that completely, because there the next bar arrives immediately.

Timer and data have separate jobs, and conflating them has cost us before:

    the alert says WHEN TO LOOK        the bars say WHETHER WE MAY DECIDE

The alert only sets `_due`. Deciding needs a panel that actually reaches the last completed
session, so a late feed defers to `on_bar` rather than polling — bar arrival IS the event being
waited on. An alert that fires with `_due` still set from last time means a session came and went
without tradeable data, and says so. Firing early on a half-formed panel is the failure mode that
once ranked a universe of one at 09:30.
"""

from __future__ import annotations

import traceback

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.capabilities import EXECUTES_DELTAS, require
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.strategies.momentum_rotation.envelopes import (
    REGISTERED as REGISTERED_ENVELOPES)
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin

import math

import asyncio
from asyncio import CancelledError as asyncio_CancelledError
from collections import defaultdict, deque

import pandas as pd
from kumo_strategies.runtime.nautilus.sides import (
    LONG, closing_quantity, refusal_reason)
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import order_status_to_str, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
from kumo_strategies.strategies.momentum_rotation.engine import (
    apply_gates, decide, needs_panel_stats, panel_stats, score_panel, trailing_atr,
    trailing_returns)
from kumo_strategies.runtime.nautilus.contract import (
    report_wrong_sided_positions, protective_close, SENT_BASIS, RegistrationMixin, session_outcome,
    call_record_terminal, terminal_fields)
from kumo_strategies.strategies.momentum_rotation.slots import validate as validate_slots
from kumo_strategies.strategies.momentum_rotation.exits import (
    evaluate_exits, needs_atr, needs_highs, reconstruct_trail)

from kumo_strategies.runtime.calendar import elapsed_slots, next_slot_fire

STRATEGY_NAME = "MOMENTUM"
EXTERNAL_ID = "MOMENTUM"
STRATEGY_LABEL = "BCT momentum rotation"

#: Cockpit allocates the `order_id_tag` — it is the only place that sees every strategy in ONE
#: trader and it is what calls `Trader.add_strategy`. This default exists so an existing deployment
#: keeps the id it already has; under NETTING the position id is `{instrument}-{strategy_id}`, so a
#: tag that moves after anything has traded orphans real positions. Callers SHOULD pass it.
DEFAULT_ORDER_ID_TAG = "002"
STRATEGY_TAG = DEFAULT_ORDER_ID_TAG  # deprecated alias; cockpit owns this
SESSION_ALERT = "session_decide"
SOURCE_TIMER = "source_refresh"
SOURCE_REFRESH_SECS = 60


def default_slots(open_offset_minutes: int) -> tuple[str, ...]:
    """The single slot implied by a plain open offset — what this strategy did before slots existed.

    A module function rather than an inline expression so the LIVE default is directly assertable.
    MOMENTUM-002 holds real positions on this schedule; "multi-slot is additive" has to be a tested
    claim, and a test on the constructor's `decision_slots=None` default does not make it — None is
    still None however this derivation changes.
    """
    return (f"open+{open_offset_minutes}m",)



class MomentumRotationStrategy(MarketAwareMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin, Strategy):
    EXTERNAL_ID = 'MOMENTUM'
    #: The pre-registered distributions this lane may place itself in (#147). Set here rather than
    #: looked up inside the hook so BCTROT inherits it — and so a lane WITHOUT one is visibly
    #: without one rather than silently falling through to someone else's envelope.
    ENVELOPE_REGISTRY = REGISTERED_ENVELOPES

    def envelope_key(self) -> str | None:
        """WHICH registered distribution describes this lane's CONFIGURATION, not merely its name.

        The gap rule is part of the identity: `MOMENTUM-002` and `MOMENTUM-002+gap` are different
        strategies with different distributions, and the deployed lanes are asymmetric — MOMENTUM
        runs one slot with no gap, BCTROT three slots with a 1.5% gap. A key derived from the lane
        name alone would place BCTROT in the no-gap distribution and read its percentile off a
        strategy it is not running.

        The ORDER ID TAG is what makes the key, because the registry is keyed on the wire id
        (`MOMENTUM-002`), and a lane's tag is allocated by cockpit rather than guessed here.
        """
        try:
            tag = getattr(self, "order_id_tag", None) or getattr(self, "_order_id_tag", None)
            if not tag:
                return None
            key = f"{self.EXTERNAL_ID}-{tag}"
            gap = getattr(getattr(self._cfg, "execution", None), "min_abs_gap_pct", None)
            if gap:
                key = f"{key}+gap"
            return key if key in (self.ENVELOPE_REGISTRY or {}) else None
        except Exception:                                               # noqa: BLE001
            return None

    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG
    LABEL = 'BCT momentum rotation'

    """Rotation over a pluggable candidate pool. Logic lives in the pure engine, not here."""

    def __init__(
        self,
        cfg: MomentumRotationConfig,
        source: CandidateSource,
        instrument_ids: list[InstrumentId] | None = None,
        # KEYWORD-ONLY, DELIBERATELY. Inserted as a positional parameter this silently shifts
        # every argument after it: a caller passing `(cfg, source, ids, suffix)` would bind the
        # suffix to `symbols`, and `_init_symbols` would then see both and refuse at BUILD — which
        # takes every other lane in the trader down with it (the #377 shape). Two repos share this
        # signature across a version pin, so a positional insert is not a change either side can
        # make safely.
        *,
        symbols: list[str] | None = None,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        instrument_type: dict[str, str] | None = None,
        equity_per_position: float | None = None,
        history_days: int | None = None,
        calendar: object | None = None,
        open_offset_minutes: int = 5,
        decision_slots: tuple[str, ...] | None = None,
        read_slots: object | None = None,
        max_stale_days: int | None = None,
        session_runner: object | None = None,
        #: Named, so a lane that deliberately computes without trading can be built. Only consulted
        #: when a capability the runner lacks is actually REQUIRED — see the `rebalance_band` guard
        #: below; a lane with no band never reaches it.
        shadow_only: bool = False,
        session_jobs: object | None = None,
        account_topic: str = "broker.account",
        external_order_claims: list[InstrumentId] | None = None,
        order_id_tag: str = DEFAULT_ORDER_ID_TAG,
        strategy_name: str = STRATEGY_NAME,
    ) -> None:
        # external_order_claims: which instruments THIS strategy owns unattributed activity for.
        #
        # Without it, when the venue reports a position flat before the local fill has closed the
        # cached one, Nautilus synthesises a flatting order and — finding no claimant — books it under
        # StrategyId("EXTERNAL"). Under NETTING the position id is `{instrument}-{strategy}`, so that
        # order OPENS a phantom position on EXTERNAL instead of CLOSING the real one on this strategy.
        # That is how HSBC became a permanent -93 short the broker had never heard of
        # (kumo-trading-platform issue 197 B8): synthetic SELL 93 @ 107.69, the CACHED average rather than the real
        # 102.91 fill, tagged RECONCILIATION.
        #
        # With a claim the same synthetic order is attributed here, lands on THIS strategy's position,
        # and closes it — which is what reconciliation was trying to express in the first place.
        #
        # Claims are EXCLUSIVE across the node: `ExecutionEngine.register_external_order_claims`
        # raises InvalidConfiguration if two strategies claim one instrument, and that happens during
        # `Trader.add_strategy` — so an over-broad claim does not degrade, it stops the node booting.
        # Pass only what this strategy genuinely owns; the caller decides, because only it knows what
        # the other strategies hold.
        super().__init__(config=StrategyConfig(
            # From the PARAMETER, not this module's constant. A subclass could previously override
            # the tag but not the name, so BCTROT came up as MOMENTUM-004: its own STRATEGY_NAME was
            # declared and never read. Under NETTING the position id is `{instrument}-{strategy_id}`,
            # so that is the cycle-attribution key — BCTROT's positions and P&L would have landed in
            # a book named MOMENTUM, permanently from the first fill.
            strategy_id=strategy_name, order_id_tag=order_id_tag,
            external_order_claims=list(external_order_claims) if external_order_claims else None))
        # Kept for `claimed_instruments` (kumo-trading-platform issue 39): cockpit needs to read back what
        # this strategy claims, and Nautilus does not expose it once configured.
        self._claims = list(external_order_claims or [])
        self._cfg = cfg
        self._source = source
        # SYMBOLS OR IDS, EXACTLY ONE (kumo-trading-platform issue 622). See `symbol_resolution`.
        self._init_symbols(instrument_ids, symbols)
        self._suffix = bar_type_suffix
        self._itype = instrument_type or {}
        # None means 'not overridden' so the config is the operator-visible source (#32).
        self._equity = (equity_per_position if equity_per_position is not None
                        else cfg.execution.equity_per_position)
        # Live only. In a backtest the catalog is preloaded and pushed through the subscription, so
        # there is nothing to request and no data client to request it from. Leaving this on is what
        # made the first backtest fail at on_start.
        self._history_days = history_days

        # rolling per-symbol history, capped at what the longest indicator needs
        need = max(cfg.score.vol_window, cfg.score.lookback) + cfg.gates.corporate_action_window
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=need + 5))
        self._need = need

        # book state, driven by FILL events — never by submit
        self._held: set[str] = set()
        self._pending: dict[str, str] = {}     # symbol -> "enter" | "exit"
        self._last_decision: pd.Timestamp | None = None

        # session clock. Import the calendar from `runtime.calendar`, NOT `runtime.executor.calendar`
        # — the latter is a package whose __init__ imports SQLAlchemy, which has no business being
        # dragged into a backtest.
        if calendar is None:
            from kumo_strategies.runtime.calendar import elapsed_slots, next_slot_fire, build_calendar
            calendar = build_calendar()
        self._calendar = calendar
        self._open_offset = open_offset_minutes
        # ONE decision per session was an assumption baked into the clock, not a config choice: a
        # single alert at a single offset, re-armed with override=True. The backtest runner and the
        # decision store's (session, slot) key had both moved on while the thing that actually
        # decides could still only fire once.
        #
        # Defaults to the single slot implied by `open_offset_minutes`, so an existing deployment
        # keeps its exact schedule. MOMENTUM-002 is live on this path.
        # PRECEDENCE: explicit argument, then the config, then the open offset. The config is the
        # operator-visible path (#32) and the argument exists for tests and for a caller that has
        # already resolved a schedule; the offset fallback is what this strategy did before slots.
        if decision_slots:
            self._slots = tuple(decision_slots)
        elif tuple(cfg.execution.decision_slots) != ("open+5m",):
            self._slots = tuple(cfg.execution.decision_slots)
        else:
            self._slots = default_slots(open_offset_minutes)
        # Validated HERE rather than at fire time. A slot that cannot be resolved must not become a
        # session that silently decides fewer times than configured — the #26 shape.
        validate_slots(self._slots)
        #: Optional `() -> tuple[str,...] | None`, re-read at every `_arm()`. THE SETTINGS KNOB WAS
        #: DEAD WITHOUT IT (kumo-trading-platform issue 514): cockpit resolved `*_SLOTS` once at build and passed a
        #: VALUE, this captured it, and `set_time_alert(..., override=True)` re-armed from the
        #: boot-time copy forever. An operator could edit the slots, see it accepted, and watch the
        #: lane fire at the old times until the next redeploy. Same dead-knob class as
        #: `qc27.ALLOCATED_EQUITY`.
        #:
        #: A CALLABLE beside the captured value, exactly as `qc27_runner.read_state` sits beside the
        #: frozen `lifecycle` value object. MUST BE SYNC AND CHEAP — `_arm` runs on the Nautilus
        #: clock callback. `None` is today's behaviour byte-for-byte.
        self._read_slots = read_slots
        self._armed_slot: str | None = None
        self._due_slot: str | None = None
        self._max_stale_days = (max_stale_days if max_stale_days is not None
                                else cfg.execution.max_stale_days)
        self._armed_session: pd.Timestamp | None = None   # session the pending alert belongs to
        self._due: pd.Timestamp | None = None             # session whose open we are trading now
        self.missed_sessions: list[pd.Timestamp] = []     # alerts that never found tradeable data
        self._last_blocked_due: pd.Timestamp | None = None  # dedup guard, see `_try_decide`

        # LIVE decision path. When present, the alert hands the panel to `PgSessionRunner`, which
        # owns everything a backtest has no notion of: pool freshness, lifecycle gating, the
        # ownership boundary against foreign positions, the daily-loss halt, and the journal row
        # that makes a session idempotent. Absent (backtest), `_decide_for` runs the local path.
        #
        # Both call the SAME apply_gates/score_panel/decide. What differs is the safety and
        # persistence wrapper around them, which only exists where there is an operator and an
        # account — not a second copy of the strategy.
        # `rebalance_band` IS MEASURED IN RESEARCH AND UNREADABLE LIVE (#184). It is read in
        # `backtesting/` twice and in `runtime/` nowhere: this runner models trading as an ENTRY
        # sized once at arrival and an EXIT that sells everything, and nothing revisits a position's
        # size. Setting the knob changes a backtest and changes NOTHING here.
        #
        # Guarded rather than ticketed because the field's own docstring cites a measurement and
        # quotes the person who asked for it (2026-09-04: "There is a sizing. The sizing needs
        # to be executed."), which is exactly what makes someone confident enough to set it. A knob
        # the live path cannot read is SILENT the day it is set — the lane keeps entry-and-full-exit
        # and the operator believes it is banding.
        #
        # THE SAME CAPABILITY AS SMHGLD'S, aimed at a different lane: resizing a held position IS
        # delta execution, so this is its second consumer rather than a parallel mechanism.
        #
        # CONDITIONAL, so no live lane changes. Every deployed config has `rebalance_band=None` and
        # never reaches this; the guard exists for the day one does not.
        # A DIRECT ATTRIBUTE READ, NOT `getattr(..., "rebalance_band", None)`. Both are safe;
        # only one is VISIBLE. A name passed as a string is invisible to every structural
        # check in this repo — `test_runtime_STILL_reads_it_nowhere` found zero reads while
        # this line executed on every construction, and the same `getattr` form hid
        # `market_view` from the backtest-only detector earlier today. `PortfolioConfig`
        # always carries the field, so the defensive form buys nothing and costs visibility.
        band = cfg.portfolio.rebalance_band
        if band is not None and not shadow_only:
            require(
                session_runner, EXECUTES_DELTAS, lane=f"{EXTERNAL_ID} (rebalance_band={band!r})",
                needs=(f"`rebalance_band={band!r}` asks the runner to RESIZE a held position when "
                       f"its weight drifts outside the band. This runner enters a position once at "
                       f"arrival and exits it whole; it never revisits a size, so the band is read "
                       f"by the backtest and by nothing live (#184). Left unguarded it would change "
                       f"your research result and nothing about what trades."))
        self._runner = session_runner
        # The runner refreshes due sources as step 1 and treats a stale one as a HARD BLOCK. Pass
        # the job runner or that step is skipped entirely: a source can sit past its cadence but
        # inside the staleness window, so the fetch that would now fail is never attempted and the
        # session decides on an old pool believing it is fresh.
        self._jobs = session_jobs
        self._session_task = None
        self._refresh_task = None
        # The event loop, captured in on_start. Nautilus LiveClock fires timer callbacks from its own
        # thread, so `asyncio.get_running_loop()` inside a callback raises RuntimeError — and both
        # call sites treated that as "not a live node" and returned SILENTLY. The result was a
        # strategy that started, armed, subscribed to 71 instruments, logged that it was watching
        # them, and then never refreshed a source or ran a session. Nothing errored; it simply did
        # nothing, which is the failure mode this whole runtime exists to avoid.
        self._loop = None

        # Nautilus `AccountState` models CASH ONLY, and cockpit's Alpaca client sets
        # AccountBalance.total to the cash figure. Equity for a risk limit has to be net
        # liquidation, so it comes off the broker's own snapshot on the bus (refreshed ~3s), which
        # carries Alpaca's portfolio_value. The topic is a constructor arg rather than an import so
        # this package keeps no dependency on cockpit.
        self._account_topic = account_topic
        self._broker_account: dict | None = None

    # -- lifecycle ---------------------------------------------------------------------------
    def on_start(self) -> None:
        # A RESTART IS THE NORMAL CASE, and `_held` is built empty. Until it is seeded, every
        # position predating this boot is invisible to `protective_close`, so the venue's answer
        # to a protective stop-out goes unrecorded and the claim outlives the position (#194).
        self.seed_held_from_positions()
        # Guarded at the CALL SITE, not only inside the method: `self.cache` is a read-only Cython
        # attribute on Nautilus's Actor, so evaluating it is what raises on any host that does not
        # have one. A strategy constructed from ids has nothing to resolve and must not need one.
        self._resolve_symbols_if_needed()
        for iid in self._iids:
            bt = self._bar_type(iid)
            if self._history_days:
                # The instrument DEFINITION, not just its bars. Nautilus needs it to size and price
                # an order, and `cache.instrument()` returning None is a hard refusal at submit —
                # which is exactly what happened on the first live run: every entry priced and
                # journalled, then rejected with "no instrument definition cached". Bars alone do
                # not populate it; the cockpit's own UiFeedStrategy requests definitions explicitly
                # (engine_node.py:470) and this did not.
                # ALREADY CACHED IS THE COMMON CASE since #622 — the adapter pushed every
                # instrument into the Cache during connect, and symbol resolution read them from
                # there. Asking again costs a request against an IBKR budget of ~60 per 10 minutes
                # that four lanes share, and the overflow comes back as an empty array with no
                # error (kumo-trading-platform issue 617). The definition is still requested when it is absent.
                self.request_instrument_if_missing(iid)
                # live: pull history first so warmup can complete before the first live bar
                start = self.clock.utc_now() - pd.Timedelta(days=self._history_days)
                self.request_bars(bt, start)
            self.subscribe_bars(bt)
        # on_start DOES run on the event loop; timer callbacks do not. Capture it here so the
        # callbacks can dispatch onto it from whatever thread they arrive on.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None      # backtest: no loop, and none needed — nothing dispatches
        self.msgbus.subscribe(self._account_topic, self._on_broker_account)
        # Deferred, and OFF the loop: `_arm` reaches the broker's calendar, and the first reach is a
        # blocking `urlopen`. See `RegistrationMixin.begin_arming` — this exact line, in BCTROT's
        # inherited copy of this method, took `Trader.START` down on 2026-08-22.
        self.begin_arming()
        if self._jobs is not None:
            # Sources refresh on their own cadence, INDEPENDENT of the session. A pool that only
            # updates when someone runs a session is a pool that is always stale on the morning it
            # matters, and the runner treats a stale source as a hard block — so without this the
            # first thing the session would find is a reason not to trade.
            self.clock.set_timer(SOURCE_TIMER, pd.Timedelta(seconds=SOURCE_REFRESH_SECS),
                                 callback=self._on_source_timer)
        self.log.info(f"{self.id} watching {len(self._iids)} instruments, "
                      f"warmup needs {self._need} bars, first session {self._first_session_note()}")

    def on_stop(self) -> None:
        self.cancel_arming()
        for name in (SESSION_ALERT, SOURCE_TIMER):
            if name in self.clock.timer_names:
                self.clock.cancel_timer(name)
        # Cancelling the timers is not enough. A session already in flight is a raw asyncio task,
        # not something Nautilus manages, so it outlives stop() and keeps running against the
        # lifecycle it read when it STARTED. An operator who pauses mid-session would watch the
        # strategy submit anyway — the stop button appearing to work is the whole problem.
        self._due = None
        for task in (self._session_task, self._refresh_task):
            if task is not None and not task.done():
                task.cancel()
        self._session_task = self._refresh_task = None

    # -- session clock -------------------------------------------------------------------------
    # `_report_missed_on_start` and `_arm_now` LIVED HERE and are now on `RegistrationMixin`.
    # Keeping them on this adapter is how QC345-003 lost its 2026-08-25 session: the check was
    # written for MOMENTUM-002's 2026-08-17 loss, sat on this class, and the two lanes that schedule
    # off an offset instead of a slot tuple never got it. One implementation, on the mixin every
    # lane already uses to arm, so a fourth lane cannot be added without it.


    def _reread_slots(self) -> None:
        """Refresh `_slots` from settings, or keep what we have.

        NEVER LETS A SETTINGS READ STOP THE LANE SCHEDULING. A lane that stopped arming looks exactly
        like one that decided to hold, which is why `_on_session_alert` re-arms first. A raising
        reader, a None, an empty tuple, or one that fails `validate_slots` all leave the current
        schedule in place and log — validated BEFORE adoption, so a bad edit cannot become a session
        that silently decides fewer times than configured (the #26 shape).
        """
        if self._read_slots is None:
            return
        try:
            got = self._read_slots()
        except Exception as exc:                                       # noqa: BLE001
            self.log.warning(f"{self.id}: could not re-read decision slots ({exc}) — "
                             f"keeping {list(self._slots)}")
            return
        if not got:
            return
        slots = tuple(got)
        if slots == self._slots:
            return
        try:
            validate_slots(slots)
        except Exception as exc:                                       # noqa: BLE001
            self.log.warning(f"{self.id}: settings gave unusable decision slots {list(slots)} "
                             f"({exc}) — keeping {list(self._slots)}")
            return
        self.log.info(f"{self.id}: decision slots moved {list(self._slots)} -> {list(slots)} "
                      f"by settings")
        self._slots = slots

    def _arm(self, after) -> None:
        """Arm the next decision across ALL configured slots.

        `override=True` still reuses the one Nautilus alert — there is only ever one alert pending,
        for whichever slot comes next. What changed is that "next" is now a minimum across slots
        rather than tomorrow's single offset, so two slots in one session arm in sequence.
        """
        self._reread_slots()
        session, slot, fire_at = next_slot_fire(self._calendar, after, self._slots)
        self._armed_session = pd.Timestamp(session)
        self._armed_slot = slot
        self.clock.set_time_alert(SESSION_ALERT, fire_at, self._on_session_alert, override=True)

    def _on_session_alert(self, event=None) -> None:
        fired, self._armed_session = self._armed_session, None
        fired_slot, self._armed_slot = self._armed_slot, None
        # Re-arm FIRST. Anything below may raise, and a strategy that has stopped scheduling looks
        # exactly like a strategy that decided to hold — silent, and wrong for as long as nobody looks.
        # GUARDED RE-ARM (kumo-trading-platform issue 628). Still FIRST, for the reason above — but `_arm` can now
        # refuse, because a venue calendar knows only a rolling window. An unguarded raise here would
        # abort this callback: the session that just fired would never be marked due, no decision
        # would be taken for it, and no future alert would be set. Silent, permanent, mid-session.
        self.rearm_after_alert(self.clock.utc_now())
        if self._due is not None:
            self.log.warning(f"{self.id}: session {self._due.date()} closed without tradeable "
                             f"data — no decision was made")
            self.missed_sessions.append(self._due)
        self._due = fired
        self._due_slot = fired_slot
        self._try_decide()

    def _try_decide(self) -> None:
        """Decide for `_due` if the panel has caught up. Called by the alert and by every bar."""
        due = self._due
        if due is None:
            return                      # the common, every-bar case — nothing pending, say nothing
        # READ ONCE, THEN USE THE LOCAL (2026-08-21). This method passed a None-guard and then read
        # `self._due` four more times across twenty lines, calling out to `self._panel()` in between.
        # Line 375 below is the only writer that sets it back to None, so any path that re-enters
        # between the guard and a later read walks past a check that has already been satisfied. It
        # did: `panel.date < self._due` raised
        #     TypeError: Invalid comparison between dtype=datetime64[ns] and NoneType
        # inside `on_bar`, so Nautilus terminated the whole TradingNode ("System will terminate
        # immediately to prevent operation in degraded state") mid-session with positions open, and
        # BCTROT-004 lost its 12:00 decision slot to the restart gap.
        #
        # The local closes the entire class, not just the one re-entry path that happened to fire.
        if self._pending or not self.warm:
            # Logged once per `_due` session, not once per bar: `on_bar` fires per SYMBOL, so an
            # undeduped log here would flood the moment warmup or an in-flight order stalls for more
            # than one bar. `_last_blocked_due` bounds it to one line per blocked session, which is
            # exactly the visibility this branch never had -- a strategy stuck here for its entire
            # first live day produced zero decisions and left NOTHING in the log distinguishing
            # "still warming" from "correctly found nothing to trade" (BCTROT, 2026-08-19).
            if self._last_blocked_due != due:
                self._last_blocked_due = due
                ready = sum(1 for v in self._bars.values() if len(v) >= self._need)
                need = self._cfg.portfolio.n_hold + self._cfg.portfolio.buffer
                reason = "orders in flight" if self._pending else f"still warming ({ready}/{need} symbols ready)"
                self.log.warning(
                    f"{self.id}: session {due.date()} waiting — {reason}, will retry on the next bar")
            return                      # orders in flight or still warming — retry on the next bar
        self._last_blocked_due = None
        panel = self._panel()
        if panel.empty:
            return
        prior = panel.loc[panel.date < due, "date"].max()
        if pd.isna(prior):
            return                      # no completed session yet; keep waiting
        session, self._due = due, None
        self._report_dataless()
        stale = (session - prior).days
        if stale > self._max_stale_days:
            # Deciding here would rank on a picture of the market that is days old. A gap this size
            # is a data outage, not a holiday, and the safe response is to hold what we have.
            self.log.error(f"{self.id}: newest bar {prior.date()} is {stale}d before session "
                           f"{session.date()} — refusing to decide on stale data")
            self.missed_sessions.append(session)
            return
        # Score on bars up to `prior` ONLY. Both live and backtest now put session `session`'s own
        # bar AFTER this alert -- live because a daily bar cannot exist at its own open, backtest
        # because `bars_for` stamps it at the session close in ET. So this trim is belt-and-braces
        # rather than load-bearing, and it stays that way deliberately: the corporate-action gate
        # reads a window around each row, so any future change that moves bar timestamps earlier
        # would otherwise let `session`'s own prices vote on a decision for `prior` -- lookahead
        # that exists only in backtest and would flatter the number we trade on.
        usable = panel[panel.date <= prior]
        # TODAY'S OPENS, carried SEPARATELY from the panel. The trim above is correct and stays:
        # today's prices must not vote on today's ranking. But `min_abs_gap_pct` needs the overnight
        # gap, which is `today's open / yesterday's close`, and the trimmed panel by construction
        # contains neither of those as "today". An earlier version read `day["open"]` inside the
        # runner and got YESTERDAY's open against the day-before's close — the previous session's
        # gap, applied silently at all three slots and journalled as healthy (found in review,
        # 2026-09-05).
        #
        # `self._bars` DOES hold today's forming daily bar: Alpaca republishes it and `_ingest`
        # replaces rather than appends. So the open is available here and only here — it is the one
        # piece of today the runner is allowed to see, because it is the price the runner is about
        # to trade against rather than an input to the ranking.
        # Guarded on COLUMNS, not just emptiness: a panel can arrive with rows and without an
        # `open` column, and an AttributeError here reaches `on_bar` and kills the session — the
        # gap filter must be able to be inert, never fatal.
        opens: dict = {}
        if {"ticker", "open", "date"} <= set(panel.columns):
            today = panel[panel.date == session]
            if len(today):
                opens = dict(zip(today["ticker"], today["open"]))
        if self._runner is not None:
            self._run_session(session, usable, self._due_slot or self._slots[0], opens=opens)
        else:
            self._decide_for(prior, usable)

    # -- live session --------------------------------------------------------------------------
    def _run_session(self, session: pd.Timestamp, panel: pd.DataFrame,
                     slot: str, opens: dict | None = None) -> None:
        """Hand the session to the Postgres runner, on the node's own event loop.

        The runner is async and this callback is not, so the work is scheduled rather than awaited —
        blocking here would stall the event loop that delivers fills for the orders being placed.

        One task at a time. A second alert while the first session is still running would race two
        runners at the same journal row; the database would refuse the duplicate decision, but only
        after both had read the same book.
        """
        import asyncio

        if self._session_task is not None and not self._session_task.done():
            self.log.warning(f"{self.id}: session {session.date()} skipped — the previous "
                             f"one is still running")
            self.missed_sessions.append(session)
            return
        if self._loop is None:
            # Refusing is the honest answer: silently falling back to the local path would run the
            # session WITHOUT the lifecycle gate, so a DISABLED or SHADOW strategy would submit.
            self.log.error(f"{self.id}: no event loop captured — refusing to run session "
                           f"{session.date()} outside the live runtime")
            self.missed_sessions.append(session)
            return
        self._session_task = asyncio.run_coroutine_threadsafe(
            self._session_coro(session, panel, slot, opens), self._loop)

    def _on_broker_account(self, snapshot: dict) -> None:
        self._broker_account = snapshot

    def broker_equity(self) -> float:
        """Net liquidation, from the broker's own snapshot.

        NOT `account.balance_total(USD)`. Nautilus models cash only, and cockpit's Alpaca client
        puts the cash figure in `AccountBalance.total` -- so reading it would compare a $20k cash
        balance against a $100k session-start equity and halt the strategy for a drawdown that
        never happened, while missing a real mark-to-market drawdown entirely because cash does not
        move when positions do.

        Raises until the first snapshot arrives. The daily-loss limit is the only automatic stop
        this strategy has; running it against a number that might be cash is worse than not running
        it, and returning 0.0 would disarm it silently.

        NON-FINITE IS NOT A NUMBER. `float(x or 0.0)` does not guard a NaN — **NaN is truthy**, so
        the `or` never fires — and `equity <= 0` does not catch one either, because every comparison
        with NaN is False. A nan therefore reached the daily-loss halt, whose `now < start * (1 -
        frac)` is then also False, so the halt NEVER FIRES. That is exactly what "returning 0.0
        would disarm it silently" was written to prevent; the guard caught the zero and let the
        NaN through (found 2026-08-24, sweeping for the `or`-default trap behind 4d46290).
        """
        if self._broker_account is None:
            raise RuntimeError(
                f"no broker account snapshot on {self._account_topic!r} yet — refusing to evaluate "
                f"the daily-loss limit against an unknown equity")
        raw = self._broker_account.get("equity")
        equity = float(raw) if raw is not None else 0.0
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"broker reported equity {raw!r} — refusing to gate risk on it")
        return equity

    def _session_date(self) -> str:
        """Today's ET trading date. Sessions are ET everywhere else in this system, and after
        19:00 ET the UTC date is already tomorrow -- so a UTC-derived key files an evening refresh
        under the NEXT session, where a staleness failure for today would never be looked for."""
        from kumo_strategies.runtime.calendar import elapsed_slots, next_slot_fire, ET
        return str(self.clock.utc_now().astimezone(ET).date())

    def _on_source_timer(self, event=None) -> None:
        """Ask each source whether it is due. Failure is isolated per source by the job runner."""
        import asyncio

        if self._refresh_task is not None and not self._refresh_task.done():
            return                      # previous refresh still running; skip rather than pile up
        if self._loop is None:
            self.log.error(f"{self.id}: no event loop captured — source refresh cannot run")
            return
        # run_coroutine_threadsafe, NOT create_task: this callback arrives on the clock's thread.
        self._refresh_task = asyncio.run_coroutine_threadsafe(self._refresh_coro(), self._loop)

    async def _refresh_coro(self) -> None:
        try:
            await self._jobs.refresh_due(session=self._session_date())
            await self._check_urgent()
        except Exception as e:                                        # noqa: BLE001
            # Never let this kill the timer. A refresh loop that dies leaves the pool frozen at its
            # last good set, which stays inside the staleness window for hours before anything says so.
            await self._record("source refresh failed", e)

    async def _record(self, what: str, exc: BaseException) -> None:
        """Log AND journal. A log line is not a record -- it is not queryable, it does not survive a
        container restart, and nobody reads it on a morning when the strategy simply did nothing.

        The failures that reach here are the ones that happen BEFORE the runner can write its own
        risk row: a source refresh that raised, a database that was unreachable at session start.
        Those are exactly the cases where the strategy looks healthy and idle, so they are the ones
        that most need to be durable. Journalling is best-effort by necessity -- if Postgres is what
        failed, this cannot work either -- but the log line is emitted first so it is never lost to
        a second failure.
        """
        self.log.error(f"{self.id}: {what}: {type(exc).__name__}: {exc}")
        journal = self.session_journal()
        if journal is None:
            return
        try:
            await journal.write("risk", f"{what}: {type(exc).__name__}: {exc}",
                                session=self._session_date())
        except Exception as e:                                        # noqa: BLE001
            self.log.error(f"{self.id}: could not journal the failure above: {e}")

    async def _check_urgent(self) -> None:
        """Give the operator's emergency controls a response time measured in a minute, not a day.

        LIQUIDATING and a blacklist on a held name are both the operator saying they know something
        the ranking does not. Honouring them only at the next scheduled alert meant a 10:00 decision
        to flatten did nothing until the following morning -- with the UI reporting LIQUIDATING the
        whole time. This rides the refresh timer that already runs every minute rather than adding
        a second clock.

        The runner's urgent path submits exits ONLY and writes no decision row, so triggering it
        here cannot consume or collide with the session's idempotency key.
        """
        urgent = getattr(self._runner, "urgent_pending", None)
        if urgent is None or self._session_task is not None and not self._session_task.done():
            return
        if not await urgent():
            return
        session = pd.Timestamp(self._session_date())
        self.log.warning(f"{self.id}: urgent operator action pending — running {session.date()} "
                         f"outside the schedule")
        panel = self._panel()
        self._session_task = asyncio.get_running_loop().create_task(
            # An OUT-OF-SCHEDULE run, so there is no armed slot to attribute it to. It takes the
            # first configured slot rather than inventing a name: the operator-urgent path is a
            # substitute for that session's decision, and a slot the config does not contain would
            # sit outside the (session, slot) key every other check reasons about.
            self._session_coro(session, panel[panel.date < session] if len(panel) else panel,
                               self._slots[0]))

    async def _session_coro(self, session: pd.Timestamp, panel: pd.DataFrame,
                            slot: str, opens: dict | None = None) -> None:
        try:
            # A POSITION ON THE SIDE THIS LANE CANNOT MANAGE IS REPORTED EVERY SESSION (#66).
            # The exit path already REFUSES one, but only when the lane is trying to exit that
            # name — WHD sat in the book for twelve hours because nobody was. Before the
            # decision, because a book we do not understand is context for what follows.
            report_wrong_sided_positions(self, session=str(session.date()))
            result = await self._runner.run(panel, str(session.date()), jobs=self._jobs,
                                            slot=slot, opens=opens)
            submitted = getattr(result, "submitted", 0)
            blocked = getattr(result, "blocked", None)
            outcome = session_outcome(result.state, decided=result.decided, blocked=blocked,
                                      sent=submitted)
            self.log.info(f"{self.id} {session.date()}: {outcome}")
            # DURABLE outcome, every session, decided or not. A refused session used to leave only
            # this log line — so at the default WARNING level the journal showed nothing at all and
            # there was no record of WHY nothing happened. "Session ran and declined" and "session
            # never ran" looked identical, which is the single most expensive ambiguity here.
            journal = self.session_journal()
            if journal is not None:
                await journal.write(
                    "state", f"session outcome — {outcome}", session=str(session.date()),
                    detail={"state": result.state, "decided": result.decided,
                            "blocked": blocked, "submitted": submitted,
                            # The integer keeps its name for cockpit's existing readers; this says
                            # what it MEANS, so local acceptance cannot be read as venue acceptance
                            # by anyone holding only the JSON (kumo-trading-platform issue 512).
                            "submitted_basis": SENT_BASIS,
                            "entered": list(getattr(result, "entered", ()) or ()),
                            "exited": list(getattr(result, "exited", ()) or ())})
        except asyncio_CancelledError:
            raise                       # on_stop cancelled us; that is not a failure to record
        except Exception as e:                                        # noqa: BLE001
            # The runner journals its own failures once it is running. This catch covers everything
            # BEFORE that point, and stops a raise from killing the task silently and leaving the
            # next alert to believe the previous session succeeded.
            self.missed_sessions.append(session)
            await self._record(f"session {session.date()} failed", e)

    def _bar_type(self, iid: InstrumentId):
        from nautilus_trader.model.data import BarType
        return BarType.from_str(f"{iid}{self._suffix}")

    # -- data --------------------------------------------------------------------------------
    def on_historical_data(self, data) -> None:
        if isinstance(data, Bar):
            self._ingest(data)

    def on_bar(self, bar: Bar) -> None:
        """Ingest, then retry a decision the alert could not complete.

        Bars arrive one symbol at a time, so the panel is only partial for most of them. This does
        not trigger anything by itself — `_try_decide` is a no-op unless an alert has armed `_due`.
        """
        self._ingest(bar)
        self._try_decide()

    def _ingest(self, bar: Bar) -> None:
        """Keep exactly ONE bar per session per symbol — the latest version of it.

        Alpaca republishes the CURRENT session's daily bar as it updates, so appending every arrival
        put one copy of today's bar in the panel per update. After thirteen hours of uptime each
        symbol carried ~103 copies of a single date. `day = panel[date == date.max()]` then selected
        all of them (7289 rows across 71 tickers), and the trailing 20-bar window for the newest rows
        held nothing but repeats of one price: zero variance, NaN score. The 2026-08-05 session found
        0 of 7289 candidates rankable and correctly refused to decide, so the strategy sat inert.

        It survived the first day only because that engine had restarted eight minutes before its
        session and had not accumulated any duplicates yet — the bug needed uptime to appear.
        """
        sym = bar.bar_type.instrument_id.symbol.value
        bars = self._bars[sym]
        day = pd.Timestamp(bar.ts_event, unit="ns").normalize()
        if bars and pd.Timestamp(bars[-1].ts_event, unit="ns").normalize() == day:
            bars[-1] = bar          # same session, republished — replace rather than append
        else:
            bars.append(bar)

    @property
    def warm(self) -> bool:
        """True once enough symbols carry enough history to rank meaningfully.

        Live data arrives asynchronously, so this is a real state, not a formality — acting before
        it is true means ranking on half-formed indicators.
        """
        ready = sum(1 for v in self._bars.values() if len(v) >= self._need)
        return ready >= self._cfg.portfolio.n_hold + self._cfg.portfolio.buffer

    # -- decision ----------------------------------------------------------------------------
    def _decide_for(self, session: pd.Timestamp, panel: pd.DataFrame | None = None) -> None:
        """Score the just-completed `session` and act at the open now underway.

        `panel` is passed in already trimmed to `session`; the fallback is for direct callers.
        """
        if self._last_decision == session or not self.warm or self._pending:
            return          # one decision per session; never while orders are in flight
        self._last_decision = session

        panel = self._panel() if panel is None else panel
        if panel.empty:
            return
        gated = apply_gates(panel, self._cfg, self._itype)
        scored = score_panel(gated, self._cfg)
        today = scored[scored.date == session]
        if today.empty:
            return

        # Trailing correlation/volatility, without which `max_correlation` and
        # `inverse_vol_sizing` are inert HERE too — `_diversified` returns the plain ranking on
        # `corr is None` and `_weights` returns equal weight on `vol is None`, and neither logs
        # (#26). Cockpit runs this adapter (backend/strategies/momentum.py:262), so the flags were
        # settable on a live path and did nothing.
        #
        # No `as_of`: this panel is the trailing history this strategy has been fed, so it is
        # point-in-time by construction. Same helper as the backtest and the Pg runner — a second
        # copy is how the three drift apart unnoticed.
        corr = vol = None
        if needs_panel_stats(self._cfg):
            corr, vol = panel_stats(trailing_returns(panel), self._cfg.portfolio.corr_window,
                                    sorted(set(self._held) | set(today["ticker"])))
        d = decide(today, self._source, self._cfg, set(self._held), corr=corr, vol=vol)

        # ExitConfig reaches behaviour only through `evaluate_exits`, and this adapter never called
        # it — so give_back_frac, stall_days, off_peak_pct and max_hold_days all did nothing here,
        # on a path Cockpit imports (#26). Positions left on the ranking alone.
        #
        # The trail is RECONSTRUCTED from bars on every decision rather than remembered. That is the
        # whole reason this is safe without a store: #197 B1 was a remembered peak that did not
        # survive a restart, and an in-memory trail here would reproduce it behind a green suite. A
        # rebuilt trail has nothing to lose — a fresh process derives the same state from the same
        # bars.
        forced = self._trail_exits()
        exits = tuple(sorted(set(d.exit) | forced))
        for sym in exits:
            self._close(sym)
        # Sizing as a MULTIPLE of the equal-weight slot, so an equal-weighted book is byte-identical
        # to the flat `equity_per_position` this used before.
        entering = tuple(x for x in d.enter if x not in exits)
        for sym in entering:
            scale = 1.0
            if d.weights and sym in d.weights:
                scale = d.weights[sym] * len(d.weights)
            self._open(sym, today, scale)

    def _trail_exits(self) -> set[str]:
        """Capital-efficiency exits, from a trail rebuilt out of bars rather than carried in memory.

        Entry price comes from the Nautilus position — the real average fill — not from a quote.
        Seeding entry from today's price is what made the trail fiction in #197 B1.

        A position whose bar history does not reach back to its open is marked ADOPTED by
        `reconstruct_trail`, so peak-relative rules skip it rather than trusting a peak that may
        never have happened.
        """
        if not self._held:
            return set()
        states, prices = {}, {}
        for pos in self.cache.positions_open(strategy_id=self.id):
            sym = pos.instrument_id.symbol.value
            if sym not in self._held:
                continue
            bars = list(self._bars.get(sym, ()))
            if not bars:
                continue
            opened = pd.Timestamp(pos.ts_opened, unit="ns").normalize()
            held_bars = [b for b in bars
                         if pd.Timestamp(b.ts_event, unit="ns").normalize() >= opened]
            # Covered only if the deque actually reaches back to the open — the oldest bar we hold
            # must not be NEWER than the session the position opened in.
            covers = bool(bars) and pd.Timestamp(
                bars[0].ts_event, unit="ns").normalize() <= opened
            closes = [b.close.as_double() for b in held_bars]
            states[sym] = reconstruct_trail(float(pos.avg_px_open), closes, covers_entry=covers)
            prices[sym] = bars[-1].close.as_double()
        if not states:
            return set()
        # THE INPUTS THE RULES NEED, THE WAY pgrunner AND qc345 SUPPLY THEM (#222). This call passed
        # neither `atr` nor `highs`, so an ATR-scaled rule (`take_profit_atr`, `stop_loss_atr`,
        # `give_back_min_peak_atr`) made `evaluate_exits` RAISE here — inside `_try_decide`, on the
        # path a lane takes when built without a runner. Dead for every deployed lane, a landmine
        # for a shadow lane or a test host. Same `trailing_atr` as the runner, off the same bars.
        atr = trailing_atr(self._panel()) if needs_atr(self._cfg.exits) else None
        highs = ({s2: float(self._bars[s2][-1].high.as_double()) for s2 in states}
                 if needs_highs(self._cfg.exits) else None)
        plan = evaluate_exits(self._cfg.exits, prices, states, atr=atr, highs=highs)
        return set(plan.exits)

    def _report_dataless(self) -> None:
        """Name the watched instruments that carry no usable history into this session.

        A symbol whose venue will not resolve is refused loudly at startup. A symbol that resolves
        and is then never sent a single bar is refused by nothing — it just never scores, so it
        falls out of every ranking in silence. FTNR sat in the pool through a whole session exactly
        that way (Alpaca has no such asset and returns zero bars) and the log never mentioned it
        once, while JEPO's louder venue failure was visible from startup. Same practical outcome,
        opposite visibility.
        """
        watched = {i.symbol.value for i in self._iids}
        empty = sorted(s for s in watched if not self._bars.get(s))
        thin = sorted(s for s in watched if 0 < len(self._bars.get(s) or ()) < self._need)
        if empty:
            self.log.warning(
                f"{self.id}: {len(empty)} watched symbols have NO bars — they cannot be "
                f"ranked or traded: {', '.join(empty)}")
        if thin:
            self.log.info(
                f"{self.id}: {len(thin)} watched symbols below the {self._need}-bar warmup: "
                f"{', '.join(thin)}")

    def _panel(self) -> pd.DataFrame:
        rows = []
        for sym, bars in self._bars.items():
            for b in bars:
                rows.append({
                    "ticker": sym,
                    "date": pd.Timestamp(b.ts_event, unit="ns").normalize(),
                    "open": b.open.as_double(), "high": b.high.as_double(),
                    "low": b.low.as_double(), "close": b.close.as_double(),
                    "volume": b.volume.as_double(),
                })
        return pd.DataFrame(rows).sort_values(["ticker", "date"]) if rows else pd.DataFrame()

    # -- orders ------------------------------------------------------------------------------
    def _iid_for(self, symbol: str) -> InstrumentId | None:
        return next((i for i in self._iids if i.symbol.value == symbol), None)

    def _open(self, symbol: str, today: pd.DataFrame, scale: float = 1.0) -> None:
        iid = self._iid_for(symbol)
        if iid is None:
            return
        px = float(today.loc[today.ticker == symbol, "close"].iloc[0])
        qty = int(self._equity * scale / px)
        if qty < 1:
            return
        self._pending[symbol] = "enter"
        self.submit_order(self.order_factory.market(iid, OrderSide.BUY, Quantity.from_int(qty)))

    def _close(self, symbol: str) -> None:
        iid = self._iid_for(symbol)
        if iid is None:
            return
        pos = next((p for p in self.cache.positions_open(strategy_id=self.id)
                    if p.instrument_id == iid), None)
        if pos is None:
            self._held.discard(symbol)
            return
        # LONG ONLY, ASSERTED (#88). A negative quantity used to reach `Quantity.from_int` and raise
        # from a line that reads as an ordinary exit. Refused by name instead.
        qty = closing_quantity(pos.quantity, side=LONG)
        if qty is None:
            why = refusal_reason(pos.quantity, side=LONG)
            if why:
                self.log.error(f"{self.id}: {symbol} — {why}")
            self._held.discard(symbol)
            return
        self._pending[symbol] = "exit"
        self.submit_order(self.order_factory.market(iid, OrderSide.SELL, Quantity.from_int(qty)))

    # -- order events are the ONLY thing that moves book state -------------------------------
    #
    # AND NOT EVERY EVENT THAT ARRIVES IS OURS (kumo-trading-platform issue 748). Nautilus routes order events by
    # the ORDER's strategy_id — `Strategy.submit_order` publishes to `events.order.{strategy_id}` —
    # so once cockpit stamps each protective stop with the lane whose shares it covers, this lane
    # receives fills, cancels and rejections for `PROT-*` orders it never placed. These handlers
    # assumed otherwise: `on_order_filled` popped `_pending` by SYMBOL alone.
    def on_order_filled(self, event) -> None:
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

        # A SELL CANNOT COMPLETE AN ENTRY, whoever sent it. This check needs no shared vocabulary
        # with cockpit and catches the case the prefix would miss if it ever drifted: a protective
        # SELL landing on a pending "enter" was read as that entry completing, so the strategy
        # believed it held a position A STOP HAD JUST SOLD — and would size its next decision off
        # that belief and skip re-entering a name it thinks it owns.
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
        self._record_terminal(event, ok=True, detail="filled")

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
        # Cockpit cancels and re-arms protective stops on a 60s reconciler tick, so a foreign cancel
        # is not rare — it is routine. `_forget` drops the pending intent by SYMBOL, which would clear
        # the in-flight marker for an order still working and unblock a second decision on it.
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
        """Forward the venue's actual answer to the injected runner's journal (#51).

        No-op in the standalone path (`self._runner is None`) and no-op if the loop is not running
        (backtest) -- this closes an observability gap on the live path, it does not gate anything.
        The session is recovered from the order's own tags (`nautilus/broker.py`'s `submit` tags every
        order `session:{session}`) rather than tracked in a parallel in-process map, so it survives a
        restart between submit and this event the same way the order itself does, via Nautilus's own
        durable cache.
        """
        if self._runner is None or self._loop is None:
            return
        record = getattr(self._runner, "record_terminal", None)
        if record is None:
            return
        order = self.cache.order(event.client_order_id)
        session = next((t.split(":", 1)[1] for t in (order.tags or ()) if t.startswith("session:")),
                       None) if order is not None else None
        if session is None:
            return                      # a fill for an order this adapter did not tag -- not ours to record
        sym = event.instrument_id.symbol.value
        # OBSERVED, not fire-and-forget: the discarded Future is how a failed write vanished
        # (kumo-trading-platform issue 549 — DELL had 4 venue fills and 3 journal rows).
        self.fire_and_report(
            call_record_terminal(record, session, sym, ok, detail, log=self.log.warning,
                                 **terminal_fields(self, event)),
            self._loop, f"terminal row for {sym}")
        # THE CLAIM FOLLOWS THE VENUE'S ANSWER, NOT THE SUBMIT (kumo-trading-platform issue 829). pgrunner writes
        # the claim when Nautilus accepts the submit LOCALLY; a denial 3 ms later reverted nothing,
        # and MOMENTUM-002 carried a 260-share LAND claim on a position that never existed. Under
        # NETTING the lane's own position `{instrument}-{strategy_id}` IS the quantity the claim
        # exists to record, so every terminal event re-syncs the claim to it: fill → the filled
        # quantity, rejection/denial of a fresh entry → flat → the claim is dropped, full exit →
        # flat → dropped (the stale VCTR 16 of #830). Optional on the runner, like `record_terminal`.
        sync = getattr(self._runner, "sync_claim", None)
        if sync is not None:
            # NEVER RAISE INTO NAUTILUS'S DISPATCH: a cache read or a price conversion that fails
            # must cost the claim sync, not the handler that is telling us a real order filled.
            try:
                qty = self._lane_quantity(event.instrument_id)
                px = float(event.last_px) if ok and getattr(event, "last_px", None) is not None else None
            except Exception as exc:                                       # noqa: BLE001
                self.log.error(f"{self.id}: claim sync for {sym} NOT attempted — "
                               f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]}")
                return
            self.fire_and_report(sync(sym, qty, px), self._loop, f"claim sync for {sym}")

    def _lane_quantity(self, instrument_id) -> int:
        """Net signed quantity THIS lane holds in the instrument, per Nautilus's own attribution."""
        return int(sum(int(p.signed_qty) for p in
                       self.cache.positions_open(instrument_id=instrument_id, strategy_id=self.id)))

    def _forget(self, event) -> None:
        """A rejected or cancelled order must not leave the book believing it traded."""
        sym = getattr(getattr(event, "instrument_id", None), "symbol", None)
        if sym is not None:
            self._pending.pop(sym.value, None)
