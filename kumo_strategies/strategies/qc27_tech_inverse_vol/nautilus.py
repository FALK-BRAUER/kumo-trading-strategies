"""Nautilus binding for QC27 — Tech Momentum with Inverse Volatility Allocation (#33).

The SAME decision code runs here and in backtest. This class contains no strategy logic: it collects
bars, asks the pure `decide()` what to hold, and turns the answer into orders. Modelled on
`qc345_rotation.py`, which is modelled on `momentum_rotation.py` — so the scars those two earned are
inherited rather than re-earned:

  warmup refusal      live history arrives asynchronously; ranking before it lands ranks noise.
  order-event book    a submit is a request, a fill is a fact. Book state moves ONLY on events.
  calendar alert      not a bar-rollover trigger. A 1-DAY bar for session D prints at D's close, so
                      a rollover fires a full day after the open it meant to trade.
  captured loop       `LiveClock` fires timer callbacks from its OWN thread, so
                      `asyncio.get_running_loop()` inside a callback raises. Captured in on_start.

IDENTITY. `order_id_tag` must be UNIQUE across every strategy in one trader — it is the suffix of the
StrategyId and of every generated client order id, and under NETTING the position id is
`{instrument}-{strategy_id}`. MANUAL holds 001, MOMENTUM 002, QC345 003, BCTROT 004.
`research/qc27/ACTIVATION.md` proposed `TECH_IVOL-003`, which COLLIDES with QC345 — two strategies
sharing a tag does not degrade, it raises at `Trader.add_strategy` and the node does not boot. That
is the same 003 -> 004 churn BCTROT already went through. So there is NO DEFAULT here: cockpit
allocates the tag and must pass it, because only cockpit sees every strategy in one trader.

MONTHLY CADENCE, same construction as QC345. The alert fires per SESSION and the monthly rule is
applied on top by asking the pure `rebalance_dates()` whether the due session is a month's first. A
second, adapter-local notion of "is it rebalance day" would be a second derivation of a fact the
backtest already derives, and two derivations of one fact disagree.

WHAT THIS ADAPTER DOES NOT OWN. Lifecycle gating, the journal, idempotency, the budget gate and risk
limits all live in the cockpit-side gateway, exactly as they do for QC345 (`strategies/qc345.py`).
When `session_runner` is None this strategy submits DIRECTLY with none of those controls — which is
correct for a backtest and would be dangerous live, so the caller must pass one in production.
"""

from __future__ import annotations

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin

import math

import asyncio
from asyncio import CancelledError as asyncio_CancelledError
from collections import defaultdict, deque
import traceback

import pandas as pd
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.runtime.nautilus.contract import (
    report_wrong_sided_positions, protective_close, SENT_BASIS, RegistrationMixin, session_outcome,
    call_record_terminal, terminal_fields)
from kumo_strategies.runtime.nautilus.sides import (
    LONG, closing_quantity, refusal_reason)
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    QC27TechInverseVolConfig,
    build_feature_panel,
    decide,
    rebalance_dates,
)

STRATEGY_NAME = "TECHIVOL"
EXTERNAL_ID = "QC27"
STRATEGY_LABEL = "QC27 tech momentum, inverse-vol weighted"
SESSION_ALERT = "qc27_session_decide"

#: What `_ingest` can actually put in the panel, because it is what a Nautilus `Bar` carries.
#: Anything the pure engine asks for beyond this cannot exist on the live path.
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


def warmup_bars_needed(cfg: QC27TechInverseVolConfig) -> int:
    """Bars per symbol before a decision is trustworthy.

    MAX, not sum: every window is a TRAILING window ending at the same bar, so they overlap. The
    momentum term needs `lookback + 2` observations to be non-NaN (two shifts, not a safety margin).
    `warmup_sessions` is QC27's own stated 100-session warmup and is included so a config change
    cannot silently under-count.

    Deciding short of this does not fail loudly — `eligible` simply goes near-empty and the strategy
    "chooses" to hold almost nothing. That is why warmup is a refusal with a log line, not a filter.
    """
    return max(
        cfg.lookback_sessions + 2,
        cfg.warmup_sessions,
        cfg.liquidity_window + 1,
        cfg.realized_vol_window + 1,
    )


class QC27RotationStrategy(MarketAwareMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin, Strategy):
    """QC27 on the live clock. Logic lives in the pure engine, not here."""

    EXTERNAL_ID = EXTERNAL_ID
    #: STATED, not inherited. `order_provenance.completes` and `broker.py` both ask the lane
    #: which side it holds, and neither has a default — a short lane that forgot to say so
    #: would read its own entry fill as foreign noise (#88, #123).
    POSITION_SIDE = LONG
    LABEL = STRATEGY_LABEL

    def __init__(
        self,
        cfg: QC27TechInverseVolConfig,
        instrument_ids: list[InstrumentId] | None = None,
        *,
        # KEYWORD-ONLY, DELIBERATELY. Inserted as a positional parameter this silently shifts
        # every argument after it: a caller passing `(cfg, source, ids, suffix)` would bind the
        # suffix to `symbols`, and `_init_symbols` would then see both and refuse at BUILD — which
        # takes every other lane in the trader down with it (the #377 shape). Two repos share this
        # signature across a version pin, so a positional insert is not a change either side can
        # make safely.
        symbols: list[str] | None = None,
        order_id_tag: str,
        strategy_name: str = STRATEGY_NAME,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        equity: float = 20_000.0,
        history_days: int | None = None,
        calendar: object | None = None,
        open_offset_minutes: int = 5,
        read_open_offset: object | None = None,
        session_runner: object | None = None,
        account_topic: str = "broker.account",
        external_order_claims: list[InstrumentId] | None = None,
    ) -> None:
        # `order_id_tag` is REQUIRED and deliberately has no default — see the module note. The
        # activation doc's proposed 003 belongs to QC345 and would stop the node booting.
        super().__init__(config=StrategyConfig(
            strategy_id=strategy_name, order_id_tag=order_id_tag,
            external_order_claims=list(external_order_claims) if external_order_claims else None))
        self._claims = list(external_order_claims or [])
        self._cfg = cfg
        # SYMBOLS OR IDS, EXACTLY ONE (kumo-trading-platform issue 622). See `symbol_resolution`.
        self._init_symbols(instrument_ids, symbols)
        self._suffix = bar_type_suffix
        self._equity = equity
        self._history_days = history_days

        # A NAUTILUS BAR CARRIES OHLCV AND NOTHING ELSE. `build_feature_panel` requires
        # `cfg.momentum_price_field` as a COLUMN, and its default is `close_adj` -- a
        # split/dividend-adjusted series that the live feed cannot produce and never will. Left
        # unchecked this does not fail at startup: it fails at the FIRST REBALANCE, weeks later,
        # with `bars missing momentum price field 'close_adj'` -- the same shape as QC345's
        # KeyError: 'eligible', which sat latent until the strategy was finally enabled and then
        # booked nothing on its first live session.
        #
        # Refuse at CONSTRUCTION instead. The node fails to boot, loudly, in front of whoever is
        # deploying, rather than silently a month later in front of nobody.
        #
        # The fix for a live config is `momentum_price_field="close"`, which is what QC345's own
        # live_config does -- but QC345 pairs it with a 252-session corporate_action_window to
        # neutralise splits, and QC27 HAS NO SUCH FIELD. Raw close therefore leaves QC27 exposed:
        # a 2:1 split prints as -50% and destroys that name's momentum rank. That gap is recorded
        # in research/qc27/BUILD_STATE.md and is a strategy decision, not one this adapter may take.
        if cfg.momentum_price_field not in _BAR_FIELDS:
            raise ValueError(
                f"momentum_price_field={cfg.momentum_price_field!r} cannot be supplied by a live "
                f"Nautilus bar, which carries only {sorted(_BAR_FIELDS)}. The strategy would run "
                f"until its first rebalance and then raise. Set momentum_price_field='close' for "
                f"the live config -- and note QC27 has no corporate_action_window, so raw close is "
                f"unprotected against splits (see research/qc27/BUILD_STATE.md).")

        self._need = warmup_bars_needed(cfg)
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._need + 5))

        # Book state, driven by ORDER EVENTS — never by the fact that submit was called. A submit
        # that is rejected, denied or partially filled leaves this untouched, which is the point.
        self._held: set[str] = set()
        self._pending: dict[str, str] = {}          # symbol -> "enter" | "exit"

        if calendar is None:
            # `runtime.calendar`, NOT `runtime.executor.calendar` — the latter's package __init__
            # imports SQLAlchemy, which has no business in a backtest.
            from kumo_strategies.runtime.calendar import build_calendar
            calendar = build_calendar()
        self._calendar = calendar
        self._open_offset = open_offset_minutes
        self._read_open_offset = read_open_offset
        self._armed_session: pd.Timestamp | None = None
        self._due: pd.Timestamp | None = None
        self.missed_rebalances: list[pd.Timestamp] = []
        self.warmup_deferrals = 0        # sessions skipped because the universe was not yet warm
        self.skipped_sessions = 0

        self._runner = session_runner
        self._session_task = None
        self._loop = None
        self._account_topic = account_topic
        self._broker_account: dict | None = None

    @property
    def _by_symbol(self) -> dict[str, InstrumentId]:
        """DERIVED ON READ, never stored (kumo-trading-platform issue 622).

        This was a dict built in `__init__`. Once `symbols` became the input, `_iids` was EMPTY at
        that moment and only filled in at `on_start` — so the dict froze as `{}` and stayed that way
        for the life of the process, while `_iids` resolved correctly and every health probe read
        normal. Both `_open` and `_close` consult it before doing anything and return on `None`, so
        the lane could neither enter nor exit: silent in both directions.

        A property cannot go stale, which is why this is not simply a rebuild inside
        `_resolve_symbols` — that fixes today's copy and leaves the next one to be written.
        """
        return {i.symbol.value: i for i in self._iids}

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
                # The instrument DEFINITION, not just its bars. `cache.instrument()` returning None
                # is a hard refusal at submit — every entry priced and journalled, then rejected
                # with "no instrument definition cached". Bars alone do not populate it.
                # ALREADY CACHED IS THE COMMON CASE since #622 — the adapter pushed every
                # instrument into the Cache during connect, and symbol resolution read them from
                # there. Asking again costs a request against an IBKR budget of ~60 per 10 minutes
                # that four lanes share, and the overflow comes back as an empty array with no
                # error (kumo-trading-platform issue 617). The definition is still requested when it is absent.
                self.request_instrument_if_missing(iid)
                self.request_bars(bt, self.clock.utc_now() - pd.Timedelta(days=self._history_days))
            self.subscribe_bars(bt)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None          # backtest: no loop, and none needed
        self.msgbus.subscribe(self._account_topic, self._on_broker_account)
        # Deferred, and OFF the loop — `_arm` reaches the calendar over a blocking socket. See
        # `RegistrationMixin.begin_arming`.
        self.begin_arming()
        self.log.info(
            f"{self.id} watching {len(self._iids)} instruments, warmup needs {self._need} bars, "
            f"next session {self._first_session_note()}, portfolio_size={self._cfg.portfolio_size}")

    def on_stop(self) -> None:
        self.cancel_arming()
        if SESSION_ALERT in self.clock.timer_names:
            self.clock.cancel_timer(SESSION_ALERT)
        # Cancelling the alert is not enough: a session already in flight is a raw asyncio task, not
        # something Nautilus manages, so it outlives stop() and keeps running against the lifecycle
        # it read when it STARTED — an operator who pauses mid-session would watch it submit anyway.
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

        Per session rather than per month is what makes a mid-month restart safe: the strategy is
        never more than one session from noticing the next rebalance. A month-long alert armed once
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
        # exactly like one that decided to hold — silent, and wrong until somebody looks.
        # GUARDED RE-ARM (kumo-trading-platform issue 628). Still FIRST, for the reason above — but `_arm` can now
        # refuse, because a venue calendar knows only a rolling window. An unguarded raise here would
        # abort this callback: the session that just fired would never be marked due, no decision
        # would be taken for it, and no future alert would be set. Silent, permanent, mid-session.
        self.rearm_after_alert(self.clock.utc_now())
        if self._due is not None:
            # WARMUP IS NOT A MISS. `_try_decide` returns early and leaves `_due` set while the
            # universe is still warming, so every subsequent alert used to see a stale `_due` and
            # record a missed rebalance. Measured on a 2025-01..2026-08 backtest: 115 "missed"
            # against ~20 real rebalance opportunities -- the first ~100 sessions were the strategy
            # CORRECTLY refusing to rank on partial history.
            #
            # An alarm that fires 100 times for the expected case is worse than no alarm: it buries
            # the one firing that matters. Same failure cockpit hit when a widened fetch turned the
            # #269 divergence detector into a klaxon. So a miss is only recorded when the strategy
            # was WARM and still could not act -- which is the condition anyone would want paged on.
            if self.warm:
                self.log.warning(f"{self.id}: rebalance {self._due.date()} closed without tradeable "
                                 f"data — no decision was made")
                self.missed_rebalances.append(self._due)
            else:
                self.warmup_deferrals += 1
        self._due = fired
        self._try_decide()

    def _is_rebalance(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """Delegates to the pure `rebalance_dates()` rather than re-deriving 'first session of the
        month' here. Two derivations of one fact disagree, and cadence is exactly the sort of thing
        that gets tuned in research and forgotten in production."""
        if panel.empty:
            return False
        return pd.Timestamp(session) in set(
            rebalance_dates(panel["date"], self._cfg.rebalance_period))

    def _try_decide(self) -> None:
        """Decide for `_due` if this is a rebalance day and the panel has caught up.

        Called by the alert AND by every bar: the alert says WHEN TO LOOK, the bars say WHETHER WE
        MAY DECIDE. A 1-DAY bar for session D is emitted at D's close, so a late feed must defer to
        `on_bar` rather than be polled — bar arrival IS the event being waited on.
        """
        if self._due is None or self._pending:
            return
        if not self.warm:
            return
        panel = self._panel()
        if panel.empty or self._due not in set(panel["date"]):
            return
        prior = panel.loc[panel["date"] < self._due, "date"].max()
        if pd.isna(prior):
            return
        session, self._due = self._due, None
        if not self._is_rebalance(session, panel):
            self.skipped_sessions += 1
            return
        if self._runner is not None:
            self._run_session(session, panel.loc[panel["date"] <= session].copy())
        else:
            self._decide_for(session, panel)

    @property
    def warm(self) -> bool:
        """Enough history on enough symbols to rank them against each other.

        A cross-sectional strategy warms as a UNIVERSE, not per symbol: ranking three warm names out
        of fifty is not an early answer, it is a different and wrong strategy.
        """
        ready = sum(1 for v in self._bars.values() if len(v) >= self._need)
        return ready >= max(self._cfg.portfolio_size, 1)

    def _panel(self) -> pd.DataFrame:
        rows = [row for sym in self._bars for row in self._bars[sym]]
        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    # -- decision --------------------------------------------------------------------------------
    def _run_session(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        if self._session_task is not None and not self._session_task.done():
            self.log.warning(f"{self.id}: rebalance {session.date()} skipped — the previous one is "
                             f"still running")
            self.missed_rebalances.append(session)
            return
        if self._loop is None:
            # Refusing is the honest answer: falling back to the local path would run the session
            # WITHOUT the lifecycle gate, so a DISABLED or SHADOW strategy would submit.
            self.log.error(f"{self.id}: no event loop captured — refusing to run rebalance "
                           f"{session.date()} outside the live runtime")
            self.missed_rebalances.append(session)
            return
        self._session_task = asyncio.run_coroutine_threadsafe(
            self._session_coro(session, panel), self._loop)

    async def _session_coro(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        try:
            # A POSITION ON THE SIDE THIS LANE CANNOT MANAGE IS REPORTED EVERY SESSION (#66).
            # The exit path already REFUSES one, but only when the lane is trying to exit that
            # name — WHD sat in the book for twelve hours because nobody was. Before the
            # decision, because a book we do not understand is context for what follows.
            report_wrong_sided_positions(self, session=str(session.date()))
            result = await self._runner.run(panel, str(session.date()), slot=self._fired_slot or self._slot_name)
            submitted = getattr(result, "submitted", 0)
            outcome = session_outcome(
                getattr(result, "state", "UNKNOWN"),
                decided=bool(getattr(result, "decided", False)),
                blocked=getattr(result, "blocked", None), sent=submitted)
            self.log.info(f"{self.id} {session.date()}: {outcome}")
            # DURABLE OUTCOME, every session, decided or not (kumo-trading-platform issue 587). This lane wrote NO
            # outcome row at all — not a wrong attribute like QC345's, simply absent — so "ran and
            # declined" and "never ran" were indistinguishable for TECHIVOL-005 in the only record
            # an operator reads. It went unreported because the lane has barely run; the ticket named
            # QC345 alone, and aiming at the lane rather than the class would have left this one.
            journal = self.session_journal()
            if journal is not None:
                await journal.write(
                    "state", f"session outcome — {outcome}", session=str(session.date()),
                    slot=self._fired_slot or self._slot_name,
                    detail={"state": getattr(result, "state", "UNKNOWN"),
                            "decided": bool(getattr(result, "decided", False)),
                            "blocked": getattr(result, "blocked", None),
                            "submitted": submitted,
                            # The integer keeps its name; this names its basis (kumo-trading-platform issue 512).
                            "submitted_basis": SENT_BASIS,
                            "entered": list(getattr(result, "entered", ()) or ()),
                            "exited": list(getattr(result, "exited", ()) or ())})
        except asyncio_CancelledError:
            raise
        except Exception as exc:                                       # noqa: BLE001
            self.missed_rebalances.append(session)
            self.log.error(f"{self.id}: rebalance {session.date()} failed: "
                           f"{type(exc).__name__}: {exc}")

    def _decide_for(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        """The PURE decision, turned into orders. Backtest path only — no lifecycle, no journal."""
        scored, _diag = build_feature_panel(panel, self._cfg)
        day = scored.loc[scored["date"] == session]
        if day.empty:
            self.log.warning(f"{self.id}: no feature rows for {session.date()} — holding")
            return
        d = decide(day, self._cfg, set(self._held))
        self.log.info(f"{self.id} {session.date()}: hold {list(d.hold)} enter {list(d.enter)} "
                      f"exit {list(d.exit)} cash_proxy {d.cash_proxy_weight:.2f}")
        today = panel.loc[panel["date"] == session].set_index("ticker")
        for sym in d.exit:
            self._close(sym)
        for sym in d.enter:
            self._open(sym, today, d.weights.get(sym, 0.0))

    # -- orders ----------------------------------------------------------------------------------
    def _bar_type(self, iid: InstrumentId):
        from nautilus_trader.model.data import BarType
        return BarType.from_str(f"{iid}{self._suffix}")

    def _open(self, symbol: str, today: pd.DataFrame, weight: float) -> None:
        """INVERSE-VOL WEIGHTED, not equal weight. `weight` comes from the pure layer's
        `select_portfolio`, which normalises 1/realized_vol across the ranked names — sizing is part
        of QC27's rule, not a portfolio-construction detail added here."""
        iid = self._by_symbol.get(symbol)
        if iid is None or symbol not in today.index or weight <= 0:
            return
        if self.cache.instrument(iid) is None:
            self.log.error(f"{self.id}: no instrument definition for {symbol} — not entering")
            return
        price = float(today.loc[symbol, "close"])
        if price <= 0:
            return
        qty = int((self._equity * weight) / price)
        if qty < 1:
            return
        self._pending[symbol] = "enter"
        self.submit_order(self.order_factory.market(
            instrument_id=iid, order_side=OrderSide.BUY, quantity=Quantity.from_int(qty),
            time_in_force=TimeInForce.DAY))

    def _close(self, symbol: str) -> None:
        iid = self._by_symbol.get(symbol)
        if iid is None:
            return
        pos = next((p for p in self.cache.positions_open(strategy_id=self.id)
                    if str(p.instrument_id) == str(iid)), None)
        if pos is None or pos.quantity == 0:
            self._held.discard(symbol)
            return
        # LONG ONLY, ASSERTED (#88). See `sides.py` — nothing in this package opens a short, so one
        # that exists came from elsewhere and selling it would double it.
        qty = closing_quantity(pos.quantity, side=LONG)
        if qty is None:
            why = refusal_reason(pos.quantity, side=LONG)
            if why:
                self.log.error(f"{self.id}: {symbol} — {why}")
            self._held.discard(symbol)
            return
        self._pending[symbol] = "exit"
        self.submit_order(self.order_factory.market(
            instrument_id=iid, order_side=OrderSide.SELL,
            # `qty` not `abs(pos.quantity)` (#88). `reduce_only=True` looks like it already covers
            # this, and on IBKR it does not: that adapter DROPS the flag, so the only protection on
            # the venue that is now live is the refusal above.
            quantity=Quantity.from_int(qty),
            time_in_force=TimeInForce.DAY, reduce_only=True))

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
        sym = str(event.instrument_id.symbol)
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
        """Forward the venue's ACTUAL answer to the runner's journal (kumo-trading-platform issue 383).

        THE SUBMIT-TIME ROW IS NOT THIS. It records Nautilus accepting the order, never what the venue
        said — so without this the journal says an order went out and never says what came back, and
        the retry counter that reads `phase == "terminal" and not ok` can never advance: a symbol the
        venue refused is silently treated as done.

        THE MIRROR OF #383, AND I WROTE THE LESSON INTO THE METHOD I THEN NEVER CALLED. MOMENTUM had
        the CALLER wired for weeks while cockpit's gateway defined no `record_terminal`, so every call
        no-opped through a getattr: 22 fills, zero terminal rows. Here `QC27SessionRunner.record_terminal`
        exists — with a docstring saying precisely that — and NO handler called it. Wiring one end is
        half a fix; the other end has to land in the same change or it is decoration.

        Best-effort by design: it runs in a live event handler outside the session's control flow, so a
        missing method or a dead loop degrades rather than raising inside Nautilus's dispatch.
        """
        if self._runner is None or self._loop is None:
            return
        record = getattr(self._runner, "record_terminal", None)
        if record is None:
            return
        sym = str(event.instrument_id.symbol)
        session = str(pd.Timestamp.utcnow().date())
        try:
            # OBSERVED, not fire-and-forget — see `RegistrationMixin.fire_and_report`.
            self.fire_and_report(
                call_record_terminal(record, session, sym, ok, detail, log=self.log.warning,
                                     **terminal_fields(self, event)),
                self._loop, f"terminal row for {sym}")
        except Exception as exc:                                       # noqa: BLE001
            self.log.warning(f"{STRATEGY_NAME}: could not record terminal for {sym}: {exc}")
        sync = getattr(self._runner, "sync_claim", None)
        if sync is not None:
            try:
                qty = int(sum(int(p.signed_qty) for p in self.cache.positions_open(
                    instrument_id=event.instrument_id, strategy_id=self.id)))
                px = float(event.last_px) if ok and getattr(event, "last_px", None) is not None else None
            except Exception as exc:                                       # noqa: BLE001
                self.log.error(f"{self.id}: claim sync for {sym} NOT attempted — "
                               f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]}")
                return
            self.fire_and_report(sync(sym, qty, px), self._loop, f"claim sync for {sym}")

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
        """Keep exactly ONE bar per session per symbol — the latest version of it.

        Alpaca republishes the CURRENT session's daily bar as it updates, so appending every arrival
        put one copy of today's bar in the panel per update. That bug left ~103 copies of a single
        date per symbol after thirteen hours of uptime, giving the trailing window zero variance and
        a NaN score, and the strategy sat inert (momentum_rotation.py's `_ingest` note).
        """
        sym = str(bar.bar_type.instrument_id.symbol)
        row = {"ticker": sym,
               "date": pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
               "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
               "close": float(bar.close), "volume": float(bar.volume)}
        bars = self._bars[sym]
        if bars and pd.Timestamp(bars[-1]["date"]).normalize() == row["date"]:
            bars[-1] = row
        else:
            bars.append(row)

    def _on_broker_account(self, snapshot: dict) -> None:
        self._broker_account = snapshot

    def broker_equity(self) -> float | None:
        """Net liquidation from the broker's own snapshot. Nautilus `AccountState` models CASH ONLY
        and cockpit's Alpaca client sets `AccountBalance.total` to the cash figure, so equity for any
        risk limit has to come off the bus.

        A METHOD, NOT A PROPERTY. `NautilusBroker.equity()` is `return self.strategy.broker_equity()`
        and assumes a method; as a property it evaluates first and the `()` lands on its result, so a
        legitimate None becomes `None()` and raises. QC345 shipped exactly this and lost a live
        session to it; QC27 carried the identical defect on a branch that merged after the test
        guarding it was written, and the test named its strategies by hand.

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
        """
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
