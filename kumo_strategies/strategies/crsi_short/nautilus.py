"""CRSISHORT on the live engine — the first lane in this package that holds the SHORT side (#123).

This class contains no strategy logic. It collects daily bars, asks `crsi_short.engine` what to be
short and `crsi_short.exits` what to cover, and hands the session to the runner that owns safety.

WHAT IS DIFFERENT ABOUT IT, and each difference is a thing that was assumed everywhere else:

  side        `POSITION_SIDE = SHORT`. `broker.py` and `order_provenance.completes` both ask a lane
              which side it holds and neither has a default, so this declaration is what makes an
              entry SELL an ENTRY rather than a reduce-only exit, and what makes the covering BUY
              complete the exit rather than look like an entry that filled.
  universe    NOT a candidate list. Every rotation lane here is handed the names it may hold; this
              one SCREENS a broad universe on features it computes itself (prev-day dollar volume,
              price, 100-session annualised vol, ConnorsRSI). The screen lives in `engine.apply_gates`
              and nowhere else.
  prices      MUST be split-adjusted, and that is refused at construction rather than assumed. On raw
              prices a reverse split reads as a -1000% short loss on an ALREADY-OPEN position, which
              no entry-time corporate-action gate can reach — it inverted the lab's 2025-26 verdict
              on 7.4% of trades.
  borrow      A locate fee per symbol is REQUIRED whenever `cfg.max_borrow_fee_annual` is set, which
              it is by default. `engine.decide` raises without it. Wiring the gate without the data
              would look armed and be inert, which is the #26 failure mode this repo has paid for
              twice — so the provider is demanded at construction, where it is a boot failure, not
              mid-session, where it is a lane that stopped deciding.
  exits       STRUCTURAL and per-position, never ranking-driven. `decide` returns no exits at all;
              `evaluate_short_exits` owns them, against a trail this class persists.

IT CANNOT TRADE YET, AND THAT IS NOT THIS FILE'S TO FIX. CRSISHORT's entry is a sell-short LIMIT
resting one session, cancelled if the name leaves the universe overnight, and resting orders consume
slots. `broker.py` can send a MARKET order and nothing else (#131). So this lane REGISTERS: it
computes, journals and publishes what it would do, and the operator moves it to TRADING only after
#131 lands and its published decisions have been reconciled against the backtest's signals.
"""

from __future__ import annotations

import asyncio
import math
import traceback
from asyncio import CancelledError as asyncio_CancelledError
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

import pandas as pd
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.contract import (
    report_wrong_sided_positions, SENT_BASIS, RegistrationMixin, protective_close, session_outcome)
from kumo_strategies.runtime.nautilus.order_provenance import completes, fill_side, is_foreign
from kumo_strategies.runtime.nautilus.sides import SHORT, closing_order_side
from kumo_strategies.runtime.executor.broker import OrderRequest
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import (
    ADJUSTED, build_feature_panel, decide)
from kumo_strategies.strategies.crsi_short.exits import (
    SessionBar, ShortTrailState, evaluate_short_exits, open_state)

#: The operator-facing lane name. NO NUMBER: cockpit allocates the `order_id_tag` and maps
#: `EXTERNAL_ID` to it. A name like "CRSISHORT-001" here is what once made a tag look claimed from
#: this side and free from cockpit's.
STRATEGY_NAME = "CRSISHORT"
EXTERNAL_ID = "CRSISHORT"
STRATEGY_LABEL = "CRSISHORT — overbought high-volatility short book"

#: ONE ALERT PER LANE, named, so `on_stop` can cancel the one this lane owns.
SESSION_ALERT = "crsi_short_session_decide"

#: CAN THIS LANE ACTUALLY SEND ITS ENTRY? Not yet, and the code says so rather than a ticket.
#:
#: The entry is a sell-short LIMIT resting into the OPENING AUCTION: 109 of the 231 accepted trades
#: fill at the opening print and carry 78% of the return. Reproducing that on IB takes TWO orders for
#: one intent — a limit-on-open (`AT_THE_OPEN` -> `OPG`) for the gap arm, and a day limit at the same
#: price after the auction cancels it, for the rest. That decomposition is designed and NOT BUILT
#: (#131), and this lane's `decision_slots` still defaults to `open+5m`, which is after the auction.
#:
#: Flip this to True in the commit that lands the decomposition, and delete the guard below with it.
#: It is a statement about the ORDER PATH, never a way to silence the refusal for a lane that still
#: cannot act.
#: The two-order decomposition (#131) is BUILT: `opening_leg` (AT_THE_OPEN -> OPG),
#: `replacement_leg`, `replacement_is_owed` with `_AUCTION_LEG_GONE` enumerated rather than
#: inferred, `NautilusBroker.order_status` to read the auction leg's fate, and `allow_pre_open`
#: derived from the lane's own slots at BOTH calendar call sites.
#:
#: WHAT THIS FLAG NO LONGER STANDS FOR, AND WHY THE GUARD BELOW CHANGED SHAPE. It used to mean two
#: things at once — "the decomposition exists" AND "this lane is configured to use it" — and only
#: the first is a property of the code. Flipping it alone would have let a lane built with the
#: DEFAULT slots (`open+5m`, after the cross) construct and trade, sending only the day limit: 108
#: of 231 accepted trades fill at the opening print and carry 77.6% of the return, so that lane
#: would trade a different strategy from the one #123 measured, at roughly a quarter of its edge,
#: with every surface reading normal. The second half is now checked directly, per lane, from the
#: slots it was actually given.
#: ITS READER IS IN ANOTHER REPO, SO DO NOT DELETE IT AS UNUSED. Nothing in this package branches on
#: it any more — the construction guard that did was removed when it flipped, on its own test's
#: instruction, because a refusal nobody reaches is worse than no refusal. kumo-trading-platform's gateway
#: imports this constant LIVE rather than mirroring it, so a revision whose path is NOT complete
#: cannot be wired to submit; two copies of this fact in two repos is the drift it exists to prevent.
ORDER_PATH_COMPLETE = True

# ==================================================================================================
# ONE INTENT, TWO ORDERS — the limit-on-open decomposition (#131)
# ==================================================================================================
#
# The entry is a sell-short LIMIT resting into the OPENING CROSS. 108 of the 231 accepted trades
# fill at the opening print and carry 77.6% of the return (+9.99%/trade against +2.53%), so where
# the order rests is not an execution detail — it is most of the strategy.
#
# IB CANNOT EXPRESS THAT AS ONE ORDER. A limit-on-open (`AT_THE_OPEN` -> OPG) participates in the
# cross and the venue CANCELS it if it does not fill there; an ordinary day limit never participates
# in the cross at all. So one intent becomes two orders — and a lane that sent only the day limit
# would trade a different strategy from the one #123 measured, with every surface reading normal.
#
# THE RULE LIVES HERE, THE SUBMITTING DOES NOT. This package never calls a broker (AGENTS.md). What
# it owns is WHICH ORDERS an intent becomes, which is a rule a test can pin against the backtest;
# the alternative is the decomposition living as prose inside a gateway that nothing can check.

#: Venue answers after which the auction leg is provably GONE and the intent still stands.
#:
#: ENUMERATED, NEVER INFERRED, and the inverse would be the dangerous spelling: "anything that is
#: not FILLED" quietly includes ACCEPTED — an order still live in the auction — and resting a second
#: order against a live one is the double fill.
_AUCTION_LEG_GONE = frozenset({"EXPIRED", "CANCELED", "CANCELLED"})


def opening_leg(*, symbol: str, qty: int, limit_px: float, session: str, strategy_id: str,
                slot: str = "") -> OrderRequest:
    """The order that participates in the opening cross.

    `AT_THE_OPEN` is load-bearing and has no fallback: `nautilus/broker.py` maps it to Nautilus'
    `TimeInForce.AT_THE_OPEN`, which the IB adapter sends as OPG. A silent downgrade to DAY would
    turn this into an ordinary resting order and forgo three quarters of the lane's return without
    failing anywhere.
    """
    q = int(qty)
    if q <= 0:
        raise ValueError(
            f"entry quantity must be a positive magnitude, got {qty!r}. The SIDE says SELL; a "
            f"negative quantity here would flip it into a buy and open the position backwards.")
    px = float(limit_px)
    # REFUSED, NOT ROUNDED. `crsi_short.engine.apply_gates` already rounds to the penny — "a limit at
    # 12.3449 rests at 12.34, not 12.35" — and rounding a second time here is a SECOND derivation of
    # one number, which is where two derivations disagree. A caller that computed the price
    # differently must find out, not be quietly corrected into a different fill.
    if round(px, 2) != px:
        raise ValueError(
            f"limit price {limit_px!r} is not a whole penny. `apply_gates` rounds it; rounding it "
            f"again here would be a second derivation of one number and the live fill would differ "
            f"from the measured one with nothing to show for it.")
    # A DECISION MADE AFTER THE CROSS CANNOT REACH IT, and this is the contradiction that would
    # otherwise ship silently. `decision_slots` defaults to `("open+5m",)` — 09:35, five minutes
    # after the auction it is meant to join. A lane wired that way builds a perfectly well-formed
    # OPG order that the venue has nothing left to match, and the whole decomposition is inert while
    # every other check passes.
    #
    # The signal permits an earlier decision: `apply_gates` computes the limit from the PRIOR close,
    # so the order is knowable before the open. What is missing is the DECLARATION, and refusing
    # here is how that stays visible instead of becoming a comment nobody reads.
    if not _reaches_the_auction(slot):
        raise ValueError(
            f"slot {slot!r} is after the opening cross, so an AT_THE_OPEN order decided there can "
            f"never reach the auction — the leg would be well-formed and inert. This lane's limit "
            f"comes from the PRIOR close, so a pre-open slot is expressible; declare one (#131).")
    return OrderRequest(symbol=str(symbol), side="SELL", qty=q, session=str(session),
                        strategy_id=str(strategy_id), slot=str(slot), limit_px=px,
                        time_in_force="AT_THE_OPEN")


def _reaches_the_auction(slot) -> bool:
    """Can a decision taken at `slot` still join the opening cross?

    PARSED WITH THE REPO'S OWN SLOT GRAMMAR, never with a second one. The first version of this
    guard matched a hand-written list of literal names — "open", "preopen", "premarket" — and would
    have REFUSED `open-10m`, the very slot the decomposition needs. `slots._parse` already defines
    what a slot spec means; a second definition is exactly where two derivations disagree, and this
    one disagreed on its first real input.

    A WALL CLOCK IS REFUSED RATHER THAN ASSUMED. `09:20` is before the open on an ordinary day and
    after it on a half-day with a shifted open, and this function has no calendar. The failure it
    would hide is silent — a well-formed OPG order the auction has already passed — so an
    open-relative offset is required.
    """
    from kumo_strategies.strategies.momentum_rotation.slots import SlotError, _parse

    name = str(slot or "").strip()
    if not name:
        return True                       # unnamed: the auction itself
    try:
        parsed = _parse(name)
    except SlotError:
        return False
    if not isinstance(parsed, tuple):
        return False                      # a wall clock: unprovable without a calendar
    anchor, minutes = parsed
    return anchor == "open" and minutes <= 0


def replacement_leg(auction_leg: OrderRequest) -> OrderRequest:
    """The intraday day-limit that carries the SAME intent after the auction ends without a fill.

    SAME PRICE, DELIBERATELY. The backtest models one limit for the whole session — `max(limit,
    open)` — so a replacement at a re-derived price would be a different decision from the one that
    was measured.

    The two legs must hash to DIFFERENT `client_order_id`s, and `OrderRequest` already arranges that
    by appending the time-in-force when it is not DAY. It matters more than it looks: Nautilus denies
    a duplicate id LOCALLY, before the venue sees it, and `submit()` then reads the FIRST order back
    out of the cache and reports ITS answer as the second's — which once reported a seven-hour-old
    rejection as that morning's.

    Callers must gate this on `replacement_is_owed`; constructing it does not make it safe to send.
    """
    if auction_leg.time_in_force != "AT_THE_OPEN":
        raise ValueError(
            f"only the AT_THE_OPEN auction leg has a replacement; got "
            f"{auction_leg.time_in_force!r}. Replacing a DAY limit would rest a SECOND order "
            f"against one slot, and both can fill — on a short that exposure is unbounded.")
    return replace(auction_leg, time_in_force="DAY")


def replacement_is_owed(auction_status) -> bool:
    """Has the auction leg provably ended WITHOUT taking the slot?

    THE ONLY QUESTION THAT MATTERS IS WHETHER A SECOND ORDER CAN DOUBLE THE POSITION. Every answer
    below is that question, not a judgement about the trade:

      EXPIRED / CANCELED    the auction ended, nothing filled, the intent stands            -> yes
      ACCEPTED              still live IN the auction; a second order can fill alongside it -> no
      FILLED                the entry is on                                                 -> no
      PARTIALLY_FILLED      the slot is taken. Topping up is a DIFFERENT decision from
                            replacing an unfilled order, and this lane does not make it     -> no
      REJECTED              the venue refused it. Resting the same intent again as a day
                            limit converts a refusal into a silent retry at another time    -> no
      anything else         we cannot tell whether the leg is live                          -> no

    AN UNRECOGNISED STATUS IS A NO, and that asymmetry is the point: refusing costs one session's
    entry, guessing costs an unbounded short. This is the same three-state discipline as
    `Verdict` — nothing in this package acts on UNKNOWN.
    """
    if auction_status is None:
        return False
    name = getattr(auction_status, "name", auction_status)
    return str(name).upper() in _AUCTION_LEG_GONE


#: WHAT THIS LANE ASKS OF A HISTORY REQUEST, stated once, venue-neutral (kumo-trading-platform issue 1124).
#:
#: `adjustment: split` — the bars must be SPLIT-ADJUSTED, which is the same fact the constructor
#: refuses to build without (`price_adjustment=ADJUSTED`). On IB the TRADES bars arrive adjusted
#: whatever is asked and the adapter ignores an unknown key; on Alpaca the REST path is `raw` unless
#: the request says otherwise (measured: NVDA 2024-06-07 `split` = 120.89 = IB, `all` = 120.54), and
#: 29 of the 130 acceptance names carry a split inside the 200-day warmup window. Asked HERE rather
#: than in the data client so the lane that needs the fact is the one that states it.
#:
#: `disable_historical_cache: True` — this request's bars are delivered to `on_historical_data` and
#: NOT written into the node's shared bar cache (Nautilus 1.229 `DataEngine._handle_response`
#: reads the key off the response params, per request). The 1-DAY bar type is shared with the
#: chart and with lanes that were measured on raw bars; letting an adjusted series overwrite their
#: rows for the same timestamps would be last-write-wins on a fact two consumers disagree about.
HISTORY_REQUEST_PARAMS: dict = {"adjustment": "split", "disable_historical_cache": True}

#: A live Nautilus Bar carries OHLCV and nothing else.
_BAR_FIELDS = ("open", "high", "low", "close", "volume")


def _as_price(px) -> float | None:
    """A Nautilus `Price` or None, read without inventing one.

    Carried for the journal only, so an unreadable price costs a field and never a reservation —
    the opposite trade from `_start_trail`, where an unusable price must NOT become a trail.
    """
    if px is None:
        return None
    try:
        out = float(px)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None



#: The only two things an order this lane places can be doing. A THIRD value is not a new feature,
#: it is a silent defect: `completes()` returns True for any string it does not recognise
#: (`order_provenance.py`, the fall-through), so a typo'd kind makes EVERY fill for that symbol
#: complete the intent and land in the entry branch. Validated where the value ENTERS, because the
#: place it does damage cannot tell a typo from a decision.
PENDING_KINDS = ("enter", "exit")


@dataclass(frozen=True)
class PendingOrder:
    """One order this lane has committed a slot to and has not yet seen resolved.

    A RESERVATION, NOT A POSITION. The lab computes `room = slots - len(pos) - len(pend)`, and with a
    limit resting into the next opening auction "committed" and "filled" are a session apart by
    design — so a slot has to be spent at submit and returned on fill, cancel, expiry, rejection or
    denial. Every one of those five has to clear it; four of them did not exist (#172).

    The price and the id are carried for the journal and for reconciliation, never for the decision:
    `decide` is handed symbols alone.
    """

    kind: str
    client_order_id: str
    limit_px: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in PENDING_KINDS:
            raise ValueError(
                f"pending kind {self.kind!r} is not one of {PENDING_KINDS}. `completes()` returns "
                f"True for an unrecognised kind, so this value would not fail at the fill — it "
                f"would make every fill for this symbol read as an ENTRY.")


@dataclass(frozen=True)
class SessionIntent:
    """One session's wanted book change. WHAT, never HOW — no order type, no venue, no sizing.

    `cover_px` is the price each cover is DEFINED at, carried out of `evaluate_short_exits` because
    the price and the condition are one fact: a driver that detected the flat cover and then booked
    the fill at the session high would report a loss the rule never took. The driver still owns
    slippage and fees; it must not own WHICH price the rule meant.
    """

    cover: dict[str, str] = field(default_factory=dict)
    cover_kind: dict[str, str] = field(default_factory=dict)
    cover_px: dict[str, float] = field(default_factory=dict)
    enter: tuple[str, ...] = ()
    limits: dict[str, float] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)


def warmup_bars_needed(cfg: CrsiShortConfig) -> int:
    """MAX, never SUM — every window is a trailing one ending at the same bar, so they overlap.

    `cfg.warmup_sessions` is already `max(vol_window, crsi_rank_period) + 2` and is DERIVED, so this
    does not restate it: a second definition of warmup is a second thing to disagree.
    """
    return cfg.warmup_sessions



# -- diagnostics --------------------------------------------------------------------------------------
#
# OBSERVATION IS NOT ALLOWED TO BREAK THE PATH IT OBSERVES. The first version of these lines
# (312d904, deployed to ibkr-paper) read `self._slots`, `self._min_warm`, `self._ingested_bars` and
# `self._panel_diag` inline in f-strings on `on_start`, the bar ingest and the session alert. On a
# host that lacks one of them — every narrow test double, and any boot where the attribute is set
# later than the line — the AttributeError took the SESSION down: eleven tests red on the branch's
# own base, and live the same shape would read as "session failed" for a log line. So these are
# MODULE functions, not methods (a host that binds one real method must not need a second), the
# message is built lazily inside a guard, every attribute is read by name with a default, the
# counters live in the object's own dict, and nothing here can raise into the caller.

def _diag(host, build, *, level: str = "info") -> None:
    """Log one CRSISHORT-DIAG line for `host`. `build` is a str or a zero-arg callable; a failure to
    build the message is itself logged at debug and never propagates."""
    log = getattr(host, "log", None)
    if log is None:
        return
    ident = getattr(host, "id", "CRSISHORT")
    try:
        msg = build() if callable(build) else build
    except Exception as exc:                                                      # noqa: BLE001
        fn = getattr(log, "debug", None)
        if fn is not None:
            fn(f"{ident}: CRSISHORT-DIAG line unavailable: {type(exc).__name__}: {exc}")
        return
    fn = getattr(log, level, None) or getattr(log, "info", None)
    if fn is not None:
        fn(f"{ident}: CRSISHORT-DIAG {msg}")


def _attr(host, name: str, default="?"):
    """An attribute for a log line — the default when the host does not carry it."""
    return getattr(host, name, default)


def _ready_count(host) -> int:
    bars = getattr(host, "_bars", None) or {}
    need = getattr(host, "_need", None)
    if need is None:
        return -1
    return sum(1 for v in bars.values() if len(v) >= need)


def _panel_diag(host, panel: pd.DataFrame, session: pd.Timestamp | None = None) -> str:
    if panel is None or panel.empty:
        return "panel empty"
    dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    newest, oldest = dates.max(), dates.min()
    parts = [
        f"rows={len(panel)}",
        f"symbols={panel['ticker'].astype(str).nunique()}",
        f"ready={_ready_count(host)}/{_attr(host, '_min_warm')}",
        f"need={_attr(host, '_need')}",
        f"oldest={oldest.date() if pd.notna(oldest) else 'none'}",
        f"newest={newest.date() if pd.notna(newest) else 'none'}",
    ]
    if session is not None:
        target = pd.Timestamp(session).normalize()
        parts.append(f"session_rows={panel.loc[dates == target, 'ticker'].astype(str).nunique()}")
    return " ".join(parts)


class CrsiShortStrategy(MarketAwareMixin, HeldSeedMixin, SymbolResolutionMixin, RegistrationMixin, Strategy):
    EXTERNAL_ID = EXTERNAL_ID
    LABEL = STRATEGY_LABEL
    #: THE FIRST LANE THAT IS NOT LONG. Read the module docstring before changing it: `broker.py`
    #: derives open-vs-exit from this, and `completes` derives which fill completes which intent.
    POSITION_SIDE = SHORT

    def __init__(
        self,
        cfg: CrsiShortConfig,
        instrument_ids: list[InstrumentId] | None = None,
        *,
        symbols: list[str] | None = None,
        calendar: object | None = None,
        decision_slots: tuple[str, ...] = ("open+5m",),
        read_slots: object | None = None,
        order_id_tag: str,
        strategy_name: str = STRATEGY_NAME,
        bar_type_suffix: str = "-1-DAY-LAST-EXTERNAL",
        session_runner: object | None = None,
        history_days: int | None = None,
        shadow_only: bool = False,
        price_adjustment: str,
        min_warm_symbols: int,
        borrow_rates: object | None = None,
        account_topic: str = "broker.account",
    ) -> None:
        super().__init__(config=StrategyConfig(strategy_id=strategy_name,
                                               order_id_tag=order_id_tag))
        # ADJUSTMENT IS DECLARED BY THE CALLER, at construction, with no default to inherit. Only the
        # component that pulled the bars knows how they were pulled; this class cannot tell an
        # adjusted series from a raw one by looking at it, and neither can the engine — which is why
        # the engine refuses at its own boundary too rather than trusting this one.
        if price_adjustment != ADJUSTED:
            raise ValueError(
                f"CRSISHORT requires split-adjusted bars (price_adjustment={ADJUSTED!r}), got "
                f"{price_adjustment!r}. On raw prices a reverse split books as a -1000% short loss "
                f"on an already-open position — not reachable by any entry-time gate (#123).")
        # THE UNIVERSE BREADTH IS A DEPLOYMENT DECISION AND IT IS STATED, not defaulted. Signalling
        # on twenty warm names out of a universe of thousands is not an early answer, it is a
        # different and much narrower strategy that would still produce a plausible curve.
        if int(min_warm_symbols) < 1:
            raise ValueError(f"min_warm_symbols must be >= 1, got {min_warm_symbols!r}")
        # A FEE CEILING WITHOUT FEE DATA IS AN INERT GATE. `engine.decide` raises rather than
        # silently skipping the check, and this refuses earlier still: at construction it is a boot
        # failure the operator sees, mid-session it is a lane that stopped deciding and looks like
        # one that decided to hold.
        if cfg.max_borrow_fee_annual is not None and borrow_rates is None:
            raise ValueError(
                f"cfg.max_borrow_fee_annual={cfg.max_borrow_fee_annual!r} needs a borrow_rates "
                f"provider — a locate ceiling with no locate data is armed and inert (#26). Pass a "
                f"callable symbols -> {{symbol: annual fee or None}}, or set the ceiling to None "
                f"and record that the gate is off.")
        # A MERGED ADAPTER READS AS DEPLOYABLE. This one is not, so it refuses to be built unless
        # the caller states it wants the shadow lane — register, compute, journal, publish, submit
        # nothing, which is platform issue 853's first phase and the only thing this lane can honestly do.
        # The refusal names what is missing, because the next reader needs to know which piece to
        # look for rather than being told "not ready".
        # A LANE THAT CANNOT REACH THE AUCTION MUST NOT TRADE, EVEN THOUGH THE PATH EXISTS (#210).
        # The decomposition being built says nothing about whether THIS lane's slots can use it, and
        # those are different facts that one flag used to carry together. A lane whose slots all
        # resolve after the cross sends only the day limit — 108 of the 231 accepted trades fill at
        # the opening print and carry 77.6% of the return (+9.99%/trade against +2.53%), so it
        # trades roughly a quarter of the strategy #123 measured while every surface reads normal.
        #
        # REFUSED AT CONSTRUCTION rather than reported per session, for `capabilities.require`'s
        # reason: a lane that cannot act must not exist. A warning here would be read once and
        # forgotten, and the lane would keep publishing figures under a name whose evidence does not
        # describe it.
        if not shadow_only and not any(_reaches_the_auction(s) for s in decision_slots):
            raise ValueError(
                f"CRSISHORT was given decision_slots={tuple(decision_slots)!r}, none of which "
                f"resolves BEFORE the open, so it can never place the limit-on-open leg its rule "
                f"needs — it would send only the day limit.\n\n"
                f"108 of the 231 accepted trades in #123 fill at the OPENING PRINT and carry 77.6% "
                f"of the return (+9.99%/trade against +2.53%), so a day-limit-only lane is not a "
                f"degraded version of this strategy, it is a different one — and every surface "
                f"would read normal.\n\n"
                f"Give it a pre-open slot for the auction leg and one after the cross for the "
                f"replacement — ('open-10m', 'open+5m') is the deployed pair — or pass "
                f"shadow_only=True to register a lane that computes and publishes without trading.")
        self._shadow_only = bool(shadow_only)
        self._cfg = cfg
        self._init_symbols(instrument_ids, symbols)
        self._calendar = calendar
        self._slots = tuple(decision_slots)
        self._read_slots = read_slots
        self._adjustment = price_adjustment
        self._min_warm = int(min_warm_symbols)
        self._borrow_rates = borrow_rates
        self._armed_session = None
        self._armed_slot = None
        self._loop = None
        self._session_task = None
        self._suffix = bar_type_suffix
        self._history_days = history_days
        self._need = warmup_bars_needed(cfg)
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._need + 5))
        self._ingested_bars = 0
        self._last_ready_count = 0
        # BOOK STATE MOVES ON ORDER EVENTS ONLY, never because submit was called.
        self._held: set[str] = set()
        #: {symbol: PendingOrder}. WRITTEN by `record_pending`, cleared by `clear_pending`, and
        #: rebuilt from the venue by `reconcile_pending` on start. It is process memory with a
        #: durable source, never a durable record of its own.
        self._pending: dict[str, PendingOrder] = {}
        self.missed_sessions: list[pd.Timestamp] = []
        #: The trail per open short, advanced by `evaluate_short_exits` and persisted here. The flat
        #: exit is derived from `best_px` alone — `ever_in_profit` is a property, never a stored
        #: field beside it, because two fields for one fact disagree and the disagreement is
        #: invisible: the flat exit simply stops arming.
        self._trail: dict[str, ShortTrailState] = {}
        self._runner = session_runner
        # Nautilus `AccountState` models CASH ONLY, so equity for a risk limit comes off the
        # broker's own snapshot on the bus, which carries the portfolio value. The topic is a
        # constructor arg rather than an import so this package keeps no dependency on cockpit.
        self._account_topic = account_topic
        self._broker_account: dict | None = None


    # -- lifecycle -------------------------------------------------------------------------------

    def on_start(self) -> None:
        # BEFORE ANYTHING DECIDES. A restart with a limit still resting at the venue would otherwise
        # read zero reserved slots and submit a second order against a slot that is already taken.
        self.reconcile_pending()
        # A RESTART IS THE NORMAL CASE, and `_held` is built empty. Until it is seeded, every
        # position predating this boot is invisible to `protective_close`, so the venue's answer
        # to a protective stop-out goes unrecorded and the claim outlives the position (#194).
        self.seed_held_from_positions()
        self._resolve_symbols_if_needed()
        # A LANE THAT CANNOT SIGNAL FOR MONTHS MUST SAY SO, ONCE, LOUDLY.
        #
        # `history_days` defaults to None in every lane here, and a falsy value means the loop below
        # requests nothing at all — silently. For a rotation lane that is a slow start; for this one
        # it is `warmup_sessions` = 102 SESSIONS PER NAME of sitting inert, which on every surface
        # an operator reads is indistinguishable from a lane that decided to hold. That ambiguity is
        # the single most expensive one in this system and it does not get to happen by omission.
        if not self._history_days:
            self.log.error(
                f"{self.id}: no history_days, so NO bar history will be requested. This lane needs "
                f"{self._need} sessions per name before it can signal at all and a live daily feed "
                f"delivers one bar per name per session — it will publish nothing for roughly "
                f"{self._need} sessions and will look exactly like a lane that decided to hold. "
                f"Pass history_days (>= {self._need} trading sessions of calendar days).")
        _diag(self, lambda: (
            f"start symbols={len(_attr(self, '_iids', ()))} slots={list(_attr(self, '_slots', ()))} "
            f"history_days={_attr(self, '_history_days')} need={_attr(self, '_need')} "
            f"min_warm={_attr(self, '_min_warm')} bar_suffix={_attr(self, '_suffix')!r}"))
        for iid in self._iids:
            bt = self._bar_type(iid)
            if self._history_days:
                # The instrument DEFINITION, not just its bars: `cache.instrument()` returning None
                # is a hard refusal at submit. Asked only when absent — every needless request spends
                # an IBKR historical allowance four lanes share.
                self.request_instrument_if_missing(iid)
                # HISTORY FIRST, OR THIS LANE CANNOT SIGNAL FOR FIVE MONTHS. Warmup here is 102
                # SESSIONS PER NAME (`max(vol_window, crsi_rank_period) + 2`), and a live daily feed
                # delivers one bar per name per session — so without this the lane accumulates its
                # own warmup in real time and sits inert until roughly February. It would look
                # exactly like a lane that decided to hold, on every surface, for months.
                #
                # The three lanes that trade all do this (momentum_rotation.py:290-306,
                # qc27_rotation.py:212-223, qc345_rotation.py:258). `template_rotation` does NOT,
                # which is where this omission came from — filed separately, because a template that
                # omits the hard part teaches that the hard part is optional.
                start = self.clock.utc_now() - pd.Timedelta(days=self._history_days)
                _diag(self, lambda: f"request_bars instrument={iid} bar_type={bt} start={start}", level="debug")
                self.request_bars(bt, start, params=HISTORY_REQUEST_PARAMS)
            _diag(self, lambda: f"subscribe_bars instrument={iid} bar_type={bt}", level="debug")
            self.subscribe_bars(bt)
        # CAPTURE THE LOOP HERE. `LiveClock` fires timer callbacks from its own thread, where
        # `asyncio.get_running_loop()` raises, so a session coroutine scheduled from the alert has
        # nowhere to go unless the loop was captured while still on it.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        # THE HANDLER WAS DEFINED AND NEVER SUBSCRIBED (#174). `_on_broker_account` existed from the
        # first commit of this lane and nothing ever called it, so `broker_equity()` returned None
        # forever with no error anywhere. Measured on ibkr-paper: BCTROT passed equity on the same
        # node and the same clock while this lane returned None four minutes later.
        #
        # It costs nothing until the order path lands and everything after: sizing is
        # `-int(slot_notional(equity) / fill)` and an unfunded slot is a refusal, so a None equity
        # refuses every entry forever while the lane reads TRADING, armed and warm.
        self.msgbus.subscribe(self._account_topic, self._on_broker_account)
        self.begin_arming()

    def on_stop(self) -> None:
        self.cancel_arming()
        if SESSION_ALERT in self.clock.timer_names:
            self.clock.cancel_timer(SESSION_ALERT)

    def _arm(self, after) -> None:
        """Arm the next decision, re-reading the slots first so an operator edit lands without a
        redeploy — a schedule captured at construction is a setting that silently does nothing."""
        from kumo_strategies.runtime.calendar import next_slot_fire

        self._reread_slots()
        # !! PRE-OPEN IS DERIVED FROM THE SLOTS, not passed as a flag (#131). `open-10m` clamps
        # FORWARD to the open without it, which places the auction leg after the OPG cutoff — a
        # well-formed order the cross has already passed, failing nowhere.
        from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

        session, slot, fire_at = next_slot_fire(
            self._calendar, after, self._slots, allow_pre_open=has_pre_open(self._slots))
        self._armed_session, self._armed_slot = pd.Timestamp(session), slot
        self.clock.set_time_alert(SESSION_ALERT, fire_at, self._on_session_alert, override=True)

    def _reread_slots(self) -> None:
        """Refresh `_slots` from settings, or keep what we have.

        NEVER LETS A SETTINGS READ STOP THE LANE SCHEDULING. A lane that stopped arming looks exactly
        like one that decided to hold. A raising reader, a None or an empty tuple all leave the
        current schedule in place and log.
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
        if slots != self._slots:
            self.log.info(f"{self.id}: decision slots {list(self._slots)} -> {list(slots)}")
            self._slots = slots

    def _on_session_alert(self, event=None) -> None:
        fired, self._armed_session = self._armed_session, None
        fired_slot, self._armed_slot = self._armed_slot, None
        # RE-ARM FIRST, through the guarded path: everything below may raise, and a lane that has
        # stopped scheduling is indistinguishable from one that decided to hold.
        self.rearm_after_alert(self.clock.utc_now())
        if fired is None:
            _diag(self, "session alert fired with no armed session", level="error")
            return
        if not self.warm:
            self._not_ready(fired, fired_slot, "warmup")
            return
        _diag(self, lambda: (
            f"alert fired session={fired.date()} slot={fired_slot} "
            f"next_session={getattr(_attr(self, '_armed_session', None), 'date', lambda: None)()} "
            f"next_slot={_attr(self, '_armed_slot', None)}"))
        panel = self.panel()
        _diag(self, lambda: f"before stale check {_panel_diag(self, panel, fired)}")
        if panel.empty:
            _diag(self, "panel empty at session alert; no runner call", level="error")
            self._not_ready(fired, fired_slot, "empty panel")
            return
        if self._refuse_a_stale_feed(fired, panel):
            _diag(self, "stale-feed refusal before runner", level="error")
            return
        panel = self._with_a_row_for(fired, panel)
        _diag(self, lambda: f"after placeholder {_panel_diag(self, panel, fired)}")
        if self._loop is not None:
            _diag(self, lambda: f"scheduling session coroutine slot={fired_slot}")
            self._session_task = self.fire_and_report(
                self._session_coro(fired, panel, fired_slot), self._loop, "session")
        else:
            _diag(self, "no asyncio loop captured; runner not scheduled", level="error")

    def _not_ready(self, session, slot, reason: str) -> None:
        """Record that a fired slot could not decide because live data was not ready yet."""
        warmth = self._warmth()
        msg = (f"{self.id}: rung {session} fired and this lane CANNOT DECIDE — {reason}: "
               f"{warmth}. No decision was made.")
        self.log.warning(msg)
        journal = self.session_journal()
        if journal is None or self._loop is None:
            return

        async def write_not_ready() -> None:
            await journal.write(
                "state", f"session not ready — {reason}: {warmth}",
                session=str(pd.Timestamp(session).date()), slot=slot,
                detail={"state": "NOT_READY", "reason": reason,
                        "position_side": self.POSITION_SIDE, **self._warmth_detail()})

        self.fire_and_report(write_not_ready(), self._loop, "not-ready session")

    async def _session_coro(self, session, panel, slot) -> None:
        """THE RUNNER OWNS SAFETY — lifecycle, journal, idempotency, risk and budget all live in the
        cockpit-side gateway. Deciding and submitting here directly bypasses every one of them."""
        try:
            _diag(self, lambda: (
                f"runner entered session={session.date()} slot={slot} {_panel_diag(self, panel, session)}"))
            # A POSITION ON THE SIDE THIS LANE CANNOT MANAGE IS REPORTED EVERY SESSION (#66).
            # The exit path already REFUSES one, but only when the lane is trying to exit that
            # name — WHD sat in the book for twelve hours because nobody was. Before the
            # decision, because a book we do not understand is context for what follows.
            report_wrong_sided_positions(self, session=str(session.date()))
            # SHADOW IS LIVE HERE, NOT JUST AT CONSTRUCTION (#209). `_shadow_only` was assigned at
            # `__init__` and READ NOWHERE: it gated a build-time raise and nothing at runtime, so the
            # moment a submission path was wired the lane would have traded while every reader —
            # cockpit's flag, this lane's own docstring, the operator — believed that flag stopped
            # it. A knob wired to nothing and a knob wired to something that never fires look
            # identical, and this one looked identical for as long as there was nothing to fire.
            #
            # IT IS A DIFFERENT QUESTION FROM `ORDER_PATH_COMPLETE`, which is why both exist:
            #     ORDER_PATH_COMPLETE  is the path BUILT?           a property of the CODE
            #     shadow_only          does THIS DEPLOYMENT trade?  a property of the CALLER
            # Collapsing them would cost the compute-only registration that is platform issue 853's first
            # phase, and it is the safest way to validate any new lane.
            if self._shadow_only or self._runner is None:
                # SEPARATE REASONS, SEPARATE WORDS. "Asked not to trade" and "nothing to trade
                # through" are different facts, and reporting them identically is how a lane that
                # lost its runner reads as a deliberate shadow deployment for a week.
                why = ("shadow_only=True — registered to compute and publish, not to trade"
                       if self._shadow_only else
                       "no session runner attached, so there is nothing to submit through")
                self.log.info(f"{self.id} {session.date()}: deciding without submitting — {why}")
                self._decide_for(session, panel)
                return
            result = await self._runner.run(panel, str(session.date()), slot=slot)
            submitted = getattr(result, "submitted", 0)
            outcome = session_outcome(
                getattr(result, "state", "UNKNOWN"),
                decided=bool(getattr(result, "decided", False)),
                blocked=getattr(result, "blocked", None), sent=submitted)
            self.log.info(f"{self.id} {session.date()}: {outcome}")
            _diag(self, lambda: (
                f"runner result decided={bool(getattr(result, 'decided', False))} "
                f"submitted={submitted} blocked={getattr(result, 'blocked', None)!r}"))
            # ONE OUTCOME ROW PER SESSION, decided or not. Without it "ran and declined" and "never
            # ran" are indistinguishable in the only record an operator reads.
            journal = self.session_journal()
            if journal is not None:
                await journal.write(
                    "state", f"session outcome — {outcome}", session=str(session.date()), slot=slot,
                    detail={"state": getattr(result, "state", "UNKNOWN"),
                            "decided": bool(getattr(result, "decided", False)),
                            "blocked": getattr(result, "blocked", None),
                            "submitted": submitted, "submitted_basis": SENT_BASIS,
                            "position_side": self.POSITION_SIDE,
                            "entered": list(getattr(result, "entered", ()) or ()),
                            "exited": list(getattr(result, "exited", ()) or ())})
        except asyncio_CancelledError:
            raise
        except Exception as exc:                                       # noqa: BLE001
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.log.error(f"{self.id}: session {session.date()} failed: "
                           f"{type(exc).__name__}: {exc}\n{trace}")

    def _bar_type(self, iid: InstrumentId):
        from nautilus_trader.model.data import BarType

        return BarType.from_str(f"{iid}{self._suffix}")

    # -- data ------------------------------------------------------------------------------------

    def on_historical_data(self, data) -> None:
        """Requested bars arrive HERE, not on `on_bar`. A lane that requests history and does not
        implement this discards every bar it asked for, spends the venue's paced allowance to do it,
        and still warms at one bar a session."""
        if isinstance(data, Bar):
            self._ingest(data, source="historical")

    def on_bar(self, bar: Bar) -> None:
        self._ingest(bar, source="live")

    def _ingest(self, bar, *, source: str = "unknown") -> None:
        sym = str(bar.bar_type.instrument_id.symbol)
        row = {"date": pd.Timestamp(bar.ts_event, tz="UTC").normalize().tz_localize(None),
               "ticker": sym, "open": float(bar.open), "high": float(bar.high),
               "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume)}
        dq = self._bars[sym]
        before = len(dq)
        d = self.__dict__
        d["_ingested_bars"] = d.get("_ingested_bars", 0) + 1
        # Alpaca REPUBLISHES the current session's daily bar as it updates. Appending every arrival
        # put ~103 copies of one date per symbol after thirteen hours, which gave the trailing window
        # zero variance and a strategy that sat inert.
        if dq and dq[-1]["date"] == row["date"]:
            dq[-1] = row
        else:
            dq.append(row)
        ready = _ready_count(self)
        _diag(self, lambda: (
            f"bar source={source} symbol={sym} date={row['date'].date()} len={len(dq)} "
            f"was={before} ready={ready}/{_attr(self, '_min_warm')} total={d['_ingested_bars']}"), level="debug")
        last = d.get("_last_ready_count", 0)
        min_warm = getattr(self, "_min_warm", None)
        if ready != last and min_warm is not None and (ready >= min_warm or ready % 10 == 0 or ready < 5):
            _diag(self, lambda: (
                f"warm breadth {last}->{ready} of min_warm={min_warm} after {source} {sym} {row['date'].date()}"))
        d["_last_ready_count"] = ready

    @property
    def warm(self) -> bool:
        """Warms as a UNIVERSE, against a breadth the caller had to state.

        Per-name warmup is the engine's own — `apply_gates` marks a name ineligible until it has
        `cfg.warmup_sessions` behind it. What this answers is the different question the engine
        cannot: is the screen wide enough yet for today's answer to be the strategy that was
        measured?
        """
        ready = sum(1 for v in self._bars.values() if len(v) >= self._need)
        return ready >= self._min_warm

    def _warmth_detail(self) -> dict:
        """Machine-readable warmup counts for operator rows."""
        counts = {str(sym): len(dq) for sym, dq in sorted(self._bars.items())}
        ready = [sym for sym, have in counts.items() if have >= self._need]
        short = {sym: have for sym, have in counts.items() if have < self._need}
        return {"ready_symbols": len(ready), "required_symbols": self._min_warm,
                "bars_needed_per_symbol": self._need, "symbols_with_bars": len(counts),
                "short_symbols": short}

    def _warmth(self) -> str:
        """Human-readable warmup counts against the configured breadth."""
        detail = self._warmth_detail()
        bits = [
            f"warm symbols {detail['ready_symbols']}/{detail['required_symbols']}",
            f"{detail['symbols_with_bars']} symbols with bars",
            f"need {detail['bars_needed_per_symbol']} bars/name",
        ]
        short = detail["short_symbols"]
        if short:
            examples = list(short.items())[:8]
            bits.append("short examples " + ", ".join(
                f"{sym} {have}/{detail['bars_needed_per_symbol']} SHORT"
                for sym, have in examples))
        return "; ".join(bits)

    def panel(self) -> pd.DataFrame:
        rows = [r for dq in self._bars.values() for r in dq]
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", *_BAR_FIELDS])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    # -- decision --------------------------------------------------------------------------------

    def _with_a_row_for(self, session: pd.Timestamp, panel: pd.DataFrame) -> pd.DataFrame:
        """Add a placeholder row for an intraday session whose daily bar does not exist yet.

        CRSISHORT decides before the US open. IB serves a daily bar only after that session completes,
        so `features.date == session` is empty intraday even when history is perfectly warm. Slicing to
        the prior row is not equivalent: `build_feature_panel` shifts every feature, so the prior row
        carries the close before the newest completed session. A NaN placeholder invents no price and
        lets the shift put the latest completed close into the session being decided.
        """
        if panel.empty:
            return panel
        session = pd.Timestamp(session).normalize()
        dates = pd.to_datetime(panel["date"]).dt.normalize()
        seen = set(panel.loc[dates == session, "ticker"].astype(str))
        tickers = sorted(set(panel["ticker"].astype(str)))
        missing = [sym for sym in tickers if sym not in seen]
        if not missing:
            return panel
        rows = [{"date": session, "ticker": sym, "open": float("nan"), "high": float("nan"),
                 "low": float("nan"), "close": float("nan"), "volume": 0.0} for sym in missing]
        return pd.concat([panel, pd.DataFrame(rows)], ignore_index=True)

    def _refuse_a_stale_feed(self, session: pd.Timestamp, panel: pd.DataFrame) -> bool:
        """Refuse when the placeholder would make a stale feed look like a valid session."""
        if panel.empty:
            return False
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
            f"on stale data. The short book holds what it has.")
        self.missed_sessions.append(session)
        return True

    def session_intent(self, session: pd.Timestamp, panel: pd.DataFrame) -> "SessionIntent":
        """WHAT this lane wants for one session — covers, entries and the limit for each.

        PUBLIC, AND THE ONLY PLACE THAT COMPUTES IT. A session runner attached from cockpit must call
        this rather than re-deriving the intent from the panel: `engine.decide` deliberately returns
        no exits, so a driver that grew its own would give this book two exit authorities that must
        agree and cannot be tested together — which is the defect `engine.decide`'s own docstring
        refuses to allow inside the decision layer.

        Three rules, three clocks, and they are deliberately not merged:

          covers      read the SESSION's own bar against the trail carried since entry.
          departures  read whether a held name can still be priced at all, and whether it may be
                      held below the ENTRY floor (`hold_through`).
          entries     read features as of the PRIOR close, and rest a limit during this session.
        """
        panel = self._with_a_row_for(session, panel)
        features = build_feature_panel(panel, self._cfg, adjustment=self._adjustment)
        day = features.loc[features["date"] == pd.Timestamp(session)]
        if day.empty:
            dates = []
            if not features.empty and "date" in features:
                dates = sorted(str(x) for x in pd.to_datetime(features["date"]).dt.date.unique())[-5:]
            _diag(self, lambda: (
                f"feature panel has no row for {session.date()} features_rows={len(features)} "
                f"last_feature_dates={dates}"), level="error")
            return SessionIntent()
        manageable = int(day.get("manageable", pd.Series(dtype=bool)).fillna(False).sum())
        in_universe = int(day.get("in_universe", pd.Series(dtype=bool)).fillna(False).sum())
        _diag(self, lambda: (
            f"intent day rows={len(day)} manageable={manageable} in_universe={in_universe} "
            f"held={len(_attr(self, '_held', ()))} pending={len(_attr(self, '_pending', ()))}"))

        # THE BOOK AS IT STOOD AT THE LAST CLOSE, captured before a single cover is applied.
        #
        # `runner_crsi_short` — the runner that produced the accepted curve — passes exactly this and
        # says why: using the post-exit book "would let a name covered at this morning's open be
        # re-shorted by an order that could not have existed", because this lane's entry is a LIMIT
        # PLACED THE NIGHT BEFORE. The rotation lanes reproduce same-session re-entry on purpose;
        # this one must not, and the difference is not a preference — it is whether the live lane
        # trades the strategy that was measured.
        held_at_open = set(self._held)

        plan = evaluate_short_exits(self._cfg, self._session_bars(panel, session), self._trail)
        self._trail = dict(plan.state)
        covers = dict(plan.exits)
        kinds = dict(plan.kind)

        for sym, reason, kind in self._departures(day):
            # A cover the exit rules already chose is not overwritten: the trail's reason is the one
            # the backtest booked, and relabelling it would move a trade between the exit-mix buckets
            # that #123's acceptance checks (flat 48% / reversal 48% / nodata 4%).
            if sym in covers:
                continue
            covers[sym] = reason
            kinds[sym] = kind

        borrow = self._borrow_for(day)
        dec = decide(day, self._cfg, held_at_open, borrow=borrow, pending=set(self._pending))
        _diag(self, lambda: (
            f"intent result enter={list(dec.enter)} limits={dict(dec.limits)} cover={covers} "
            f"refused={dict(dec.refused)}"))
        return SessionIntent(cover=covers, cover_kind=kinds, cover_px=dict(plan.exit_px),
                             enter=tuple(dec.enter), limits=dict(dec.limits),
                             refused=dict(dec.refused))

    def _departures(self, day: pd.DataFrame):
        """Held names that leave the book for a reason that is not one of the exit rules.

        TWO DIFFERENT QUESTIONS, and conflating them was one of the lab's four engine defects:

          NOT MANAGEABLE   the name has left the tape — below the $1M priceable floor, or its
                           indicators are NaN. The lab's signal table has no row for it, and the
                           position is closed at its last mark and booked `nodata`. That is 4% of
                           #123's trades: a rule with a measured cost, not a data-quality footnote.
          BELOW THE ENTRY  the $200M floor gates ENTRIES only. `hold_through=True` (the frozen spec)
          FLOOR            manages such a position to its exit; False closes it. The difference is
                           worth 30 points of return — +80.7% against +50.9% — because closing on
                           liquidity closes WINNERS for a reason unrelated to the trade.

        A held name missing from today's rows entirely is `nodata` too: absent from the table is the
        same fact as present-but-unpriceable, and treating it as "hold" would age a position through
        a data outage.
        """
        rows = {r.ticker: r for r in day.itertuples(index=False)}
        for sym in sorted(self._held):
            row = rows.get(sym)
            if row is None or not bool(row.manageable):
                yield sym, ("no usable price or indicator this session — closed at the last mark "
                            "(nodata)"), "nodata"
            elif not self._cfg.hold_through and not bool(row.in_universe):
                yield sym, (f"below the ${self._cfg.min_dollar_volume:,.0f} entry floor and "
                            f"hold_through is off"), "liquidity"

    def _decide_for(self, session: pd.Timestamp, panel: pd.DataFrame) -> None:
        """Compute and PUBLISH one session's intent. Submits nothing, ever."""
        intent = self.session_intent(session, panel)
        self.log.info(
            f"{self.id} {session.date()}: short {len(self._held)} · cover {intent.cover} "
            f"({intent.cover_kind}) · enter {list(intent.enter)} at {intent.limits} · "
            f"refused {intent.refused}")

    def _session_bars(self, panel: pd.DataFrame, session: pd.Timestamp) -> dict[str, SessionBar]:
        """This session's own OHLC per HELD symbol.

        A symbol with no bar is omitted rather than defaulted. `evaluate_short_exits` then leaves it
        alone entirely and does NOT age its state — a data outage must not quietly walk a position
        through a holding rule.
        """
        today = panel.loc[panel["date"] == pd.Timestamp(session)]
        out: dict[str, SessionBar] = {}
        for row in today.itertuples(index=False):
            if row.ticker not in self._trail:
                continue
            out[row.ticker] = SessionBar(open=float(row.open), high=float(row.high),
                                         low=float(row.low), close=float(row.close))
        return out

    def _borrow_for(self, day: pd.DataFrame) -> dict[str, float | None] | None:
        """Locate fees for the names in play, or None when the gate is deliberately off.

        A provider that RAISES is not a provider that answered "no locate": the first is an outage
        and the second is a fact about a symbol. So a failure propagates as an empty answer for
        every name — which `engine.decide` refuses on, name by name, with a reason recorded — rather
        than as a silently permissive one.
        """
        if self._cfg.max_borrow_fee_annual is None:
            return None
        symbols = sorted(set(day["ticker"]) | set(self._held))
        try:
            got = self._borrow_rates(symbols)
        except Exception as exc:                                       # noqa: BLE001
            self.log.warning(f"{self.id}: borrow provider failed ({type(exc).__name__}: {exc}) — "
                             f"treating every name as having NO locate for this session")
            return {sym: None for sym in symbols}
        return {sym: got.get(sym) for sym in symbols}

    # -- the slot reservation, the half that did not exist (#172) ----------------------------------

    def record_pending(self, symbol: str, client_order_id: str, limit_px: float | None,
                       kind: str) -> None:
        """A slot is spent HERE, at submit — not at the fill.

        Called by whoever submits (the cockpit-side session gateway). `_pending` was read in three
        places and written in none, so the reservation the decision layer sizes against could never
        exist: `decide` saw an empty set every session and would re-signal a name whose limit was
        already resting, submitting a second order against one slot.

        IT MUST BE CALLED BEFORE THE ORDER GOES OUT, not after. A fill can arrive before the submit
        call returns, and a fill for a symbol with no reservation moves nothing (see
        `on_order_filled`) — so recording afterwards trades a leaked slot for a lost position.
        """
        sym = str(symbol)
        prior = self._pending.get(sym)
        if prior is not None:
            # TWO LIVE ORDERS ON ONE SLOT IS THE THING THE SLOT EXISTS TO PREVENT. Overwriting would
            # make the second order's fill clear a reservation the first still needs, and the first
            # fill would then find nothing and be discarded.
            raise ValueError(
                f"{sym} already has a pending {prior.kind} ({prior.client_order_id}); recording a "
                f"second reservation on one slot would discard whichever fill arrives first")
        self._pending[sym] = PendingOrder(kind=kind, client_order_id=str(client_order_id),
                                          limit_px=None if limit_px is None else float(limit_px))
        self.log.info(f"{self.id}: {sym} slot reserved for {kind} "
                      f"({client_order_id}, limit {limit_px})")

    def clear_pending(self, symbol: str, *, reason: str) -> None:
        """Return the slot, and say what returned it.

        `reason` is REQUIRED and keyword-only. A fill, a cancel, an expiry, a rejection and a denial
        all empty this slot and are five different facts; #172 is what a slot whose state carries no
        record of how it got there costs to diagnose. A default would be how every caller comes to
        omit it.
        """
        sym = str(symbol)
        gone = self._pending.pop(sym, None)
        if gone is None:
            return
        self.log.info(f"{self.id}: {sym} slot released after {gone.kind} "
                      f"({gone.client_order_id}) — {reason}")

    def pending(self) -> set[str]:
        """The reserved SYMBOLS, which is what `decide(pending=...)` counts.

        A set rather than the mapping: handing out the dict would make `len()` right by accident and
        let a caller mutate the reservation through the back door.
        """
        return set(self._pending)

    def reconcile_pending(self) -> None:
        """Rebuild the reservations from the VENUE after a restart.

        `_pending` is process memory. A restart with a limit still resting would come back reading
        zero reserved slots and submit a second order against a slot that is already taken.

        FROM THE CACHE, NOT FROM A JOURNALLED COPY OF THE DICT. Persisting `_pending` separately
        would make two records of one fact, and they disagree silently the moment a cancel lands
        while the process is down — the shape that cost `sync_claim_to` a 260-share claim on a
        position that never existed. The venue's own open and in-flight orders ARE the fact; this
        method has no second opinion to reconcile against.

        IN-FLIGHT COUNTS. Submitted and not yet acknowledged is still committed, and those are
        exactly the orders sent moments before the process died.
        """
        cache = getattr(self, "cache", None)
        if cache is None:
            return
        found: dict[str, PendingOrder] = {}
        for getter in ("orders_open", "orders_inflight"):
            fn = getattr(cache, getter, None)
            if fn is None:
                continue
            try:
                orders = fn(strategy_id=self.id) or ()
            except TypeError:
                orders = fn() or ()
            for order in orders:
                sym = str(order.instrument_id.symbol)
                if sym in found:
                    continue
                # THE SIDE TELLS AN ENTRY FROM A COVER, and it is not decorative: a resting BUY on
                # this lane is a cover, and reconciling it as an entry would make its fill land in
                # `on_order_filled`'s entry branch and RE-OPEN the short it was closing.
                side = fill_side(order)
                if side is None:
                    self.log.error(
                        f"{self.id}: resting order {getattr(order, 'client_order_id', '?')} on "
                        f"{sym} has an unreadable side — the slot is NOT reserved and the lane may "
                        f"submit against it")
                    continue
                kind = "exit" if side == closing_order_side(self.POSITION_SIDE) else "enter"
                found[sym] = PendingOrder(
                    kind=kind, client_order_id=str(getattr(order, "client_order_id", "")),
                    limit_px=_as_price(getattr(order, "price", None)))
        self._pending = found
        if found:
            self.log.info(f"{self.id}: {len(found)} slot(s) reconciled from the venue — "
                          + ", ".join(f"{s} {p.kind}" for s, p in sorted(found.items())))

    # -- book state, from ORDER EVENTS only --------------------------------------------------------

    def on_order_filled(self, event) -> None:
        # A FOREIGN FILL CAN STILL HAVE CLOSED US (#133). `is_foreign` answers "did someone else
        # place this"; it does not answer "did our position change". A protective stop stamped with
        # this lane fills and the short is gone, whoever placed the order — and for THIS lane the
        # trail goes with it, or the next session evaluates covers against a book that is not there.
        if protective_close(self, event):
            return
        if is_foreign(event):
            return
        sym = str(event.instrument_id.symbol)
        order = self._pending.get(sym)
        # NO RESERVATION, NO BOOK MOVE — and this is the branch that was live for the whole time
        # `_pending` was never written (#172). `completes(None, ...)` returns True by design, for a
        # terminal-recording caller that must not assert an outcome; here the fall-through means an
        # unrecorded fill reaches the `else` below and is recorded as an ENTRY. A covering BUY would
        # add the name back to `_held` and open a fresh trail at the COVER price, so the lane closes
        # a short and immediately believes it opened one.
        #
        # Refused rather than guessed from the side: a SELL this lane did not reserve is an
        # unattributed order, not evidence that this lane is short.
        if order is None:
            self.log.error(
                f"{self.id}: {event.instrument_id} filled with NO reserved slot "
                f"(side={fill_side(event)!r}); book state NOT moved. Either an order went out "
                f"without record_pending, or this fill belongs to someone else.")
            return
        intent = order.kind
        # THE SIDE IS THIS LANE'S, not the shop's. For a short lane the entry is a SELL and the exit
        # is a BUY; under the long mapping this handler would refuse its own entry fill and read the
        # cover as the entry completing.
        if not completes(intent, event, side=self.POSITION_SIDE):
            self.log.warning(
                f"{self.id}: {event.instrument_id} fill does not complete pending intent "
                f"{intent!r} (side={fill_side(event)!r}); book state NOT moved")
            return
        self.clear_pending(sym, reason="filled")
        if intent == "exit":
            self._held.discard(sym)
            self._trail.pop(sym, None)
        else:
            self._held.add(sym)
            self._start_trail(sym, event)

    def _start_trail(self, sym: str, event) -> None:
        """Start the trail at the price this short was actually opened at.

        The FILL price, never the limit: the rule rests a limit 3% above the close and fills at the
        better of the limit and the open, so the two differ on exactly the gap sessions where the
        flat exit matters most. Seeding from the limit would arm the flat cover at a level the
        position never traded through.
        """
        px = getattr(event, "last_px", None) or getattr(event, "avg_px", None)
        try:
            entry = float(px)
        except (TypeError, ValueError):
            entry = float("nan")
        if not math.isfinite(entry) or entry <= 0:
            # No usable fill price: refuse to invent one. Without a trail the flat cover cannot arm,
            # so this is journalled loudly rather than left to be discovered as a position that
            # never exits.
            self.log.error(f"{self.id}: {sym} filled with an unusable price {px!r} — no trail "
                           f"opened, the flat cover CANNOT arm for this position")
            return
        bar = self._bars.get(sym)
        last = bar[-1] if bar else None
        if last is None:
            self.log.error(f"{self.id}: {sym} filled with no bar on hand — no trail opened")
            return
        self._trail[sym] = open_state(
            entry, SessionBar(open=float(last["open"]), high=float(last["high"]),
                              low=float(last["low"]), close=float(last["close"])))

    def on_order_rejected(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event, reason="rejected by the venue")

    def on_order_denied(self, event) -> None:
        if is_foreign(event):
            return
        self._forget(event, reason="denied before the venue")

    def on_order_canceled(self, event) -> None:
        """A cancelled entry RETURNS ITS SLOT. The entry is a limit that rests into the next opening
        auction, so a cancel — the venue's, or ours when the name leaves the universe overnight — is
        an ordinary outcome, not an error. Nautilus' base handler runs otherwise and moves nothing,
        which leaks the slot for the life of the process (#172)."""
        if is_foreign(event):
            return
        self._forget(event, reason="canceled")

    def on_order_expired(self, event) -> None:
        """Same slot, different ending. A limit-on-open that the auction does not fill EXPIRES
        rather than cancels, and that is this lane's most common non-fill by design."""
        if is_foreign(event):
            return
        self._forget(event, reason="expired")

    def _forget(self, event, *, reason: str = "rejected or denied") -> None:
        """A denied EXIT must leave the position HELD. Forgetting the intent AND the holding makes a
        real position read as foreign, and nothing ever covers it."""
        sym = str(event.instrument_id.symbol)
        self.clear_pending(sym, reason=reason)

    # -- account ---------------------------------------------------------------------------------

    def _on_broker_account(self, snapshot: dict) -> None:
        self._broker_account = snapshot

    def broker_equity(self) -> float | None:
        """A METHOD, not a @property: `NautilusBroker.equity()` calls it, so as a property the `()`
        would land on its result and a legitimate None would become `None()`.

        The key is `equity`. Cockpit reads the venue's `portfolio_value` and REPUBLISHES it under
        that name; there is no `portfolio_value` key on the message, and reading that name returns
        None every time, which sizes every entry to zero shares.
        """
        if not self._broker_account:
            return None
        v = self._broker_account.get("equity")
        if v is None:
            return None
        # NON-FINITE IS NOT A NUMBER. A nan passes `is not None` and `float()` happily and then
        # disarms every comparison downstream, the daily-loss halt included — which fails by never
        # halting. None is already a legal return here and callers handle it.
        out = float(v)
        return out if math.isfinite(out) else None
