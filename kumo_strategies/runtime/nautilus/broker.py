"""`Broker` implemented against a live Nautilus strategy.

The session runner was submitting Alpaca REST orders directly. That worked, and it meant the
TradingNode had no idea those orders existed: no RiskEngine pre-trade check, nothing in the cache,
nothing to reconcile against, and a cockpit UI that could not see a position the strategy held.
Routing through Nautilus puts execution back where CLAUDE.md says it belongs.

Three properties are preserved deliberately, because the runner depends on each:

  replay safety   `AlpacaBroker` leaned on Alpaca rejecting a duplicate `client_order_id`. Nautilus
                  generates its own ids, so the deterministic one is passed EXPLICITLY and a
                  duplicate is denied locally. That denial arrives as an order STATE, not an
                  exception, which is why `submit()` reads the order back rather than trusting
                  `submit_order()` to return or raise. The journal's unique index is the first line
                  against a replayed session; this is the second.
  ownership       `positions()` returns EVERY open position, not just this strategy's. The runner
                  subtracts what it owns to find foreign holdings and leave them alone. Filtering by
                  strategy_id here would make `foreign` permanently empty and silently delete that
                  check.
  submit != held  `submit()` reports whether the order was ACCEPTED for submission, never that it
                  filled. Book state moves on fill events, in the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

from nautilus_trader.model.enums import (
    OrderSide, OrderStatus, TimeInForce, order_status_to_str)
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from kumo_strategies.runtime.executor.broker import OrderRequest, OrderResult
from kumo_strategies.runtime.nautilus.sides import LONG, closing_order_side

#: `OrderRequest.side` is a plain string. Anything else used to fall through to BUY, because the old
#: line asked only whether it was "SELL" -- so a mistyped side placed a real order in the wrong
#: direction and reported success.
_ORDER_SIDES = frozenset({"BUY", "SELL"})

# Terminal states a LOCAL rejection lands in, synchronously, before submit_order() returns. The
# msgbus is in-process and the RiskEngine denies on the calling thread, so by the time we read the
# order back the denial is already applied -- this is not a poll and there is no window to lose.
_FAILED_ON_SUBMIT = frozenset({OrderStatus.DENIED, OrderStatus.REJECTED})

#: The time-in-force values a caller may name, resolved HERE rather than passed through as a string.
#:
#: `AT_THE_OPEN` is the one CRSISHORT needs and the reason this mapping exists. The IB adapter maps
#: it to IB's `"OPG"` (`adapters/interactive_brokers/parsing/execution.py:42`) — a limit-on-open that
#: participates in the OPENING CROSS. Measured on that lane's 231 accepted trades: 108 of them
#: (46.8%) gap through the resting limit and fill at the opening print, and those carry 77.6% of the
#: total return at +9.99%/trade against +2.53% for an ordinary limit fill. An order that merely rests
#: during the session earns the weaker arm.
#:
#: A NAME THIS DOES NOT KNOW IS REFUSED, never defaulted. Falling back to DAY would turn a
#: limit-on-open into an order that expires at the close, and the lane would never learn its entry
#: had become something else.
_TIME_IN_FORCE = {
    "DAY": TimeInForce.DAY,
    "GTC": TimeInForce.GTC,
    "IOC": TimeInForce.IOC,
    "FOK": TimeInForce.FOK,
    "AT_THE_OPEN": TimeInForce.AT_THE_OPEN,
    "AT_THE_CLOSE": TimeInForce.AT_THE_CLOSE,
}


def _tick_error(px: float) -> str | None:
    """Why this limit price cannot be sent, or None.

    NOT ROUNDED HERE. `crsi_short.engine.apply_gates` already rounds `limit_px` to the penny, and its
    comment says why: "a limit at 12.3449 rests at 12.34, not 12.35", which changes which fills
    happen at all. Rounding a second time in this layer would silently correct a caller that computed
    the price differently, and the live fill would then differ from the measured one with nothing to
    show for it. Two roundings of one number is two derivations; the second is where they disagree.
    """
    if px is None:
        return None
    try:
        v = float(px)
    except (TypeError, ValueError):
        return f"limit price {px!r} is not a number"
    if not math.isfinite(v):
        return (f"limit price {px!r} is not finite — a nan passes `is not None` and `float()` and "
                f"then disarms every comparison downstream")
    if v <= 0:
        return f"limit price {v} must be positive"
    # Every name this lane trades is above its $5 price floor, so the tick is a cent.
    if abs(v * 100 - round(v * 100)) > 1e-9:
        return (f"limit price {v!r} is not a whole number of pennies, and this layer refuses to "
                f"round it — the caller that computed it must, or the live fill stops matching the "
                f"measured one (a limit at 12.3449 rests at 12.34, not 12.35)")
    return None


@dataclass
class NautilusBroker:
    """Adapts a running `Strategy` to the `Broker` protocol the session runner expects."""

    strategy: object                      # the MomentumRotationStrategy, once registered
    instrument_ids: list = None           # InstrumentId list; resolved by symbol
    feed: object = None
    """cockpit's `UiFeedStrategy`, threaded in at construction. NOT reachable through `strategy` --
    that is a different Nautilus `Strategy` instance with no shared base beyond Nautilus's own, and
    no `__getattr__` delegation between them. `release_for_exit` lives only here."""

    def _iid(self, symbol: str) -> InstrumentId | None:
        """The instrument for `symbol`, asked of whoever actually resolved it (kumo-trading-platform issue 622).

        THE STRATEGY IS THE AUTHORITY. It resolves symbols at `on_start`, from the instruments the
        attached adapter loaded, and holds them in `_iids`. This used to keep its OWN copy, built by
        the same cockpit call — two derivations of one fact, and they drift.

        `instrument_ids` is still honoured when supplied, and that is not a fallback: it is the
        pre-#622 caller, kept so the pin can move without changing the tenant that is trading. It is
        authoritative when given, because a caller that passes an explicit list means it.

        `strategy` is None between construction and `momentum.py:724`; anything asking in that window
        gets a refusal rather than an AttributeError that would take the build down.
        """
        ids = self.instrument_ids
        if ids is None:
            ids = getattr(self.strategy, "_iids", None) or []
        return next((i for i in ids if i.symbol.value == symbol), None)

    def submit(self, req: OrderRequest) -> OrderResult:
        iid = self._iid(req.symbol)
        if iid is None:
            return OrderResult(False, None, f"{req.symbol} is not a subscribed instrument", req)
        inst = self.strategy.cache.instrument(iid)
        if inst is None:
            return OrderResult(False, None, f"no instrument definition cached for {req.symbol}", req)
        coid = ClientOrderId(req.client_order_id)
        tick = _tick_error(req.limit_px)
        if tick is not None:
            return OrderResult(False, None, tick, req)
        tif = _TIME_IN_FORCE.get(req.time_in_force)
        if tif is None:
            return OrderResult(False, None,
                               f"time_in_force {req.time_in_force!r} is not one of "
                               f"{sorted(_TIME_IN_FORCE)} — refusing to fall back to DAY, which "
                               f"would turn a resting order into one that expires at the close", req)
        if req.side not in _ORDER_SIDES:
            return OrderResult(False, None,
                               f"order side {req.side!r} is neither 'BUY' nor 'SELL' — refusing to "
                               f"guess one", req)
        try:
            # OPEN OR EXIT IS DECIDED BY THE LANE'S DECLARED SIDE, never by the word "SELL".
            #
            # This read `is_exit = req.side == "SELL"`. Every lane written before #123 is LONG, so a
            # SELL really was always an exit and the line was correct for four deployed lanes at
            # once -- which is why it survived: there was no lane that could disagree with it.
            # CRSISHORT (#123) holds the other side. Its ENTRY is a sell-short, and under the old
            # line that entry went out `reduce_only=True` against a flat book, where the venue has
            # nothing to reduce and rejects it. Not a weaker order: a different one.
            #
            # THE BOOK IS DELIBERATELY NOT CONSULTED HERE. It is tempting to ask the cache whether a
            # position exists and call the order an exit only then, but that is a second source of
            # truth for a fact the lane already states, and the two drift exactly when it matters --
            # mid-session, between a fill and the cache seeing it. The declared side is knowable
            # without the venue and cannot go stale.
            #
            # `reduce_only` IS NOT WHAT MAKES THAT SAFE, and an earlier version of this comment said
            # it was. MEASURED on the installed Nautilus: the Interactive Brokers adapter contains
            # the string `reduce_only` ZERO times -- nineteen other venue adapters implement it, IB's
            # own API has no such field, so on IB the flag is accepted here, carried through, and
            # SILENTLY DROPPED at the adapter boundary. ibkr-paper, where the first short lane goes,
            # is the IBKR instance. A safety argument that rests on it is void exactly where it is
            # first needed.
            #
            # What actually guards the exit is one level up and always was: `sides.closing_quantity`
            # refuses a flat or wrong-sided position at the adapter, before any order is built, and
            # every `_close` path is required to call it (`test_position_side_is_declared`). The flag
            # stays because on a venue that honours it it is real defence in depth -- an order that
            # outlives the position it was closing cannot open the opposite one --  but nothing may
            # DEPEND on it. `test_the_order_path_does_not_depend_on_reduce_only` pins that, and
            # re-takes the IB observation on every run rather than quoting it.
            is_exit = req.side == closing_order_side(getattr(self.strategy, "POSITION_SIDE", LONG))
            build = (self.strategy.order_factory.market if req.limit_px is None
                     else self.strategy.order_factory.limit)
            extra = {} if req.limit_px is None else {
                "price": Price.from_str(f"{float(req.limit_px):.2f}")}
            order = build(
                **extra,
                instrument_id=iid,
                # The REQUESTED side, once. The line this replaces derived it a second time from
                # `is_exit`, i.e. from `req.side` through a variable that means something else --
                # two derivations of one fact, agreeing for a long lane and disagreeing for a short.
                order_side=OrderSide.SELL if req.side == "SELL" else OrderSide.BUY,
                quantity=Quantity.from_int(int(req.qty)),
                client_order_id=coid,
                # DAY, not Nautilus's GTC default. A market order that cannot fill -- a halted
                # symbol, say -- would otherwise REST. The next session still needs that exit, so it
                # submits a second SELL under a new session's client_order_id, and when the halt
                # lifts both fill and the position is oversold into a short.
                time_in_force=tif,
                # reduce_only on exits, so an order that outlives the position it was closing cannot
                # open the opposite one.
                reduce_only=is_exit,
                # The strategy's own event handlers (`on_order_filled`/`on_order_rejected`/…) are
                # where the venue's actual answer arrives, but they receive only the Nautilus event,
                # not the session this order was submitted under. `req.session` is not derivable from
                # the fill event or from `client_order_id` (a one-way hash) any other way, and Nautilus
                # persists tags on the order in its own durable cache -- so a restart between submit
                # and fill does not lose the correlation the way an in-process dict would (#51).
                tags=[f"session:{req.session}"],
            )
            self.strategy.submit_order(order)
        except Exception as e:                                        # noqa: BLE001
            return OrderResult(False, None, f"{type(e).__name__}: {e}", req)

        # `submit_order()` RETURNING IS NOT SUCCESS. It accepts the command into Nautilus's order
        # lifecycle and returns; the venue has not seen it, and a local denial -- risk-engine
        # rejection, or a duplicate ClientOrderId -- surfaces as an order STATE, synchronously, not
        # as an exception. Reporting ok=True here let the runner write durable position state for an
        # order Nautilus had already denied: phantom ownership on a denied buy, and on a denied sell
        # the strategy forgetting it owns a live position, which the next session then reads as
        # someone else's and refuses to manage.
        placed = self.strategy.cache.order(coid)
        if placed is None:
            return OrderResult(False, None, "order not in the cache after submit", req)
        # Compare the ENUM, never a stringified one. `str(OrderStatus.DENIED)` is "2" -- the enum
        # stringifies to its integer value -- so a substring test for "DENIED" silently never
        # matches and every denied order reads as accepted. `order_status_to_str` is the name, and
        # it is used only for the human-readable detail.
        status = placed.status
        if status in _FAILED_ON_SUBMIT:
            reason = getattr(getattr(placed, "last_event", None), "reason", "") or "no reason given"
            return OrderResult(False, req.client_order_id,
                               f"{order_status_to_str(status).lower()}: {reason}", req)
        # Anything else is in flight, which is all submit can honestly claim. Whether it FILLS is an
        # order event, and the strategy's fill handlers are what may move the book.
        return OrderResult(True, req.client_order_id,
                           f"submitted via nautilus ({order_status_to_str(status).lower()})", req)


    def _say(self, level: str, msg: str) -> None:
        """Log without ever becoming the reason an order path failed.

        `self.strategy` is None between construction and registration, and `Actor.log` is a
        read-only Cython attribute that a narrow test host may not carry — so reaching for it must
        not be what raises. The notice never costs the act.
        """
        try:
            getattr(self.strategy.log, level)(msg)
        except Exception:                                             # noqa: BLE001
            pass

    def order_status(self, client_order_id: str) -> str | None:
        """The venue's status for one order, or None when it is NOT IN THE CACHE (#131).

        `crsi_short.replacement_is_owed` needs this at the post-cross slot to decide whether the
        auction leg ended without taking the slot. It reads the same seam `cancel` does, one line
        below — `self.strategy.cache.order(ClientOrderId(...))` — rather than a second lookup path.

        ABSENT IS ITS OWN ANSWER AND IS NOT A STATUS, deliberately. `replacement_is_owed` treats
        EXPIRED and CANCELED as "the intent stands, send the replacement" and everything else as no,
        so folding "not in the cache" into a status string would either invent a replacement that
        must not happen or suppress one that must. None is the only honest value, and
        `replacement_is_owed(None)` is already False — refusing costs one session's entry, guessing
        costs an unbounded short.

        !! AND THE ABSENT CASE IS THE ONE THAT DECIDES WHETHER THE DECOMPOSITION WORKS AT ALL, so it
        must be MEASURED rather than assumed. If Nautilus keeps a venue-cancelled OPG order in the
        cache, this returns CANCELED and the replacement fires. If it PURGES it, this returns None,
        `replacement_is_owed` answers False, and THE REPLACEMENT NEVER FIRES — silently, every
        session, while the auction leg does its job and the rest of the intent is simply dropped.
        Nobody in either repo knows which it is today. The distinct log line below exists so the
        first live session answers it by grep instead of by inference.

        REPORTS, NEVER RAISES, for the reason `cancel` gives: this runs on the session path beside
        other symbols, and an order that already filled, was already cancelled, or was never placed
        is an ORDINARY answer here rather than an error.
        """
        try:
            order = self.strategy.cache.order(ClientOrderId(client_order_id))
        except Exception as exc:                                      # noqa: BLE001
            # A CACHE THAT CANNOT ANSWER IS NOT AN ABSENT ORDER, and the two must not share a line.
            # Both yield None, because there is no safe third answer on the order path — but only
            # one of them is a broken system, and an operator needs to tell them apart.
            self._say("error", f"order_status({client_order_id}) could not read the cache "
                               f"({type(exc).__name__}: {exc}) — treating it as ABSENT, so no "
                               f"replacement will be sent for this leg this session")
            return None
        if order is None:
            self._say("info", f"order_status({client_order_id}): NOT IN CACHE — no replacement "
                              f"will be sent. If the venue cancelled this leg and Nautilus purged "
                              f"it, the decomposition is inert and this line is the evidence (#131)")
            return None
        status = getattr(order, "status", None)
        return str(getattr(status, "name", status)) if status is not None else None

    def cancel(self, client_order_id: str) -> OrderResult:
        """Pull a resting order.

        CRSISHORT cancels an entry whose name dropped out of the universe overnight, and the lab
        does that every session — `for tkr in [t for t in pend if t not in entry_pool.index]: del
        pend[tkr]`. Dropping the cancel changes the trade set, so it is part of the rule rather than
        housekeeping.

        REPORTS, NEVER RAISES. It runs on the session path beside other symbols, and an order that
        has already filled, already been cancelled, or was never placed is an ORDINARY answer here —
        the auction cancels an unfilled limit-on-open by itself, so "not in the cache" is the
        expected outcome on many sessions, not an error. Raising would stop the symbols queued
        behind it.
        """
        req = OrderRequest(symbol="", side="BUY", qty=0, session="", strategy_id="")
        order = self.strategy.cache.order(ClientOrderId(client_order_id))
        if order is None:
            return OrderResult(False, client_order_id,
                               f"{client_order_id} is not in the cache — already filled, already "
                               f"cancelled, or never placed", req)
        try:
            self.strategy.cancel_order(order)
        except Exception as e:                                        # noqa: BLE001
            return OrderResult(False, client_order_id, f"{type(e).__name__}: {e}", req)
        # Like `submit`, this reports that the CANCEL was accepted, never that the venue honoured it.
        # A cancel racing a fill is decided at the venue and arrives as an order event.
        return OrderResult(True, client_order_id, "cancel submitted via nautilus", req)

    async def exit(self, req: OrderRequest) -> OrderResult:
        """A plain `submit()` SELL is refused at the venue when a resting protective stop already
        reserves the full quantity -- `available: 0` means unreserved is 0, not held is 0. FSM and
        VCTR both hit this: reconciliation confirmed the shares were held, and the exit was refused
        anyway (issue 48, kumo-trading-platform issue 358).

        `release_for_exit` is cockpit's, on `feed` (`UiFeedStrategy`) -- NOT `strategy`, a separate
        Nautilus `Strategy` instance `submit()` uses that shares nothing with `feed`. Suppresses
        protection for this position (TTL-bounded, dropped on every exit path so a hung release
        cannot leave it permanently unprotected), cancels the specific resting stop, and waits for
        the venue to confirm the shares are actually free before returning. False means the release
        did not land in time; the position KEEPS its protection, which is the safe end state, so
        nothing is sent -- firing an exit against an unconfirmed release is how a naked position
        happens. That includes when protection is disabled or the market is closed: cockpit refuses
        the release rather than cancel something it cannot re-arm.

        `strategy_id` NAMES THE CANCELLER, and it is a different argument from the one deleted in
        #245/#252 despite the identical name. Read the distinction before touching this line:

          FILTER         which resting order to cancel and wait on. Stays cockpit's own
                         `MANUAL-001`, because that is who the protective stops actually carry.
                         Passing the LANE here was tried and was wrong -- it matched nothing, so the
                         cancel-confirm step cleared instantly against a stop still resting and
                         returned True having waited for nothing. A naked position presenting as a
                         successful exit, strictly worse than the defect it was meant to fix.
          CANCELLER      who is asking. This is the lane, and it is an AUTHORISATION.

        Passing it flips cockpit's `proxy` concession to False and engages the strict ownership rule
        on the exit path for the first time (kumo-trading-platform issue 462): MOMENTUM exiting BCTROT's WHD is
        REFUSED rather than silently cancelling BCTROT's protection. Without it, every exit from
        every lane arrived as MANUAL-001 and "exiting my own stop" and "exiting yours" were the same
        call -- the #437 incident, mechanically.

        LANDED ONLY AFTER THE ACCEPTOR WAS VERIFIED RUNNING. `inspect.signature` was read inside the
        live api container on 2026-08-23, not from cockpit's repo, because an image tag is a claim
        and the 2026-08-22 stale build was byte-identical in its log to a correct one. Passing an
        argument the deployed acceptor does not take kills every exit with `TypeError` -- with
        protection already suppressed. `test_the_feed_double_matches_the_signature_RUNNING_IN_THE_CONTAINER`
        pins that shape.
        """
        iid = self._iid(req.symbol)
        if iid is None:
            return OrderResult(False, None, f"{req.symbol} is not a subscribed instrument", req)
        if self.feed is None:
            return OrderResult(False, None,
                               "broker.feed is not wired — cannot release the shares the "
                               "protective stop is holding", req)
        released = await self.feed.release_for_exit(str(iid), int(req.qty),
                                                   strategy_id=req.strategy_id)
        if not released:
            return OrderResult(False, None, "shares not released — protection still holds them", req)
        return self.submit(req)

    def positions(self) -> dict[str, int]:
        """Every open position the node knows about -- see "ownership" in the module docstring."""
        return self._sum_positions(self.strategy.cache.positions_open())

    def strategy_positions(self) -> dict[str, int]:
        """Only the quantity THIS strategy holds, per Nautilus's own attribution.

        Ownership was symbol-only: a symbol appeared in our claims, so the WHOLE account quantity
        for it was treated as ours. If MOMENTUM opened 10 AAPL and a manual trade holds 90, an exit
        submitted SELL 100 -- liquidating someone else's position to flatten ours.

        Under NETTING the position id is `{instrument}-{strategy_id}`, so Nautilus already keeps the
        split; nothing here needs to infer it. Note the ADR's caveat: the broker verifies only the
        NET, so this attribution is ours, not the broker's, and a position reconciled in from
        outside will not carry our id -- which is correct, it is not ours to sell.
        """
        return self._sum_positions(
            self.strategy.cache.positions_open(strategy_id=self.strategy.id))

    def other_strategy_positions(self) -> dict[str, int]:
        """The quantity every OTHER lane holds, per Nautilus's attribution, net per symbol (#830).

        Feeds `retire_claims(theirs=...)`: a claim of ours on a symbol whose whole account quantity
        is attributed to other lanes is dead. Positions reconciled in from outside carry the
        `EXTERNAL` strategy id and no lane — they are excluded here, because "nobody's" is the
        silence the retirement rule already refuses to read as evidence. MANUAL-001 COUNTS: a
        manual position is positively somebody else's, which is exactly the evidence wanted.
        """
        mine = str(self.strategy.id)
        return self._sum_positions(
            p for p in self.strategy.cache.positions_open()
            if str(p.strategy_id) not in (mine, "EXTERNAL"))

    def position_entries(self) -> dict[str, float]:
        """Symbol -> the average price THIS strategy actually opened at.

        Exists so an adopted position can be seeded with a real entry instead of today's price
        (kumo-trading-platform issue 197 B1). Nautilus already carries `avg_px_open` on every Position; the runner
        simply had no way to ask, because `strategy_positions()` returns quantity and discards
        everything else — which is why the trail was seeded from a quote and asserted a peak that
        never happened.

        Only positions Nautilus attributes to THIS strategy. Anything reconciled in from outside
        carries a different id and is deliberately absent: we do not know what it opened at.
        """
        out: dict[str, float] = {}
        for p in self.strategy.cache.positions_open(strategy_id=self.strategy.id):
            avg = getattr(p, "avg_px_open", None)
            if avg:
                out[p.instrument_id.symbol.value] = float(avg)
        return out

    def last_price(self, symbol: str, *, max_age_ns: int | None = None) -> float | None:
        """The freshest price the node holds, or None to let the caller fall back to the close.

        Sizing off the prior close is wrong by exactly the overnight gap, and the position cap
        counts positions rather than dollars, so an oversized entry passes every downstream check.

        `max_age_ns` is OPT-IN (kumo-trading-platform's item 4, same design): `None` is today's behaviour
        exactly, existence-only. A bound is enforced against `ts_event`, which only the tick/bar
        objects carry — `cache.price()` returns a bare `Price` with no observation time at all, so a
        bounded caller skips it entirely rather than silently treating an unaged value as fresh.
        Without this, both of this repo's staleness guards were existence checks on a value that,
        once a session has run at all, is nearly never absent — which is why the exit-side guard has
        fired zero times in this system's life.
        """
        iid = self._iid(symbol)
        if iid is None:
            return None
        now = self.strategy.clock.timestamp_ns() if max_age_ns is not None else None
        for get in ("price", "quote_tick", "trade_tick"):
            if max_age_ns is not None and get == "price":
                continue                    # no ts_event on a bare Price -- cannot honour the bound
            fn = getattr(self.strategy.cache, get, None)
            if fn is None:
                continue
            try:
                v = fn(iid)
            except Exception:                                      # noqa: BLE001
                continue
            if v is None:
                continue
            if max_age_ns is not None and now - getattr(v, "ts_event", now) > max_age_ns:
                continue                    # too old -- try the next source rather than accept it
            for attr in ("as_double", "value"):
                if hasattr(v, attr):
                    got = getattr(v, attr)
                    return float(got() if callable(got) else got)
            for attr in ("last_px", "bid_price", "price"):
                if hasattr(v, attr):
                    return float(getattr(v, attr).as_double())
        # Fall back to the freshest BAR. This strategy subscribes to bars only — no quotes, no
        # trades — so the tick caches above are permanently empty and every entry was refused with
        # "no live price to size against". A decided session submitted nothing, all eight names, on
        # the first live run.
        #
        # The forming daily bar IS current during the session, so this is not the stale
        # yesterday's-close the sizing guard exists to reject; when the session has not opened yet
        # it is the previous close, which is the best available and no worse than the old default.
        # With a bound in effect, that previous-close case is exactly what must be rejected instead.
        try:
            bar_type = self.strategy._bar_type(iid)
            bar = self.strategy.cache.bar(bar_type)
            if bar is not None:
                if max_age_ns is not None and now - bar.ts_event > max_age_ns:
                    return None
                return float(bar.close.as_double())
        except Exception:                                          # noqa: BLE001
            pass
        return None

    @staticmethod
    def _sum_positions(positions) -> dict[str, int]:
        """Net per symbol, SIGNED.

        This summed `p.quantity`, which is Nautilus's MAGNITUDE -- "the current open quantity", never
        negative -- while `p.signed_qty` is the signed one. So BCTROT's +28 WHD and MOMENTUM's -28
        summed to 56 rather than netting to 0, and the broker reports
        `GET /v2/positions/WHD -> 404 position does not exist`. The account holds none.

        kumo-trading-platform hit the identical defect in market value (#427): a SHORT reported positive and a
        flat book rendered as 56 shares held. Same symbol, same number, same cause, other repo.

        IT FEEDS THE OWNERSHIP CEILING, which is why it is not cosmetic. `positions()` supplies
        `acct_qty` to `own_ceiling`, and an inflated account makes the ceiling too PERMISSIVE:

            unsigned  own_ceiling(56, my_claim=28, other=28) = 28   sells a position that is not there
            signed    own_ceiling(0,  my_claim=28, other=28) =  0   correct

        A lone SHORT now reports negative rather than positive, which is also the point: a long-only
        lane holding a negative is a real condition -- MOMENTUM-002 is carrying -28 WHD -- and
        reporting it as +28 hides it from every check that looks for one.
        """
        out: dict[str, int] = {}
        for p in positions:
            sym = p.instrument_id.symbol.value
            out[sym] = out.get(sym, 0) + int(getattr(p, "signed_qty", p.quantity))
        return out

    def equity(self) -> float:
        """Total account balance.

        NET LIQUIDATION, from the broker's own snapshot -- delegated to the strategy, which
        subscribes to it. Two wrong answers were live here in turn:

          portfolio.account(venue)   resolved by INSTRUMENT venue, but the account issuer is the
                                     broker (`ALPACA-master`) while instruments carry XNAS/XNYS.
                                     Found no account at all, and a mixed pool raised first.
          account.balance_total(USD) resolves, and returns CASH. Nautilus `AccountState` models
                                     cash only and cockpit's Alpaca client puts cash in
                                     `AccountBalance.total`. With $20k cash against $100k net-liq
                                     that halts the strategy for a drawdown that never happened --
                                     and hides a real one, because cash does not move with marks.

        The daily-loss halt anchors on this number. A wrong one does not fail loudly, it fails by
        never halting -- the worst way for a risk limit to break -- so every path here raises rather
        than returning a plausible-looking zero that would quietly disarm the check.
        """
        return self.strategy.broker_equity()
