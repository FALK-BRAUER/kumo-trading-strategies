"""The registration contract every strategy implements, so the platform can police them uniformly.

    kumo-trading-platform issue 39, 2026-08-15. The operator's call: uniform, not per-strategy.

THE PRINCIPLE: THE STRATEGY DECLARES, THE PLATFORM DECIDES
----------------------------------------------------------
A strategy is never asked to be trustworthy about a limit, only honest about what it is doing.
There is deliberately no `check_your_own_budget` here and there must never be one: a strategy
polices its own allocation correctly right up until the day it has a bug, and then holds more than
it was granted, silently. Enforcement belongs on the order path, which is cockpit's.

A refusal from that gate is NORMAL, not an error. Logging it as one, retrying, or halting turns an
ordinary wind-down into an incident.

WHY THE ENTRY/EXIT ASYMMETRY IS LOAD-BEARING, AND WHY THE STRATEGY DOES NOT DECIDE IT
------------------------------------------------------------------------------------
Cockpit refuses ENTRIES when a strategy is over budget and always permits EXITS. A strategy over
budget NEEDS to sell, and blocking that traps it above target permanently.

The side alone does not say which is which — a SELL is an exit on a long and an ENTRY on a short —
so this module states the rule once, as `is_entry_order`, and cockpit's `budget_gate.is_entry`
implements the same test. The two agreeing is what makes a wind-down un-blockable on both sides.

THERE WAS ALSO AN `is_entry` MEMBER ON THE CONTRACT, REMOVED 2026-08-22 (kumo-trading-platform issue 442). Its
rationale said only the strategy can answer this, because only it knows its position. That was true
about the arithmetic and false about who can do it: cockpit reads the same net position from the same
Nautilus cache (`exec_client.py:612`) and never once called the strategy's version. A required member
that nothing calls is worse than dead code — this module defined `RegistrationMixin` TWICE, and the
dead copy's `is_entry` had an empty body, which would have classified every order as not-an-entry and
disabled the budget gate. Nothing bound to it, so it never fired, and nothing would have noticed.

THE TAG IS COCKPIT'S, THE NAME IS OURS
--------------------------------------
`external_id` is this repo's stable name for a strategy. The `order_id_tag` is not ours to choose:
cockpit is the only place that sees every strategy in ONE trader and it is what calls
`Trader.add_strategy`, so it allocates. Publishing a hardcoded tag from an adapter made this repo
the authority on an id cockpit owns, and cost a full 003 -> 004 -> 003 churn.

The asymmetry is the reason it works: the INTERNAL id keys positions under NETTING
(`{instrument}-{strategy_id}`) and must never move once anything has traded, whereas an EXTERNAL id
is only a lookup key — so a strategy can be renamed freely and no position moves.
"""

from __future__ import annotations

import pandas as pd

import asyncio
from dataclasses import dataclass
from datetime import timedelta

from typing import Protocol, runtime_checkable

from kumo_strategies.runtime.calendar import OutsideCalendarWindow
#: The journal's OWN vocabulary, imported rather than restated — a second definition of
#: a kind is how a row ends up in a bucket no reader watches.
from kumo_strategies.runtime.executor.pgjournal import ORDER, RISK
from kumo_strategies.runtime.nautilus.order_provenance import fill_side, is_foreign
from kumo_strategies.runtime.nautilus.sides import LONG, closing_order_side, refusal_reason


def is_entry_order(net_position: float, side: str, qty: float) -> bool:
    """Would this order INCREASE absolute exposure? Then it is an entry.

    Pure and dependency-free so it can be tested without a Nautilus node, and so both the adapters
    and cockpit's gate can reason about the same rule.

    Stated as "increases absolute exposure" rather than "same side as the position" because that is
    the formulation that gets the two awkward cases right:

      REDUCING OR FLATTENING is always an exit, whatever the side. This is the property the
      wind-down depends on — an over-budget strategy must never be blocked from selling down, so
      anything that shrinks the position has to classify as an exit even when it is a BUY closing a
      short.

      A FLIP is an entry. Selling 50 against a +10 long leaves a -40 short: 10 of it reduced and 40
      of it opened new exposure in the other direction. Calling that an exit would let a strategy
      over budget open an unbounded short and label it a wind-down, which is exactly the dishonesty
      the platform cannot detect from outside.

    `side` is Nautilus's "BUY"/"SELL". A zero quantity is not an entry — it changes nothing.
    """
    signed = qty if side.upper() == "BUY" else -qty
    return abs(net_position + signed) > abs(net_position)


@runtime_checkable
class StrategyRegistration(Protocol):
    """What cockpit needs from a strategy in order to register and police it (kumo-trading-platform issue 39).

    A Protocol rather than a base class: the adapters already inherit from Nautilus's `Strategy`,
    and a second inheritance edge would buy nothing. Conformance is asserted by tests per adapter.
    """

    @property
    def external_id(self) -> str:
        """This repo's stable name — "MOMENTUM", "QC345". Never carries a number: the tag is
        cockpit's to allocate, and a numbered name is what made 003 look claimed on one side and
        free on the other."""

    @property
    def label(self) -> str:
        """Human-readable, for the activation screen."""

    @property
    def claimed_instruments(self) -> list:
        """Instruments this strategy owns unattributed activity for.

        EXCLUSIVE across the node: `register_external_order_claims` raises InvalidConfiguration if
        two strategies claim one instrument, during `Trader.add_strategy` — so an over-broad claim
        does not degrade, it stops the node booting.
        """

    @property
    def warmup_bars(self) -> int:
        """Bars per symbol required before any decision is trustworthy."""


    def preflight(self, broker) -> "list[Probe]":
        """OBSERVE each capability once, so the PLATFORM can judge whether this lane can act.

        DECLARED IN THE PROTOCOL, not merely on the mixin. On the mixin alone, a strategy that does
        not use the mixin satisfies the Protocol while being unprobeable — and to anything reading
        the Protocol, "unprobeable" is indistinguishable from "probed and healthy". Same shape as
        every other defect this week, so it is declared where cockpit looks.

        Cockpit's `strategy_contract.REQUIRED` needs the matching entry, or this is enforced only by
        kumo-trading-strategies' own suite.
        """
        ...



@dataclass(frozen=True)
class Probe:
    """ONE OBSERVATION. Deliberately carries no verdict (kumo-trading-platform issue 438).

    THE STRATEGY OBSERVES, THE PLATFORM JUDGES — the same principle this module already states for
    budget, applied to health. A `preflight()` returning pass/fail is self-attestation one layer up:
    it reports healthy right up until the day the thing deciding health is the thing that is broken,
    which describes every defect found between 2026-08-20 and 2026-08-22.

    There is deliberately no `ok`/`passed`/`status` field and there must never be one. If it exists,
    something will eventually set it from inside the strategy.

    WHY THE VALUE AND NOT A BOOLEAN. `equity` raising, `equity=None` and `equity=0.0` are three
    different diagnoses -- a @property called as a method, a key the publisher never sends, and an
    unfunded sleeve. A boolean flattens them into one. And TECHIVOL-005's `owned` probe is the case
    that settles it: it returned eight symbols and NO error, so it looked healthy. Those eight were
    another strategy's book, and that is only knowable against "this strategy has never filled
    anything" -- which cockpit knows and the strategy does not.
    """

    name: str
    value: object = None
    error: str | None = None




# ==================================================================================================
# A FOREIGN FILL THAT CLOSED OUR POSITION (#133)
#
# MODULE-LEVEL, NOT A MIXIN METHOD, and that is not a style choice. `PennyGapStrategy` and
# `IntradayMomentumRotation` subclass Nautilus's `Strategy` DIRECTLY and do not carry
# `RegistrationMixin` — so `self.protective_close(...)` in their fill handlers would raise
# AttributeError inside Nautilus's dispatch, on a live event, for every lane that is not a rotation
# lane. One function every handler can call is the only shape that covers all of them.
# ==================================================================================================

# -- a foreign fill that closed OUR position (#133) -------------------------------------------

def protective_close(self, event) -> bool:
    """Did this FOREIGN fill close a position of ours? Book state and the claim follow if so.

    Returns True when the event was handled here and the caller must not process it further.

    WHY THIS EXISTS. `is_foreign` answers "did someone else place this order", and every fill
    handler used that one answer for two different questions. The first is right: a foreign SELL
    must never be read as our ENTRY completing, or a protective stop marks a symbol held out of
    nothing (kumo-trading-platform issue 748). The second was never asked at all: a protective stop that FILLS
    is our position closing, whoever placed it.

    Measured on an Alpaca paper instance 2026-09-10 (#133): `PROT-SELL-TOST-XNYS-93bfd28c`, stamped TECHIVOL-005,
    filled 97 in partials; six OrderFilled events reached the lane; the handler returned at
    `is_foreign` for all six. No terminal row, `_held` still holding TOST, and the claims ledger
    asserting 97 against a book of 0 for the rest of the session — which narrows every OTHER
    lane's sellable quantity, not just this one's.

    THE BOOK IS THE AUTHORITY HERE, and that is deliberate rather than a shortcut. Partials mean
    the position is not flat until the last one lands, so the holding is dropped only when the
    lane's OWN Nautilus position reads flat, and the claim is re-synced to whatever that position
    actually says on every partial. Counting shares out of the events instead would need this
    handler to know how many arrive, which is exactly what it cannot know.

    Elsewhere in this package the book is deliberately NOT consulted (`broker.py`, open-vs-exit)
    because the lane's declared side is knowable without the venue and cannot go stale. The
    question here is the opposite kind: "how much do I still hold" is a fact only the book has.
    """
    # FOREIGN FIRST, AND THIS LINE IS THE WHOLE POINT OF THE METHOD.
    #
    # Without it the gates below — "we hold it" and "the fill is on our closing side" — are BOTH
    # TRUE OF THE LANE'S OWN ORDINARY EXIT, so every exit on every lane was captured here:
    # journalled as "closed by a protective stop — this lane did not place that order" (false,
    # and the row an operator reads during an incident), `_record_terminal` never firing, so the
    # venue's real answer went unrecorded and momentum/qc345's own `sync_claim` was skipped, and
    # qc27 never reaching `_try_decide`. Strictly worse than the defect this method fixes.
    #
    # It shipped green: `test_foreign_order_events.py` collected `ast.Name` callees only, and
    # `self.protective_close(...)` is an `ast.Attribute`. That hole is closed in the same change.
    if not is_foreign(event):
        return False
    sym = str(event.instrument_id.symbol)
    held = getattr(self, "_held", None)
    if held is None or sym not in held:
        return False
    # Only a fill on the side that REDUCES this lane's side can have closed us. A foreign BUY on
    # a long lane is someone else opening something, and none of our business.
    # THE SIDE IS REQUIRED, NOT DEFAULTED — the same rule `completes` and `sides.py` enforce.
    # A defaulted LONG here would silently give a short lane the long mapping and this method
    # would then ignore every protective close on it, which is the defect it exists to fix,
    # inverted. Every adapter in this package declares `POSITION_SIDE`.
    side = getattr(self, "POSITION_SIDE", None)
    if side is None:
        raise AttributeError(
            f"{type(self).__name__} has no POSITION_SIDE. A lane must state which side it "
            f"holds before its fills can be interpreted (#88, #133).")
    if fill_side(event) != closing_order_side(side):
        return False
    try:
        qty = int(sum(int(p.signed_qty) for p in self.cache.positions_open(
            instrument_id=event.instrument_id, strategy_id=self.id)))
    except Exception as exc:                                       # noqa: BLE001
        # Never raise into Nautilus's dispatch. Without the book we cannot say whether this
        # closed us, and guessing in either direction is worse than saying so.
        self.log.error(f"{self.id}: {sym} protective fill NOT reconciled — could not read the "
                       f"position ({type(exc).__name__}: {exc})")
        return False
    if qty == 0:
        held.discard(sym)
        pending = getattr(self, "_pending", None)
        if pending is not None:
            pending.pop(sym, None)
        # THE TRAIL GOES WITH THE POSITION. Removed from #133 when it merged to main, because no
        # lane there kept a trail and an unreachable line that LOOKS like careful handling is how a
        # mechanism comes to be believed in. CRSISHORT keeps one, so it returns here WITH the test
        # that reaches it (`test_a_protective_close_clears_the_TRAIL_not_just_the_holding`).
        #
        # Not cosmetic: `_trail` is what arms the flat cover, and `evaluate_short_exits` iterates
        # STATE rather than holdings — so a trail left behind has the next session evaluating covers
        # against a book that no longer exists.
        trail = getattr(self, "_trail", None)
        if trail is not None:
            trail.pop(sym, None)
    # LOUD, always. A position leaving the book because protection fired is not an ordinary exit
    # and an operator reading the journal must be able to tell the two apart.
    self.log.warning(f"{self.id}: {sym} closed by a protective stop ({fill_side(event)} "
                     f"{getattr(event, 'last_qty', '?')}), position now {qty} — this lane did "
                     f"not place that order")
    # THE ROW IS WRITTEN HERE AND NOT THROUGH `_record_terminal`, deliberately. That method
    # recovers the session from the ORDER'S OWN `session:` tag, which `nautilus/broker.py` stamps
    # on orders THIS adapter submits — a protective stop was placed by cockpit and carries no
    # such tag, so `_record_terminal` returns early for exactly the events this method exists
    # for. Calling it anyway would look like journalling and write nothing, which is the failure
    # mode kumo-trading-platform issue 549 and #383 both already cost this repo.
    _journal_protective_close(self, sym, event, qty)
    _sync_claim_to_book(self, sym, event, qty)
    return True

def _journal_protective_close(self, sym: str, event, qty: int) -> None:
    """One durable row saying the position left the book and that this lane did not do it.

    EVERY CAPABILITY IS RESOLVED WITH `getattr`, INCLUDING `session_journal` ITSELF. It is a
    `RegistrationMixin` method and `PennyGapStrategy` / `IntradayMomentumRotation` subclass Nautilus's
    `Strategy` directly, so calling it unguarded raises AttributeError inside Nautilus's dispatch on
    a live fill — the failure the module-level form was supposed to prevent, one call deeper.

    THE BOOK IS UPDATED BEFORE THIS RUNS AND DOES NOT DEPEND ON IT. A lane that cannot write the row
    must still stop believing it holds a position that is gone. What it must NOT do is go quiet: the
    absence of a durable trace is the whole subject of #133, so an unreachable journal is logged
    with the capability that was missing.
    """
    resolve = getattr(self, "session_journal", None)
    report = getattr(self, "fire_and_report", None)
    loop = getattr(self, "_loop", None)
    if resolve is None or report is None:
        self.log.error(
            f"{self.id}: {sym} was closed by a protective stop and NO durable row was written — "
            f"this lane has no {'session_journal' if resolve is None else 'fire_and_report'}, so "
            f"the only trace of the position leaving the book is this line (#133)")
        return
    journal = resolve()
    if journal is None or loop is None:
        self.log.error(
            f"{self.id}: {sym} was closed by a protective stop and NO durable row was written — "
            f"{'no journal is attached' if journal is None else 'no event loop was captured'} "
            f"(#133)")
        return
    session = str(pd.Timestamp.utcnow().date())
    try:
        # KIND `order`, PHASE `terminal` — the convention `record_terminal` already writes and
        # every cockpit reader keys on (/slots verdicts, the journal filters, the UI strategy
        # plane all match kind='order' AND detail.phase='terminal'). A NEW kind is accepted by
        # the column, written durably, and INVISIBLE everywhere an operator would look for it,
        # which for a row whose entire purpose is to make a silent close visible would be the
        # same defect wearing a fix's clothes.
        self.fire_and_report(
            journal.write(ORDER,
                          f"terminal: {sym} closed by a protective stop — this lane did not "
                          f"place that order", session=session, symbol=sym,
                          detail={"phase": "terminal", "ok": True, "by": "protection",
                                  "position_after": qty, "side": fill_side(event),
                                  "client_order_id": str(getattr(event, "client_order_id", "")),
                                  "issue": "issue 133"}),
            loop, f"protective-close row for {sym}")
    except Exception as exc:                                       # noqa: BLE001
        self.log.error(f"{self.id}: protective-close row for {sym} NOT written: {exc}")

def _sync_claim_to_book(self, sym: str, event, qty: int) -> None:
    """Bring the claims ledger back to what the book says (kumo-trading-platform issue 829).

    Best-effort by design: it runs inside a live event handler, so a missing method or a dead
    loop degrades rather than raising inside Nautilus's dispatch.
    """
    runner, loop = getattr(self, "_runner", None), getattr(self, "_loop", None)
    report = getattr(self, "fire_and_report", None)          # mixin-only, like session_journal
    if runner is None or loop is None or report is None:
        return
    sync = getattr(runner, "sync_claim", None)
    if sync is None:
        return
    px = float(event.last_px) if getattr(event, "last_px", None) is not None else None
    try:
        self.fire_and_report(sync(sym, qty, px), loop, f"claim sync for {sym}")
    except Exception as exc:                                       # noqa: BLE001
        self.log.error(f"{self.id}: claim sync for {sym} NOT attempted: {exc}")



def report_wrong_sided_positions(self, *, session: str) -> list[str]:
    """Journal a RISK row for every position this lane holds on the side it cannot manage.

    Returns the symbols found, so a caller can decide whether to proceed.

    WHY THIS EXISTS SEPARATELY FROM THE EXIT REFUSAL. `sides.closing_quantity` and `refusal_reason`
    already refuse a wrong-sided position — on the EXIT path, when the lane is already trying to
    exit that name. WHD sat in the book as `+28 BCTROT / -28 MOMENTUM` for twelve hours because
    nobody was trying to exit it. A refusal is not a detector.

    AND IT CANNOT BE REPAIRED AUTOMATICALLY, which is why reporting is the whole job. The broker has
    ONE net position per symbol and no opinion about whose it is: WHD's drift detector reported
    nothing and was RIGHT to, because +28 and -28 net to zero and the broker agreed. The per-strategy
    split is unanchored and always will be. Nothing in a lane that declares its side can have opened
    a position on the other one, so it came from reconciliation, a manual order or a phantom — and
    none of those make it ours to trade against.

    THIS LANE'S POSITIONS ONLY. Reading the account's book here would report another lane's short as
    our unmanageable position, which is #66's own defect inverted — TECHIVOL-005 proposed exiting
    eight names it did not own by reading the account as its own.
    """
    side = getattr(self, "POSITION_SIDE", None)
    if side is None:
        raise AttributeError(
            f"{type(self).__name__} has no POSITION_SIDE, so it cannot say which positions are "
            f"unmanageable (#88, #66).")
    try:
        held = list(self.cache.positions_open(strategy_id=self.id))
    except Exception as exc:                                       # noqa: BLE001
        # Never raise into a live handler. A book we cannot read is not a book we can judge, and
        # saying so is better than a silent empty answer that reads as "nothing wrong".
        self.log.error(f"{self.id}: could not read this lane's positions to check their side "
                       f"({type(exc).__name__}: {exc}) — wrong-sided positions NOT checked")
        return []

    found: list[str] = []
    for p in held:
        qty = int(getattr(p, "signed_qty", 0) or 0)
        why = refusal_reason(qty, side=side)
        if why is None:                       # correctly sided, or flat — both are ordinary
            continue
        sym = str(p.instrument_id.symbol)
        found.append(sym)
        # LOUD, EVERY SESSION, UNTIL A HUMAN RESOLVES IT. This is not a transient: it survives
        # restarts, and a lane that mentioned it once would let it age out of the log.
        self.log.error(f"{self.id}: holds {sym} on the side it cannot manage — {why}")
        _journal_risk(self, session, sym, qty, why)
    return found


def _journal_risk(self, session: str, sym: str, qty: int, why: str) -> None:
    """Best effort, and it does not go quiet when the durable path is unreachable — two lanes carry
    no `RegistrationMixin`, so `session_journal` and `fire_and_report` do not exist on them. The log
    line above has already been written whatever happens here."""
    resolve = getattr(self, "session_journal", None)
    report = getattr(self, "fire_and_report", None)
    loop = getattr(self, "_loop", None)
    if resolve is None or report is None or loop is None:
        return
    journal = resolve()
    if journal is None:
        return
    try:
        report(journal.write(RISK, f"{sym} is held on the side this lane cannot manage",
                             session=session, symbol=sym,
                             detail={"symbol": sym, "signed_qty": qty,
                                     "position_side": self.POSITION_SIDE, "reason": why,
                                     "issue": "issue 66"}),
               loop, f"wrong-sided row for {sym}")
    except Exception as exc:                                       # noqa: BLE001
        self.log.error(f"{self.id}: wrong-sided row for {sym} NOT written: {exc}")


class RegistrationMixin:
    """Default implementations for adapters that keep the conventional attributes.

    Everything here is DECLARATIVE — a name, a label, the instruments claimed, the warmup needed —
    plus `preflight`, which observes without judging, and `begin_arming`, which schedules. Nothing
    here decides whether the strategy may act. That is cockpit's, on the order path.
    """

    # NOTE 2026-08-22: this module defined `RegistrationMixin` TWICE. The second shadowed the first,
    # so every adapter inherited the second and the first was dead code — including an `is_entry`
    # with an EMPTY BODY, which would have returned None and classified every order as not-an-entry,
    # so cockpit's budget gate would never have refused one. Nothing bound to it, so it never fired;
    # it was found only because a `preflight` added to the first silently did not exist on any
    # strategy. Guarded by `test_no_class_in_this_package_is_defined_twice`.
    # -- preflight ---------------------------------------------------------------------------------
    #: The calls a lane must be able to make between deciding and submitting. NOT per strategy: every
    #: one of the eight defects found 2026-08-20..22 sits on the runner<->broker interface, which is
    #: what that interface is for. A lane needing a seventh overrides this and says so in code.
    #
    # `universe` was EMITTED before it was DECLARED (#80, caught by cockpit's cross-repo contract
    # check). That gap is the mirror of the defect the probe itself was written for: the probe
    # existed, the contract did not know about it, so a lane that quietly stopped emitting it would
    # still have satisfied this tuple. Emitted-but-undeclared and written-but-unread are the same
    # family — a mechanism nothing is holding to account.
    PREFLIGHT_PROBES: tuple[str, ...] = ("armed", "equity", "owned", "price", "universe")


    def preflight(self, broker) -> "list[Probe]":
        """OBSERVE each capability once. Never raises; a call that fails reports `error`.

        Read-only plus arithmetic — a preflight that traded would be a preflight nobody dares run.
        The order path is proven after the fact by cockpit's slot-outcome detectors on real intent.

        `lifecycle` and `budget` are deliberately absent: those are cockpit's own state, and asking
        the strategy to report them would be asking it to attest to something it does not own.
        """
        out: list[Probe] = []

        def observe(name, call):
            try:
                out.append(Probe(name, call()))
            except Exception as exc:                                   # noqa: BLE001
                out.append(Probe(name, None, repr(exc)))

        # ARMED IS A CAPABILITY, and the one cockpit cannot observe from outside. A lane whose
        # calendar never answered probes healthy on every other line here while being unable to ever
        # decide; TECHIVOL-005's `owned` probe is the precedent — eight symbols, no error, another
        # strategy's book.
        observe("armed", lambda: self.is_armed)
        # THE UNIVERSE THIS LANE CAN ACTUALLY TRADE (#80). Three facts were recorded and read by
        # nothing in-process: `_unresolved_symbols`, `_ambiguous_symbols`, and an empty pool. A lane
        # that resolved 60 of 97 symbols reported `armed=True` with every other probe answering, and
        # only a log line said otherwise — which `self.log` being Nautilus's logger makes unreadable
        # from here. 64 symbols carry two venue identities on staging (kumo-trading-platform issue 625) and nothing
        # could see it without a screenshot.
        #
        # ALWAYS EMITTED, including for a lane built from `instrument_ids`, which has nothing to
        # resolve. An omitted probe reads as "we never checked" and is indistinguishable from "not
        # applicable" — absence reported as an all-clear, which is what this whole preflight exists
        # to end.
        #
        # COUNTS, NOT A VERDICT. `requested == resolved + len(unresolved)` is an invariant a reader
        # can check, and it disagrees loudly if any path stops recording. `ambiguous` is a SUBSET of
        # resolved — those symbols did resolve, to an identity we picked among several — so it is
        # reported separately rather than folded into either number.
        observe("universe", self._universe_probe)
        observe("equity", lambda: broker.equity())
        observe("owned", lambda: broker.strategy_positions())
        # THE SUBJECT IS WHAT THE LANE TRADES, NOT WHAT IT ADOPTS (kumo-trading-platform issue 544). This read
        # `claimed_instruments` — i.e. `external_order_claims`, documented as "instruments this
        # strategy owns UNATTRIBUTED activity for". That is a registry of foreign positions a lane
        # adopts, not its universe. MOMENTUM-002 is the only lane that has any, so it was the only
        # lane that ever emitted this probe: QC345-003 watches 164 instruments, claims none, and
        # reported `no such probe` on every boot. Surfaced once cockpit's #532 removed the platform
        # noise that had been hiding it.
        #
        # AND THE PROBE IS ALWAYS EMITTED, even with nothing to price. An omitted probe reads as "we
        # never checked" and is indistinguishable from "not applicable" — absence reported as an
        # all-clear, which is the shape this whole preflight exists to end. "No instruments" is a
        # fact about the lane and belongs in the report.
        iids = list(getattr(self, "_iids", ()) or ()) or list(
            getattr(self, "claimed_instruments", ()) or ())
        if iids:
            first = str(iids[0]).split(".")[0]
            observe("price", lambda: broker.last_price(first))
        else:
            out.append(Probe("price", None, "no instruments to price — the lane trades nothing"))
        return out

    # -- arming ------------------------------------------------------------------------------------
    #: How long to wait before asking the calendar again. A minute is short enough that a transient
    #: outage costs one slot at worst, and long enough not to hammer a venue that is genuinely down.
    ARM_RETRY_SECS: int = 60

    def begin_arming(self) -> None:
        """Resolve this lane's schedule WITHOUT blocking the loop and WITHOUT taking the node down.

        THE FAILURE THIS EXISTS FOR, 2026-08-22: BCTROT-004 raised `URLError(<urlopen error timed
        out>)` out of `on_start`. Nautilus re-raised it from `Trader.START`, and QC345-003 and
        TECHIVOL-005 — neither of which had anything wrong with it — never started, because they were
        queued behind it in the start sequence. One dead socket, three lanes down.

        The chain was four calls long and entirely ours: `on_start` -> `_arm` -> `next_slot_fire` ->
        `AlpacaCalendar.day` -> `urlopen(timeout=15)`. Every lane had it. Only one was ever reached.

        TWO DEFECTS SAT ON THAT ONE LINE and both are fixed here, because fixing either alone leaves
        the node broken in a way that looks fixed:

          FATAL   a raise propagated to the start sequence. Now it retries, and the lane that cannot
                  reach the calendar is the ONLY lane affected.
          BLOCKING  a 15s synchronous socket read on the event loop stalls every other strategy's
                  start, the data feeds and the message bus. The fetch now runs off the loop.

        WHAT IS DELIBERATELY NOT SOFTENED: an unresolved calendar still means NOT ARMED. There is no
        fallback to the holiday-unaware `WeekdayCalendar` and there must never be one — that would
        turn a visible outage into a session scheduled on a market holiday. A lane that cannot say
        WHEN it decides is not the same thing as a lane that must take the node down, and this
        separates those two; it does not make the first one harmless.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # NO RUNNING LOOP. Usually a backtest — the calendar is injected rather than fetched, so
            # there is no socket to fail and an ARMING failure is a real bug that must stay loud. It
            # still does: `_arm_now` only guards the missed-slot REPORT, and a raise from `_arm`
            # propagates straight out of here.
            #
            # A REPORT failure no longer propagates, and that is deliberate. It is announced at
            # ERROR instead, and the harness is where loudness belongs: research runs should treat
            # an unexpected ERROR record as a failure. Making the guard conditional on loop-absence
            # was considered and rejected — "no running loop" does NOT mean backtest here, because
            # Nautilus's `LiveClock` fires timer callbacks from a thread with no loop (see
            # `rearm_after_alert`). Keying fatality on that would make a LIVE lane die on a
            # diagnostic failure, on the path hardest to test, which is the defect this guard exists
            # to prevent. (Fable, reviewing this branch.)
            self._arm_now()
            return
        self._arming = loop.create_task(self._arm_until_resolved())

    async def _arm_until_resolved(self) -> None:
        """Retry until the calendar answers. Announces every failure; never gives up quietly."""
        from kumo_strategies.runtime.calendar import warm

        while True:
            try:
                # OFF the loop. A timer that fires the same blocking `urlopen` back onto the event
                # loop would be the identical defect with a delay on it.
                #
                # ONLY THIS CALL IS INSIDE THE `try`, and the scope is the point. `_arm_now` resolves
                # slots, computes the next fire and sets a time alert — none of which can fail for a
                # network reason. Catching it here would retry a DETERMINISTIC failure every
                # ARM_RETRY_SECS forever while reporting it as a socket problem: one malformed
                # `*_SLOTS` entry in operator-editable settings ("open+45", missing the `m`) raises
                # `SlotError` out of `slots.resolve`, and the lane would start, never arm, never
                # decide, and blame the venue in every log line. A bug laundered into a business
                # condition, on the lane already hardest to diagnose. Found in review of 7de760a.
                await asyncio.to_thread(warm, self._calendar, self.clock.utc_now())
            except asyncio.CancelledError:
                raise
            except Exception as exc:                                   # noqa: BLE001
                # UNARMED IS ANNOUNCED, EVERY TIME. A lane that starts, silently fails to resolve its
                # schedule and sits quiet is counted as running while being incapable of ever
                # deciding — indistinguishable from a lane that is armed and simply holding. That is
                # a worse outcome than the crash this replaces, because the crash was visible.
                self.log.error(
                    f"{self.id}: UNARMED — the trading calendar did not answer ({exc!r}). This "
                    f"strategy CANNOT DECIDE until it arms. Retrying in {self.ARM_RETRY_SECS}s. No "
                    f"fallback calendar is used on purpose: guessing the session would schedule a "
                    f"decision into a closed venue.")
                await asyncio.sleep(self.ARM_RETRY_SECS)
                continue

            try:
                self._arm_now()
            except asyncio.CancelledError:
                raise
            except OutsideCalendarWindow as exc:
                # RETRYABLE, and it must be classified here rather than falling into the branch
                # below. The rule that branch encodes is "a failure arming is deterministic, so
                # retrying it forever while blaming the venue is worse than stopping" — correct for
                # a malformed `*_SLOTS` entry, and exactly wrong for this one.
                #
                # A venue calendar knows a ROLLING window (IB's is about six days). Running past the
                # end of it is not a bug in the lane, it is the venue not having said yet, and it
                # resolves by itself as the window advances. Treating it as permanent would leave a
                # lane unarmed until somebody restarted the node — the silent-forever outcome this
                # whole exception exists to prevent, reintroduced by the machinery meant to report it.
                self.log.error(
                    f"{self.id}: UNARMED — the calendar cannot see far enough yet ({exc!r}). This "
                    f"is a rolling-window limit, not a fault: retrying in {self.ARM_RETRY_SECS}s. No "
                    f"day is assumed open in the meantime.")
                await asyncio.sleep(self.ARM_RETRY_SECS)
                continue
            except Exception as exc:                                   # noqa: BLE001
                # NOT RETRIED, and named as itself. Retrying cannot fix it, and "the calendar did not
                # answer" said about a slot-parsing bug is worse than saying nothing: it is a
                # confident, specific, wrong answer, and it is the first thing anyone reads.
                self.log.error(
                    f"{self.id}: UNARMED — the calendar answered but arming FAILED: {exc!r}. This is "
                    f"not a network problem and will not be retried. This strategy CANNOT DECIDE "
                    f"until the cause is fixed and it is restarted.")
            return

    #: Alert name for the re-arm retry. Distinct from each lane's SESSION_ALERT on purpose — reusing
    #: that one would re-enter `_on_session_alert`, which does session bookkeeping (records a missed
    #: session, clears `_due`) that must not happen for a retry.
    REARM_ALERT = "calendar_window_rearm"

    def rearm_after_alert(self, now) -> None:
        """Re-arm from inside a fired alert, surviving a calendar that cannot see far enough yet.

        WHY THIS IS NOT JUST `self._arm(now)`. Every lane re-arms as the FIRST statement of its
        session-alert handler, deliberately: everything after it may raise, and a lane that has
        stopped scheduling looks exactly like a lane that decided to hold. But that ordering means a
        raise from `_arm` itself aborts the whole callback — so the session that just fired is never
        marked due, no decision is taken for it, AND no future alert is set. The lane goes silent
        permanently, mid-session, which is strictly worse than the failure the ordering prevents.

        Harmless until a calendar could refuse: `AlpacaCalendar` answers for any date and
        `WeekdayCalendar` answers for all of them. A venue calendar knows a rolling window
        (kumo-trading-platform issue 628), so refusing is now a normal thing for `_arm` to do and this path is
        reachable.

        The retry goes through the CLOCK, not the event loop. This runs on Nautilus's `LiveClock`
        thread where there is no running loop, so `begin_arming` would take its no-loop branch and
        call `_arm_now()` straight back into the same refusal. `set_time_alert` is what the lane
        already uses from this thread and is the one mechanism known to work here.
        """
        try:
            self._arm(now)
        except OutsideCalendarWindow as exc:
            retry_at = now + timedelta(seconds=self.ARM_RETRY_SECS)
            self.log.error(
                f"{self.id}: UNARMED — the calendar cannot see past its window yet ({exc!r}). The "
                f"session that just fired is unaffected and will still be decided. Retrying the "
                f"re-arm at {retry_at}. No day is assumed open in the meantime.")
            self.clock.set_time_alert(self.REARM_ALERT, retry_at, self._on_rearm_retry,
                                      override=True)

    def _on_rearm_retry(self, event=None) -> None:
        """Try to arm again, and keep trying. Never touches session state.

        The lifecycle check is for the RACE that cancelling the timer cannot close: this callback can
        already be executing on the timer thread when `on_stop` runs, and it would then arm a
        SESSION_ALERT on a stopped strategy. `is_running` is Nautilus's own component state, so it
        answers for the real lifecycle rather than a flag of ours that could disagree with it.
        """
        if not getattr(self, "is_running", True):
            return
        self.rearm_after_alert(self.clock.utc_now())

    def request_instrument_if_missing(self, iid) -> bool:
        """Ask the venue for an instrument DEFINITION only when the cache does not already hold it.

        WHY THE REQUEST EXISTS AT ALL, and why this guards rather than deletes it: `cache.instrument()`
        returning None is a hard refusal at submit. On the first live run every entry was priced and
        journalled and then rejected with "no instrument definition cached". Bars alone do not
        populate it. That guarantee is unchanged here — when the definition is absent, it is still
        requested.

        WHY IT IS NOW USUALLY REDUNDANT. Since kumo-trading-platform issue 622 the adapter's instruments are already
        in the Cache by `on_start`: the kernel awaits `_await_engines_connected()` before starting the
        trader, and IB's data client pushes every instrument the provider loaded into the Cache during
        that connect. Symbol resolution reads them from exactly there — a lane that resolved a symbol
        has, by construction, proved the definition is cached.

        WHAT IT COSTS TO ASK ANYWAY — AND NOT WHAT THIS FIRST CLAIMED. `request_instrument` becomes
        `reqContractDetails`, which has its OWN IB allowance. It is not `reqHistoricalData`, which
        carries the ~60-per-10-minutes historical pacing that #617 is about. Confirmed by cockpit
        against their adapter: only `request_bars` goes through their paced queue.

        So skipping these does NOT relieve the pacing that starves warmup. The first version of this
        docstring said it halved a 744-request burst against the pacing budget; that was wrong twice
        over — wrong limit, and 744 was a grep total from one 45-second window that was never broken
        down by source. Cockpit's own feed derives ~742 from seven granularities over ~106
        instruments, so their number and mine COUNT DIFFERENT THINGS and land together by accident.
        Two derivations agreeing is not corroboration when they measure different quantities.

        THE REASON THAT SURVIVES is the one that did not depend on any of that: since #622 the
        instruments are already in the Cache by `on_start`, so this is asking the venue for something
        we have provably already got. That is worth not doing whichever budget it draws on.

        Returns whether a request was actually made, so a caller can report how much it asked for
        rather than guessing.
        """
        if self.cache.instrument(iid) is not None:
            return False
        self.request_instrument(iid)
        return True

    #: Where a session journal may live, in the order it is looked for. Cockpit's `SessionGateway`
    #: stores it PRIVATELY as `_journal` (`momentum.py:47`); `PgJobRunner` exposes it publicly as
    #: `journal`. Both are legitimate and neither is this repo's to rename.
    _JOURNAL_HOLDERS: tuple[str, ...] = ("_runner", "_jobs")
    _JOURNAL_NAMES: tuple[str, ...] = ("journal", "_journal")

    def session_journal(self):
        """The journal to write session rows to, or None when there is genuinely nowhere.

        THE DEFECT THIS REPLACES (kumo-trading-platform issue 587). Each adapter guessed at the attribute, and only
        one guessed usefully:

            qc345   getattr(self._runner, "journal", None)                      -> None, silent
            momentum  getattr(self._runner, "journal", None)
                      or getattr(self._jobs, "journal", None)                   -> found on _jobs

        Cockpit's gateway stores it as `_journal`, so the FIRST lookup fails on both. Momentum only
        works because it happens to have a second thing to try, and `PgJobRunner` happens to expose
        the same journal publicly. QC345 has no such fallback, so it returned None and wrote nothing
        for the lane's entire life — `state`/`session outcome` rows for QC345-003 on paper: zero.

        AND THE RETURN WAS INDISTINGUISHABLE FROM THE LEGITIMATE CASE. `_journal_session` documents
        "degrades to a no-op with no runner attached (the backtest and standalone paths)" — true, and
        it is exactly what a wrong attribute name also looks like. A guard written for a real absence
        silently absorbed a real defect. Absence readable as permission, in the journal path whose
        whole purpose is making silence impossible.

        So the two cases are now SEPARATED. No holder at all is a backtest and returns None quietly.
        A holder that exists and carries no journal under any known name is a WIRING DEFECT and says
        so — once, because this runs per session and a repeated line would bury it.
        """
        holders = [(n, getattr(self, n, None)) for n in self._JOURNAL_HOLDERS]
        attached = [(n, h) for n, h in holders if h is not None]
        if not attached:
            return None                       # backtest / standalone: nowhere to write, by design
        for _, holder in attached:
            for name in self._JOURNAL_NAMES:
                journal = getattr(holder, name, None)
                if journal is not None:
                    return journal
        if not getattr(self, "_journal_wiring_reported", False):
            self._journal_wiring_reported = True
            self.log.error(
                f"{self.id}: a session runner is attached ({', '.join(n for n, _ in attached)}) but "
                f"carries no journal under any of {self._JOURNAL_NAMES}. Session outcome rows will "
                f"NOT be written, so 'ran and declined' and 'never ran' are indistinguishable in "
                f"the record (kumo-trading-platform issue 587).")
        return None

    def _universe_probe(self) -> dict:
        """What this lane asked for, what it got, and what it could not tell apart.

        `source` is here because the two construction paths make `requested` mean different things:
        a lane built from `symbols` has a pool it is trying to resolve, while one built from
        `instrument_ids` was handed the answer. Without it, `requested=0, resolved=164` on the ids
        path looks like a broken invariant rather than a different question.
        """
        symbols = list(getattr(self, "_symbols", ()) or ())
        iids = list(getattr(self, "_iids", ()) or ())
        unresolved = list(getattr(self, "_unresolved_symbols", ()) or ())
        ambiguous = dict(getattr(self, "_ambiguous_symbols", {}) or {})

        # RESOLUTION IS NOT USABILITY, and the first version of this probe reported only the first.
        # Measured on ibkr-paper-retired 2026-08-28 13:05 ET, BCTROT-004's first IBKR decision:
        #
        #   this probe would have said   requested 109, resolved 109, unresolved []
        #   the journal actually said    bars missing for 24/109 — refusing to rank
        #
        # So on the one night the distinction mattered, the detector written to make a degraded lane
        # look degraded would have reported it perfectly healthy. A symbol can resolve, be
        # subscribed, and never receive a single bar — `_report_dataless` has said exactly that in
        # its docstring since FTNR sat in the pool for a whole session that way.
        #
        # `min_bar_coverage` is enforced in `pgrunner` from this same dict, so reporting it here
        # makes the probe PREDICT the refusal instead of contradicting it.
        bars = getattr(self, "_bars", None) or {}
        watched = [i.symbol.value for i in iids]
        no_bars = sorted(s for s in watched if not bars.get(s))
        need = getattr(self, "warmup_bars", None)
        thin = sorted(s for s in watched
                      if bars.get(s) and isinstance(need, int) and 0 < len(bars[s]) < need)
        return {
            "source": "symbols" if symbols else "instrument_ids",
            "requested": len(symbols) if symbols else len(iids),
            "resolved": len(iids),
            "unresolved": sorted(unresolved),
            "ambiguous": {k: sorted(v) for k, v in sorted(ambiguous.items())},
            # Bars are reported as COUNT PLUS NAMES, like unresolved: the count is what a coverage
            # floor compares against, the names are what an operator needs to chase.
            "with_bars": len(watched) - len(no_bars),
            "no_bars": no_bars,
            "below_warmup": thin,
        }

    def _decision_slot_names(self) -> tuple[str, ...]:
        """This lane's slots as names, however it happens to store them.

        Two shapes exist: `momentum_rotation` keeps a `_slots` tuple, while `qc345_rotation` and
        `qc27_rotation` keep an `_open_offset` int and DERIVE the name from it (their `_slot_name`
        property). Both are legitimate; the missed-slot check needs the names, so it asks here rather
        than each lane growing its own copy of the check — which is exactly how the check came to
        exist on one lane and not the others.
        """
        slots = getattr(self, "_slots", None)
        if slots:
            return tuple(slots)
        offset = getattr(self, "_open_offset", None)
        return () if offset is None else (f"open+{int(offset)}m",)

    def report_missed_on_start(self, now) -> list:
        """Say loudly when starting up has skipped a decision that was due today.

        `_arm` asks for the NEXT fire, so a process that starts after today's slot arms tomorrow and
        today never happens — no alert, no decision row, nothing. A lane's `missed` list does not
        cover it either: that is appended when a session was already DUE, and a restart before the
        alert fires means it never became due. There is nothing to resume, because the resume path
        begins at the journal write and this never reached it.

        THE ESTABLISHED CASE: MOMENTUM-002, 2026-08-17. Engine restarted at 09:35 ET, exactly
        the decision minute, armed forward to the 18th, no decision row for the day. From the outside
        "skipped because we restarted" and "held because nothing ranked" were the same picture.

        A SECOND CASE WAS SUSPECTED AND IS REFUTED: QC345-003 produced nothing on 2026-08-25. That
        silence is measured, and it is CORRECT — QC345 is monthly, 08-25 was neither a month-start
        nor in the operator's `forced_rebalance_dates`, and a non-rebalance session journals nothing
        by design. It was blamed on a deploy landing on the lane's slot, using a value from
        `/strategies.last_decision_slot`, which reports where a lane LAST decided rather than where
        it fires next — and that value came from a settings override already removed. Left here
        because a defect this mechanism does NOT explain is worth as much as one it does.

        WHAT DID NOT DEPEND ON EITHER STORY: two of four lanes could not report a missed slot at all,
        because the fix for the 08-17 incident lived on ONE adapter. It is here now, on the mixin
        every lane already uses to arm, so a fifth lane cannot be added without it.

        THIS DOES NOT DECIDE LATE, and must not. Firing a decision hours after its slot trades a
        stale ranking at prices that have moved, and the slot exists precisely to fix when that
        happens. The honest response is to skip it visibly rather than quietly.
        """
        from kumo_strategies.runtime.calendar import elapsed_slots

        # ONCE PER PROCESS, and the key is a FLAG rather than membership of the record itself.
        #
        # WHY THE GATE EXISTS. `_arm_until_resolved` retries `_arm_now` every ARM_RETRY_SECS, and
        # since #628 a rolling-window refusal from `_arm` is RETRYABLE — so the report ahead of it
        # re-runs on every retry. `elapsed_slots` is stateless and would report the same session
        # again each time: an hour of stall adds ~60 duplicate entries to a list operators read as
        # "sessions we did not trade", plus an ERROR line a minute saying the same thing.
        #
        # WHY NOT `stamped in self.missed_on_start`, which was the first fix. Two live edges, both
        # found by Fable:
        #
        #   A STALL OUTLASTS A SESSION. The refusal this gate exists for can run past midnight.
        #   Next session the stamp differs, the gate passes, and the lane announces "STARTED AFTER
        #   ... the session was lost in the gap" for a day on which the process did not start. It
        #   has been up and announcing UNARMED the whole time; those slots are missed-WHILE-UNARMED,
        #   which the retry branch already reports. A confident, specific, wrong answer — the shape
        #   this method's own docstring condemns.
        #
        #   THE LIST IS INJECTABLE. Adapters and tests can pre-populate it, and this exact list has
        #   already been a doubles vehicle once (see the note below). Any pre-seeded entry for today
        #   would silently disable a genuine start-up report. Control state must be private and
        #   unforgeable; `missed_on_start` is DATA.
        #
        # Set on COMPLETION, not on entry, so a report that raises before recording still retries on
        # the next arming attempt. Set even when nothing was missed: start-up's answer was "nothing",
        # and a slot elapsing later during a stall is not a start-up miss.
        #
        # AT THE TOP, before any calendar call. Placed after `elapsed_slots` it still made
        # that call on every retry — and once the rolling window slips past today, that call
        # RAISES, the guard in `_arm_now` swallows it, and an ERROR is logged every retry:
        # the same log spam this gate exists to stop, reintroduced by a different route.
        if getattr(self, "_missed_report_done", False):
            return []

        specs = self._decision_slot_names()
        if not specs:
            return []
        # BOTH OR NEITHER (#131). `next_slot_fire` derives this from the same specs; a detector
        # that clamps a pre-open slot forward would report a slot the lane never armed on.
        from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

        missed = elapsed_slots(self._calendar, now, specs,
                               allow_pre_open=has_pre_open(specs))
        if not missed:
            self._missed_report_done = True     # start-up's answer was "nothing missed"
            return []
        names = ", ".join(f"{slot} ({when:%H:%M %Z})" for _, slot, when in missed)
        session = missed[0][0]
        stamped = pd.Timestamp(session)
        self.log.error(
            f"{self.id}: STARTED AFTER {len(missed)} of today's decision slots — {session} "
            f"{names} did not run and will NOT be run late. If this was a restart, the session was "
            f"lost in the gap; if the process has been down longer, so were the days before it.")
        # Recorded under its own name so it cannot be confused with "was due and found no data",
        # which is what each lane's own `missed_*` list means. Appended to that too, where it
        # exists, because momentum's counter has been read as "sessions we did not trade" since
        # before this check existed.
        # ASSIGNED BACK, NOT `getattr(...).append(...)`. That form appended to a FRESH LIST whenever
        # the attribute did not exist — and no adapter initialised it, so in production every record
        # landed on a throwaway and evaporated. The test hid it by injecting `missed_on_start` on the
        # double: an attribute production did not have, in the fix for a defect about writes leaving
        # no trace (found by 2026-08-26). The mixin owns the list now, so no adapter has to
        # remember to declare it — which is how a fifth lane would have reintroduced this.
        if not isinstance(getattr(self, "missed_on_start", None), list):
            self.missed_on_start = []
        # ONCE PER SESSION, and this is not tidiness. `_arm_until_resolved` retries `_arm_now` every
        # ARM_RETRY_SECS, and since #628 a rolling-window refusal from `_arm` is RETRYABLE — so the
        # report ahead of it re-runs on every retry. `elapsed_slots` is stateless and would append
        # the same session again each time: restart after the close on the last day the venue's
        # window describes, and an hour of stall adds ~60 duplicate entries to a list operators read
        # as "sessions we did not trade", plus an ERROR line a minute saying the same thing.
        #
        # The report is a statement about start-up, and start-up happened once. Found by Fable
        # reviewing the guard below, which is what made the retry reachable.
        self.missed_on_start.append(stamped)
        legacy = getattr(self, "missed_sessions", None)
        if legacy is not None:
            legacy.append(stamped)
        self._missed_report_done = True
        return missed

    def _arm_now(self) -> None:
        """Report what start-up missed, then arm — and NEVER let the report stop the arming.

        Both halves are here deliberately: `report_missed_on_start` reaches the calendar too
        (`elapsed_slots` -> `cal.day`), so leaving it in `on_start` would have kept a blocking socket
        there while the arming below looked fixed.

        THE GUARD IS THE POINT, AND ITS ABSENCE WAS A TRAP I INTRODUCED. `_arm_until_resolved` splits
        failures in two: `warm()` is network and retries forever, `_arm_now()` is deterministic and is
        NOT retried — its message says "This is not a network problem and will not be retried. This
        strategy CANNOT DECIDE until the cause is fixed and it is restarted." That split is right,
        and it rests on a precondition this method used to satisfy: the mixin's `_arm_now` was a bare
        `self._arm(...)` documented as "assuming the calendar can answer WITHOUT I/O".

        Adding the missed-slot report broke that precondition without moving the caller's assumption.
        One transient calendar blip during arming then landed in the non-retryable branch and
        permanently unarmed the lane, while reporting it as not a network problem — a confident,
        specific, wrong answer, which is the exact failure that comment exists to prevent.

        `momentum_rotation` carried its own report-then-arm `_arm_now` before the mixin did, and
        dispatch is virtual, so MOMENTUM-002 and BCTROT-004 have had the exposure longest.

        DIAGNOSTICS MAY FAIL; ARMING MAY NOT FAIL BECAUSE DIAGNOSTICS FAILED. A lane that cannot say
        what it missed has worse diagnostics. A lane that cannot arm never decides again until
        someone restarts it. Never caught in production — kumo-trading-platform grepped both tenants for the
        exact announcements on 2026-08-27 and found zero of either — so this is a trap, not an
        incident.
        """
        now = self.clock.utc_now()
        try:
            self.report_missed_on_start(now)
        except Exception as exc:                                       # noqa: BLE001
            # ERROR, NOT WARNING. Every comparable announcement in this class is an error —
            # "UNARMED", "STARTED AFTER", "arming FAILED" — and what is lost here is the same class
            # of fact: a decision session may have been skipped invisibly, which is the 2026-08-17
            # incident. ERROR-filtered monitoring is what the tenants are actually grepped with, so
            # a warning here is a line nobody sees.
            #
            # AND IT DOES NOT PROMISE ANYTHING ABOUT ARMING. This used to end "the schedule below is
            # unaffected", asserted BEFORE `_arm` had run — so when the report and the arming share
            # a cause (a dead injected calendar), the log carried a confident, specific, wrong claim
            # directly above the real failure. That is the exact shape the docstring above condemns.
            self.log.error(
                f"{self.id}: could not report what start-up missed ({exc!r}) — arming anyway. This "
                f"lane may have skipped a decision slot without recording it. Whether arming then "
                f"succeeded is reported separately, below.")
        self._arm(now)

    def fire_and_report(self, coro, loop, what: str):
        """Schedule `coro` on `loop` from a Nautilus event thread, and REPORT if it fails.

        Every adapter did this as a bare `asyncio.run_coroutine_threadsafe(...)` with the Future
        DISCARDED — so any exception inside the coroutine was captured in that Future and never read.
        Nothing logged, nothing retried, and the write simply did not happen.

        MEASURED CONSEQUENCE (kumo-trading-platform issue 549, 2026-08-24). Against the venue:

            Alpaca FILL events : MRVL 2, LRCX 2, INTC 4, DELL 4, AMAT 1
            journal terminal   : MRVL 2, LRCX 2, INTC 4, DELL 3, AMAT 1

        Four of five match — so these rows are per-FILL — and DELL is ONE SHORT, in the same session
        and the same code path as three that landed. A lost terminal row is not cosmetic:
        `retry.attempts_for` reads exactly these to decide a `client_order_id`, so a swallowed failure
        row means the next retry regenerates an id already used.

        BEST EFFORT STILL, AND DELIBERATELY. This runs inside Nautilus's dispatch, so it must never
        raise there — a bookkeeping failure must not take down the event handler that is telling us a
        real order filled. What changes is that failing is now VISIBLE instead of silent.
        """
        import asyncio as _asyncio

        try:
            fut = _asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:                                       # noqa: BLE001
            # CLOSE THE COROUTINE. It was created by the caller and never awaited, so without this
            # Python emits "coroutine was never awaited" — a second, noisier warning on the path that
            # is already reporting a failure, obscuring the one that matters (2026-08-26).
            try:
                coro.close()
            except Exception:                                          # noqa: BLE001
                pass
            self.log.error(f"{self.id}: could not schedule {what} — {type(exc).__name__}: {exc}")
            return None

        def _report(f) -> None:
            try:
                f.result()
            except Exception as exc:                                   # noqa: BLE001
                self.log.error(
                    f"{self.id}: {what} WAS NOT WRITTEN — {type(exc).__name__}: {exc}. The venue's "
                    f"answer for this order is missing from the journal.")

        fut.add_done_callback(_report)
        return fut

    def cancel_arming(self) -> None:
        """Stop retrying. Called from `on_stop`, so a stopped lane does not keep polling the venue.

        BOTH RETRY MECHANISMS, and they are not the same object. `_arming` is the start-up asyncio
        task; `REARM_ALERT` is a clock timer set by `rearm_after_alert` from Nautilus's timer thread,
        where there is no loop to hold a task. Cancelling only the first leaves a stopped lane
        rescheduling itself every ARM_RETRY_SECS and, on the retry that finally succeeds, arming a
        SESSION_ALERT on a strategy an operator has stopped — the stop button appearing to work,
        which is the failure `on_stop` already documents for in-flight session tasks.

        Fixed here rather than in each lane's `on_stop`: all three already call this, so a fourth
        lane inherits the cancellation by calling the same one method.
        """
        task = getattr(self, "_arming", None)
        if task is not None and not task.done():
            task.cancel()
        self._arming = None
        clock = getattr(self, "clock", None)
        if clock is not None and self.REARM_ALERT in getattr(clock, "timer_names", ()):
            clock.cancel_timer(self.REARM_ALERT)

    def _first_session_note(self) -> str:
        """What to print for "next session" at start, when arming has not happened yet.

        The start log used to read `self._armed_session.date()`, which was safe only because arming
        was synchronous and fatal. Now that a lane can start unarmed, that attribute is None and the
        log line itself would raise inside `on_start` — reintroducing the very failure this change
        removes, one line further down.
        """
        session = getattr(self, "_armed_session", None)
        return str(session.date()) if session is not None else "NOT YET ARMED (calendar pending)"

    @property
    def is_armed(self) -> bool:
        """Whether this lane has resolved WHEN it next decides. Read by `preflight`."""
        return getattr(self, "_armed_session", None) is not None

    #: Overridden per adapter.
    EXTERNAL_ID: str = ""
    LABEL: str = ""

    @property
    def external_id(self) -> str:
        return self.EXTERNAL_ID

    @property
    def label(self) -> str:
        return self.LABEL or self.EXTERNAL_ID

    @property
    def claimed_instruments(self) -> list:
        return list(getattr(self, "_claims", None) or [])

    @property
    def warmup_bars(self) -> int:
        return int(getattr(self, "_need", 0))

def session_outcome(state: str, *, decided: bool, blocked: str | None, sent: int) -> str:
    """The one-line session outcome, written by every adapter that runs a session.

    SAYS ONLY WHAT IS KNOWN AT WRITE TIME. This read `(submitted {n})`, and on ibkr-paper-retired
    2026-08-24 it said `TRADING: decided (submitted 8)` while `/orders` held ZERO — all eight
    rejected asynchronously inside the IBKR exec client. That row is what made two other IBKR
    defects invisible: both repos read it and concluded the stack was fine (kumo-trading-platform issue 512).

    The count is `submitted += int(ok)` and `ok` is `OrderResult.ok`, which `NautilusBroker`'s own
    docstring already defines honestly -- *"`submit()` reports whether the order was ACCEPTED for
    submission, never that it filled"*. So it is LOCAL acceptance: `initialized` is not `accepted`,
    and `accepted` is not `filled`. The broker was honest; this row was not.

    Blocking the session on terminal outcomes is the wrong trade -- the venue answer is async and
    waiting for it inside `run()` costs more than it buys. So the row names its BASIS instead, which
    is the distinction an operator needs and could not previously recover from a bare integer.

    ONE DERIVATION, called by both adapters. The sentence was written out twice and could drift, and
    QC345's outcome row did not exist at all until recently -- that asymmetry is what left 2026-08-21
    unexplainable from the journal.
    """
    verdict = "decided" if decided else "NO DECISION"
    reason = f" — {blocked}" if blocked else ""
    return f"{state}: {verdict}{reason} (sent {sent} to nautilus; venue outcome not yet known)"


#: Detail key carried alongside the integer, so a reader that only has the JSON can still tell local
#: acceptance from venue acceptance. The integer keeps its name for compatibility with cockpit's
#: existing readers; this names what it MEANS.
SENT_BASIS = "accepted-by-nautilus-locally, not venue-confirmed"


#: Which runners have already been reported as taking or dropping the richer terminal fields, so
#: the record is ONE LINE PER RUNNER PER PROCESS rather than one per order event. A degradation
#: notice that fires on every fill is noise, and noise is how a real one gets missed.
_SEAM_REPORTED: set[tuple[str, frozenset]] = set()


def _report_seam(record, dropped: frozenset, taken: frozenset, log=None) -> None:
    """Say ONCE that this runner takes, or does not take, the per-order fields.

    A SHIM THAT SILENTLY DROPS FIELDS IS A FALLBACK, and this repo's rule is that a fallback is a
    silent wrong answer unless it is as correct as the primary. This one is not: a runner with the
    old signature keeps receiving poorer rows indefinitely and nothing anywhere says so. The
    failure is not a crash — it is someone reading a terminal row with no `filled_qty` in six
    months, concluding the fill data was never written, and re-deriving this entire investigation.

    Raised by the coordinator on #166. Deliberately NOT a health surface and NOT something to gate
    on: one record, so the degradation is legible and whoever reads a poor row can find out it was
    poor BY DESIGN and which side needs widening.

    THREE STATES, as everywhere else here: supplied and taken, supplied and dropped, never asked.
    """
    owner = getattr(record, "__qualname__", None) or type(record).__name__
    key = (owner, dropped)
    if key in _SEAM_REPORTED:
        return
    _SEAM_REPORTED.add(key)
    if log is None:
        return
    # THE NOTICE MUST NEVER COST THE ROW. Third instance of this shape in one change: the fix for
    # an observability gap reintroducing the gap. `terminal_fields` did it by reading the cache
    # outside the guard; this would do it with a logger that raises. Both sit inside an adapter
    # body wrapped in `try/except Exception: log`, so the row is what disappears.
    try:
        if dropped:
            log(f"terminal rows to {owner} are DEGRADED BY DESIGN: it does not accept "
                f"{sorted(dropped)}, so those fields are dropped and the row cannot be attributed "
                f"to an order. Widen its record_terminal signature to take them (#164). Taken: "
                f"{sorted(taken) or 'none'}.")
        else:
            log(f"terminal rows to {owner} carry the full per-order outcome {sorted(taken)} (#164)")
    except Exception:                                                   # noqa: BLE001
        pass


def call_record_terminal(record, session, symbol, ok, detail, log=None, **extra):
    """Call a runner's `record_terminal`, passing only the keywords ITS SIGNATURE ACCEPTS (#164).

    THIS SEAM IS CROSS-REPO AND PINNED BY REVISION, which makes adding a keyword to it a breaking
    change rather than an enhancement. `record_terminal` is reached by
    `getattr(self._runner, "record_terminal", None)` — kumo-trading-platform supplies its own runner, and it
    is pinned to a revision of this repo. Passing a keyword an older implementation does not accept
    raises `TypeError` INSIDE NAUTILUS'S DISPATCH, on the handler that is telling us a real order
    filled.

    That is not hypothetical. `bctrot_rotation` records the same lesson about `read_slots`: cockpit
    passes a kwarg only to adapters whose SIGNATURE accepts it, "because this repo is pinned by
    revision and an unknown kwarg raises TypeError at build, taking every other lane down with it
    (#377)". The direction is reversed here — we are the caller — but the seam and the failure are
    the same.

    So the signature is INSPECTED, not assumed. A runner that accepts the richer fields gets them;
    one that does not gets the call it has always received, and the row is simply poorer. Degrading
    the row is the right failure: #164's keying falls back to per-symbol counting for rows with no
    order id, which over-counts and therefore only ever refuses a retry.
    """
    import inspect

    try:
        params = inspect.signature(record).parameters
    except (TypeError, ValueError):                     # a C callable or an exotic mock
        _report_seam(record, frozenset(extra), frozenset(), log)
        return record(session, symbol, ok, detail)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        # **kwargs ACCEPTS THE CALL AND MAY STILL IGNORE THE VALUES — "forwarding through **kwargs
        # is not a public seam". It cannot raise, so pass everything; whether the callee stores it
        # is the callee's contract to keep.
        _report_seam(record, frozenset(), frozenset(extra), log)
        return record(session, symbol, ok, detail, **extra)
    accepted = {k: v for k, v in extra.items() if k in params}
    _report_seam(record, frozenset(extra) - set(accepted), frozenset(accepted), log)
    return record(session, symbol, ok, detail, **accepted)


# ---------------------------------------------------------------------------------------------------
# ANYTHING ADDED BELOW RUNS INSIDE AN ADAPTER'S TERMINAL PATH. THE NOTICE MUST NOT COST THE ROW.
#
# Every adapter's `_record_terminal` body is wrapped in `try/except Exception: log`, deliberately —
# a live Nautilus event handler must not raise into the dispatch. The consequence is that ANYTHING
# here which raises does not produce an error: IT PRODUCES A MISSING TERMINAL ROW. The swallow is
# correct and it is also what makes this path silently lossy.
#
# TWICE IN ONE CHANGE (#164/#166), by two different routes:
#   * `terminal_fields` read `self.cache.order(...)` outside its guard -> a cache without `order`
#     suppressed all four of qc27's terminal rows. "recorded nothing", not "recorded wrongly".
#   * `_report_seam`'s logger raised -> the same, from the code added to make the FIRST one
#     observable.
#
# Both were fixes for an observability gap that reintroduced the observability gap. If you are
# adding a third thing here — a metric, a trace, a health ping — it must be unable to raise, and
# there must be a test that a failing version of it still lets the row through.
# ---------------------------------------------------------------------------------------------------


def terminal_fields(strategy, event) -> dict:
    """The order's identity and outcome for a terminal row (#164). NEVER RAISES.

    THE ROW IS THE POINT AND THESE FIELDS ARE A BONUS, so the enrichment must not be able to
    suppress the record. The first version of this read `self.cache.order(...)` outside the guard,
    and in an adapter whose whole `_record_terminal` body sits inside `try/except Exception: log`,
    an AttributeError there swallowed the ENTIRE ROW — turning a missing `filled_qty` into a
    missing terminal row, which is the observability gap #51 exists to close. Caught by a test
    double whose cache had no `order` method; in production it would have needed a cache miss.

    Returns `{}` when nothing can be read. `call_record_terminal` drops unknown keys anyway, and
    `attempts_for` falls back to per-symbol counting for rows with no order id — which over-counts,
    and over-counting only ever refuses a retry.
    """
    out: dict = {}
    try:
        coid = str(getattr(event, "client_order_id", "") or "") or None
        out["client_order_id"] = coid
        try:
            out["filled_qty"] = int(getattr(event, "last_qty", 0) or 0)
        except (TypeError, ValueError):
            out["filled_qty"] = 0
        order = None
        cache = getattr(strategy, "cache", None)
        getter = getattr(cache, "order", None)
        if getter is not None and getattr(event, "client_order_id", None) is not None:
            order = getter(event.client_order_id)
        if order is not None:
            try:
                out["filled_qty"] = int(getattr(order, "filled_qty", out["filled_qty"])
                                        or out["filled_qty"])
            except (TypeError, ValueError):
                pass
            status = getattr(order, "status", None)
            out["status"] = str(status).split(".")[-1].lower() if status is not None else None
            raw = getattr(order, "side", None)
            out["side"] = ("BUY" if "BUY" in str(raw).upper() else "SELL") if raw is not None else None
        else:
            out.setdefault("status", None)
            out.setdefault("side", None)
    except Exception:                                                   # noqa: BLE001
        return {}
    return out
