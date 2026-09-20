"""The `Broker` protocol, the order request/result types, and the dry-run implementation.

THIS MODULE NO LONGER TALKS TO A VENUE. It once held `AlpacaBroker`, which submitted straight to
Alpaca's REST API; that class is deleted (the operator's ruling, 2026-08-27 — see AGENTS.md and #75).
Venue access goes through Nautilus, which holds the connectors: orders via
`runtime.nautilus.broker.NautilusBroker`, instruments via `cache.instrument()`, prices via the data
client, account and positions via the `broker.account` topic and `cache.positions_open()`.

What remains here is the SEAM, not an implementation of it:
  - `Broker`, the protocol every runner is written against.
  - `OrderRequest` / `OrderResult`, including the `client_order_id` derivation — (strategy, session,
    symbol, side, slot, attempt) — which is the replay guard. Nautilus denies a duplicate id
    locally, so getting this wrong is either a denied retry or, in the other direction, a second
    position.
  - `DryRunBroker` as the DEFAULT. Submitting anywhere real requires a deliberately constructed
    Nautilus-backed broker; nothing in this module can reach money.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol


#: NO DEFAULT LANE. `OrderRequest.strategy_id` used to default to "MOMENTUM-002" — a real, live,
#: position-holding lane rather than a sentinel — and `pgrunner`'s BUY path never passed it, so every
#: entry from BCTROT-004 and QC345-003 was built claiming to be MOMENTUM-002 (found 2026-08-22).
#:
#: Agreed with kumo-trading-platform for #462, in their words: "a default that is a real position-holding lane
#: is not a default, it is a wrong answer with good manners." Once cockpit threads the owner through
#: `release_for_exit` as the CANCELLER, this field stops being a label and starts deciding whose
#: protective stop may be cancelled — and a forgotten argument would then authorise cancelling
#: MOMENTUM-002's protection while looking entirely well-formed.
#:
#: Kept as an empty sentinel rather than deleted because cockpit's side distinguishes "no lane named"
#: (its permissive proxy concession) from a named one, and that branch was UNREACHABLE from this repo
#: while the default was a real id.
DEFAULT_STRATEGY_ID = ""


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str          # BUY | SELL
    qty: int
    session: str
    strategy_id: str
    """REQUIRED. Feeds `client_order_id`, so it must match the strategy that actually places the
    order — the id is the replay guard and the only thing tying a broker order back to its owner.

    A `TypeError` at construction is the point: it is the one failure mode that cannot be mistaken
    for a correct order. A defaulted lane id produced a byte-identical `client_order_id` to a chosen
    one and was well-formed everywhere it was read, so nothing downstream could tell them apart."""
    slot: str = ""
    """Feeds `client_order_id`. A new decision slot is a NEW DECISION, not a retry of an earlier
    one — `uq_exec_one_decision_per_session` is already keyed on (strategy_id, session, slot), and
    the order id disagreed. MOMENTUM-002's open+5m attempt for FSM this morning and its open+130m
    retry hashed to the SAME id, since neither `session` nor `attempt` had changed: Nautilus denied
    the second as a local duplicate before it ever reached the venue (issue 48 follow-up,
    2026-08-19 15:40 UTC).

    That collision cost more than the denial itself: `NautilusBroker.submit()` reads the order back
    from the cache BY this id to report the outcome, and for a duplicate that lookup returns the
    EARLIER order — this morning's, genuinely rejected by Alpaca with "insufficient qty available".
    The journal then reported a fresh venue rejection that was actually seven hours old, quoting a
    different order's real answer as though it were this one. Distinct ids per slot make the cache
    lookup find the order actually being placed, closing both defects at once.

    Defaulted to "" rather than made required: every existing caller before this passed no slot, and
    "" hashes identically to omitting it entirely — attempt=0, slot="" reproduces today's id
    byte-for-byte, same as `attempt`'s own zero case."""
    limit_px: float | None = None
    """The price a LIMIT order rests at. `None` is a MARKET order, which is what every lane written
    before CRSISHORT sends, so the default preserves all four live lanes exactly.

    ALREADY ROUNDED BY THE CALLER. `crsi_short.engine.apply_gates` rounds `limit_px` to the penny
    because "a limit at 12.3449 rests at 12.34, not 12.35" and that changes which fills happen. The
    order path REFUSES a sub-penny price rather than rounding it a second time: two roundings of one
    number is two derivations, and the second is where they disagree."""

    time_in_force: str = "DAY"
    """DAY is what every lane sends today. CRSISHORT needs `AT_THE_OPEN` — IB's `OPG`, a
    limit-on-open that participates in the OPENING CROSS, which is where 108 of its 231 accepted
    trades (77.6% of the return, +9.99%/trade against +2.53%) actually fill. A silent fallback to
    DAY would turn that into an ordinary resting order and quietly halve the strategy."""

    attempt: int = 0
    """Which submission of this (session, symbol, side, slot) this is — 0 for the first.

    `nautilus/broker.py`'s own docstring: "a duplicate [client_order_id] is denied locally" — that is
    deliberate replay safety, and it was also silently defeating #51's retry: a terminally REJECTED
    order resubmitted with the same id got denied by Nautilus as a duplicate before ever reaching the
    venue a second time, and the local denial journalled identically to a genuine second rejection.
    `attempt` breaks that tie only when it needs to. `attempt=0` hashes to EXACTLY today's id — every
    first submission, which is nearly all of them, is byte-for-byte unaffected."""

    @property
    def client_order_id(self) -> str:
        raw = f"{self.strategy_id}:{self.session}:{self.symbol}:{self.side}"
        if self.slot:
            raw = f"{raw}:{self.slot}"
        if self.attempt:
            raw = f"{raw}:{self.attempt}"
        # APPENDED ONLY WHEN SET, exactly as `slot` and `attempt` are, so every id this repo has ever
        # produced is byte-for-byte unchanged. A market order and a limit order for the same
        # (session, symbol, side) are DIFFERENT orders — the limit-on-open that the auction cancelled
        # and the intraday limit that replaces it both exist on the same day — and if they hashed the
        # same, Nautilus would deny the second locally as a duplicate and `submit()` would read the
        # FIRST order back out of the cache and report its answer as this one's. That exact confusion
        # once reported a seven-hour-old rejection as this morning's.
        if self.limit_px is not None:
            raw = f"{raw}:lmt{self.limit_px:.2f}"
        if self.time_in_force != "DAY":
            raw = f"{raw}:{self.time_in_force}"
        return f"kumo-{hashlib.sha1(raw.encode()).hexdigest()[:20]}"


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    order_id: str | None
    detail: str
    request: OrderRequest


class Broker(Protocol):
    def submit(self, req: OrderRequest) -> OrderResult: ...
    async def exit(self, req: OrderRequest) -> OrderResult: ...
    def positions(self) -> dict[str, int]: ...
    def equity(self) -> float: ...

    # DECLARED 2026-08-22 because runners were already calling them. The protocol listed four methods
    # while `qc27_runner` used six, and the two it added were not checked by anything: one existed on
    # every implementation but was undeclared, and the OTHER DID NOT EXIST AT ALL.
    #
    #   strategy_positions   real, on every broker, undeclared. The entire ownership contract rests
    #                        on it -- a broker satisfying the old protocol would not have had it.
    #   last_price           real. `qc27_runner` called `broker.price()`, which exists NOWHERE; its
    #                        `except Exception: return 0` turned the AttributeError into "0 shares"
    #                        and TECHIVOL-005 sized every entry to zero.
    #
    # A protocol narrower than its callers is not a contract, it is a suggestion.
    def cancel(self, client_order_id: str) -> OrderResult: ...
    """Pull a resting order. CRSISHORT cancels an entry whose name left the universe overnight, and
    the lab does that every session — dropping it changes the trade set."""

    def strategy_positions(self) -> dict[str, int]: ...
    def last_price(self, symbol: str, **kwargs) -> float | None: ...

    # Optional. A broker that can attribute positions to this strategy may also report what they
    # actually opened at, which lets an adopted position carry a REAL entry instead of today's
    # price. Absent on brokers that cannot know it (a plain REST account view), so callers must
    # use getattr rather than assume it.
    # def position_entries(self) -> dict[str, float]: ...


@dataclass
class DryRunBroker:
    """Records what WOULD have been sent. The default, and what ARMED uses.

    AS UNFORGIVING AS PRODUCTION, DELIBERATELY. This is the tool a strategy proves itself with before
    it trades, and until 2026-08-22 it could not fail:

      * it offered only `positions()`, the ACCOUNT's book, with no `strategy_positions()`. A gateway
        reading the account for OWNERSHIP passed here and then, live, proposed exits against other
        strategies' holdings -- TECHIVOL-005 formed eight liquidation orders against BCTROT's and
        MOMENTUM's book on 2026-08-21 with every test green.
      * `equity()` returned a constant, so a strategy that cannot read equity in production reads a
        healthy 100,000 here. Both of QC345's live failures were equity reading as None.
      * there was no `last_price()`, so sizing was never exercised -- and "sizing yielded 0 shares"
        was QC345's second failure: money, a price, and still nothing to buy.

    A double more forgiving than production hides live defects. That is true of any double and worse
    of this one, because this one ships AS the dry run.
    """
    sent: list[OrderRequest] = field(default_factory=list)
    _pos: dict[str, int] = field(default_factory=dict)
    #: Positions on the same account belonging to OTHER strategies. Visible to `positions()`, never
    #: to `strategy_positions()` -- which is the only configuration in which a gateway reading the
    #: wrong one shows up at all.
    _foreign: dict[str, int] = field(default_factory=dict)
    #: `None` is a legal, testable answer: it is what production returns before an account frame
    #: arrives, and returning a plausible number instead is how the dry run certifies a broken lane.
    starting_equity: float | None = 100_000.0
    prices: dict[str, float] = field(default_factory=dict)

    def adopt_foreign(self, positions: dict[str, int]) -> None:
        """Seed another strategy's holdings on this account."""
        self._foreign.update(positions)

    def adopt_own(self, positions: dict[str, int]) -> None:
        """Seed THIS strategy's holdings. Without it a dry run can never exercise an EXIT path, so a
        gateway that fails to submit exits at all passes — found by mutation-biting the template
        integration test, where deleting the entire exit loop survived."""
        self._pos.update(positions)

    def submit(self, req: OrderRequest) -> OrderResult:
        self.sent.append(req)
        d = req.qty if req.side == "BUY" else -req.qty
        self._pos[req.symbol] = self._pos.get(req.symbol, 0) + d
        if self._pos[req.symbol] == 0:
            self._pos.pop(req.symbol)
        return OrderResult(True, req.client_order_id, "dry-run, not sent", req)

    async def exit(self, req: OrderRequest) -> OrderResult:
        return self.submit(req)

    def positions(self) -> dict[str, int]:
        """The ACCOUNT's book -- ours plus everyone else's. Not an ownership answer."""
        out = dict(self._foreign)
        for sym, qty in self._pos.items():
            out[sym] = out.get(sym, 0) + qty
        return out

    def strategy_positions(self) -> dict[str, int]:
        """What THIS strategy holds. The read every ownership decision must use."""
        return dict(self._pos)

    def last_price(self, symbol: str, **_) -> float | None:
        """None for an unpriced symbol, never a plausible default -- sizing off a fabricated price is
        how an oversized entry passes every downstream check."""
        return self.prices.get(symbol)

    def equity(self) -> float | None:
        return self.starting_equity


@dataclass
class ReadOnlyBroker:
    """Real positions and equity, refused submissions.

    SHADOW used to get a bare `DryRunBroker`, which reports an EMPTY account. Two things followed,
    both bad:

      the dry run was not one   SHADOW is meant to compute exactly what TRADING would, minus the
                                submit. Against an empty book it computed entries for names already
                                held and no exits at all, so the decisions it published -- the whole
                                basis for trusting the strategy before arming it -- were fiction.
      reconciliation misfired   the session retires position claims that the broker does not back.
                                An empty account means every claim looks unbacked, so a single
                                SHADOW session would release the lot.

    Reads pass through to the real broker; `submit` never does.

    EVERY read, via `__getattr__`, and NOT an explicit whitelist. It was a whitelist of four methods
    with no forwarding, so it silently dropped five of the wrapped broker's: `feed`,
    `instrument_ids`, `last_price`, `position_entries`, `strategy_positions`. It CONFORMED to the
    four-method protocol while being strictly less capable than the object it stands in for -- it
    satisfied the contract and broke the code.

    What that cost, at pgrunner's own `getattr(broker, x, None)` sites:

      strategy_positions -> None   `mine = None`, so ADOPTION NEVER RUNS. An unclaimed position "is
                                   read as foreign, gets leaving-them-alone, and is never exited by
                                   anything, including liquidation".
      last_price -> None           every trail evaluated against yesterday's close. "Entry 10, peak
                                   20, yesterday 18, opening at 12: the rule is breached, and reading
                                   18 holds it anyway."
      position_entries -> None     entry attribution lost, falls back to the quote.

    Every one biases toward HIDING exits, and every one is silent -- a missing capability becomes a
    quieter answer rather than an error. This class already said, three paragraphs up, that SHADOW
    "is meant to compute exactly what TRADING would, minus the submit". The docstring was right and
    the method list was wrong; the same claim of equivalence, derived twice, disagreeing.

    A read-only wrapper degrades WRITES. Anything else it withholds, the caller does without.
    """

    #: Refused by `__getattr__` as well as defined explicitly above. BELT AND BRACES, DELIBERATELY:
    #: normal attribute lookup finds the explicit `submit`/`exit`, so `__getattr__` is never reached
    #: for them today and this tuple is unreachable through ordinary use. It is kept because it stops
    #: being unreachable the moment someone deletes or renames one of those methods, at which point
    #: writes would forward silently to the real broker -- and "silently" is the whole complaint.
    #: Tested by calling `__getattr__` directly, so it is live code rather than a comment.
    _WRITES = ("submit", "exit")

    inner: Broker
    sent: list[OrderRequest] = field(default_factory=list)

    def submit(self, req: OrderRequest) -> OrderResult:
        self.sent.append(req)
        return OrderResult(False, None, "read-only broker: not submitted", req)

    async def exit(self, req: OrderRequest) -> OrderResult:
        return self.submit(req)

    def positions(self) -> dict[str, int]:
        return self.inner.positions()

    def __getattr__(self, name: str):
        """Forward any read the wrapped broker has. Writes are defined above and never reach here.

        Dunders are excluded so this cannot accidentally satisfy a protocol check or a copy/pickle
        hook on the inner object's behalf.
        """
        if name.startswith("_") or name in self._WRITES:
            raise AttributeError(name)
        return getattr(self.inner, name)

    def equity(self) -> float:
        return self.inner.equity()


#: `AlpacaBroker` WAS HERE AND IS DELETED (the operator's ruling, 2026-08-27). It submitted orders straight
#: to Alpaca's REST API, bypassing Nautilus entirely — no RiskEngine pre-trade check, nothing in the
#: cache, nothing to reconcile against, and no `release_for_exit`, so an exit could not free shares a
#: protective stop was holding.
#:
#: It was never constructed anywhere; the risk was that it existed. Every safeguard this runtime has
#: gained lives on the Nautilus path, so a caller reaching for this class would have silently opted
#: out of all of them at once — and the class's own docstring ("Construct explicitly — nothing
#: defaults to this") reads as permission rather than a warning.
#:
#: Venue access goes through Nautilus. Orders: `NautilusBroker`. Instruments: `cache.instrument()`.
#: Prices: the data client. Account and positions: the `broker.account` topic and
#: `cache.positions_open()`. See AGENTS.md and #75.
