"""One session, on Postgres. The order of operations IS the safety model, so it is fixed here.

  1. refresh sources. A failed or stale source is a HARD BLOCK — never decide on yesterday's pool.
  2. bail if this session already decided. The journal is the idempotency key.
  3. check bar coverage. Ranking whatever bars happen to exist silently narrows the universe.
  4. compute exits for EVERY held name — pool membership governs buying only.
  5. rank and decide with the same pure functions the backtest uses.
  6. journal the decision WITH its basis, before any order is sent.
  7. submit only if the lifecycle permits. SHADOW runs 1-6 and stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
import math

import numpy as np
import pandas as pd

from kumo_strategies.runtime.executor import daily_loss
from kumo_strategies.runtime.executor.broker import Broker, DryRunBroker, OrderRequest
from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.runtime.executor.pgjournal import (
    DECISION, DEFAULT_STRATEGY_ID, ERROR, ORDER, POOL, RISK, STATE, DuplicateDecision, PgJournal)
from kumo_strategies.runtime.executor.store import DEFAULT_SLOT
from kumo_strategies.runtime.executor.pgpool import PgSymbolPool
from kumo_strategies.runtime.executor.store import PositionState, utcnow
from kumo_strategies.runtime.executor.retry import (
    already_attempted, attempts_for, held_sessions_before, submitted_ok)
from kumo_strategies.runtime.executor.runner import RiskLimits, SessionResult, _FixedSource
from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig, unsupported_live_exits)
from kumo_strategies.strategies.momentum_rotation.engine import (
    apply_gates, decide, filter_entries_by_gap, needs_panel_stats, panel_stats, score_panel,
    sizing_denominator, trailing_atr, trailing_returns)
from kumo_strategies.strategies.momentum_rotation.exits import (
    ADOPTED, LIVE, TrailState, evaluate_exits, needs_atr, needs_highs)


_MAX_SUBMIT_ATTEMPTS = 3
#: How many consecutive SESSIONS entries may be held for the same refused exits before the lane
#: enters anyway. THE LEAD'S RULING ON platform issue 1058 (jkdr9txx, 2026-09-13), not a tuning knob:
#: unbounded never-enter is a silently shrinking book with every surface green; bounded by sessions
#: (not slots) the sells get two full days of slots, and on the third the lane trades its top-K
#: with a loud row naming what is still refused. A 1 would release after a single day of refused
#: sells — before cockpit's protection release has had a full day of slots to land — and reproduce
#: the budget-rejection chain the rule exists to stop; a 3 is another day of a lane that cannot
#: rotate while every surface reads TRADING. Change it with the lead, and say why in the ticket.
_MAX_HELD_SESSIONS = 2
"""How many FAILED submits of one symbol in one session before resume stops retrying it. Failures
are retryable because they are usually environmental — a missing instrument definition, an
unreachable broker — and those get fixed while the session is still open. A genuine venue rejection
will simply fail three times and stop."""


def own_ceiling(acct_qty: int, my_claim: int, other_claims: int) -> int:
    """The MOST this strategy can possibly own — derivable without attribution.

    2026-08-22: "strategies are not supposed to trade among each other." On 2026-08-21
    MOMENTUM-002 sold 28 WHD that BCTROT-004 had bought the day before. The account is genuinely flat
    (+28 / -28 = 0), so reconcile drift reports nothing and is RIGHT to; the split underneath is wrong
    and no reconciliation can ever see it.

    Sizing narrowed with `min(account, claim, attribution)`, but the attribution term is SKIPPED
    whenever `mine` is empty -- and it is legitimately empty for a position reconciled in after a
    restart, which comes back with no strategy id. The fallback exists so LIQUIDATING cannot strand a
    position we did open, and that reasoning is sound. What it never asked is whether ANYONE ELSE
    claims the symbol:

        sole claimant, no attribution   the fallback is safe; whatever is there is ours
        two claimants, no attribution   we cannot tell ours from theirs, so we may not take theirs

    `acct_qty - other_claims` is the most that can possibly be ours whatever attribution says, and it
    needs no attribution to compute. Combined with our own claim as the cap on what we opened, that
    makes the invariant TRUE rather than merely detected -- `over_claimed` reports a broken ledger
    after the fact; this stops a sell being sized into someone else's position in the first place.

    LONG-ONLY BY NAME AND BY CONTRACT (#173). This answers "how many shares may I SELL", and every
    caller feeds the result into `held_qty` and sells it. A lane holding the SHORT side reduces by
    BUYING, and handing a cover size back through this door would make the long-only runner sell a
    short position — doubling it rather than closing it. So a short claim answers ZERO here, exactly
    as it did before, and the capability lives in `reducible` where the answer carries its side.

    Floors at zero: over-claimed books are real, and a negative size would flip a sell into a buy.
    """
    got = reducible(acct_qty, my_claim, other_claims)
    return got.qty if got.side == SELL else 0


#: The two reducing directions. A lane REDUCES by selling if it is long and by buying if it is
#: short, and the two must never be confused: `pgrunner` is long-only and sells whatever quantity it
#: is given.
SELL = "SELL"
BUY = "BUY"


@dataclass(frozen=True)
class Reduction:
    """How much a lane may unwind, AND IN WHICH DIRECTION.

    THE SIDE TRAVELS WITH THE QUANTITY, deliberately, because a bare positive integer is ambiguous
    in exactly the way that costs a position. `own_ceiling` returned one for years and it was safe
    only because every lane was long; CRSISHORT makes the same integer mean "buy 50 to cover" for
    one lane and "sell 50" for another, and a caller cannot tell which by looking.
    """

    qty: int
    side: str

    def __post_init__(self) -> None:
        if self.side not in (SELL, BUY):
            raise ValueError(f"side must be {SELL!r} or {BUY!r}, got {self.side!r}")
        if self.qty < 0:
            raise ValueError(
                f"qty is a MAGNITUDE and cannot be negative ({self.qty}); the direction is `side`")


def reducible(acct_qty, my_claim, other_claims) -> Reduction:
    """The MOST this lane may unwind of its own position, in the direction that unwinds it (#173).

    ONE ARITHMETIC, TWO DIRECTIONS. `own_ceiling` asks "how much may I sell", which is the right
    question only while every lane is long. This asks "how much may I REDUCE", which both sides can
    answer:

        residue = acct_qty - other_claims        what could possibly be ours at the venue
        long  (mine > 0)   SELL  max(0, min( mine,  residue))
        short (mine < 0)   BUY   max(0, min(-mine, -residue))

    THE SYMMETRY IS NOT WHAT THE FIRST VERSION OF THIS DOCSTRING CLAIMED, and the correction
    matters because SIGNING MADE A NEW CASE REACHABLE. It said `-residue` is positive only when the
    residue is genuinely short, "the mirror of a long lane being unable to sell against a short
    one". FALSE: subtracting a neighbour's SIGNED short INCREASES a long lane's residue.

        acct=-20  mine=+10 (stale long)  other=-30  ->  residue +10  ->  SELL 10

    The book is SHORT 20 and that sells ten more of it. Under magnitude storage `other` was +30, the
    residue -50, and the answer 0 — so this is a case the sign change CREATED. The mirror is real
    too: `acct=+20, mine=-10, other=+30` covers against a long book.

    IT IS NOT CLAMPABLE. Three clamps were tried and each broke `MOM +30, CRSI -10, acct +20 ->
    SELL 30`, because the arithmetic cannot tell a phantom `mine` from a real one: both are a claim
    whose sign differs from the account's.

    ATTRIBUTION IS WHAT COVERS IT. With `mine` populated, `pgrunner.run` narrows by
    `q = min(q, attributed_qty)`, and a lane holding none of a short book has no positive attributed
    quantity, so the symbol drops. The exposure is exactly the NO-ATTRIBUTION case — a position
    reconciled in after a restart, which this package explicitly supports — combined with a stale
    claim, which its own docstrings call routine. `claims_that_contradict_the_account` is what
    surfaces it when attribution is absent.

    WHAT THE `residue` TERM IS FOR, since a simpler rule ("a lane may unwind its own claim") is the
    obvious answer and is wrong. `acct_qty - other_claims` is the only bound a lane has against ITS
    OWN STALE CLAIM. Claims once recorded on ACCEPT, before the venue answered:
    `sync_claim_to`'s docstring cites MOMENTUM-002 carrying a 260-share LAND claim on a position
    that never existed, and TECHIVOL-005 claiming 97 TOST against a book of 0. Stale is routine here
    rather than exceptional. Drop the term and a lane with a phantom +28 sells 28 shares belonging
    to the lane that actually holds them — the WHD incident, restored.

    The account is not a neighbour's opinion. It is the VENUE, and it is the only ground truth
    either lane has.

    CLAIMS MUST BE STORED SIGNED for this to work, and that migration is FREE ONLY UNTIL THE FIRST
    SHORT CLAIM IS WRITTEN: for a long lane signed and magnitude are the same number, so every
    existing row is already correct, but the moment an `abs()` records a short the table holds two
    conventions and no row says which it is. See `store.sync_claim_to`.
    """
    mine = int(my_claim or 0)
    residue = int(acct_qty or 0) - int(other_claims or 0)
    if mine > 0:
        return Reduction(max(0, min(mine, residue)), SELL)
    if mine < 0:
        return Reduction(max(0, min(-mine, -residue)), BUY)
    # NO CLAIM IS NOT A DIRECTION. Zero qty makes the side inert, and SELL is named rather than left
    # to a default so the value is never read as "this lane is long".
    return Reduction(0, SELL)


def claims_that_contradict_the_account(account: dict, claims_by_strategy: dict) -> dict:
    """Lanes whose claim SIGN disagrees with the account's side. `{(lane, symbol): (claim, net)}`.

    WHY TOTALS ARE NOT ENOUGH. `over_claimed` sums, so the composition that produces the reachable
    stale-claim case in `reducible` reports CLEAN: `{A:+10 phantom, B:-30}` against an account of
    -20 totals to exactly -20. Nothing is over-claimed and A holds none of it.

    ONE COMPARISON SEES IT. A lane claiming LONG against a SHORT account holds nothing of that
    account, and the reverse likewise — the sign is the whole test, and no magnitude reasoning is
    needed.

    A DETECTOR, NEVER A GATE, exactly as `over_claimed` is. A false positive costs a journal row; a
    gate here would cost a frozen exit, which is the failure the whole signed-claims change exists
    to remove. It is also the ONLY thing that surfaces the stale-long case when attribution is
    absent, which is the case `reducible` cannot defend against on its own.

    A FLAT ACCOUNT CONTRADICTS NOTHING. Zero is not a side, here as everywhere else in this module.
    """
    out = {}
    for lane, claims in sorted((claims_by_strategy or {}).items()):
        for sym, qty in sorted((claims or {}).items()):
            q, net = int(qty or 0), int((account or {}).get(sym, 0) or 0)
            if q == 0 or net == 0:
                continue
            if (q > 0) != (net > 0):
                out[(lane, sym)] = (q, net)
    return out


def retire_claims(owned: set, account: dict, mine: dict | None,
                  foreign: dict | None = None, theirs: dict | None = None) -> list:
    """Which of THIS strategy's claims are provably dead. Pure, so it needs no database.

    THE FREEZE THIS FIXES, found live by kumo-trading-platform at 23:14 ET on 2026-08-22:

        BCTROT-004   own_ceiling(79, my_claim=79, other=77) = 2
        MOMENTUM-002 own_ceiling(79, my_claim=77, other=79) = 0
                                                    TOTAL     2 sellable of 79 actually held

    BETA went FLAT at 15:35:22 on 2026-08-20 and reopened 25 minutes later under the other lane. It
    was therefore never ABSENT from a session-start account read, and claims retired only on
    `owned - set(account)` -- a set difference with no quantity comparison -- so MOMENTUM's claim of
    77 survived a position it no longer held any part of.

    The result is not an over-sell: `own_ceiling` prevents that and is right. It is worse in a
    quieter way -- 97% of a real position becomes unexitable by ANY lane on ANY path, LIQUIDATING
    included, because each lane's ceiling subtracts the other's stale claim. A position that cannot
    be exited and has no protective stop is the shape nobody looks for.

    WHY NOT PRO-RATA. Scaling 156 claims down to the 79 held gives MOMENTUM 39 -- 39 shares of a
    position it demonstrably holds NONE of, and selling them is reaching into BCTROT's position,
    which is the WHD incident. Pro-rata does not repair an unanchored split, it invents one.

    THE RULE IS UNCHANGED: RETIRE ON POSITIVE EVIDENCE OF ABSENCE, NEVER ON THE ABSENCE OF EVIDENCE.
    What changes is that attribution counts as evidence, where before only account membership did:

      account holds NONE of it       retired. Missing, zero, or negative -- one predicate, because
                                     these lanes are long-only and a claim on a non-positive net is
                                     dead by definition. It was key membership alone, which left
                                     present-with-zero surviving forever.
      attributed to us, POSITIVELY   kept. The strongest evidence there is. An attributed SHORT is
                                     not ownership and does not protect the claim.
      attribution silent about it,   kept. Silence is evidence of NOTHING -- the strategy id is
      and nobody else claims it      missing for reasons unrelated to ownership, and a position with
                                     no owner on record is what a reconciled-in position looks like.
      attribution silent about it,   RETIRED. The new case, and the only new case. Someone else
      and ANOTHER lane claims the    positively accounts for the whole position, so ours is dead.
      WHOLE account quantity
      account read entirely empty    retired NOTHING, as before -- indistinguishable from a failed
                                     read, and the guard must not gain a second way past it.

    WHY "ANOTHER LANE CLAIMS IT" AND NOT "ATTRIBUTION LOOKS COMPLETE" (kumo-trading-platform caught the first
    version of this before it shipped). `mine` is not all-or-nothing, it is PER POSITION: a restart
    reconciles positions in WITHOUT a strategy id, so they are absent from `mine`, and if the lane
    then opens ONE new position `mine` becomes NON-EMPTY AND INCOMPLETE. A rule guarding only the
    empty case retired all six of a real book against that state -- the freeze arriving through the
    fix for the freeze, in exactly the restart the fix has to ship through.

    Book-level completeness cannot work either. `set(mine) >= set(account) & owned` is false for BETA
    precisely BECAUSE BETA is the symbol missing from `mine`, so it never fires on the case it exists
    for.

    The discriminator is not how much attribution says. It is whether a DIFFERENT lane accounts for
    the position. That is positive evidence about THIS symbol rather than an inference from silence,
    which is the same standard the rest of this function already holds to.

    `foreign` is read to INTERPRET evidence, never to act on another lane's ledger: the result is
    always a subset of `owned`. Retiring our own claim can only REDUCE what we may sell, since
    `my_claim` caps us -- the safe direction, and the same failure mode the empty-read guard already
    chooses. Retiring someone ELSE'S would RAISE our ceiling into their position, which is the WHD
    incident exactly.

    A PARTIAL foreign claim does not account for the position, so it does not retire ours: the safe
    direction is to keep the claim and let `over_claimed` alarm on the inconsistent ledger.

    ONLY THIS STRATEGY'S CLAIMS, and the signature is the guarantee: there is no parameter through
    which another lane's claim could arrive. Retiring our own can only REDUCE what we may sell, since
    `my_claim` caps us -- the safe direction, and the same failure mode the empty-read guard already
    chooses. Retiring someone ELSE'S would RAISE our ceiling into their position, which is the WHD
    incident exactly. The other lane's stale claim clears when that lane next reconciles against its
    own attribution, never from here.
    """
    if not account and owned:
        # An empty read is indistinguishable from a failed one. Caller journals this.
        return []
    # NON-POSITIVE IS NOT SOMETHING A LONG-ONLY LANE CAN OWN, and the same predicate answers both
    # inputs. This was `set(owned) - set(account)`, which tested KEY MEMBERSHIP: two opposing OPEN
    # legs net to zero but leave the symbol PRESENT as a key, so present-with-zero was neither
    # "absent" nor "held" and the claim survived forever. That is the residue a cross-strategy sell
    # leaves behind -- kumo-trading-platform read it off the live cache on 2026-08-23:
    #
    #     WHD.XNYS  BCTROT-004 LONG 28 (+28)   MOMENTUM-002 SHORT 28 (-28)   both OPEN  -> net 0
    #     XLV.ARCX  both lanes FLAT                                          both CLOSED -> absent
    #
    # XLV retired on the old rule because `positions_open()` excludes flat. WHD did not, and
    # OVER-CLAIMED WHD would have fired forever without ever resolving. No order could form either
    # way -- `own_ceiling(0, 28, 28)` is 0 -- so the cost is a permanent false alarm, which is its own
    # failure: an alarm that never clears is one nobody reads on the morning it means something.
    absent = {s for s in owned if int(account.get(s, 0) or 0) <= 0}
    disowned: set = set()
    if mine:
        for sym in owned:
            if sym in absent:
                continue
            # `> 0`, not truthy. An attributed SHORT is not ownership: a long-only lane holding a
            # short leg holds no shares to preserve, and truthiness let an attributed -28 protect a
            # claim that cannot be true -- the same inversion as the account-side zero, on the other
            # input.
            if int(mine.get(sym, 0) or 0) > 0:
                continue
            # Attribution is silent about this symbol. On its own that means nothing. It becomes
            # evidence only when ANOTHER lane positively claims the WHOLE position.
            #
            # No separate "foreign must be positive" test, deliberately: `absent` has already removed
            # every non-positive account quantity, so inside this loop `account[sym] > 0` and
            # `foreign_qty >= account[sym]` implies it. A guard that cannot fail is not a guard, and
            # one written anyway reads as protection that is not there.
            if int((foreign or {}).get(sym, 0) or 0) >= int(account.get(sym, 0) or 0):
                disowned.add(sym)
                continue
            # THE SIBLING'S ATTRIBUTION, not only its claim (#830). Measured on an Alpaca paper instance 2026-09-09:
            # MOMENTUM-002 claimed LFST 152, BCTROT-004 claimed 2, and Nautilus attributed all 154
            # to BCTROT. The claim test above needs the sibling's CLAIM to cover the position, and a
            # sibling whose claim is stale in the same direction never does — so two wrong ledgers
            # kept each other alive while the sizer, reading attribution, refused to size an exit
            # off this claim every slot. Two readers of one fact must hold the same standard: when
            # attribution speaks (`mine` non-empty — the guard the sizer already relies on), does
            # not name the symbol, and gives the WHOLE quantity to other lanes, the claim is dead.
            # `theirs` excludes reconciled-in (EXTERNAL) positions, which carry no lane and are
            # exactly the silence the rule above refuses to read as evidence.
            if int((theirs or {}).get(sym, 0) or 0) >= int(account.get(sym, 0) or 0):
                disowned.add(sym)
    return sorted(absent | disowned)


def over_claimed(account: dict, claims_by_strategy: dict) -> dict:
    """Symbols where the strategies together claim MORE than the account holds.

    Returns {symbol: (claimed_total, account_qty)} for each breach, {} when the ledger is consistent.

    WHY A PROPERTY AND NOT A RULE. The per-strategy split is unanchored: the broker has one net
    position per symbol and no opinion about whose it is, so it cannot be repaired by reconciliation
    and has to be protected at write time (kumo-trading-platform issue 437, issue 66). Every write-time
    mechanism we have argued about -- FIFO by opening lot, pro-rata, absence-only retirement -- is a
    MECHANISM, and three days of this week were spent on mechanisms that were subtly wrong while
    nothing checked the property they exist to maintain.

    Claims are retired only when a symbol is ENTIRELY absent (`owned - set(account)`, a set difference
    with no quantity comparison), so a PARTIAL close leaves the claim at full size:

        BCTROT claims 11, MOMENTUM 19, account 30. An account-level stop sells 11 -> account 19. The
        position is not absent, so no claim retires. Claims sum to 30 against 19, and BCTROT reads
        min(19, 11) = 11 -- believing it owns shares that were sold.

    Whatever mechanism turns out to be right, a ledger claiming more than the account holds is wrong.
    Checking the property fires the same session instead of twelve hours later, and it keeps firing
    if someone later changes the mechanism and gets it wrong again.

    SIGNED, SO IT SEES A SHORT OVER-CLAIM TOO (#173). `total > net` is the breach only while every
    claim is positive. With CRSISHORT storing signed claims, a lane claiming -50 against an account
    of -20 has over-claimed by 30 shares in exactly the same way — and `-50 > -20` is False, so the
    original test read the worst short breach as a consistent ledger. The breach is that the claimed
    quantity exceeds what the account holds ON THAT SIDE.

    UNDER-CLAIMING IS NOT REPORTED HERE and that is deliberate. Unattributed shares — a position
    reconciled in after a restart, carrying no lane — make the claims sum to LESS than the account,
    and that is a legitimate, routine state rather than a breach. `reducible`'s residue term already
    keeps it safe: shares nobody claims are shares no lane may reach for.

    A DETECTOR, NEVER A GATE. This reports; it narrows nobody's sizing. A disagreement that silently
    becomes a refusal to exit is the freeze this arithmetic was changed to remove — on 2026-08-22,
    97% of a real position became unexitable by ANY lane on ANY path, LIQUIDATING included, because
    each lane's ceiling subtracted the other's stale claim. A lane must always be able to get out; a
    wrong ledger is an alarm, not a handcuff.

    PURE, so it can be tested without a database, a broker or a session.
    """
    totals: dict[str, int] = {}
    for claims in claims_by_strategy.values():
        for sym, qty in (claims or {}).items():
            totals[sym] = totals.get(sym, 0) + int(qty or 0)
    out = {}
    for sym, total in totals.items():
        net = int(account.get(sym, 0) or 0)
        # ZERO IS NOT A SIDE — the rule `reducible`'s own third branch already applies. `total >= 0`
        # put a net-zero claim set on the LONG branch, so `{A:+10, B:-10}` against an account of -5
        # reported a breach while claiming NOTHING against the same account reported clean: the same
        # real state, two answers, and under-claiming is documented here as not-a-breach.
        if total == 0:
            continue
        # The claimed side's magnitude against the account's, rather than a raw `>`: a long total
        # over-claims above `net`, a short total over-claims below it.
        breach = total > net if total > 0 else total < net
        if breach:
            out[sym] = (total, net)
    return out


@dataclass
class PgSessionRunner:
    """One session. See the module docstring: the ORDER of operations is the safety model."""

    pool: PgSymbolPool
    journal: PgJournal
    lifecycle: Lifecycle
    cfg: MomentumRotationConfig
    broker: Broker = field(default_factory=DryRunBroker)
    limits: RiskLimits = field(default_factory=RiskLimits)
    instrument_type: dict[str, str] = field(default_factory=dict)
    strategy_id: str = DEFAULT_STRATEGY_ID
    """REQUIRED IN PRACTICE — see `__post_init__`. Validated rather than made non-default because
    this field sits after fields that DO have defaults, and a non-default field cannot follow one;
    reordering would silently change what every positional caller means.

    It is not a label. It drives `PositionState.strategy_id == self.strategy_id` and
    `StrategyState.strategy_id == self.strategy_id`, so a runner that defaulted to a real lane would
    read THAT lane's book and act on it."""
    tradable_symbols: set[str] | None = None   # what the runtime can actually reach; None = no limit
    recheck_state: object = None
    """async () -> State. Re-read of the operator's lifecycle, called immediately before submitting.

    The state captured at session start can be many seconds old -- source refresh alone can stall
    that long -- and an operator who HALTs in that window watched the session submit anyway. This is
    the last read before money moves; without it the halt only takes effect next session."""

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError(
                "PgSessionRunner requires an explicit strategy_id. It used to default to the string "
                "'MOMENTUM-002', a real live lane — so a runner built without one would query "
                "MOMENTUM's positions and lifecycle and act on them, while being well-formed "
                "everywhere it was read.")

    def _finite_equity(self) -> float | None:
        """The account equity as a number, or None when it is not one.

        None rather than 0.0, because the two are different facts and the caller must not confuse
        them: 0.0 is a real account value and would read as a catastrophic loss against any baseline.
        `qc27_runner._account_equity` returns 0.0 for its own purpose — sizing, where zero correctly
        degrades to inaction — and that is the right answer there and the wrong one here.
        """
        try:
            value = float(self.broker.equity())
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    # `_session_start_equity` / `_previous_session_equity` DELETED, not left for reference. They
    # were the UNGUARDED half of #111 — a truthy `.get("equity")` that returned a NaN as a usable
    # baseline — and they had no callers once `daily_loss.baseline` replaced them. A dead copy of a
    # known-defective read is a defect waiting for its next caller.
    async def _j(self, kind, summary, **kw):
        """EVERY journal write goes through here, so the session's slot cannot be forgotten.

        It was forgotten on 31 call sites. `PgJournal.write` takes `slot` as OPTIONAL with a default,
        and only the DECISION write was passed one — so every ORDER, RISK and ERROR row landed on
        DEFAULT_SLOT. On 2026-08-19 MOMENTUM-002 ran three slots and all their order rows filed under
        `open+5m`, while the decision rows correctly said open+5m / open+130m / open+177m. The two
        halves of one session disagreed inside one table.

        The cost was a wrong CONCLUSION, not a wrong label: `select slot, kind, count(*)` reads
        "open+130m decided and formed no orders", and open+177m had in fact filled FSM 933 and
        VCTR 88 for +$866.00 and +$795.52. Slot-outcome detectors are built on this column.

        A wrapper rather than 31 fixes, because an optional defaulted argument WILL be forgotten
        again — the default is what made this invisible. `self._slot` is set at the top of `run()`;
        anything writing before that gets the runner's configured default, which is correct because
        no slot has begun.
        """
        if "slot" in kw:
            # THE OPPOSITE MISTAKE TO THE ONE THIS WRAPPER WAS BUILT FOR. It exists because `slot`
            # was forgotten at 31 of 33 call sites; what it did not anticipate was a caller
            # REMEMBERING to pass it. Python's own `TypeError: got multiple values for keyword
            # argument 'slot'` names the journal, so the reader looks there instead of at the caller
            # — and it lands on the DECISION ROW write, which stops the session before any order goes
            # out. Refuse by name instead. `test_no_j_call_passes_slot` stops it shipping.
            raise ValueError(
                f"_j() supplies `slot` itself (currently {getattr(self, '_slot', DEFAULT_SLOT)!r}); "
                f"the caller passed slot={kw['slot']!r}. Drop it — `run()` sets `self._slot` for the "
                f"whole session, and two sources for one label is what `_j` exists to prevent.")
        return await self.journal.write(kind, summary, slot=getattr(self, "_slot", DEFAULT_SLOT), **kw)

    async def _owned(self) -> dict[str, float]:
        """{symbol: claimed quantity}. The ownership boundary, and how much of it is ours."""
        async with self.journal.sessionmaker() as s:
            rows = (await s.execute(select(PositionState).where(
                PositionState.strategy_id == self.strategy_id))).scalars().all()
        return {r.symbol: float(r.qty or 0.0) for r in rows}

    async def _foreign_claims(self) -> dict[str, float]:
        """{symbol: quantity claimed by OTHER strategies on this account}.

        Needed because attribution can be legitimately absent -- a position reconciled in after a
        restart carries no strategy id -- and in that case the only thing separating our shares from
        someone else's is what they have claimed. Same table as `_owned`, inverted filter.
        """
        async with self.journal.sessionmaker() as s:
            rows = (await s.execute(select(PositionState).where(
                PositionState.strategy_id != self.strategy_id))).scalars().all()
        out: dict[str, float] = {}
        for r in rows:
            out[r.symbol] = out.get(r.symbol, 0.0) + float(r.qty or 0.0)
        return out

    async def _owned_symbols(self) -> set[str]:
        return set(await self._owned())

    async def _load_state(self, symbols) -> dict[str, TrailState]:
        if not symbols:
            return {}
        async with self.journal.sessionmaker() as s:
            rows = (await s.execute(select(PositionState).where(
                PositionState.strategy_id == self.strategy_id,
                PositionState.symbol.in_(list(symbols))))).scalars().all()
        return {r.symbol: TrailState(
            entry_px=r.entry, peak_px=r.peak,
            opened_session=r.opened_at.date() if r.opened_at else None,
            sessions_held=r.sessions_held or 0,
            sessions_since_high=r.sessions_since_high or 0,
            quality=r.quality or ADOPTED) for r in rows}

    async def _save_state(self, sym: str, st: TrailState, qty: float | None = None) -> None:
        """Persist the WHOLE trail. Writing only `peak` was survivable while give-back was the one
        rule; `max_hold_days` and `stall_days` count sessions and would silently reset on restart.

        THROUGH `store.write_claim`, in call order with every other writer of the key (#124): a
        decision-time save here and a fill's `drop_claim` in cockpit write the same row, and a
        cockpit-side lock could not have seen this one.

        FAIL-CLOSED, DELIBERATELY. `write_claim` raises `ClaimWriteStalled` after
        `CLAIM_WRITE_TIMEOUT_S` behind a slow Postgres, and this method does not catch it — the
        post-decision save (`run()`, before `_submit`) then aborts the session. A trail the database
        would not take is not a trail to size orders against; waiting silently is what this replaced."""
        from kumo_strategies.runtime.executor.store import trail_upsert_stmt, write_claim

        await write_claim(self.journal, self.strategy_id, sym, trail_upsert_stmt(
            self.strategy_id, sym, entry=st.entry_px, peak=st.peak_px, quality=st.quality,
            sessions_held=st.sessions_held, sessions_since_high=st.sessions_since_high, qty=qty))

    async def _drop_state(self, sym: str) -> None:
        from kumo_strategies.runtime.executor.store import drop_claim_stmt, write_claim

        await write_claim(self.journal, self.strategy_id, sym, drop_claim_stmt(self.strategy_id, sym))

    async def _trail_bookkeeping(self, session: str, sym: str, write) -> bool:
        """A trail write that stalls AFTER the venue has answered, or beside other symbols still to
        be processed, costs a journal row — not the session (#124 step-9 review; #377's shape).

        `ClaimWriteStalled` ONLY. Any other failure keeps the behaviour it had: a database error on
        a trail write has always propagated, and this wrapper must not widen what is absorbed. The
        one site that must NOT use this is the pre-submit save (`_persist_pending_trail`): there the
        trail is what orders are sized against, and a stall aborts before money moves.
        """
        from kumo_strategies.runtime.executor.store import ClaimWriteStalled

        try:
            await write
            return True
        except ClaimWriteStalled as exc:
            await self._j(RISK, f"trail write for {sym} STALLED — {exc}; continuing: the row is "
                                f"bookkeeping and the venue has already answered",
                          session=session, symbol=sym)
            return False


    async def sync_claim(self, symbol: str, qty, px=None) -> bool:
        """The claim follows the venue's answer, not the submit (#133, kumo-trading-platform issue 829).

        Called from an adapter's order-event handlers, including
        `RegistrationMixin.protective_close` — a protective stop that fills is this lane's position
        closing, and the claim has to follow it down.

        THIS METHOD'S ABSENCE WAS THE DEFECT. The adapters reach it through
        `getattr(runner, "sync_claim", None)`, which cannot tell "missing" from "working", so on
        every runner in this package the re-sync was silently inert while cockpit's gateways had it.
        `test_sync_claim_exists_on_the_real_runners` pins it on the shipped class for that reason.
        """
        from kumo_strategies.runtime.executor.store import sync_claim_to

        return await sync_claim_to(self.journal, self.strategy_id, symbol, qty, px)

    async def _retire_claim(self, sym: str, *, session: str) -> bool:
        """One retirement in the reconcile loop — a stall on one symbol must not stop the rest."""
        return await self._trail_bookkeeping(session, sym, self._drop_state(sym))

    async def _persist_pending_trail(self) -> None:
        """The pre-submit save. FAIL-CLOSED, DELIBERATELY: `ClaimWriteStalled` propagates and the
        session aborts before `_submit` — a trail the database would not take is not a trail to
        size orders against. The only trail write in this file that is allowed to abort a session."""
        for sym, (trail, qty) in self._pending_trail.items():
            await self._save_state(sym, trail, qty=qty)
        self._pending_trail = {}

    async def _forced_exits(self, held: dict[str, int]) -> dict[str, str]:
        """Exits the operator has demanded. These are not subject to the data gates: the operator
        knows something the ranking does not, which is the entire point of the control."""
        out = {}
        for sym in held:
            if await self.pool.must_liquidate(sym):
                out[sym] = "operator blacklist"
        return out

    def _entry_prices(self) -> dict[str, float]:
        """What each position ACTUALLY opened at, where the broker can attribute it to us.

        Seeding an adopted position from today's quote is what made the trail fiction (#197 B1); the
        real average is already on the Nautilus position and simply was not being asked for. Both
        adoption paths go through here — the orphan sweep in `_reconcile` used to seed from the quote
        on its own, and because it persists a row FIRST, `_trail_exits` then loaded that row and
        never re-adopted. Fixing one caller and not the other fixed nothing on the live path.

        Empty dict when the broker cannot attribute (a plain REST account view): callers fall back to
        the quote and record which source they used.
        """
        fn = getattr(self.broker, "position_entries", None)
        if fn is None:
            return {}
        try:
            return fn() or {}
        except Exception:                                               # noqa: BLE001
            # No logger in this module by design — the journal is the audit surface. A missing entry
            # price is not worth a row on its own; the adoption line names the source per symbol.
            return {}

    def _max_age_kw(self) -> dict:
        """The `last_price(..., max_age_ns=...)` kwarg, or `{}` when no bound is configured.

        One definition, two call sites (`_trail_exits`, `_submit`'s entry sizing) — duplicated
        inline before this, which is how the two could have silently drifted if the bound's shape
        ever changed in only one of them (code review)."""
        if self.limits.max_price_age_seconds is None:
            return {}
        return {"max_age_ns": int(self.limits.max_price_age_seconds * 1e9)}

    async def _trail_exits(self, held: dict[str, int], last: dict[str, float],
                           atr: dict[str, float] | None = None,
                           highs: dict[str, float] | None = None) -> dict[str, str]:
        """Every configured exit rule, via the SHARED evaluator. One implementation, two drivers.

        This used to be a private give-back that was the only one of four rules the live runner
        implemented; `stall_days`, `max_hold_days` and `off_peak_pct` typechecked, backtested and
        were silently ignored here (#197 B12).

        A position with no state row is ADOPTED, not seeded from today's price. The old code did the
        latter, asserting a peak that never happened: on 6 Aug that left give-back unarmed on seven
        positions and firing on a 0.06% artefact for the eighth. An adopted position is skipped by
        peak-relative rules until it makes a new observed high, at which point the peak is real.
        """
        state = await self._load_state(set(held))
        # Price the trail on the CURRENT market, not the last completed daily bar. The session fires
        # at the open, so `last` is yesterday's close -- and a position that gapped through the
        # give-back level overnight is exactly the one that needs exiting. Entry 10, peak 20,
        # yesterday 18, opening at 12: the rule is breached, and reading 18 holds it anyway.
        quote = getattr(self.broker, "last_price", None)
        real_entry = self._entry_prices()
        prices: dict[str, float] = {}
        stale, adopted = [], []
        # No try/except around the call: `NautilusBroker.last_price` is the only implementation of
        # this optional method and already accepts `max_age_ns=None`, so there is no real broker to
        # fall back FOR -- and a bare `except TypeError` would also swallow a genuine bug inside a
        # future implementation, silently doubling the price lookup and returning an unbounded price
        # with no journal row, the opposite of this module's own discipline.
        max_age_kw = self._max_age_kw()
        for sym in held:
            live_px = quote(sym, **max_age_kw) if quote is not None else None
            # The close is a last resort and it is recorded as one. Evaluating the trail against
            # yesterday cannot see the gap that breached it, so the exit reads as "not triggered"
            # when it is simply unobservable -- a distinction worth having in the journal, because
            # the two look identical in the book.
            if not live_px:
                stale.append(sym)
            px = live_px or last.get(sym)
            if px and np.isfinite(px):
                prices[sym] = float(px)
            if sym not in state and px and np.isfinite(px):
                # No record. Entry comes from the broker's own average where it can attribute the
                # position, and only falls back to the current quote when it cannot.
                #
                # The PEAK is unknown either way and is declared so rather than fabricated: quality
                # stays ADOPTED, so peak-relative rules are skipped until the position makes a new
                # observed high. A correct entry with an invented peak is still fiction — it is what
                # produced "gave back 30825% of a 0.1% peak" on 6 Aug.
                entry = real_entry.get(sym) or float(px)
                state[sym] = TrailState(entry_px=entry, peak_px=float(px), quality=ADOPTED)
                adopted.append(f"{sym}(entry={'broker' if sym in real_entry else 'quote'})")

        plan = evaluate_exits(self.cfg.exits, prices, state, atr=atr, highs=highs)
        # NOT persisted here. Two runners can reach this point in the same session -- only one wins
        # the decision row -- and the loser would still have advanced `sessions_held` and could
        # overwrite a higher peak with its own staler one. The advanced trail is handed back and
        # written once the session is known to be the one that counts.
        self._pending_trail = {s2: (st2, held.get(s2)) for s2, st2 in plan.state.items()}

        if stale:
            await self._j(
                RISK, f"exits evaluated on stale closes for {len(stale)} names — no live price",
                session=str(pd.Timestamp.utcnow().date()), detail={"stale_price": sorted(stale)})
        if adopted:
            await self._j(
                RISK, f"adopted {len(adopted)} position(s) with no trail on record — peak-relative "
                      f"exits are SKIPPED for them until they make a new high",
                session=str(pd.Timestamp.utcnow().date()), detail={"adopted": sorted(adopted)})
        return plan.exits

    async def run(self, panel: pd.DataFrame, session: str,
                  jobs: "object | None" = None, slot: str = DEFAULT_SLOT,
                  opens: dict | None = None) -> SessionResult:
        """`slot` names WHICH decision of the session this is (#29, #32).

        Defaulted, so a single-slot caller is unchanged — MOMENTUM-002 runs through here.

        It is not decoration: the unique index is `(strategy_id, session, slot)`, so a strategy
        deciding twice a day and writing both under the default would have its SECOND decision
        rejected as a duplicate of its first. The close decision would simply never happen, and the
        journal would show one tidy decision per session — the failure looking exactly like success.
        """
        # The slot every journal write in this session is filed under. See `_j`.
        self._slot = slot
        st = self.lifecycle.state

        # Refresh due sources FIRST. Without this a source can be past its cadence yet still inside
        # the 36h stale window, so a fetch that would now fail is never attempted and the session
        # decides on an old pool instead of being blocked.
        if jobs is not None:
            await jobs.refresh_due(session=session)

        # An ENABLED source that has never produced a pool row has no health record, so checking
        # pool.sources() alone would let it pass unnoticed while the UI shows it stale.
        health = {h.source: h for h in await self.pool.sources()}
        stale = [h.source for h in health.values()
                 if h.is_stale(self.limits.max_source_age_hours)]
        if jobs is not None:
            stale += [r["name"] for r in await jobs.status()
                      if r["enabled"] and r["name"] not in health]
        if stale and st is not State.LIQUIDATING:
            # LIQUIDATING is exempt on purpose. Flattening what we own needs no pool, no ranking and
            # no candidate scores -- it needs the broker's position list. Gating it on source
            # freshness meant an operator who hit liquidate during exactly the kind of upstream
            # outage that makes someone want to liquidate got nothing, with the refusal recorded as
            # a routine staleness block.
            msg = f"pool sources stale or failed: {', '.join(stale)}"
            await self._j(RISK, msg, session=session, detail={"stale": stale})
            return SessionResult(session, st.value, False, blocked=msg)
        if stale:
            await self._j(
                RISK, f"LIQUIDATING with stale sources ({', '.join(stale)}) — flattening anyway, "
                      f"it needs positions not a pool", session=session, detail={"stale": stale})

        # daily-loss halt. RiskLimits.daily_loss_frac said "halts the strategy" and nothing read
        # it, so the account could breach the limit and the runtime would keep deciding.
        #
        # THE RULE MOVED TO `daily_loss`, unchanged, because qc27_runner had no halt at all
        # (kumo-trading-platform issue 548) and a second copy is how the two halves of "a usable equity" diverged:
        # #111 guarded the equity read at decision time and left the RECORDED baseline unguarded, so
        # a NaN anchor let a measured 99% fall pass without halting. One derivation now.
        if st.may_submit_entries:
            verdict = await daily_loss.enforce(
                journal=self.journal, lifecycle=self.lifecycle,
                write=lambda k, m, detail=None: self._j(k, m, session=session, detail=detail),
                equity=self.broker.equity, frac=self.limits.daily_loss_frac, session=session)
            if verdict.action is daily_loss.Action.BLOCK:
                return SessionResult(session, st.value, False, blocked=verdict.reason)
            if verdict.action is daily_loss.Action.HALT:
                return SessionResult(session, State.HALTED.value, False,
                                     blocked="daily loss limit breached — strategy halted")

        account = self.broker.positions()
        claims = await self._owned()
        foreign = await self._foreign_claims()
        owned = set(claims)
        attributed = getattr(self.broker, "strategy_positions", None)
        mine = attributed() if attributed is not None else None
        # What the OTHER lanes are attributed, for `retire_claims` (#830). Optional on the broker
        # protocol: absent → None → the rule that reads it never fires, exactly as before.
        others = getattr(self.broker, "other_strategy_positions", None)
        theirs = others() if others is not None else None

        # ADOPT what the broker says is ours but Postgres has not recorded. A buy that filled while
        # the process was dying leaves a real position with no claim -- and an unclaimed position is
        # read as foreign, gets "leaving them alone", and is never exited by anything, INCLUDING
        # liquidation. Nautilus attributes it to this strategy under NETTING, so the evidence is
        # there; refusing to act on it is the failure, not the safeguard.
        # Only meaningful where the broker attributes per strategy: without that there is no way to
        # tell our unrecorded position from someone else's, and adopting on a guess would be the
        # foreign-position bug wearing a different hat.
        orphans = {s2: q for s2, q in (mine or {}).items() if s2 not in claims and q > 0}
        if orphans:
            quote = getattr(self.broker, "last_price", None)
            real_entry = self._entry_prices()
            adopted = []
            for s2, q in orphans.items():
                # The trail needs an entry price and we do not know what it actually filled at, so
                # this is marked ADOPTED: peak-relative rules are SKIPPED for it until it makes a new
                # observed high, rather than firing against a peak we invented. Without a price we
                # cannot seed it at all, and entry=0 would divide by zero in the trail.
                px = quote(s2) if quote is not None else None
                if not px:
                    continue
                entry = real_entry.get(s2) or float(px)
                if not await self._trail_bookkeeping(
                        session, s2, self._save_state(s2, TrailState(entry_px=entry, peak_px=float(px),
                                                                     quality=ADOPTED), qty=q)):
                    continue                      # not adopted; the remaining orphans still are
                claims[s2] = q
                adopted.append(s2)
            orphans = {s2: q for s2, q in orphans.items() if s2 in adopted}
            owned |= set(orphans)
            if orphans:
                await self._j(
                    RISK, f"adopted {len(orphans)} broker positions this strategy opened but had no "
                          f"record of — they would otherwise never be exited by anything, including "
                          f"liquidation", session=session, detail={"adopted": sorted(orphans)})
        # RECONCILE BEFORE ANYTHING READS OWNERSHIP. A row here is a CLAIM made when a buy was
        # submitted, not proof of a fill: in live, submit_order() schedules the venue call, so an
        # order can be accepted locally and rejected by Alpaca minutes later. Left alone, the two
        # failure directions are very different:
        #
        #   claim with no position   harmless for selling (held_qty intersects with the account, so
        #                            it can never produce an order) but it inflates the position cap
        #                            and hides real capacity.
        #   position with no claim   DANGEROUS. It reads as foreign, gets "leaving them alone", and
        #                            the strategy never exits a position it actually opened.
        #
        # Dropping the claim at SELL-submit time created exactly the second case whenever the venue
        # rejected the sell. So claims are now retired here, once the position is genuinely absent
        # from the broker, rather than when we hoped it would be.
        # RETIRE ON POSITIVE EVIDENCE OF ABSENCE, NEVER ON THE ABSENCE OF EVIDENCE.
        #
        # `account` comes from the Nautilus CACHE, which after a restart is filled by reconciliation --
        # and reconciliation SKIPS a symbol whose venue read fails. kumo-trading-platform observed exactly that
        # on 2026-08-22: an Alpaca 503 made one pass log
        #     "Skipping position reconciliation for AEM.XNYS: failed to query venue"   failure -> UNKNOWN
        #     "Reconciling PROT-SELL-BDX: not found at venue, marking as REJECTED"     failure -> ABSENT
        # Same outage, same pass, opposite conclusions from the identical error.
        #
        # A skipped symbol is absent from the cache, absent from `account`, and this line RETIRES its
        # claim -- writing the retirement to Postgres, so it outlives the outage and the strategy
        # believes it owns nothing it actually holds. A restart during a venue wobble is Monday's shape.
        #
        # An account read that came back COMPLETELY EMPTY while claims exist is indistinguishable from
        # a failed one, so it retires nothing and says so. A genuinely flat book with stale claims then
        # keeps them until a session sees a non-empty account -- the safe direction, and `over_claimed`
        # reports the ledger either way.
        if not account and owned:
            await self._j(RISK,
                          f"account read returned NO positions while {len(owned)} claims exist — "
                          f"retiring none: an empty read is indistinguishable from a failed one",
                          session=session, detail={"claims": sorted(owned)})
        # QUANTITY, not presence. A claim also retires when attribution SPEAKS and does not name the
        # symbol — the case where a position went flat between reconciles and reopened under another
        # lane, leaving a full-size claim that freezes the real owner out of its own shares. See
        # `retire_claims`; kumo-trading-platform measured BETA at 79 held and 2 sellable.
        stale_claims: list[str] = retire_claims(owned, account, mine, foreign, theirs)
        if stale_claims:
            # READ THE TRAIL BEFORE DESTROYING IT. `_drop_state` is a hard DELETE of the entry price,
            # the running peak, the sessions-held counters and the quality flag — and this row used
            # to record only the symbol NAMES. So a claim retired in error was unrecoverable: nothing
            # anywhere held what had been deleted.
            #
            # The rule that retires is right and is not weakened here. "Retire on positive evidence
            # of absence" fixed a real freeze (BETA, 2026-08-22: 97% of a live position unexitable by
            # any path including LIQUIDATING). What it assumes is that the account read is TRUTH.
            #
            # It currently is not. kumo-trading-platform issue 635: reconciliation invents mirror SHORT positions in
            # the Nautilus cache — three on MOMENTUM-002 today, each beside the real long it mirrors
            # — and the venue itself reports 19 positions, every quantity positive, zero shorts. A
            # mirrored long NETS TO ZERO, zero is "account holds none of it", and the claim for a
            # position we really hold is retired and its trail deleted.
            #
            # This does not fix that; it makes the loss reversible. An irreversible destruction of
            # durable state that logs only a name is the same shape as everything else this week.
            # BEST EFFORT, and it must be. This read exists to make a retirement recoverable; if
            # it could RAISE it would take down the session it is only meant to observe, and a
            # recovery mechanism that can break the thing it records is worse than none. A failed
            # read costs the reconstruction detail, not the session — and the symbol names are still
            # in the row either way.
            try:
                doomed = await self._load_state(stale_claims)
            except Exception:                                          # noqa: BLE001
                doomed = {}
            for sym in stale_claims:
                await self._retire_claim(sym, session=session)
            owned -= set(stale_claims)
            await self._j(
                POOL, f"released {len(stale_claims)} position claims this strategy does not hold",
                session=session,
                detail={"released": stale_claims,
                        "absent_from_account": sorted(set(stale_claims) - set(account)),
                        "not_attributed_to_us": sorted(set(stale_claims) & set(account)),
                        # THE STATE ITSELF, so a wrong retirement can be reconstructed from the
                        # record rather than mourned. Keyed by symbol, values exactly as stored.
                        "retired_state": {
                            sym: {"entry": getattr(st, "entry_px", None),
                                  "peak": getattr(st, "peak_px", None),
                                  "opened_session": str(getattr(st, "opened_session", "") or ""),
                                  "sessions_held": getattr(st, "sessions_held", None),
                                  "sessions_since_high": getattr(st, "sessions_since_high", None),
                                  "quality": getattr(st, "quality", None)}
                            for sym, st in sorted(doomed.items())},
                        # And what the account said, because that is the evidence the retirement was
                        # based on — the thing to check first when a retirement turns out wrong.
                        "account_qty": {sym: float(account.get(sym, 0) or 0)
                                        for sym in stale_claims}})
        # QUANTITY, never membership. Being in `owned` says this strategy opened SOMETHING in that
        # symbol, not that it opened all of it, and taking the account figure meant 10 ours beside a
        # manual 90 exiting as SELL 100.
        #
        # The size we may sell is capped by OUR OWN CLAIM first -- that is the record of what this
        # strategy opened, and it is the cap that fixes the original bug regardless of what the
        # broker can tell us. Where the broker can also attribute per strategy (Nautilus can, under
        # NETTING), that narrows it further.
        #
        # Falling back to the ACCOUNT quantity when attribution is missing was the bug, and an
        # earlier version of this fix reintroduced it: a stale claim with no attribution would have
        # sold the whole account position. Falling back to zero instead is also wrong -- it would
        # mean a broker that cannot attribute can never exit anything, which is every broker except
        # the live Nautilus one. The claim is the answer to both.
        held_qty = {}
        for s2, acct_qty in account.items():
            if s2 not in owned:
                continue
            # A STRATEGY MAY NEVER SELL ANOTHER STRATEGY'S SHARES (2026-08-22). `own_ceiling`
            # caps us at `acct_qty - other_claims` as well as at our own claim -- derivable WITHOUT
            # attribution, which is exactly the case that let MOMENTUM-002 sell 28 WHD that
            # BCTROT-004 had bought the day before. Sole claimant and no attribution is still safe
            # and still sells; two claimants and no attribution now yields zero rather than theirs.
            q = own_ceiling(acct_qty, claims.get(s2) or 0, foreign.get(s2, 0.0))
            # Attribution NARROWS, but only where it actually reports the symbol. Absence is not
            # evidence of zero: with the durable cache off, a position reconciled in from the broker
            # after a restart comes back without a strategy id, so `mine` is empty while the position
            # is real and ours. Treating that as "we own none of it" meant give-back and even
            # LIQUIDATING would refuse to sell a position we had opened -- the mirror of the bug this
            # narrowing was added for, and just as unexitable.
            #
            # Safety does not depend on attribution anyway: the CLAIM already caps us at what we
            # opened, which is what stops us reaching into a manual position.
            if mine:
                attributed_qty = mine.get(s2)
                if attributed_qty:
                    q = min(q, attributed_qty)
                elif q > 0:
                    # A POPULATED ATTRIBUTION THAT OMITS THIS SYMBOL IS EVIDENCE OF ZERO
                    # (kumo-trading-platform issue 692). The paragraph above is right that ABSENCE is not evidence
                    # of zero — but it could not tell the two apart, because `if attributed_qty:`
                    # treats "not in the dict" and "attributed 0" identically to "no attribution at
                    # all". Zero read as absent, one more time.
                    #
                    # MEASURED ON AN ALPACA PAPER INSTANCE 2026-08-28 15:40 ET. BDX:
                    #
                    #     claims ledger   MOMENTUM 45 / BCTROT 10
                    #     Nautilus cache  MOMENTUM 55 / BCTROT FLAT
                    #
                    # Totals agree, the SPLIT disagrees. `own_ceiling` cannot see it — it computes
                    # `min(claim, acct - other_claims)` = `min(10, 55-45)` = 10, self-consistent
                    # within a ledger that is itself wrong. So BCTROT sized an exit for 10 shares it
                    # does not hold, and the sell was a cross-strategy reach VIA the claims ledger.
                    # MOMENTUM's trailing stop blocked it, correctly, and that block is the only
                    # reason this was visible at all.
                    #
                    # `mine` being NON-EMPTY is what makes this evidence: the venue attributed this
                    # strategy's book and did not put the symbol in it. When `mine` is empty the
                    # attribution is simply unavailable — a position reconciled in after a restart
                    # carries no strategy id — and the claim still stands, unchanged.
                    #
                    # A STRATEGY MAY NEVER SELL ANOTHER STRATEGY'S SHARES (2026-08-22). When
                    # the two records disagree about whose they are, the answer is not to pick the
                    # one that lets us trade.
                    await self._j(
                        RISK, f"{s2}: the claims ledger says this strategy holds {int(q)} but the "
                              f"venue attributes it NONE — refusing to size an exit off a claim the "
                              f"cache contradicts. The split is wrong even though the totals agree; "
                              f"selling here would reach into another strategy's position.",
                        session=session,
                        detail={"symbol": s2, "claimed": int(q), "attributed": 0,
                                "account_total": float(acct_qty)})
                    q = 0
            if q > 0:
                held_qty[s2] = int(q)
        # HELD IS NOT SELLABLE (#224, platform issue 1058). `held_qty` is what this strategy MAY SELL — a
        # claim the venue attributes to nobody is zeroed above, correctly, because a strategy may
        # never sell another strategy's shares. But zeroing it also dropped the symbol from the
        # set handed to `decide`, so the engine saw a FREE SLOT where a real position sits: BCTROT-004
        # journaled `hold 9 · enter 4 · exit 0` at n_hold 8, the venue answered "holds +9 AEM; a BUY
        # 9 adds to it", and every such entry was then budget-rejected. Refusing to SIZE the exit
        # is right; refusing to COUNT the position is the defect.
        #
        # So the ranking, the hold tuple and the sizing denominators read `held` — every claimed
        # symbol with a long position on the account, sellable or not — and only the exit loop
        # reads `held_qty`. A stale claim (owned, not on the account) is retired above and is in
        # neither; a short is in neither (long-only lane, reported separately below).
        held = set(held_qty) | {s2 for s2, acct_qty in account.items()
                                if s2 in owned and (acct_qty or 0) > 0}
        # CARRIED ON THE INSTANCE, not threaded through `_resume`/`_submit`'s signatures: both are
        # seams that tests bind narrowly (a double overriding `_resume(session, st, held_qty, panel,
        # slot=None)` would raise on a new kwarg), and `_suppressed` already travels this way. Reset
        # per run; `_book_slots()` falls back to `held_qty` for a caller that never set it.
        self._held_slots: set = held
        self._unfunded_exits: tuple = ()
        # A SHORT IS NOT "WE OWN NONE OF IT" (#88). `own_ceiling` floors at zero, correctly — a
        # negative size would flip a sell into a buy. But that makes a NEGATIVE account quantity
        # indistinguishable from a symbol we simply do not hold, so the symbol never enters
        # `held_qty` and give-back, stall, forced exits and LIQUIDATING all skip it in silence.
        #
        # Nothing in this package opens a short, so one that exists came from reconciliation, a
        # manual order, or a phantom — and it is unmanageable from here either way. What must not
        # happen is that it is unmanageable AND unmentioned: kumo-trading-platform issue 197 B8 was a permanent -93
        # HSBC short that nothing looked for.
        shorts = {k: v for k, v in account.items() if (v or 0) < 0}
        if shorts:
            await self._j(
                RISK, f"{len(shorts)} SHORT positions on this account cannot be exited by this "
                      f"strategy, which is long-only: {', '.join(f'{k} {v:g}' for k, v in sorted(shorts.items()))}. "
                      f"They are skipped by every exit path including LIQUIDATING and must be "
                      f"resolved outside the strategy.",
                session=session, detail={"shorts": {k: float(v) for k, v in sorted(shorts.items())}})
        stale_claims = sorted(owned - set(account))
        foreign = sorted(set(account) - owned)
        if foreign:
            await self._j(
                RISK, f"{len(foreign)} account positions are not this strategy's — leaving them "
                      f"alone", session=session, detail={"foreign": foreign[:50]})

        # URGENT EXITS DO NOT WAIT FOR TOMORROW. An operator who blacklists a name after bad news,
        # or flips to LIQUIDATING, is acting on something the ranking cannot see -- that is the whole
        # point of the control. Both used to be honoured only by the next scheduled session, and if
        # this one had already decided, the resume path replayed the OLD decision instead: the UI
        # said liquidating while the account stayed long.
        #
        # This path submits exits only. It writes no decision row, so it cannot collide with the
        # session's idempotency key, and it leaves the normal rotation to run as usual on a day where
        # nothing urgent is pending.
        already = await self.journal.decided_this_session(session, slot=slot)
        urgent = await self._forced_exits(held_qty)
        if st is State.LIQUIDATING:
            urgent = {s2: "liquidating" for s2 in held_qty}
        if urgent and (already or st is State.LIQUIDATING):
            if not st.may_submit_exits:
                return SessionResult(session, st.value, False,
                                     blocked=f"{len(urgent)} urgent exits pending but {st.value} "
                                             f"cannot submit")
            await self._j(
                RISK, f"urgent exit outside the daily decision: {', '.join(sorted(urgent))}",
                session=session, detail={"exits": urgent, "state": st.value})
            n = await self._submit(session, (), tuple(sorted(urgent)), held_qty, {}, st,
                                   decision_slot=slot)
            return SessionResult(session, st.value, True, (), tuple(sorted(urgent)), (), n,
                                 detail={"urgent": urgent})

        if already:
            # The decision row is the idempotency key, and it is written BEFORE any order goes out
            # -- deliberately, so a crash cannot leave orders with no audit record. The cost is the
            # mirror case: crash or stop between journalling and submitting, and the next attempt
            # sees "already decided" and does nothing, forever. A give-back exit computed at 09:35
            # and never sent would simply never be sent.
            #
            # So a decided session is RESUMED, not re-decided. The recorded decision is replayed and
            # only the symbols with no order row are submitted. Re-deciding would be wrong: the book
            # and the prices have moved, and the journal already says what today's answer was.
            return await self._resume(session, st, held_qty, panel, slot=slot)
        if not st.decides:
            return SessionResult(session, st.value, False,
                                 blocked=f"state {st.value} does not decide")

        # OWNERSHIP: the broker reports every position in the account, including manual trades and
        # any other strategy's. Treating them all as ours means a name we never bought is absent
        # from our ranking, reads as "fell out of the book", and gets SOLD. exec_position_state is
        # the record of what this strategy actually opened, so it is the authority on ownership.
        #
        # This runs BEFORE the already-decided bail deliberately. Retiring a claim the broker does
        # not back is housekeeping, not part of the decision, and a session that has already decided
        # is exactly when a rejected order from the first attempt needs cleaning up.
        px = panel.sort_values(["ticker", "date"]).copy()
        px["date"] = pd.to_datetime(px["date"])
        scored = score_panel(apply_gates(px, self.cfg, self.instrument_type), self.cfg)
        day = scored[scored["date"] == scored["date"].max()]
        last = day.set_index("ticker")["close"].to_dict()

        buyable = await self.pool.symbols()
        # The pool refreshes while the node runs; instrument subscriptions were fixed at startup.
        # A name added to the pool today has no bars, no instrument definition and no subscription,
        # so it can be ranked and chosen and then fail at submission -- consuming a slot with an
        # order that can never fill. Worse, it counts toward bar coverage, so the coverage floor
        # reads healthy while the universe silently narrows.
        if self.tradable_symbols is not None:
            unreachable = sorted(buyable - self.tradable_symbols)
            if unreachable:
                buyable = buyable & self.tradable_symbols
                await self._j(
                    RISK, f"{len(unreachable)} pool names are not subscribed in this node — "
                          f"excluded from ranking until it restarts",
                    session=session, detail={"unreachable": unreachable[:50]})
        have = set(day.ticker)
        missing = sorted(buyable - have)
        cover = 1.0 - len(missing) / max(len(buyable), 1)
        if missing:
            await self._j(
                RISK, f"bars missing for {len(missing)}/{len(buyable)} pool names "
                      f"({100*cover:.0f}% coverage)", session=session,
                detail={"missing": missing[:50], "coverage": round(cover, 3)})
        # A forced exit -- operator blacklist or LIQUIDATING -- needs the broker's positions and a
        # price, not a ranked universe. Blocking it on bar coverage meant a symbol blacklisted after
        # bad news stayed held because an unrelated source was stale, with the refusal recorded as a
        # routine coverage block.
        forced_now = bool(await self._forced_exits(held_qty)) or st is State.LIQUIDATING
        if cover < self.limits.min_bar_coverage and not forced_now:
            msg = (f"bar coverage {100*cover:.0f}% below the "
                   f"{100*self.limits.min_bar_coverage:.0f}% floor — refusing to rank a silently "
                   f"narrowed universe")
            await self._j(RISK, msg, session=session,
                                     detail={"missing": missing[:50], "coverage": round(cover, 3)})
            return SessionResult(session, st.value, False, blocked=msg)

        self._pending_trail: dict[str, tuple[TrailState, float | None]] = {}
        # ATR for `give_back_min_peak_atr`. Computed from the same trailing panel the ranking
        # uses, via the shared helper — a second ATR here is how live and backtest disagree.
        atr = trailing_atr(px) if needs_atr(self.cfg.exits) else None
        highs = (day.set_index("ticker")["high"].to_dict()
                 if needs_highs(self.cfg.exits) and "high" in day else None)
        forced = await self._trail_exits(held_qty, last, atr=atr, highs=highs)
        forced.update(await self._forced_exits(held_qty))

        cand = day[day.ticker.isin(buyable)]
        # `cand` being non-empty is not the same as anything being RANKABLE. If every row scores NaN
        # -- too little history, a gate blanking the column, a bad panel -- then `decide()` ranks an
        # empty frame, finds nothing to keep, and exits EVERY held position. A data fault would
        # liquidate the book and read in the journal as a normal rotation.
        rankable = int(cand["score"].notna().sum()) if "score" in cand else 0
        floor = self.limits.min_rankable_frac * len(cand) if len(cand) else 0
        thin = rankable < floor
        if (cand.empty or thin) and st is not State.LIQUIDATING:
            msg = ("no eligible candidates after gates" if cand.empty else
                   f"only {rankable}/{len(cand)} candidates carry a score, below the "
                   f"{100*self.limits.min_rankable_frac:.0f}% floor — refusing to treat a collapsed "
                   f"ranking as a decision to exit everything")
            await self._j(RISK, msg, session=session,
                                     detail={"pool": len(buyable), "candidates": len(cand),
                                             "rankable": rankable})
            return SessionResult(session, st.value, False, blocked=msg)

        if st is State.LIQUIDATING:
            # flatten everything this strategy owns; do not rank, do not enter
            exits, enters, hold = tuple(sorted(held_qty)), (), ()
            weights: dict = {}
            basis_extra = {"liquidating": True}
        else:
            # The trailing correlation/volatility estimates `max_correlation` and
            # `inverse_vol_sizing` need. WITHOUT THESE BOTH FLAGS ARE INERT HERE: `_diversified`
            # returns the plain ranking on `corr is None` and `_weights` returns equal weight on
            # `vol is None`, neither logs, and this runner passed neither — so a live config could
            # set either, typecheck, deploy, and change nothing at all (#26). Gated on
            # `needs_panel_stats` so a config asking for neither computes nothing extra and behaves
            # exactly as before.
            #
            # No `as_of` slicing: live only ever holds trailing bars, so the panel IS the history up
            # to now. That is what makes this point-in-time by construction rather than by care, and
            # it is why the same helper is safe on both sides.
            corr = vol = None
            if needs_panel_stats(self.cfg):
                corr, vol = panel_stats(trailing_returns(px), self.cfg.portfolio.corr_window,
                                        sorted(held | set(cand["ticker"])))
            # `held`, not `held_qty`: the engine must see every slot the lane occupies, including
            # the one it cannot sell (#224). An exit it then decides for an unsellable name is
            # filtered by the exit loop's `held_qty.get(sym, 0) <= 0`, and said out loud above.
            d = decide(cand, _FixedSource(buyable), self.cfg, held, corr=corr, vol=vol)
            # A held name with NO BAR TODAY is absent from `cand`, so `decide()` reads it as having
            # fallen out of the ranking and exits it. That is a data outage producing a sell: bar
            # coverage of exactly the floor passes, and the eight names missing bars get liquidated
            # as if the trader had dropped them. Selling because the pool dropped a name is
            # deliberate and valuable; selling because a feed hiccuped is not, and the two are
            # indistinguishable downstream.
            #
            # Forced exits are unaffected -- an operator blacklist and the give-back rule both still
            # fire, and give-back prices off the live quote rather than the missing bar.
            no_bar = {s2 for s2 in d.exit if s2 not in have and s2 not in forced}
            if no_bar:
                await self._j(
                    RISK, f"held {len(no_bar)} names with no bar today — holding rather than "
                          f"treating a data gap as an exit signal",
                    session=session, detail={"no_bar": sorted(no_bar)})
            exits = tuple(sorted((set(d.exit) - no_bar) | set(forced)))
            enters = tuple(x for x in d.enter if x not in exits)
            # THE GAP DEAD BAND, applied where live actually knows the gap (#199 follow-up).
            #
            # This is the LAST LOOK before the order goes out, not a ranking term: `decide()` has
            # already chosen the names and this only declines. Same predicate the backtest calls, so
            # the measured rule and the shipped rule cannot drift — the failure that left three exit
            # rules live-inert (#197 B12) was two copies, not a missing one.
            #
            # TODAY'S OPEN comes from the CALLER, never from the panel. The panel handed to this
            # runner is trimmed to sessions strictly BEFORE the one being decided — deliberately, so
            # today's prices cannot vote on today's ranking — so `day` is yesterday's row and
            # `day["open"]` is yesterday's open. An earlier version read exactly that and computed
            # the PREVIOUS session's gap, at every slot, journalled as healthy (review, 2026-09-05).
            #
            # `last` is the newest close in that trimmed panel, which IS yesterday's close, so it is
            # the correct prior close and no lookup is needed. Together: today's open over
            # yesterday's close, the overnight gap, identical at all three slots.
            enters, gap_declined = filter_entries_by_gap(enters, opens or {}, last, self.cfg)
            # A rule that is armed but has no open for a name ADMITS it (fail-open, like every other
            # missing-data path here). That is safe but silent, and at 09:35 the first republish of
            # today's daily bar may not have arrived — so the rule would be inert at precisely the
            # slot it was added for. Journalling it is what makes "inert" distinguishable from
            # "nothing to decline", which are otherwise the same empty result.
            if self.cfg.execution.min_abs_gap_pct is not None:
                blind = sorted(s2 for s2 in enters if not (opens or {}).get(s2))
                if blind:
                    await self._j(
                        RISK,
                        f"gap filter has no open for {len(blind)} of {len(enters)} entries at "
                        f"{self._slot} — admitted unfiltered",
                        session=session, detail={"no_open": blind[:50]})
            if gap_declined:
                # RISK, NOT DECISION (#831). `uq_exec_one_decision_per_session` is partial on
                # `kind='decision'`, so a decline row written under that kind TOOK the slot's one
                # decision and the real decision row 60 ms later raised DuplicateDecision — reported
                # as "already decided (concurrent run)". BCTROT-004 decided nothing for five
                # consecutive slots (2026-09-04 19:40 → 2026-09-09 13:35), entries AND exits, while
                # the filter's arithmetic was right to five decimals. A filter outcome is a RISK
                # line, like the no-open row above it; the decision row is the one below.
                await self._j(
                    RISK,
                    f"gap filter declined {len(gap_declined)} entries at {self._slot}",
                    session=session,
                    detail={"declined": gap_declined,
                            "min_abs_gap_pct": self.cfg.execution.min_abs_gap_pct})
            hold = tuple(sorted((held - set(exits)) | set(enters)))
            weights = d.weights
            basis_extra = {}

        ranked = cand.dropna(subset=["score"]).nlargest(15, "score")
        names = list(ranked.ticker)
        basis = {
            "state": st.value, "pool_size": len(buyable), "bar_coverage": round(cover, 3),
            "ranking": [{"symbol": r.ticker, "score": round(float(r.score), 4)}
                        for r in ranked.itertuples()],
            "blocked_by_gate": int(day["blocked"].sum()) if "blocked" in day else 0,
            "reasons": ({s: f"entered: rank {names.index(s)+1}" for s in enters if s in names}
                        | {s: forced.get(s, "left the ranking") for s in exits}),
            "target_book": list(hold),
            # ANCHORS THE DAILY-LOSS CHECK for this session, via `anchor()` so a non-finite value
            # is OMITTED rather than recorded. This line used to write the raw broker value, and a
            # `Decimal("NaN")` from it became a truthy baseline that made every comparison False —
            # a measured 99% fall that did not halt (#111).
            **daily_loss.anchor(self.broker.equity),
            **basis_extra,
        }
        # The decision row is the idempotency key. If it does not persist, a retry would decide
        # again — so a failed write must stop the session BEFORE any order goes out.
        try:
            decision_id = await self._j(
                DECISION, f"{st.value}: hold {len(hold)} · enter {len(enters)} · exit {len(exits)}",
                session=session, detail=basis, correlation=f"{session}/{slot}:decision")
        except DuplicateDecision:
            # lost the race to a concurrent run — the database, not the check above, is authority
            return SessionResult(session, st.value, False,
                                 blocked=f"already decided {session}/{slot} (concurrent run)")
        if decision_id is None:
            msg = "decision could not be journalled — refusing to submit without an audit record"
            await self._j(ERROR, msg, session=session)
            return SessionResult(session, st.value, False, blocked=msg, detail=basis)

        # This session is now the one that counts, so the advanced trail becomes durable. Deferring
        # to here also means a session blocked by a data gate leaves the trail untouched: it decided
        # nothing and submitted nothing, so counting it against `max_hold_days` would age positions
        # out of the book on days we refused to act.
        await self._persist_pending_trail()

        if not st.may_submit_entries and not st.may_submit_exits:
            return SessionResult(session, st.value, True, enters, exits, hold, 0,
                                 blocked="SHADOW — decisions published, nothing submitted",
                                 detail=basis)
        if (blocked := await self._state_changed_under_us(session, st)) is not None:
            return SessionResult(session, blocked, True, enters, exits, hold, 0,
                                 blocked="operator changed the state during the session — "
                                         "nothing submitted", detail=basis)
        n = await self._submit(session, enters, exits, held_qty, last, st,
                               weights=weights, decision_slot=slot)
        # Report what was SUBMITTED, not what was intended. `entered` feeds the Nautilus layer's
        # journal line and anything reading it later; returning the planned entries after the guard
        # refused them would have the audit trail claim trades that never happened -- the same class
        # of dishonesty this whole change exists to remove.
        if suppressed := getattr(self, "_suppressed", ()):
            rules = ", ".join(unsupported_live_exits(self.cfg.exits))
            return SessionResult(
                session, st.value, True, (), exits, hold, n,
                blocked=f"entries suppressed — {rules} configured but not implemented live; "
                        f"exits ran normally",
                detail={**basis, "suppressed_entries": list(suppressed),
                        "unsupported_exits": unsupported_live_exits(self.cfg.exits)})
        return SessionResult(session, st.value, True, enters, exits, hold, n, detail=basis)

    async def record_terminal(self, session: str, symbol: str, ok: bool, detail: str,
                              *, client_order_id: str | None = None,
                              filled_qty: int = 0, status: str | None = None,
                              side: str | None = None) -> None:
        """The venue's actual answer, called from the adapter's order-event handlers (#51).

        `_submit`'s `phase="result"` row is submit-time only -- Nautilus accepting the order, not the
        venue filling it. This is the row that arrives later, when it is actually known: a real fill,
        or a rejection/denial/cancellation that submit-time acceptance did not and could not predict.
        `_resume`'s `attempted` computation reads `phase="terminal"` rows precisely so a symbol whose
        submit-time row said ok=true is not silently treated as done when the venue says otherwise.

        Best-effort by design: called from a live event handler outside this session's own control
        flow, so a missing terminal row degrades to the pre-#51 behaviour (bounded retry on submit-time
        results only) rather than blocking anything. It closes the loop; it is not load-bearing for
        safety the way the intent-before-submit ordering is.
        """
        await self._j(
            ORDER if ok else ERROR,
            f"terminal: {symbol} {'filled' if ok else 'rejected'} — {detail}",
            session=session, symbol=symbol,
            # CORRELATION = THE ORDER (#164). Terminal rows carried only {ok, phase} and a null
            # correlation, so they were not attributable to an order — which is why `attempts_for`
            # had to key by symbol and why a late reject from an OLDER order on the same symbol
            # re-armed the id of a filled one. The keying fix is unreadable without this written
            # first: until a terminal row can name its order, "which order failed" is a guess from
            # symbol and timestamp.
            correlation=client_order_id,
            detail={"phase": "terminal", "ok": ok,
                    "client_order_id": client_order_id,
                    # SHARES BOOKED, not just "it failed". `ok=False` alone cannot distinguish
                    # "refused, nothing done" from "refused, 17 booked", and a resume that treats
                    # the second as a retryable attempt re-sends the WHOLE quantity.
                    "filled_qty": int(filled_qty or 0),
                    "status": status,
                    # SIDE, so a SELL's rejection cannot increment the BUY attempt for the same
                    # name in the same session.
                    "side": side})

    async def _state_changed_under_us(self, session: str, st: State) -> str | None:
        """The last look before money moves. Returns the new state if it is no longer `st`."""
        if self.recheck_state is None:
            return None
        try:
            now = await self.recheck_state()
        except Exception as e:                                    # noqa: BLE001
            # Cannot confirm the operator's intent, so do not act on a stale copy of it.
            await self._j(RISK, f"could not re-read lifecycle before submitting: {e} — "
                                           f"not submitting", session=session)
            return st.value
        if now is st:
            return None
        await self._j(
            RISK, f"lifecycle moved {st.value} -> {now.value} while the session ran — the decision "
                  f"stands but nothing is submitted under a state the operator has left",
            session=session, detail={"was": st.value, "now": now.value})
        return now.value

    async def _resume(self, session: str, st: State, held_qty, panel,
                      *, slot: str | None = None) -> SessionResult:
        """Finish a session that decided but may not have submitted everything.

        Prices are re-derived from the panel here rather than reusing the decision's -- the recorded
        decision says WHAT to do, current prices say what size it is. Sizing a resumed entry off a
        price from the failed attempt would be sizing off a number already known to be stale.

        `slot` threaded through explicitly (#54): `explain` used to take the latest decision row for
        the SESSION alone, so a multi-slot strategy resuming its close-20m decision could be handed
        its open+150m decision instead -- the wrong book, silently. `run()` already knows which slot
        it is resuming; not passing it down was the bug, not anything `explain` itself got wrong.
        """
        px = panel.sort_values(["ticker", "date"]) if len(panel) else panel
        last = ({} if not len(px) else
                px[px["date"] == px["date"].max()].set_index("ticker")["close"].to_dict())
        decision = await self.journal.explain(session, slot=slot)
        if not decision:
            return SessionResult(session, st.value, False, blocked="already decided this session")
        basis = decision.get("detail") or {}
        reasons = basis.get("reasons") or {}
        book = set(basis.get("target_book") or ())
        exits = tuple(sorted(s2 for s2 in reasons if s2 not in book))
        # `held` (slots), not `held_qty` (sellable): a claimed name the lane cannot sell is not an
        # entry to re-send (#224).
        occupied = self._book_slots(held_qty)
        enters = tuple(sorted(s2 for s2 in reasons if s2 in book and s2 not in occupied))

        # An order is "attempted" only once a RESULT row exists. The intent row is written BEFORE
        # the broker call -- on purpose, so a crash cannot leave an order with no audit trail -- so
        # counting it here would make the crash-between-intent-and-submit case unrecoverable: the
        # exit would look attempted forever and never be sent. Result rows come from ERROR too, so
        # a genuine rejection is also not retried blindly within the same session.
        # "Attempted" means SUCCESSFULLY handed to the broker, or tried too many times. Counting
        # any result row -- including failures -- meant a session that failed for an environmental
        # reason could never recover: the first live run refused all 8 entries because instrument
        # definitions were missing, and after that was fixed resume still reported "already decided"
        # because those failures looked like attempts.
        #
        # Bounded, so a genuine venue rejection cannot loop: a symbol is retried while it has fewer
        # than _MAX_SUBMIT_ATTEMPTS failed results and no successful one.
        #
        # `phase="result"` is submit-time only -- `ok=True` there means Nautilus accepted the order
        # into its local lifecycle (INITIALIZED), not that the venue filled it (broker.py's own
        # comment says so). A `phase="terminal"` row (`record_terminal`, from the adapter's
        # order-event handlers) is the venue's real answer, arriving later. This is the #51 fix:
        # VCTR's SELL read ok=true at submit, was rejected by Alpaca minutes later, and nothing ever
        # retried it, because submit-time acceptance alone used to be treated as done forever.
        #
        # A submit-time ok=True with NO terminal row yet is left ALONE here, not retried -- it may
        # still be genuinely in flight at the venue, and resubmitting it would risk a duplicate. Only
        # a TERMINAL fill marks a symbol done; a terminal REJECTION is a failure like any submit-time
        # one and falls into the same bounded retry, which is the behaviour that was missing.
        # EXTRACTED to `retry.already_attempted` (#164). It was inline, and it is the guard that
        # actually stopped the 08-31 AEM doubling — a symbol with a confirmed fill never reaches the
        # replay list, whatever was refused on it at later slots. Buried inline, a refactor could
        # change its grain silently; the cockpit session found it while reviewing the counter fix.
        attempted = already_attempted(await self.journal.tail(400), session,
                                      max_attempts=_MAX_SUBMIT_ATTEMPTS)
        todo_exits = tuple(s2 for s2 in exits if s2 not in attempted and s2 in held_qty)
        todo_enters = tuple(s2 for s2 in enters if s2 not in attempted)
        if not todo_exits and not todo_enters:
            return SessionResult(session, st.value, False, blocked="already decided this session")
        if not st.may_submit_entries and not st.may_submit_exits:
            return SessionResult(session, st.value, False,
                                 blocked="already decided this session (nothing to submit in "
                                         f"{st.value})")
        # The last look before money moves, same as the normal path. Resume reached _submit()
        # WITHOUT it: a session that had already decided, with unsubmitted exits, would replay them
        # even after an operator halted the strategy in between. The normal path blocked; this one
        # did not, and it is the path that runs after a crash -- exactly when someone is most likely
        # to be reaching for the halt button.
        if (blocked := await self._state_changed_under_us(session, st)) is not None:
            return SessionResult(session, blocked, False,
                                 blocked="operator changed the state during the session — "
                                         "nothing resubmitted")
        await self._j(
            RISK, f"resuming session {session}: {len(todo_exits)} exits and {len(todo_enters)} "
                  f"entries from the recorded decision were never submitted",
            session=session, detail={"exits": list(todo_exits), "enters": list(todo_enters)})
        # EXITS THE DECISION OWES THAT NEVER WENT OUT still hold the entries (#224 B). `attempted`
        # includes a sell exhausted after _MAX_SUBMIT_ATTEMPTS refusals — it drops out of
        # `todo_exits`, and without this the buys it was meant to fund would go on the next slot.
        # "Never went out" = no submit-time ok result and no terminal fill for (session, SELL).
        went_out = submitted_ok(await self.journal.tail(400), session, exits, side="SELL")
        self._unfunded_exits = tuple(s2 for s2 in exits if s2 in held_qty
                                     and s2 not in todo_exits and s2 not in went_out)
        n = await self._submit(session, todo_enters, todo_exits, held_qty, last, st,
                               decision_slot=slot or "")
        if suppressed := getattr(self, "_suppressed", ()):
            rules = ", ".join(unsupported_live_exits(self.cfg.exits))
            return SessionResult(
                session, st.value, True, (), todo_exits, tuple(sorted(book)), n,
                blocked=f"entries suppressed — {rules} configured but not implemented live; "
                        f"exits ran normally",
                detail={**basis, "suppressed_entries": list(suppressed),
                        "unsupported_exits": unsupported_live_exits(self.cfg.exits)})
        return SessionResult(session, st.value, True, todo_enters, todo_exits,
                             tuple(sorted(book)), n, detail=basis)

    def _book_slots(self, held_qty) -> set:
        """The slots the lane OCCUPIES (#224): every claimed long on the account, sellable or not.
        Set by `run()`; a caller that drove `_submit`/`_resume` directly gets `held_qty`."""
        slots = getattr(self, "_held_slots", None)
        return set(slots) if slots is not None else set(held_qty)

    async def _submit(self, session, enters, exits, held_qty, last, st: State,
                      *, weights: dict | None = None, decision_slot: str = "") -> int:
        # Reset per call: a runner is built fresh per session, but _submit also runs on the resume
        # path, and a stale value would misreport the second run.
        self._suppressed: tuple[str, ...] = ()
        sent = 0
        # ATTEMPT NUMBER, fetched once for every symbol this call might submit, not per symbol --
        # avoids an N-query loop over the same journal window. `client_order_id` is deterministic
        # per (session, symbol, side, attempt); a symbol resubmitted after a TERMINAL rejection needs
        # a different id or Nautilus denies it locally as a duplicate before it ever reaches the venue
        # a second time -- silently defeating #51's retry, which this counter exists to prevent.
        # ONE DERIVATION, shared with kumo-trading-platform (kumo-trading-platform issue 383). This predicate decides an
        # `attempt`, which decides a `client_order_id`, which is the replay guard — two copies
        # disagreeing produces a duplicate id or a doubled position, not a wrong dashboard number.
        # SIDE-AWARE (#164). One call per side: without it a SELL's terminal rejection increments
        # the BUY attempt for the same name in the same session, minting a fresh id for an entry
        # because an exit failed. Latent for MOMENTUM, which never exits and re-enters one name in a
        # session; live for QC345, which rebalances — and #120 phase 2 puts QC345 on this loop.
        _tail = await self.journal.tail(400)
        sell_attempts = attempts_for(_tail, session, exits, side="SELL")
        buy_attempts = attempts_for(_tail, session, enters, side="BUY")
        # THE EXITS THAT DID NOT GO OUT THIS CALL (#224 B, platform issue 1058). Symbol -> why. Filled below;
        # `unfunded` carries the ones a resume already knows never went out on an earlier slot.
        # Symbol -> {"why", "cause"}; `cause` is a closed word the operator's fix is named by
        # (release_refused | unsizable | intent_not_journalled | unsubmitted_earlier_slot), so the
        # journal alone says what to reconcile.
        refused: dict[str, dict] = {
            s2: {"why": "never submitted on an earlier slot of this session",
                 "cause": "unsubmitted_earlier_slot"}
            for s2 in getattr(self, "_unfunded_exits", ())}
        for sym in exits:
            q = held_qty.get(sym, 0)
            if q <= 0:
                # A decided exit with NO sellable quantity — the claim the venue does not attribute,
                # zeroed at the top of `run()` and said out loud there — did not go out either, and
                # the entry it was meant to fund waits like any other (#224 B). Loud, not silent:
                # this is the case that stays until an operator reconciles the claim.
                refused[sym] = {"why": "cannot be sized — the claim is not attributed to this "
                                       "strategy", "cause": "unsizable"}
                continue
            # journal the INTENT before the broker sees it: if the write fails we have not traded,
            # and if the submit succeeds but a later write fails there is still a record of intent.
            #
            # FAIL CLOSED. PgJournal.write swallows non-integrity errors and returns None, so an
            # unreachable Postgres used to mean the order went out with no durable record at all --
            # and then resume, which decides what was attempted from those records, would send it
            # again. No record, no order.
            if await self._j(ORDER, f"SELL {q} {sym}: submitting", session=session,
                                        symbol=sym, correlation=f"{session}:decision",
                                        detail={"phase": "intent"}) is None:
                await self._j(ERROR, f"SELL {sym}: intent not journalled — not sending",
                                         session=session, symbol=sym)
                refused[sym] = {"why": "intent not journalled", "cause": "intent_not_journalled"}
                continue
            # strategy_id EXPLICITLY, not the OrderRequest default. `NautilusBroker.exit` passes it
            # to cockpit's `release_for_exit` to find THIS position's protective stop; the default is
            # the literal "MOMENTUM-002", so every other runner would ask to release a stop belonging
            # to a strategy that does not own the shares, find nothing, and have its exit refused.
            # Harmless for MOMENTUM (the default already equals its id) and load-bearing for BCTROT-004
            # and QC345-003, both moved to TRADING on 2026-08-19.
            r = await self.broker.exit(OrderRequest(sym, "SELL", int(q), session,
                                                     strategy_id=self.strategy_id,
                                                     slot=decision_slot,
                                                     attempt=sell_attempts.get(sym, 0)))
            await self._j(ORDER if r.ok else ERROR, f"SELL {q} {sym}: {r.detail}",
                                     session=session, symbol=sym,
                                     correlation=f"{session}:decision",
                                     detail={"phase": "result", "ok": r.ok})
            if not r.ok:
                refused[sym] = {"why": str(r.detail), "cause": "release_refused"}
            if r.ok:
                # NOT dropping state here. `ok` means Nautilus accepted the command, not that the
                # venue filled it -- in live the submit is asynchronous, so a rejection lands after
                # this returns. Dropping the claim now and having the sell rejected leaves a live
                # position with no claim, which the next session reads as foreign and never exits.
                # The session-start reconcile retires the claim once the position is really gone.
                #
                # And NOT counting it as closed. `closed` frees capacity for new entries, so
                # treating an accepted sell as a completed one lets buys fill slots that the
                # unfilled sells have not actually vacated -- the book ends over the position cap.
                # The conservative direction is the correct one here: an exit that really did close
                # frees its slot at the NEXT session, one session late, which costs an entry we
                # could have made. The other direction costs a position we cannot carry.
                sent += 1
            # a REJECTED sell also leaves the position open. Keeping its trail matters — dropping it
            # would reseed entry/peak from today and disarm the give-back rule on a live winner.
        if not st.may_submit_entries:
            return sent
        # THE EXIT FUNDS THE ENTRY, OR THE ENTRY WAITS (#224 B, platform issue 1058). A top-K rotation
        # funds its buys with its sells; a sell that did not go out — release refused, intent not
        # journalled, or never submitted on an earlier slot of this session — leaves the buys
        # unfunded, and sending them anyway puts them in front of cockpit's budget gate, which is
        # blind to in-flight sells by design (#1006) and refuses them "0 of budget left". Measured
        # on paper: BCTROT-004 09-09, four sells "shares not released" x8, every buy budget-rejected,
        # and the stall read as a budget problem. The whole entry set waits — not "reduce room by
        # N", which is a half-funded book, the #1006 shape again.
        #
        # A WAIT, NOT A REFUSAL: nothing is journalled per entry here (no result row, no
        # `suppressed` detail), so `_resume` still sees these entries as never attempted and sends
        # them on the slot where the sells finally go. One RISK row says what was held and why.
        if refused and enters:
            # RISK rows only, and a deep window: `_tail` (400 of every kind) cannot be relied on to
            # reach two sessions back on a lane that writes ~45 pool rows a session (BCTROT, paper
            # 09-11) — and a streak that cannot see yesterday never releases, which is the unbounded
            # rule the cap exists to refuse. RISK rows are a few per session, so 2000 spans months.
            streak = held_sessions_before(await self.journal.tail(2000, kind="risk"),
                                          session, set(refused))
            if streak < _MAX_HELD_SESSIONS:
                await self._j(
                    RISK,
                    f"entries held this slot: {len(refused)} exit(s) refused — "
                    + "; ".join(f"{s2}: {r2['why']}" for s2, r2 in sorted(refused.items()))
                    + f" — the exit funds the entry, or the entry waits (#224). "
                      f"Held: {', '.join(enters)} (session {streak + 1} of {_MAX_HELD_SESSIONS})",
                    session=session,
                    detail={"refused_exits": dict(sorted(refused.items())),
                            "held_entries": list(enters), "held_count": len(enters),
                            "held_sessions_before": streak, "slot": decision_slot})
                return sent
            # THE CAP. Two sessions of slots have refused the same exits; a third of never-entering
            # would be the lane quietly narrowing itself. Enter anyway and say what is still stuck.
            await self._j(
                RISK,
                f"entries released after {streak} held session(s) — exits still refused: "
                + "; ".join(f"{s2}: {r2['why']}" for s2, r2 in sorted(refused.items()))
                + f" (#224, cap {_MAX_HELD_SESSIONS}). Entering: {', '.join(enters)}",
                session=session,
                detail={"refused_exits": dict(sorted(refused.items())),
                        "released_entries": list(enters), "released_count": len(enters),
                        "released_after_sessions": streak, "slot": decision_slot})
        # A CONFIGURED EXIT RULE THIS RUNNER CANNOT HONOUR STOPS US BUYING, NEVER SELLING. The rules
        # are implemented twice (backtest harness and here) and only give-back exists on this side, so
        # a config asking for `stall_days` gets a book entered on one set of rules and managed by a
        # smaller one -- positions opened with no way to close them on the terms that justified the
        # entry. Refusing the ENTRY is the conservative half.
        #
        # Deliberately NOT an exception. Raising here would also kill the exit paths above, operator
        # forced exits and LIQUIDATING -- a config typo would strand real positions with no way out.
        # Placed after the exit loop so everything sellable has already been submitted.
        if unsupported := unsupported_live_exits(self.cfg.exits):
            await self._j(
                RISK,
                f"entries suppressed: {', '.join(unsupported)} configured but not implemented live "
                f"— exits still run (kumo-trading-platform issue 197 B12)",
                session=session, detail={"unsupported_exits": unsupported})
            # TERMINAL PER-SYMBOL ROWS, not just the summary above. `_resume` rebuilds the entry list
            # from the decision detail and treats a symbol as attempted only when a `phase=result`
            # row NAMES it -- so a suppression recorded once, with no symbol, leaves every suppressed
            # buy looking like it was never sent. Resume would then submit exactly the orders this
            # guard refused, and if the config were fixed mid-session it would submit them against a
            # decision taken under the old one.
            #
            # `suppressed` rather than ok=False: a failed result is RETRIED up to
            # _MAX_SUBMIT_ATTEMPTS, and this is a decision not to trade, not a failure to.
            self._suppressed = tuple(enters)
            for sym in enters:
                await self._j(
                    ORDER, f"BUY {sym}: suppressed — {', '.join(unsupported)} not implemented live",
                    session=session, symbol=sym, correlation=f"{session}:decision",
                    detail={"phase": "result", "ok": False, "suppressed": True,
                            "unsupported_exits": unsupported})
            return sent
        # The strategy's OWN allocation when it has one, not the account. Sizing off the account is
        # the one-position failure this exists to prevent — ~10k a name against a 20k target, for a
        # book built to hold eight.
        #
        # THIS COMMENT USED TO DESCRIBE MACHINERY THAT IS NOT IN THIS REPO (corrected 2026-08-24). It
        # claimed a three-level precedence ("PUSHED, then configured, then the account"), a `resolve`
        # that "RAISES for a strategy cockpit says it manages whose allocation has not arrived", a
        # `managed` set, and a `getattr` choice. None of those exist here — `grep` for `resolve`,
        # `managed`, or `pushed` across `src/` finds no allocation machinery at all. There is ONE
        # source: whatever single float the caller put in `limits.allocated_equity`.
        #
        # It mattered, and not as tidiness. Both this repo and kumo-trading-platform reasoned from that raise
        # as though it were a live safeguard while diagnosing the falsy-`or` defect below, and cockpit
        # then reported that its own `momentum.py:175` does `float(resolve(...).get(id) or 0.0)` and
        # never raises either. So the "not arrived" case is not guarded ANYWHERE on the path.
        #
        # WHAT IS ACTUALLY TRUE, and the remaining hole: `None` is an unconditional fall-through to
        # the whole account, and a caller has no way to say "I manage this lane and its allocation
        # did NOT arrive". Absent and unknown are the same value here. The fix below makes 0.0 mean
        # zero; it does not close that. Closing it needs a distinct state from the caller and is a
        # behaviour change to every running config, so it is the operator's call, not this commit's.
        # ZERO IS NOT ABSENT -- see `qc27_runner._submit`. `or` sent a lane allocated 0.0 to the
        # account's book, which is the one-position failure the paragraph above exists to prevent,
        # arriving through the value rather than through a missing one. `None` is unchanged.
        alloc = self.limits.allocated_equity
        # AND IT MUST BE A NUMBER. `broker.equity()` was read straight into the arithmetic below —
        # the same unguarded read as the daily-loss halt, one function away, and only that one was
        # repaired. MOMENTUM-002 and BCTROT-004 pass a bare `RiskLimits()`, so `alloc` is None and
        # this IS their sizing path.
        #
        # What each bad value does, and neither is exotic — `NautilusBroker.equity()` reads the
        # msgbus account snapshot, which is absent until the first account update:
        #
        #   None   `None * frac` raises TypeError out of the decision path.
        #   NaN    `slot` becomes NaN, survives every downstream comparison (all of them are False
        #          against a NaN), and raises `cannot convert float NaN to integer` inside the
        #          quantity conversion PART WAY THROUGH the entry loop — after earlier symbols have
        #          already been submitted. A half-executed rotation is worse than none.
        #
        # `qc27_runner._account_equity` has said exactly this in its docstring since the NaN sizing
        # defect. This is the same rule, in the runner that has been trading.
        equity = self._finite_equity() if alloc is None else float(alloc)
        if equity is None:
            await self._j(
                RISK, f"cannot size entries — the account equity is not a number "
                      f"({self.broker.equity()!r}). No entry is submitted this session rather than "
                      f"a partial book sized off a value we could not read.",
                session=session, detail={"equity": None})
            # `sent`, NOT 0. Exits ran ABOVE this point and some may already have reached the broker;
            # returning 0 would under-report them, and this counter is what stops #51's retry
            # submitting a second time. Entries are what is refused here — matching the rule this
            # function already states for an unhonourable exit config: stop us BUYING, never selling.
            return sent
        # BOOK SIZE, not the cap. These were one number until 2026-08-21; dividing by the cap would
        # shrink every position as soon as the cap gained the headroom a rotation needs.
        #
        # `size_to_book` then asks whether to divide by the book that will EXIST after this call
        # instead of the one that was planned. Derived here from what `_submit` already receives
        # rather than added to the signature: three call sites reach this function (normal, resume,
        # urgent-exit) and a parameter is a thing one of them forgets — the `slot` argument was
        # defaulted for exactly that reason and was then dropped on 31 journal call sites.
        #
        # Set arithmetic, not len() arithmetic: on the RESUME path `enters` holds only the entries
        # not yet submitted, and some may already sit in `held_qty`. A union counts each name once;
        # `len(held) - len(exits) + len(enters)` would double-count them and size every position
        # too small on precisely the path that runs after a partial failure.
        occupied = self._book_slots(held_qty)
        book_after = len((occupied - set(exits)) | set(enters))
        slot = equity * self.limits.max_deployed_frac / sizing_denominator(
            book_after, self.limits.book_size, self.cfg)
        # `dec.weights` was computed and thrown away here, exactly as it was in the backtest: sizing
        # was one flat slot per name, so `inverse_vol_sizing` could be set live and change nothing
        # (#26). Expressed as a MULTIPLE of the equal-weight slot rather than as a raw share of
        # equity, because equal weight makes `w = 1/len(book)` and the multiple is then exactly 1.0
        # — so a config that does not ask for inverse-vol sizes identically to before, which is
        # every config running today.
        #
        # Sizing is at ENTRY ONLY and never rebalanced, matching the backtest. A name enters at a
        # size proportional to 1/sigma and then drifts; that is what this runner can actually
        # execute, and measuring anything else would be measuring a strategy we cannot run.
        # Normalised against the WEIGHT SET. An earlier version used `held_qty | enters`, which
        # still contains the names being exited this session, so every entry was sized ABOVE its
        # intended share. The weights sum to 1 over their own keys, so `w * len(w)` is the multiple
        # of equal weight regardless of what the book is doing around it.
        scale = {}
        if weights:
            scale = {s2: weights[s2] * len(weights) for s2 in enters if s2 in weights}
        # Only positions the broker no longer reports free a slot, and that is not knowable inside
        # this call -- an accepted sell may still reject. So capacity is measured against what we
        # actually hold right now, with nothing deducted for sells submitted moments ago.
        live = len(occupied)
        for sym in enters:
            if live >= self.limits.max_positions:
                await self._j(RISK, f"skipped {sym}: position cap", session=session,
                                         symbol=sym)
                continue
            # Size on the freshest price the node has, not yesterday's close. Sizing a $10k budget
            # off a $10 close when the name opens at $15 submits 1000 shares for $15k of notional --
            # 50% over the intended position, and the position cap counts positions, not dollars, so
            # nothing downstream notices.
            quote = getattr(self.broker, "last_price", None)
            # See `_trail_exits`: no try/except, no real broker needs the fallback, and swallowing
            # TypeError here would hide a genuine bug the same way.
            p = quote(sym, **self._max_age_kw()) if quote is not None else None
            if not p or not np.isfinite(p):
                # NO STALE FALLBACK FOR SIZING. Yesterday's close is wrong by exactly the overnight
                # gap: a $10k budget against a $10 close on a name that opened at $15 buys 1000
                # shares of $15k notional, and the position cap counts positions rather than
                # dollars, so nothing downstream notices. An entry skipped for one session costs an
                # opportunity; an entry sized on a stale price costs money on every one.
                await self._j(
                    RISK, f"skipped {sym}: no live price to size against", session=session,
                    symbol=sym)
                continue
            # `max_position_notional` is a hard risk limit, so it binds AFTER the weight is applied.
            # A low-vol name earning three slots of dollars must still not exceed the cap.
            budget = min(slot * scale.get(sym, 1.0), self.limits.max_position_notional)
            q = int(budget / p)
            if q < 1:
                # A silent `continue` here was safe only while `slot` came from account equity --
                # two orders of magnitude above any real share price. Sizing off `allocated_equity`
                # (#334) can put `slot` within one order of magnitude of a real price, and a name
                # priced above it now drops out with no trace: the decision said "enter 5" and fewer
                # than 5 buys reach the broker, with nothing recording which name or why. Same
                # RISK-row discipline as the no-live-price refusal just above it.
                await self._j(
                    RISK, f"skipped {sym}: budget ${budget:,.2f} at ~${p:.2f}/share buys 0 shares",
                    session=session, symbol=sym, detail={"budget": budget, "price": p})
                continue
            # BUY had no pre-submit intent row at all: a crash between submit and the result write
            # left an order with nothing recording it. Same fail-closed rule as the sell path.
            if await self._j(ORDER, f"BUY {q} {sym} @~{p:.2f}: submitting",
                                        session=session, symbol=sym,
                                        correlation=f"{session}:decision",
                                        detail={"phase": "intent"}) is None:
                await self._j(ERROR, f"BUY {sym}: intent not journalled — not sending",
                                         session=session, symbol=sym)
                continue
            # NAMED, like the SELL path above. Omitting it does not fail — `strategy_id` defaults
            # to the string "MOMENTUM-002", so every entry from BCTROT-004 and QC345-003 was built
            # claiming to be MOMENTUM-002. It feeds `client_order_id`, which is the replay guard and
            # the only thing tying a broker order back to its owner.
            r = self.broker.submit(OrderRequest(sym, "BUY", q, session,
                                                strategy_id=self.strategy_id,
                                                slot=decision_slot,
                                                attempt=buy_attempts.get(sym, 0)))
            await self._j(ORDER if r.ok else ERROR,
                                     f"BUY {q} {sym} @~{p:.2f}: {r.detail}", session=session,
                                     symbol=sym, correlation=f"{session}:decision",
                                     detail={"phase": "result", "ok": r.ok})
            if r.ok:
                # ACCEPTED LOCALLY, NOT OPENED. Nautilus accepting the command means the order is in
                # its lifecycle; the venue can still deny or reject it. Writing `PositionState` here
                # created live claims for positions that never existed (2026-09-14: CRAK/DINO after
                # budget denial). The terminal event path calls `sync_claim()` with the lane's
                # actual attributed book and the fill price, so the claim/trail follows the venue's
                # answer. Still count it inside THIS loop: an accepted entry may fill, and submitting
                # the next one as if no slot were occupied would overbook the session.
                live += 1
            sent += r.ok
        return sent
