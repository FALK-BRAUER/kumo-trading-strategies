"""The daily-loss stop, in one place, for every runner (kumo-trading-platform issue 548, #111).

WHY THIS MODULE EXISTS. `pgrunner` enforced `RiskLimits.daily_loss_frac` inline and `qc27_runner`
did not enforce it at all, so MOMENTUM-002 and BCTROT-004 could halt themselves on risk and
TECHIVOL-005 could not, under any condition. `momentum_rotation.broker_equity` calls this limit
"the only automatic stop this strategy has", which makes the gap a live question rather than a
tidiness one.

AND A CONTROL THAT CANNOT FIRE LEAVES NO TRACE. "this lane had no reason to halt" and "this lane has
no ability to halt" produce the same journal — nothing — so no operator reading it can tell them
apart. That is the more serious half of #548 and it is why `enforce` writes a row when a lane has no
limit configured, instead of returning quietly.

ONE DERIVATION OF "A USABLE EQUITY". #111 was found because there were two. `_finite_equity` guarded
the equity read at decision time; `_session_start_equity` guarded nothing, so a non-finite value
recorded as a baseline came back through a truthy `.get("equity")` and disarmed the limit:

    now < NaN * 0.95   ->   False, like every comparison with a NaN

THE EXPOSURE WAS NIL, AND THIS IS NOT WHAT IT LOOKS LIKE. Read the paragraph above as a caught
incident and you would be wrong. Measured afterwards by cockpit, against both production databases:
55 decision rows, 55 carrying a finite equity anchor, ZERO non-finite, ever. And it could not have
been otherwise — the production write path cannot represent the poison:

    float('nan')        -> '{"equity": NaN}'       bare token, postgres rejects the jsonb cast
    float('inf')        -> '{"equity": Infinity}'  same
    Decimal('NaN')      -> TypeError inside json.dumps, before the database sees anything

Either way `PgJournal.write` returns None and the session is refused on "decision was not durably
journalled". My own demonstration of a 99% fall that did not halt ran through a FIXTURE journal that
accepts anything -- a double more permissive than production, which is the exact failure this repo
keeps writing tests about. The vulnerability was real. The exposure was not.

SO THIS GUARDS A CLASS, NOT THAT BUG. The instance was unreachable; the class is not, and one member
of it is worse. A STRINGIFIED value is valid JSON and does persist:

    '{"equity": "Infinity"}'  -> ACCEPTED by jsonb, and float("Infinity") is inf, so the old
                                 `now < inf * 0.95` is always TRUE -- a PERMANENT SPURIOUS HALT.
                                 A lane that stops every session forever and looks correct doing it.

That needs only one caller anywhere to hand the journal a pre-stringified equity (a `default=str`
encoder would do it; nothing in the chain uses one today). `finite()` refuses every non-finite value
whatever route it arrives by, which is why it is framed as a class guard rather than a NaN check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Three outcomes, named, because collapsing any two of them has already cost a session.

    PROCEED and BLOCK differ in whether the lane trades today. BLOCK and HALT differ in whether it
    trades again without an operator. Both distinctions are load-bearing:

    * A missing account snapshot is TRANSIENT -- `NautilusBroker.equity()` reads the msgbus account
      snapshot, absent until the first account update -- and clears itself next session. Halting on
      it takes a lane down for a condition that resolves on its own, the same error as treating a
      rolling-window calendar refusal as fatal.
    * A breach is DURABLE. `lifecycle.halt()` stops the lane and keeps it stopped until an operator
      clears it. Blocking on a breach would let the lane resume tomorrow into the same drawdown.
    """

    PROCEED = "proceed"
    BLOCK = "block"
    HALT = "halt"


@dataclass(frozen=True)
class Verdict:
    action: Action
    reason: str | None = None
    start: float | None = None
    now: float | None = None

    @property
    def may_continue(self) -> bool:
        return self.action is Action.PROCEED


def finite(value) -> float | None:
    """The one definition of an equity this system will compare against a limit.

    None rather than 0.0, because the two are different facts and a caller must not confuse them:
    0.0 is a real account value and reads as a catastrophic loss against any baseline.
    `qc27_runner._account_equity` returns 0.0 for its own purpose -- SIZING, where zero correctly
    degrades to inaction -- and that is the right answer there and the wrong one here.

    Guards NaN explicitly. `x or 0.0` does not: NaN is truthy, so the `or` never fires, and that is
    exactly how #111 got a NaN past a `.get("equity")` truthiness check and into the baseline.
    """
    try:
        out = float(value() if callable(value) else value)
    except (TypeError, ValueError, ArithmeticError):
        # THE CALL ITSELF IS INSIDE THE GUARD. `_finite_equity` wrapped `float(broker.equity())`, so
        # a broker whose equity() raised became a BLOCK. Evaluating the callable outside and passing
        # the result in let that exception escape the session path instead — a refusal turned into a
        # crash, which is the one direction this module must never move.
        return None
    return out if math.isfinite(out) else None


def anchor(equity) -> dict:
    """The `equity` key for a DECISION row's detail, or NOTHING when it is not a number.

    THE WRITE SIDE of the same invariant `finite` enforces on the read side: **the key is present if
    and only if the value is usable.** #111 existed because the writer recorded whatever the broker
    returned -- `pgrunner:982` wrote a raw `broker.equity()` -- and the reader trusted the key's
    presence. A `Decimal("NaN")` therefore became a truthy baseline that made every comparison False.

    Omitting the key beats writing None or a null: the reader's scan already treats an absent key as
    "not a record", so old rows and new rows are handled by one rule instead of two.
    """
    value = finite(equity() if callable(equity) else equity)
    return {} if value is None else {"equity": value}


def decide(start: float | None, now: float | None, frac: float | None) -> Verdict:
    """PURE. No journal, no broker, no lifecycle -- so every case below is reachable in a test
    without a session fixture, and the fixture cannot be what makes it pass.

    `start` and `now` must already have been through `finite()`. Passing a raw broker value here is
    the #111 defect re-introduced one layer down, which is why `enforce` is the only intended caller
    and takes raw values.
    """
    if not frac:
        # NOT AN ERROR, AND NOT SILENT EITHER. The caller journals this; see `enforce`.
        return Verdict(Action.PROCEED, "no daily-loss limit configured for this lane",
                       start=start, now=now)
    if start is None:
        # No anchor. Today's behaviour for a genuinely absent baseline, and it must stay PROCEED:
        # a lane's first session ever has no prior decision to anchor on, and blocking it would mean
        # a new lane can never start.
        return Verdict(Action.PROCEED, "no baseline equity recorded yet", start=None, now=now)
    if now is None:
        # UNVERIFIABLE IS NOT WITHIN THE LIMIT. Read straight into the comparison, a NaN makes
        # `now < threshold` False and the limit is skipped in silence.
        return Verdict(Action.BLOCK, "equity unreadable — daily-loss limit unverifiable",
                       start=start, now=None)
    if now < start * (1 - frac):
        return Verdict(Action.HALT, f"daily loss {100*(1-now/start):.1f}% breached "
                                    f"{100*frac:.0f}% limit", start=start, now=now)
    return Verdict(Action.PROCEED, None, start=start, now=now)


class Baseline(Enum):
    """WHY the anchor is what it is. `None` alone cannot carry this, and the difference decides
    whether the lane trades.

    * MISSING -- no equity-bearing row at all. Legitimate: a lane's first session ever has no prior
      decision to anchor on, so it PROCEEDS. Blocking would mean a new lane can never start.
    * POISONED -- rows exist, every one of them non-finite. Also proceeds, and MUST.

    WHY POISONED CANNOT BLOCK, though it is tempting and I got this wrong once. BLOCK returns before
    the DECISION write, so a blocked session records no new anchor. Next session scans back, finds
    the SAME non-finite rows, and blocks again. MEASURED over three consecutive sessions: block,
    block, block. Nothing an operator does clears it, because nothing rewrites history -- a permanent
    stop wearing transient clothes, which inverts the BLOCK/HALT distinction this module exists to
    keep. HALT would at least announce that it needs someone.

    The asymmetry with an unreadable equity TODAY is the whole argument: a NaN now is a fact about
    the number you would trade against this instant, and the next session re-reads the live account
    and heals. A NaN recorded is a fact about a past write, and it disqualifies the RECORD, not the
    session. So a non-finite row is skipped and the scan continues -- an older finite anchor still
    counts, which is how `test_a_GOOD_prior_anchor_beats_a_corrupt_current_one` behaves -- and only
    when nothing finite survives anywhere do we land in the pre-existing no-baseline behaviour.
    Loudly: `enforce` says so every session.
    """

    FOUND = "found"
    CARRIED = "carried"          # from the previous session; this lane decides once a day
    MISSING = "missing"
    POISONED = "poisoned"


async def baseline(journal, session: str) -> tuple[float | None, Baseline]:
    """The equity this session's loss is measured against, and why.

    Reads the lane's OWN journal, so baselines differ per lane by design: a lane deciding at open+5m
    and one at open+150m anchor on different account values and cross the threshold at different
    times, or one crosses and the other does not.

    Falls back to the previous session's anchor. Without that the limit cannot fire at all on a
    once-a-day strategy: the baseline is written by the first decision of the session, and the first
    decision is the only one. An overnight gap from 100k to 93k passed straight through, and 93k was
    then recorded as today's "start", making the loss invisible on every later comparison too.
    """
    from kumo_strategies.runtime.executor.pgjournal import DECISION

    rows = await journal.tail(400, kind=DECISION)
    saw_unusable = False
    for want_current in (True, False):
        for r in rows:
            same = r.get("session") == session
            if same is not want_current:
                continue
            if not want_current and not (r.get("session") or "") < session:
                continue
            if "equity" not in (r.get("detail") or {}):
                continue
            value = finite(r["detail"]["equity"])
            if not value:
                # ZERO IS NOT AN ANCHOR, though it IS a real equity. `finite(0.0)` is 0.0 and that is
                # right for `now`; as a baseline it is vacuous-but-armed — `now < 0 * 0.95` is never
                # true, so HALT becomes unreachable while the status row proudly says "anchored on
                # 0". The old truthy scan skipped 0.0 rows and fell through to an older anchor;
                # keeping that is the conservative reading and the only one where the stop still
                # works. Grouped with non-finite here because the caller's question is the same:
                # is there something to measure against?
                # RECORDED AND UNUSABLE -- not an equity record at all. Skip the ROW and keep
                # scanning: one poisoned row must not shadow a finite anchor sitting behind it,
                # which would disarm the limit exactly when a real loss may be in progress.
                saw_unusable = True
                continue
            return value, (Baseline.FOUND if want_current else Baseline.CARRIED)
    return None, (Baseline.POISONED if saw_unusable else Baseline.MISSING)


async def enforce(*, journal, write, lifecycle, equity, frac, session: str,
                  on_halt=None) -> Verdict:
    """Check the limit, journal what happened, halt if it was breached. The whole rule, once.

    `write(kind, summary, detail=...)` is supplied by the runner because each one already knows the
    slot its rows must be filed under. This function must NOT reach for `_slot` itself: a row filed
    under the wrong slot does not match the session it describes, which is the ba37ef9 shape.

    `equity` is the RAW broker value or a callable returning one, deliberately. Taking a pre-cleaned
    float would let each caller decide again what "usable" means, and two derivations of that is
    precisely #111.

    `on_halt` is an async hook the OWNER of the durable state supplies. `lifecycle.halt()` mutates a
    value object captured when the runner was constructed; `qc27_runner._read_state` reads cockpit's
    store every session and ignores that object entirely, so a local halt there survives exactly one
    session and then evaporates -- a durable stop silently degraded to a one-session decline, on the
    runner being newly wired. Defaults to local-only, which is correct for tests and backtests where
    `lifecycle` IS the whole truth.

    THE STATUS ROW IS UNCONDITIONAL, and that is the point of #548 rather than a side effect. A lane
    with no limit, a lane with a limit and no anchor to measure it against, and a lane checking
    normally used to be indistinguishable in the journal -- all three wrote nothing. An operator
    could not tell "had no reason to halt" from "has no ability to". One row per session per lane is
    the price, and it is the same order as the carried-baseline row pgrunner already wrote.
    """
    from kumo_strategies.runtime.executor.pgjournal import RISK, STATE

    now = finite(equity)          # callable resolved INSIDE the guard; see `finite`
    # RESOLVED ONCE. The BLOCK message below quotes the raw value so an operator can see WHAT the
    # account returned, and re-invoking the callable to build that string put an unguarded broker
    # call back on the refusal path — the very path taken because the broker is misbehaving. A
    # second call can also return a different value than the one just judged.
    try:
        raw = equity() if callable(equity) else equity
    except Exception as exc:                                           # noqa: BLE001
        raw = f"<raised {type(exc).__name__}: {exc}>"

    if not frac:
        # STATE, NOT RISK. "not armed" is a static configuration fact that recurs identically
        # forever; a daily RISK row meaning "config unchanged" trains an operator to skim RISK,
        # which is how the next real one gets missed — the 115-false-alarms failure, in the CHANNEL
        # rather than the cadence. RISK stays for ARMED-BUT-UNENFORCEABLE, which is actionable, and
        # for BLOCK and HALT.
        await write(STATE, "daily-loss stop NOT ARMED for this lane — it cannot halt itself on risk "
                          "under any condition (kumo-trading-platform issue 548). The mechanism is wired and the "
                          "threshold is deliberately unset; set RiskLimits.daily_loss_frac to arm.",
                    detail={"daily_loss_frac": None, "equity": now, "enforceable": False})
        return Verdict(Action.PROCEED, "daily-loss stop not armed for this lane", now=now)

    start, why = await baseline(journal, session)

    if start is None:
        # ARMED AND UNENFORCEABLE, which is the state #548 was really about and the one that used to
        # be invisible. The limit is set, and there is nothing to measure it against — so the lane
        # is running with a stop that cannot fire, and now says so every session.
        await write(RISK, f"daily-loss stop ARMED at {100*frac:.0f}% but UNENFORCEABLE — "
                          f"{'every recorded equity anchor is non-finite' if why is Baseline.POISONED else 'this lane has written no equity anchor yet'}. "
                          f"Proceeding unchecked. Blocking instead would never clear: a blocked "
                          f"session records no new anchor, so the next one reads the same rows "
                          f"(#111).",
                    detail={"start": None, "now": now, "baseline": why.value,
                            "enforceable": False})
        return Verdict(Action.PROCEED, f"daily-loss baseline {why.value}", now=now)

    await write(STATE, f"daily-loss stop armed at {100*frac:.0f}%, anchored on {start:,.0f}"
                       + (" (carried over — no decision yet this session)"
                          if why is Baseline.CARRIED else ""),
                detail={"baseline": start, "now": now, "source": why.value, "enforceable": True})

    v = decide(start, now, frac)
    if v.action is Action.BLOCK:
        await write(RISK, f"daily-loss limit CANNOT BE CHECKED — the account equity is not a "
                          f"number ({raw!r}). Refusing this session rather than trading against "
                          f"an unverifiable limit.",
                    detail={"start": v.start, "now": None})
    elif v.action is Action.HALT:
        # HALT BEFORE THE WRITE, so a journal failure cannot un-halt the lane.
        lifecycle.halt(v.reason)
        if on_halt is not None:
            try:
                saved = await on_halt(v.reason)
                if saved not in (None, True):
                    # A RETURNED FAILURE IS STILL A FAILURE. Cockpit's `save_if_unchanged` reports
                    # OPERATOR_WON / CONTRADICTION by RETURNING them, not by raising — so reacting
                    # only to exceptions would let an UNPERSISTED halt read as persisted. Absence
                    # readable as permission, in the persistence path of a risk stop.
                    await write(RISK, f"HALT was NOT PERSISTED — the persistence hook returned "
                                      f"{saved!r} rather than saving. This lane is halted for THIS "
                                      f"session only and will resume next session unless an "
                                      f"operator sets the state.",
                                detail={"persist_result": str(saved)})
            except Exception as exc:                                   # noqa: BLE001
                # THE LOCAL HALT ALREADY HAPPENED, which is the fail-safe direction. A raising
                # persistence hook must not also cost us the row that says the lane halted — that
                # would leave it stopped with no record of why, the worst of both.
                await write(RISK, f"HALT could not be PERSISTED ({type(exc).__name__}: {exc}) — "
                                  f"this lane is halted for THIS session only and will resume next "
                                  f"session unless an operator sets the state.",
                            detail={"persist_error": str(exc)})
        await write(RISK, f"HALTED: equity {v.now:,.0f} vs session start {v.start:,.0f} "
                          f"({100*(v.now/v.start-1):.1f}%)",
                    detail={"start": v.start, "now": v.now})
    return v
