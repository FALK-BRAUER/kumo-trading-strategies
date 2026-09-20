"""How many times a symbol has already been attempted this session — ONE derivation.

`OrderRequest.attempt` feeds `client_order_id`. Get the count wrong and the id is wrong, and the id
is the replay guard:

  TOO LOW   a retry regenerates an id already used. Nautilus denies it LOCALLY as a duplicate before
            it reaches the venue, and the denial journals identically to a genuine venue rejection —
            silently defeating the retry this counter exists to enable (issue 51). Worse,
            `NautilusBroker.submit()` reads the order back BY that id to report the outcome, so a
            duplicate returns the EARLIER order and reports a stale answer as though it were this
            one: a rejection seven hours old, quoting a different order's reason.
  TOO HIGH  a fresh id for an order that may still be live at the venue — a SECOND POSITION.

kumo-trading-platform needs the same count for QC345 (kumo-trading-platform issue 383). This lives here, imported by both,
rather than copied — the disagreement between two copies would not be a wrong number on a dashboard,
it would be a duplicate order id or a doubled position, and neither repo would see it until a retry
actually happened. That seam has already produced three defects: leash validation, manager-armed
derivation, and percent-to-bps rounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def attempts_for(rows: Iterable[Mapping], session: str, symbols: Iterable[str],
                 *, side: str | None = None) -> dict[str, int]:
    """`{symbol: failed attempts}` for `session`, counting TERMINAL failures only.

    WHY TERMINAL AND NOT `phase="result"`. A result row is submit-time: `ok=True` there means
    Nautilus accepted the order into its local lifecycle (INITIALIZED), not that the venue did
    anything — `NautilusBroker`'s own docstring says so. Counting result rows as attempts is what
    broke resume once already: the first live run refused all 8 entries for missing instrument
    definitions, and after that was fixed resume still reported "already decided", because those
    failures looked like attempts.

    A `phase="terminal"` row is the venue's real answer, written by the adapter's order-event
    handlers via `record_terminal`.

    KNOWN GAP, and the caller must handle it rather than this function guessing (see
    `record_terminal`'s own contract): a refusal that happens BEFORE an order exists produces no
    Nautilus order event, so no terminal row, so no increment here. "not a subscribed instrument",
    an unwired feed, and "shares not released — protection still holds them" are all in that class.
    Those are terminal in fact — nothing will ever follow them — and the fix is for that path to
    write its own terminal row, not for this counter to start trusting result rows again.

    KEYED BY ORDER, NOT BY (SESSION, SYMBOL) — #164, and the reason is a doubling path rather than
    a wrong number. On 2026-08-31 MOMENTUM-002's AEM order FILLED at the first of five slots and the
    same symbol was refused five times at later slots, each writing an `ok=False` terminal row. The
    old keying read FIVE ATTEMPTS FOR A SYMBOL WHOSE ORDER FILLED. `attempt` feeds
    `client_order_id`, and `_resume` passes it, so a nonzero count mints a FRESH id that Nautilus
    does not deny — the local duplicate guard is not engaged. Six instances of the grain exist in
    the paper journal: AEM 1+5, VCTR 1+5, NOW 1+3, HALO 2+2, APA 1+1, TOST 1+1.

    THREE STATES, WHERE THE COUNTER HAD TWO: absent / refused-nothing-booked /
    refused-shares-booked — and "filled" was not a state it could represent at all. So:

      * an order that FILLED ends the question for its symbol; whatever was tried later, the thing
        a resume would replay is already done
      * a failure that BOOKED SHARES is not an attempt to repeat, because repeating it re-sends the
        whole quantity on top of the shares already held
      * the same order reporting twice (a reject then a cancel of one id) is ONE attempt

    BACKWARD COMPATIBLE BY DESIGN, and that is a safety property rather than politeness. Every
    terminal row already in the paper journal predates this and carries no `client_order_id`; those
    rows still count per-symbol, one each. If they stopped counting, a resume would read 0 for a
    symbol that genuinely failed three times and retry it past its cap. Over-counting only ever
    refuses a retry; under-counting sends an order.

    SIDE-AWARE, and that is not a nicety either. Without `side` a SELL's terminal rejection
    increments the BUY attempt for the same name in the same session, minting a fresh id for an
    entry because an exit failed. That is latent today only because MOMENTUM never exits and
    re-enters one name within a session — a property of its SCHEDULE, not of this counter. QC345
    rebalances, so a lane that trims and re-adds a name in one session makes it live, and #120
    phase 2 puts QC345 on this loop. Raised by the cockpit session before it could be discovered in
    the port.

    `side=None` counts every side, which is the pre-#164 behaviour. Rows that do not RECORD a side
    count under any filter, for the same backward-compatibility reason as the missing order id.

    Pure and synchronous: takes rows, not a journal, so it can be tested without a database and
    called from either repo against whatever row source it already has.
    """
    wanted = set(symbols)
    #: A symbol is keyed by ORDER. `client_order_id` identifies one; a row without one is its own
    #: unit, which is the pre-#164 behaviour and is deliberate — see the backward-compatibility note
    #: in the docstring.
    failed: dict[str, set] = {}
    filled: set[str] = set()
    for i, r in enumerate(rows):
        d = r.get("detail") or {}
        sym = r.get("symbol")
        if r.get("session") != session or sym not in wanted or d.get("phase") != "terminal":
            continue
        row_side = d.get("side")
        if side is not None and row_side is not None and row_side != side:
            continue
        if d.get("ok"):
            # AN ORDER THAT FILLED ENDS THE QUESTION FOR THIS SYMBOL. Whatever else was tried at
            # later slots, the thing a resume would replay has already been done.
            filled.add(sym)
            continue
        # SHARES BOOKED IS NOT "NOTHING HAPPENED". A cancel or reject that filled part of the order
        # is not an attempt to repeat — repeating it re-sends the WHOLE quantity on top of shares
        # already held. Missing means zero: every row written before #164 lacks the field, and
        # treating absence as "some filled" would silently stop counting genuine failures.
        if int(d.get("filled_qty") or 0) > 0:
            continue
        failed.setdefault(sym, set()).add(d.get("client_order_id") or f"__row{i}")
    return {s2: len(ids) for s2, ids in failed.items() if s2 not in filled}


def already_attempted(rows: Iterable[Mapping], session: str, *, max_attempts: int) -> set[str]:
    """Symbols a resume must NOT submit again. Extracted from `_resume` unchanged (#164).

    THIS IS THE GUARD THAT ACTUALLY STOPPED THE 08-31 DOUBLING, ahead of `held_qty` and well ahead
    of the budget gate, and it was buried inline where a refactor could change its grain without
    anything saying so. The cockpit session found it while reviewing #164 and the warning is the
    reason it now lives here: `attempts_for` being SYMBOL-grain is the defect, and `attempted` being
    SYMBOL-grain is the fix. Converting both to per-order in one pass would repair the counter and
    delete the guard covering for it — and no existing test exercised one-fill-plus-N-rejects
    through `_resume`, so the suite would very plausibly have stayed green.

    So the grain here is deliberate and must stay symbol-level: A SYMBOL WITH A CONFIRMED FILL IS
    DONE, whatever else was tried on it at later slots. `terminal_fill` is tested BEFORE
    `terminal_reject` for exactly that reason — the AEM shape is one fill and five rejects on one
    symbol, and the fill wins.

    Behaviour is byte-identical to the inline version it replaces; `max_attempts` is passed rather
    than imported so the constant keeps one home.
    """
    results: dict[str, list[bool]] = {}
    terminal_fill: set[str] = set()
    terminal_reject: set[str] = set()
    suppressed: set[str] = set()
    for r in rows:
        d = r.get("detail") or {}
        if r.get("session") != session or not r.get("symbol"):
            continue
        sym = r["symbol"]
        phase = d.get("phase")
        if phase == "result":
            results.setdefault(sym, []).append(bool(d.get("ok")))
        elif phase == "terminal":
            (terminal_fill if d.get("ok") else terminal_reject).add(sym)
        if d.get("suppressed"):
            suppressed.add(sym)

    attempted: set[str] = set()
    for sym, oks in results.items():
        if sym in terminal_fill:
            attempted.add(sym)                          # confirmed filled -- done
        elif sym in terminal_reject:
            if sum(1 for ok in oks if not ok) + 1 >= max_attempts:
                attempted.add(sym)                      # confirmed rejected, bound exhausted
        elif len(oks) >= max_attempts:
            attempted.add(sym)                          # exhausted on submit-time failures
        elif any(oks):
            attempted.add(sym)                          # accepted, no answer yet -- leave it
    # A suppressed entry is terminal for this session: the runner declined to enter on a ruleset it
    # cannot manage the position with, so retrying would defeat the guard, and doing so after a
    # mid-session config fix would submit against a decision taken under the old configuration.
    return attempted | suppressed


def submitted_ok(rows: Iterable[Mapping], session: str, symbols: Iterable[str],
                 *, side: str | None = None) -> set[str]:
    """Symbols whose order for `session` was ACCEPTED at submit (`phase="result"`, `ok=True`) or
    confirmed FILLED (`phase="terminal"`, `ok=True`) — the ones a rotation may treat as funded.

    The complement of what `_resume` needs for #224: a sell that was refused N times and is now
    "attempted" by exhaustion has NOT gone out, and the entries it was meant to fund must keep
    waiting. `already_attempted` cannot say that — exhaustion and success both read as attempted —
    so this reads the positive fact directly. Pure, row-driven, side-aware like `attempts_for`; a
    row without a recorded side counts under any filter (same backward-compatibility reason).
    """
    wanted = set(symbols)
    out: set[str] = set()
    for r in rows:
        d = r.get("detail") or {}
        sym = r.get("symbol")
        if r.get("session") != session or sym not in wanted or not d.get("ok"):
            continue
        if d.get("phase") not in ("result", "terminal"):
            continue
        row_side = d.get("side")
        if side is not None and row_side is not None and row_side != side:
            continue
        out.add(sym)
    return out


def held_sessions_before(rows: Iterable[Mapping], session: str, refused: set[str]) -> int:
    """How many CONSECUTIVE prior sessions held their entries for (a superset of) `refused` (#224 B).

    Walks the distinct sessions in `rows` newest-first, strictly before `session`, and counts while
    each carries an "entries held" row whose `refused_exits` cover today's refused set; the first
    prior session without one ends the streak. Sessions, not slots: a lane with three slots a day
    that holds all three is ONE held session. A release row (`released_entries`) ends the streak
    too — the cap fired, the count restarts.

    Pure and row-driven like its siblings, so both drivers count the same way.
    """
    by_session: dict[str, tuple[bool, bool]] = {}
    for r in rows:
        s2 = r.get("session")
        if not s2 or s2 >= session:
            continue
        d = r.get("detail") or {}
        seen, held = by_session.get(s2, (False, False))
        if "released_entries" in d:
            by_session[s2] = (True, False)
            continue
        if "held_entries" in d and refused <= set((d.get("refused_exits") or {})):
            held = True
        by_session[s2] = (True, held or by_session.get(s2, (False, False))[1])
    streak = 0
    for s2 in sorted(by_session, reverse=True):
        _, held = by_session[s2]
        if not held:
            break
        streak += 1
    return streak
