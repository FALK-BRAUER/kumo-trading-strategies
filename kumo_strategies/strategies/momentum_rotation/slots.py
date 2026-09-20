"""Decision slots — WHEN a session decides, as named anchors rather than a timer (#29).

MOMENTUM-002 decides once, at open+5. Variant A asks whether deciding 2-4 times a session is worth
it. The parameterisation is not incidental, because the live journal keys a decision on the session
date and enforces one decision per session (`pgjournal.py:51`). Cadence changes the IDENTITY of a
decision, so the shape of that identity is the design.

ANCHORS, NOT AN INTERVAL, and the reason is idempotency. A named slot yields a stable dedup key —
`2026-08-14/close-30m`. A free-running interval yields a wall-clock timestamp subject to scheduler
jitter, restarts and retries: on a retry there is no way to tell "the 12:00 decision again" from "a
new decision at 12:03". That is the #197 identity-split class, where the journal wrote under one id
while positions sat under another. A slot name makes the retry safe by construction.

Three further reasons, all measured rather than assumed:

  cost is time-shaped   Half-spreads are 3.59bps a side at 09:35 and 1.36 by 15:55 (`costs.py`). An
                        interval landing on the open pays ~2.6x one landing late. Anchors choose the
                        window; an interval lets the clock choose it.
  half-days             `runtime/calendar.py` models the 13:00 ET close. `close-30m` is well defined
                        there. An interval either overruns the close or truncates unevenly, silently
                        changing the NUMBER of decisions on those sessions — an uncontrolled
                        variable inside the sweep that is measuring cadence.
  the actual failure    2026-08-13: four miners opened near their highs and closed near their lows,
                        and the next look came after the gap. What was needed was a look BEFORE the
                        close. `every 120m` gets there only by luck.

THE COUNTER-ARGUMENT IS REAL: chosen times are free parameters, and three of them fitted on ~130
sessions is exactly what #14 objects to. So `every()` exists to sweep a SINGLE parameter — does
checking more often pay at all, net of time-of-day cost — and only if that is positive does anyone
pre-register a small set of anchors justified by mechanism. Research sweeps intervals, live sets
anchors, and both resolve through this one module so what is measured is what ships.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

#: The live default: one slot, five minutes after the open. Expresses MOMENTUM-002 exactly, so the
#: control is a one-element list rather than a special case.
DEFAULT_SLOTS: tuple[str, ...] = ("open+5m",)

_OFFSET = re.compile(r"^(open|close)([+-])(\d+)m$")
_CLOCK = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class SlotError(ValueError):
    """A slot spec that cannot be resolved. Raised at parse time, never silently dropped."""


def _parse(spec: str) -> tuple[str, int] | time:
    s = spec.strip().lower()
    m = _OFFSET.match(s)
    if m:
        anchor, sign, mins = m.group(1), m.group(2), int(m.group(3))
        return anchor, (mins if sign == "+" else -mins)
    m = _CLOCK.match(s)
    if m:
        return time(int(m.group(1)), int(m.group(2)))
    raise SlotError(
        f"unrecognised slot {spec!r}. Use 'open+5m', 'close-30m' or a wall clock 'HH:MM' "
        "in exchange local time.")


def validate(specs: tuple[str, ...]) -> None:
    """Parse every spec, raising on the first bad one.

    Called at config construction rather than at decision time. A slot that cannot be resolved must
    not become a session that silently decides fewer times than configured — that is the same
    silent-degradation shape as the inert config flags in #26.
    """
    if not specs:
        raise SlotError("at least one decision slot is required")
    for s in specs:
        _parse(s)
    if len(set(s.strip().lower() for s in specs)) != len(specs):
        raise SlotError(f"duplicate slots in {specs!r} — each slot name is an idempotency key")


def resolve(specs: tuple[str, ...], open_at: datetime, close_at: datetime, *,
            allow_pre_open: bool = False) -> list[tuple[str, datetime]]:
    """`(slot_name, when)` pairs for one session, sorted, clipped to the session.

    Clipping rather than dropping: on a half day `close-30m` is still meaningful, and `12:00` is
    after the 13:00 close only in the sense that it is late in a short session. What must NOT happen
    is a decision resolving past the close, which would fire into a shut venue — the failure
    `runtime/calendar.py` exists to prevent.

    A slot resolving before the open is clipped forward to the open UNLESS the caller asks for
    pre-open scheduling. A slot resolving after the close is DROPPED, not clipped back, because
    clipping would collapse two slots onto one time and two decisions would then share an
    idempotency key.

    `allow_pre_open` EXISTS BECAUSE THE CLAMP MADE AN AUCTION ORDER INEXPRESSIBLE (#131). CRSISHORT's
    entry is a limit-on-open and IB accepts an OPG order until roughly 09:28 ET, so the DECISION has
    to happen before the open. The signal permits it — `apply_gates` computes the entry limit from
    the PRIOR close, so nothing about the decision needs the session it trades into — but
    `open-10m` silently resolved to the open itself, the OPG order went out at 09:30, and the venue
    refused or converted it while every check upstream passed.

    THE CLAMP IS NOT A BUG AND IS NOT BEING REMOVED. It is correct for every lane that existed when
    it was written: a rotation lane cannot trade pre-open, and a mistyped `open-10m` scheduling a
    decision into a shut venue is a real hazard. What was wrong is that it applied SILENTLY to a
    caller that meant it. So the default is False — every existing caller is byte-identical — and a
    lane that genuinely needs the auction has to say so.

    KEYWORD-ONLY, so it can never be passed positionally where `close_at` belongs.

    THREE CALLERS, TWO PLUMBED, ONE DELIBERATELY NOT. `calendar.next_slot_fire` and
    `calendar.elapsed_slots` both take the flag — both or neither, or they would disagree about when
    one slot fires. `backtesting/runner_cadence` and `backtesting/runner_sessions` call this bare and
    stay bare: nothing sweeps a pre-open slot today, and a parameter no caller passes is how an
    option acquires an untested value. If CRSISHORT is ever swept pre-open in research they must be
    plumbed THEN, because a cadence measured under a different clamp than live is unverifiable
    against it — the same argument as `next_slot_fire`'s, one layer up.

    THE FLAG COVERS THE WALL-CLOCK BRANCH TOO. `_parse` accepts a bare `HH:MM`, and `09:20` is
    pre-open on a normal session while not being expressible as an offset, so it reaches this clamp
    by the other route. The clamp sits after both branches converge, which is why — and a test
    pins it, because "happens to" is how a refactor that moves the clamp into the offset branch
    passes review.
    """
    out: list[tuple[str, datetime]] = []
    for spec in specs:
        p = _parse(spec)
        if isinstance(p, time):
            when = datetime.combine(open_at.date(), p, tzinfo=open_at.tzinfo)
        else:
            anchor, mins = p
            base = open_at if anchor == "open" else close_at
            when = base + timedelta(minutes=mins)
        if when < open_at and not allow_pre_open:
            when = open_at
        if when > close_at:
            continue
        out.append((spec.strip().lower(), when))
    # Sort by time, then by name so two slots resolving to the same instant have a stable order.
    out.sort(key=lambda x: (x[1], x[0]))
    return out


def has_pre_open(specs) -> bool:
    """Does any spec in `specs` resolve BEFORE the open?

    DERIVED, NEVER FLAGGED, and that is the point. `allow_pre_open` must reach `next_slot_fire` AND
    `elapsed_slots` or NEITHER — they call the same resolver and a lane that arms on a pre-open slot
    while its missed-slot detector clamps that slot forward reports a slot it never had. A boolean a
    caller passes is a boolean a caller forgets on one of the two call sites; asking the SLOTS
    settles both from one source.

    PARSED WITH THIS MODULE'S OWN GRAMMAR. A second definition of "before the open" is where two
    derivations disagree — `crsi_short._reaches_the_auction` records the first version of exactly
    that mistake, a hand-written name list that would have refused `open-10m`.

    A WALL CLOCK IS NOT PRE-OPEN HERE. `09:20` is before the open on an ordinary day and after it on
    a half-day with a shifted open, and this function has no calendar; treating it as pre-open would
    unclamp a slot on days where that is wrong. Only an explicit negative open-offset counts.
    """
    for spec in specs or ():
        try:
            parsed = _parse(spec)
        except SlotError:
            continue                       # `validate` refuses it at construction; not this one's job
        if isinstance(parsed, tuple) and parsed[0] == "open" and parsed[1] < 0:
            return True
    return False


def every(minutes: int, *, first: str = "open+5m", last_before_close: int = 5) -> tuple[str, ...]:
    """Generate anchors at a fixed interval — the RESEARCH form, one swept parameter.

    Deliberately produces anchor NAMES rather than a separate interval mechanism, so a swept arm and
    a live config travel the same code path. `every(390)` on a normal session yields a single slot
    and reproduces MOMENTUM-002.

    Expressed as offsets from the OPEN, so a half day simply yields fewer slots rather than a
    differently-shaped grid. `last_before_close` keeps the final slot off the closing auction, where
    the measured spread widens again (1.81bps at 15:55 against a 1.36 low).
    """
    if minutes <= 0:
        raise SlotError("interval must be positive")
    first_off = _parse(first)
    if not isinstance(first_off, tuple) or first_off[0] != "open":
        raise SlotError(f"`first` must be an open-relative offset, got {first!r}")
    start = first_off[1]
    # 390 minutes is a normal RTH session. The generator does not know the calendar, so it emits the
    # full-session grid and `resolve` drops what falls past an early close.
    return tuple(f"open+{m}m" for m in range(start, 390 - last_before_close + 1, minutes))


def session_key(session: date, slot: str) -> str:
    """The idempotency key for one decision. `2026-08-14/close-30m`.

    This is the whole point of naming slots. The live journal currently enforces one decision per
    session; under cadence the key must become (session, slot), and a NAME is stable across a retry
    in a way a timestamp is not.
    """
    return f"{session.isoformat()}/{slot}"
