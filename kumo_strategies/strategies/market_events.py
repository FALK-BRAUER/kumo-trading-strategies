"""What the platform NOTIFIES and LABELS when a lane's market view changes (#147, platform issue 873).

The operator's requirement is a programme, not an interface: every strategy implements the view, it wires
into backtests, it is measured, and **it sends a notification when any event fires, with labels in
the UI**. This module is the event vocabulary and the payload contract — the parts that do not
depend on which backtest loop is running, so they can be built while #120 phase 2 unifies the loop.

THE LOAD-BEARING RULE: AN EVENT IS A TRANSITION, NOT A STATE
-------------------------------------------------------------
The platform polls. A lane that is RISK_OFF is RISK_OFF at every poll, so notifying on STATE means
a message per lane per poll — and a channel that sends a message every five minutes is a channel
nobody reads, which is strictly worse than no channel because it looks like coverage. Events fire
on CHANGE, and `transition()` is the only way to make one.

The same rule is why the previous state is part of the payload rather than implied. "TECHIVOL is
blocking entries" does not tell an operator whether something just happened; "TECHIVOL RISK_ON ->
RISK_OFF" does.

UNKNOWN IS NOT AN ALARM, AND IT IS NOT SILENCE EITHER
------------------------------------------------------
A newly registered lane answers UNKNOWN for WEEKS — `Envelope.min_live_sessions` is 13 sessions,
about 2.6 weeks —
so paging on UNKNOWN would page continuously on every new lane. But a lane that FALLS INTO unknown
having previously been computable has lost a signal the operator believes is protecting them, and
that is worth exactly one notification.

So UNKNOWN's severity depends on where it came from, which only a transition can express. This is
the third place in this package where three states beat two, and the reason is the same each time:
"could not compute" is a real answer, distinct from both yes and no.

SEVERITY IS DECLARED PER TRANSITION, NOT DERIVED FROM THE STATE
----------------------------------------------------------------
`RISK_OFF` under `EXIT_ONLY` blocks entries; under `LIQUIDATE` it sells the book. Same state, very
different thing to wake someone for. Severity therefore reads the ACTION as well as the state —
which is rule 3 of the protocol ("blocking entries and liquidating are different actions") arriving
in the notification layer, where collapsing them would page identically for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, RISK_OFF, RISK_ON, UNKNOWN, MarketAction)

__all__ = ["EventKind", "Severity", "MarketEvent", "transition", "LABELS", "SEVERITY"]


class EventKind(str, Enum):
    """What happened. One member per thing an operator would act on differently."""

    ENTRIES_BLOCKED = "entries_blocked"
    ENTRIES_UNBLOCKED = "entries_unblocked"
    EMERGENCY_EXIT = "emergency_exit"
    EMERGENCY_CLEARED = "emergency_cleared"
    VIEW_UNREADABLE = "view_unreadable"
    VIEW_RECOVERED = "view_recovered"
    OUT_OF_ENVELOPE_OBSERVED = "out_of_envelope_observed"
    OUT_OF_ENVELOPE_CONFIRMED = "out_of_envelope_confirmed"
    BACK_IN_ENVELOPE = "back_in_envelope"


class Severity(str, Enum):
    """How loudly. `INFO` is journalled and shown; `WARN` notifies; `CRITICAL` pages.

    Three, and deliberately not two: an operator who cannot distinguish "worth knowing at the end of
    the day" from "look now" ends up treating both as neither.
    """

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


#: UI labels. SHORT, and they say what is TRUE OF THE LANE rather than what the platform did — a
#: badge reading "blocked" next to a lane is ambiguous about who blocked whom, and an operator
#: reading a dashboard at speed resolves that ambiguity wrongly about half the time.
LABELS: dict[EventKind, str] = {
    EventKind.ENTRIES_BLOCKED: "not opening new positions",
    EventKind.ENTRIES_UNBLOCKED: "opening positions again",
    EventKind.EMERGENCY_EXIT: "closing its book",
    EventKind.EMERGENCY_CLEARED: "no longer closing its book",
    EventKind.VIEW_UNREADABLE: "market view unreadable",
    EventKind.VIEW_RECOVERED: "market view readable again",
    EventKind.OUT_OF_ENVELOPE_OBSERVED: "outside its own envelope (one window)",
    EventKind.OUT_OF_ENVELOPE_CONFIRMED: "outside its own envelope (confirmed)",
    EventKind.BACK_IN_ENVELOPE: "back inside its own envelope",
}


#: Payload keys the event owns. `detail` is spread over the top, so anything here must be refused
#: rather than silently overwritten.
_RESERVED = frozenset({"lane", "kind", "severity", "label", "previous", "current", "reasons",
                       "action"})


@dataclass(frozen=True)
class MarketEvent:
    """One transition, with everything needed to notify and to explain it afterwards.

    FROZEN and carrying BOTH states: a notification that says only where a lane ended up cannot tell
    an operator whether anything happened. `previous` is what makes this an event rather than a
    reading.
    """

    lane: str
    kind: EventKind
    severity: Severity
    previous: str
    current: str
    reasons: tuple[str, ...] = ()
    action: MarketAction | None = None
    detail: dict[str, object] = field(default_factory=dict)

    #: The one kind whose state legitimately does not change: a lane already OUT_OF_ENVELOPE whose
    #: EVIDENCE crossed `min_windows_to_act`.
    _MAY_REPEAT = frozenset({EventKind.OUT_OF_ENVELOPE_CONFIRMED})

    def __post_init__(self) -> None:
        """Refuse a non-event.

        `transition()`'s docstring used to say it was "the only way to build a MarketEvent". IT WAS
        NOT — this is a dataclass, and a direct construction with `previous == current` produced
        exactly the per-poll notification the whole module exists to prevent. A claim about a
        constraint, in a module whose subject is claims that outlive their mechanism, with no
        mechanism. Enforced now rather than described.
        """
        if self.previous == self.current and self.kind not in self._MAY_REPEAT:
            raise ValueError(
                f"{self.lane}: a {self.kind.value} event with previous == current ({self.current!r}) "
                f"is a STATE, not a transition. The platform polls, so one of these per poll is one "
                f"notification per poll. Build events with `transition()`, which returns None when "
                f"nothing changed.")

    @property
    def label(self) -> str:
        return LABELS[self.kind]

    def payload(self) -> dict[str, object]:
        """The notification body. FLAT and JSON-safe, because it crosses a process boundary.

        `reasons` is included in full rather than summarised. The reason a lane de-risked is the
        first thing asked afterwards and the first thing lost — QC27's inverted valve went
        unexplained for months because the valve recorded that it fired and not why.
        """
        reserved = _RESERVED & set(self.detail)
        if reserved:
            raise ValueError(
                f"{self.lane}: detail keys {sorted(reserved)} collide with the event's own fields. "
                f"They were being spread over the top, so a detail key named 'lane' silently "
                f"re-attributed the notification to another strategy — and a payload that names the "
                f"wrong lane is worse than no payload, because it is actionable and wrong.")
        return {
            "lane": self.lane,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "label": self.label,
            "previous": self.previous,
            "current": self.current,
            "reasons": list(self.reasons),
            "action": self.action.value if self.action is not None else None,
            **self.detail,
        }


def _market_kind(previous: str, current: str, action: MarketAction | None) -> EventKind | None:
    if previous == current:
        return None
    if current == RISK_OFF:
        return (EventKind.EMERGENCY_EXIT if action is MarketAction.LIQUIDATE
                else EventKind.ENTRIES_BLOCKED)
    if current == UNKNOWN:
        return EventKind.VIEW_UNREADABLE
    if current == RISK_ON:
        # FROM UNKNOWN IS A RECOVERY, NOT AN UNBLOCK. Nothing was blocked — the view could not be
        # computed — so reporting "opening positions again" would tell an operator the lane had
        # been stopped when it never was.
        if previous == UNKNOWN:
            return EventKind.VIEW_RECOVERED
        # LEAVING A LIQUIDATION IS NOT "OPENING POSITIONS AGAIN". The action matters on the way
        # OUT exactly as it does on the way in — and this was an ASYMMETRY IN MY OWN DESIGN: the
        # entry into RISK_OFF was split by action (ENTRIES_BLOCKED vs EMERGENCY_EXIT) and the exit
        # was not, so a lane that STOPPED EMPTYING ITS BOOK reported the same INFO notification as
        # one that merely resumed buying.
        #
        # Found by kumo-trading-platform driving this function from their mirror, table-driven, one row per
        # cockpit event — a second implementation asking the same question of the first, which is
        # the disagreement rule working across a repo boundary rather than inside one.
        if action is MarketAction.LIQUIDATE:
            return EventKind.EMERGENCY_CLEARED
        return EventKind.ENTRIES_UNBLOCKED
    return None


def _envelope_kind(previous: str, current: str, *, acts: bool) -> EventKind | None:
    if current == OUT_OF_ENVELOPE:
        if previous == OUT_OF_ENVELOPE:
            # The state did not change but the EVIDENCE did: one bad window became enough to act.
            # That IS a transition an operator must hear about, and it is the only case where a
            # repeated state legitimately fires.
            return EventKind.OUT_OF_ENVELOPE_CONFIRMED if acts else None
        return (EventKind.OUT_OF_ENVELOPE_CONFIRMED if acts
                else EventKind.OUT_OF_ENVELOPE_OBSERVED)
    if previous == current:
        return None
    if current == IN_ENVELOPE:
        return EventKind.BACK_IN_ENVELOPE if previous == OUT_OF_ENVELOPE else None
    if current == UNKNOWN:
        return None                       # a lane with too little evidence is not an event
    return None


#: PUBLIC, because cockpit routes on it. It was `_SEVERITY` and the tests reached through the
#: underscore to check it — a private name that three call sites depend on is public with extra
#: steps, and the underscore only discourages the people who would have read the contract.
SEVERITY: dict[EventKind, Severity] = {
    EventKind.EMERGENCY_EXIT: Severity.CRITICAL,
    #: WARN, not INFO. The END of a liquidation is a material state change on a lane that was
    #: emptying its book — not the same budget line as a lane that merely resumed buying.
    EventKind.EMERGENCY_CLEARED: Severity.WARN,
    EventKind.ENTRIES_BLOCKED: Severity.WARN,
    EventKind.ENTRIES_UNBLOCKED: Severity.INFO,
    EventKind.VIEW_UNREADABLE: Severity.WARN,
    EventKind.VIEW_RECOVERED: Severity.INFO,
    EventKind.OUT_OF_ENVELOPE_OBSERVED: Severity.INFO,
    EventKind.OUT_OF_ENVELOPE_CONFIRMED: Severity.WARN,
    EventKind.BACK_IN_ENVELOPE: Severity.INFO,
}


def transition(lane: str, *, previous: str, current: str, reasons: tuple[str, ...] = (),
               action: MarketAction | None = None, acts: bool = False,
               envelope: bool = False, detail: dict[str, object] | None = None
               ) -> MarketEvent | None:
    """The event for a state change, or None when nothing an operator needs happened.

    THE WAY TO BUILD A `MarketEvent`. Returning None for "no change" makes the quiet case the
    default rather than something every call site has to remember — the call site that forgets is
    the one nobody reviews.

    Direct construction is not forbidden (this is a dataclass, and pretending otherwise was the
    original sin here) but it is CONSTRAINED: `MarketEvent.__post_init__` refuses
    `previous == current` for every kind except OUT_OF_ENVELOPE_CONFIRMED, so the per-poll shape
    cannot be built by either route.

    `previous` has no default either. A caller with no prior state has not got one to compare
    against and must say so explicitly by passing UNKNOWN — defaulting it would turn every lane's
    first poll into an event.
    """
    kind = (_envelope_kind(previous, current, acts=acts) if envelope
            else _market_kind(previous, current, action))
    if kind is None:
        return None
    return MarketEvent(lane=lane, kind=kind, severity=SEVERITY[kind], previous=previous,
                       current=current, reasons=reasons, action=action, detail=detail or {})
