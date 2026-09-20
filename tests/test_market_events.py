"""Notifications fire on TRANSITIONS, not on states (#147, platform issue 873).

The operator's requirement is a notification when any event fires, plus UI labels. The failure mode that
makes such a channel worthless is not missing a message — it is sending one per poll, because a
channel that speaks every five minutes is one nobody reads, and that is strictly worse than no
channel because it looks like coverage.

So the property under test throughout is: **the same reading twice produces one event, not two.**
"""

from __future__ import annotations

import inspect

import pytest

from kumo_strategies.strategies import market_events as me
from kumo_strategies.strategies.market_events import (
    LABELS, EventKind, Severity, transition)
from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, RISK_OFF, RISK_ON, UNKNOWN, MarketAction)


# -- the load-bearing rule ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", [RISK_ON, RISK_OFF, UNKNOWN])
def test_AN_UNCHANGED_STATE_IS_NOT_AN_EVENT(state):
    """The platform polls. Notifying on STATE means a message per lane per poll."""
    assert transition("TECHIVOL-005", previous=state, current=state,
                      action=MarketAction.EXIT_ONLY) is None


def test_A_LANE_SITTING_RISK_OFF_ALL_DAY_NOTIFIES_ONCE():
    """The whole point, as a sequence rather than a single call."""
    readings = [RISK_ON] + [RISK_OFF] * 40
    events, prev = [], readings[0]
    for r in readings:
        e = transition("TECHIVOL-005", previous=prev, current=r, action=MarketAction.EXIT_ONLY)
        if e:
            events.append(e)
        prev = r
    assert len(events) == 1, f"{len(events)} notifications for one de-risking event"
    assert events[0].kind is EventKind.ENTRIES_BLOCKED


def test_THERE_IS_NO_WAY_TO_BUILD_AN_EVENT_WITHOUT_A_PREVIOUS_STATE():
    """`previous` has no default. A caller with no prior state must say UNKNOWN explicitly —
    defaulting it would make every lane's first poll an event, which is the per-poll failure
    arriving by a different door."""
    params = inspect.signature(transition).parameters
    assert params["previous"].default is inspect.Parameter.empty, (
        "previous has a default — every first poll becomes an event")


def test_THE_PAYLOAD_CARRIES_BOTH_STATES():
    """"TECHIVOL is blocking entries" does not tell an operator whether something just happened."""
    e = transition("TECHIVOL-005", previous=RISK_ON, current=RISK_OFF,
                   action=MarketAction.EXIT_ONLY, reasons=("index below its 50d average",))
    p = e.payload()
    assert p["previous"] == RISK_ON and p["current"] == RISK_OFF


# -- rule 3 reaches the notification layer -------------------------------------------------------------

def test_LIQUIDATE_AND_EXIT_ONLY_DO_NOT_PAGE_THE_SAME():
    """Same state, very different thing to wake someone for. Collapsing them here would undo rule 3
    of the protocol at the last step, where it is most visible to a human."""
    block = transition("A", previous=RISK_ON, current=RISK_OFF, action=MarketAction.EXIT_ONLY)
    liq = transition("A", previous=RISK_ON, current=RISK_OFF, action=MarketAction.LIQUIDATE)
    assert block.kind is EventKind.ENTRIES_BLOCKED
    assert liq.kind is EventKind.EMERGENCY_EXIT
    assert liq.severity is Severity.CRITICAL
    assert block.severity is Severity.WARN
    assert liq.severity != block.severity, "selling the book pages the same as declining to buy"


# -- UNKNOWN is neither an alarm nor silence -----------------------------------------------------------

def test_FALLING_INTO_UNKNOWN_IS_WORTH_ONE_NOTIFICATION():
    """A lane that loses a signal the operator believes is protecting them has lost something."""
    e = transition("A", previous=RISK_ON, current=UNKNOWN)
    assert e.kind is EventKind.VIEW_UNREADABLE
    assert e.severity is Severity.WARN


def test_A_NEW_LANE_SITTING_IN_UNKNOWN_FOR_MONTHS_NEVER_NOTIFIES():
    """`Envelope.min_live_sessions` guarantees this is the common case for a newly registered lane.
    Paging on it would page continuously on every new lane, which trains an operator to ignore the
    channel that also carries EMERGENCY_EXIT."""
    prev, events = UNKNOWN, []
    for _ in range(200):
        e = transition("NEW-009", previous=prev, current=UNKNOWN)
        if e:
            events.append(e)
    assert events == [], f"{len(events)} notifications from a lane that never became computable"


def test_RECOVERING_FROM_UNKNOWN_IS_NOT_AN_UNBLOCK():
    """Nothing was blocked — the view could not be computed. Reporting "opening positions again"
    would tell an operator the lane had been stopped when it never was."""
    e = transition("A", previous=UNKNOWN, current=RISK_ON)
    assert e.kind is EventKind.VIEW_RECOVERED
    assert e.kind is not EventKind.ENTRIES_UNBLOCKED
    assert e.severity is Severity.INFO


def test_RECOVERING_FROM_RISK_OFF_IS_AN_UNBLOCK():
    e = transition("A", previous=RISK_OFF, current=RISK_ON)
    assert e.kind is EventKind.ENTRIES_UNBLOCKED


# -- the envelope hook, whose evidence can change without its state changing ---------------------------

def test_ONE_BAD_WINDOW_IS_OBSERVED_AND_DOES_NOT_WARN():
    """`Assessment.acts` is False until enough independent windows agree. The observation is still
    worth showing — it just is not worth waking anyone for."""
    e = transition("A", previous=IN_ENVELOPE, current=OUT_OF_ENVELOPE, envelope=True, acts=False)
    assert e.kind is EventKind.OUT_OF_ENVELOPE_OBSERVED
    assert e.severity is Severity.INFO


def test_EVIDENCE_BECOMING_SUFFICIENT_FIRES_THOUGH_THE_STATE_DID_NOT_CHANGE():
    """THE ONE CASE WHERE A REPEATED STATE LEGITIMATELY FIRES, and the reason `acts` is an input.

    A lane sits OUT_OF_ENVELOPE for three windows. The state is identical throughout; what changed
    is that the evidence crossed `min_windows_to_act`. A transition rule keyed only on state would
    stay silent exactly when the lane became actionable.
    """
    e = transition("A", previous=OUT_OF_ENVELOPE, current=OUT_OF_ENVELOPE, envelope=True, acts=True)
    assert e is not None, "the lane became actionable and nothing fired"
    assert e.kind is EventKind.OUT_OF_ENVELOPE_CONFIRMED
    assert e.severity is Severity.WARN


def test_STILL_ONLY_OBSERVED_DOES_NOT_REFIRE():
    """The companion to the above: repeated insufficient evidence is still one observation."""
    assert transition("A", previous=OUT_OF_ENVELOPE, current=OUT_OF_ENVELOPE,
                      envelope=True, acts=False) is None


def test_AN_ENVELOPE_LANE_GOING_UNKNOWN_IS_NOT_AN_EVENT():
    """Unlike the market view. A lane whose evidence fell below `min_live_sessions` has not lost a
    protection — it never had one to lose, and this is the ordinary state after any refit."""
    assert transition("A", previous=IN_ENVELOPE, current=UNKNOWN, envelope=True) is None


def test_RETURNING_TO_THE_ENVELOPE_IS_REPORTED():
    e = transition("A", previous=OUT_OF_ENVELOPE, current=IN_ENVELOPE, envelope=True)
    assert e.kind is EventKind.BACK_IN_ENVELOPE


# -- the payload crosses a process boundary ------------------------------------------------------------

def test_THE_PAYLOAD_IS_JSON_SAFE():
    """It goes to Telegram and to a UI. An enum or a tuple that survives in-process and fails at the
    boundary is a notification that exists in testing and not in production."""
    import json
    e = transition("TECHIVOL-005", previous=RISK_ON, current=RISK_OFF,
                   action=MarketAction.LIQUIDATE, reasons=("a", "b"), detail={"window": 50})
    text = json.dumps(e.payload())
    back = json.loads(text)
    assert back["action"] == "liquidate" and back["reasons"] == ["a", "b"]
    assert back["window"] == 50


def test_THE_PAYLOAD_CARRIES_THE_REASONS_IN_FULL():
    """The first thing asked afterwards and the first thing lost. QC27's inverted valve went
    unexplained for months because it recorded that it fired and not why."""
    reasons = tuple(f"reason {i}" for i in range(6))
    e = transition("A", previous=RISK_ON, current=RISK_OFF, action=MarketAction.EXIT_ONLY,
                   reasons=reasons)
    assert tuple(e.payload()["reasons"]) == reasons


def test_EVERY_EVENT_KIND_HAS_A_LABEL_AND_A_SEVERITY():
    """DISCOVERED, NOT LISTED — #138's rule. Enumerates the ENUM and looks each member up, so a kind
    added tomorrow fails here rather than reaching a UI as a blank badge or a KeyError."""
    missing_label = [k.name for k in EventKind if k not in LABELS]
    missing_sev = [k.name for k in EventKind if k not in me.SEVERITY]
    assert not missing_label, f"no UI label: {missing_label}"
    assert not missing_sev, f"no severity: {missing_sev}"


def test_LABELS_SAY_WHAT_IS_TRUE_OF_THE_LANE():
    """A badge reading "blocked" is ambiguous about who blocked whom, and an operator reading a
    dashboard at speed resolves that wrongly about half the time."""
    for kind, text in LABELS.items():
        assert text == text.lower(), f"{kind.name}: {text!r} is not a lower-case phrase"
        assert len(text) <= 48, f"{kind.name}: {text!r} is too long for a badge"
        assert not text.endswith("."), f"{kind.name}: labels are phrases, not sentences"


def test_ONLY_EMERGENCY_EXIT_IS_CRITICAL():
    """Severity is a budget. If three things page, the operator mutes the channel and the one that
    mattered is muted with them."""
    critical = [k.name for k, s in me.SEVERITY.items() if s is Severity.CRITICAL]
    assert critical == [EventKind.EMERGENCY_EXIT.name], critical


# -- what the self-review found ------------------------------------------------------------------------

def test_A_MARKET_EVENT_CANNOT_BE_BUILT_WITH_previous_EQUAL_current():
    """`transition()`'s docstring claimed it was "the only way to build a MarketEvent". IT WAS NOT.

    This is a dataclass. Direct construction with `previous == current` produced exactly the
    per-poll notification the module exists to prevent — a claim about a constraint, with no
    mechanism, in a module whose whole subject is claims that outlive their mechanism. Third
    instance of that shape found in one day, and the first two were also mine.

    Enforced now instead of described.
    """
    with pytest.raises(ValueError, match="STATE, not a transition"):
        me.MarketEvent(lane="A", kind=EventKind.EMERGENCY_EXIT, severity=Severity.CRITICAL,
                       previous=RISK_OFF, current=RISK_OFF)


def test_THE_ONE_KIND_THAT_MAY_REPEAT_STILL_CAN():
    """The guard must not break the legitimate case: evidence crossing `min_windows_to_act` while
    the state stays OUT_OF_ENVELOPE. A guard that also blocks this would silence the lane exactly
    when it became actionable."""
    e = me.MarketEvent(lane="A", kind=EventKind.OUT_OF_ENVELOPE_CONFIRMED, severity=Severity.WARN,
                       previous=OUT_OF_ENVELOPE, current=OUT_OF_ENVELOPE)
    assert e.kind is EventKind.OUT_OF_ENVELOPE_CONFIRMED


def test_A_DETAIL_KEY_CANNOT_RE_ATTRIBUTE_THE_NOTIFICATION():
    """`detail` is spread over the payload, so a key named `lane` silently replaced the event's own
    lane — a notification naming the WRONG STRATEGY. Worse than no payload, because it is
    actionable and wrong. Found by self-review, not by a test that already existed."""
    e = transition("TECHIVOL-005", previous=RISK_ON, current=RISK_OFF,
                   action=MarketAction.EXIT_ONLY, detail={"lane": "QC345-003"})
    with pytest.raises(ValueError, match="collide"):
        e.payload()


@pytest.mark.parametrize("key", ["lane", "kind", "severity", "label", "previous", "current",
                                 "reasons", "action"])
def test_EVERY_FIELD_THE_EVENT_OWNS_IS_RESERVED(key):
    """Enumerated rather than spot-checked: `lane` was the one that mattered, and the next one to
    matter will be whichever is left out of a hand-written list."""
    e = transition("A", previous=RISK_ON, current=RISK_OFF, action=MarketAction.EXIT_ONLY,
                   detail={key: "hijacked"})
    with pytest.raises(ValueError, match="collide"):
        e.payload()


def test_AN_ORDINARY_DETAIL_KEY_STILL_WORKS():
    """The guard must not make `detail` useless — it carries the window, the threshold, the
    percentile."""
    e = transition("A", previous=RISK_ON, current=RISK_OFF, action=MarketAction.EXIT_ONLY,
                   detail={"window": 50, "percentile": 13.0})
    assert e.payload()["window"] == 50


def test_THE_SEVERITY_MAP_IS_PUBLIC():
    """Cockpit routes on it. A private name that three call sites depend on is public with extra
    steps, and the underscore only discourages the people who would have read the contract."""
    assert "SEVERITY" in me.__all__
    assert hasattr(me, "SEVERITY")


def test_LEAVING_A_LIQUIDATION_IS_NOT_OPENING_POSITIONS_AGAIN():
    """THE ASYMMETRY IN MY OWN DESIGN, found by cockpit driving this function from their mirror.

    The entry into RISK_OFF was split by action — ENTRIES_BLOCKED for EXIT_ONLY, EMERGENCY_EXIT
    for LIQUIDATE. The EXIT was not. So a lane that had been EMPTYING ITS BOOK and stopped
    reported "opening positions again" at INFO, the same notification as a lane that merely
    resumed buying after a bear signal cleared.

    The action matters on the way out exactly as it does on the way in.
    """
    out = transition("L", previous=RISK_OFF, current=RISK_ON, action=MarketAction.LIQUIDATE)
    assert out.kind is EventKind.EMERGENCY_CLEARED
    assert out.severity is Severity.WARN, "the end of a liquidation is not an INFO line"
    assert out.label == "no longer closing its book"


def test_LEAVING_A_BLOCK_IS_STILL_AN_UNBLOCK():
    """The other half. A fix that reported EMERGENCY_CLEARED for both would lose the distinction
    in the opposite direction."""
    out = transition("L", previous=RISK_OFF, current=RISK_ON, action=MarketAction.EXIT_ONLY)
    assert out.kind is EventKind.ENTRIES_UNBLOCKED
    assert out.severity is Severity.INFO


def test_THE_ENTRY_AND_EXIT_ARE_BOTH_SPLIT_BY_ACTION():
    """The symmetry stated as one assertion, so neither half can be collapsed without failing.

    Four transitions, four distinct kinds — two in, two out. Three would mean one direction has
    stopped distinguishing the actions.
    """
    kinds = {
        transition("L", previous=RISK_ON, current=RISK_OFF, action=a).kind
        for a in (MarketAction.EXIT_ONLY, MarketAction.LIQUIDATE)
    } | {
        transition("L", previous=RISK_OFF, current=RISK_ON, action=a).kind
        for a in (MarketAction.EXIT_ONLY, MarketAction.LIQUIDATE)
    }
    assert len(kinds) == 4, f"the two directions do not both distinguish the actions: {kinds}"


def test_STILL_ONLY_EMERGENCY_EXIT_PAGES():
    """Adding a kind must not spend the severity budget. EMERGENCY_CLEARED is WARN — worth
    knowing, not worth waking someone."""
    critical = [k.name for k, v in me.SEVERITY.items() if v is Severity.CRITICAL]
    assert critical == [EventKind.EMERGENCY_EXIT.name], critical
