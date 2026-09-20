"""A lane that concludes its edge is gone must have something to SAY (#147, #873).

The operator's three conditions, in his words:

    emergency_exit    "market explodes, get out"     LIQUIDATE   — act NOW
    entries_blocked   "today is not the day"         EXIT_ONLY   — bear, stop adding
    self_assessment   "I'm a loser, help me"         STAND_DOWN  — the lane says it is not working

THE THIRD IS THE INVERSE OF THE OTHER TWO. For those, the action existed and no trigger did. Here
the trigger and the evidence discipline were built first — #147's envelopes place a lane's live
window in its own pre-registered distribution, and `Assessment` already carried `state`, `reasons`,
`evidence_sufficient` and `acts` — and THE ACTION DID NOT EXIST. A lane could already conclude "I
am outside my own envelope and I have enough independent windows to be sure" and had nothing to
say next. `acts` returned a bool into the void.
"""

from __future__ import annotations

import inspect

import pytest

from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, UNKNOWN, Assessment, MarketAction, MarketViewConfig,
    MarketSignal, SelfAction)


def test_A_LANE_OUT_OF_ITS_ENVELOPE_WITH_ENOUGH_EVIDENCE_STANDS_DOWN():
    a = Assessment(OUT_OF_ENVELOPE, ("outside on return_pct",), evidence_sufficient=True)
    assert a.acts is True
    assert a.action is SelfAction.STAND_DOWN


@pytest.mark.parametrize("state,evidence", [
    (OUT_OF_ENVELOPE, False),   # one bad window is not a verdict
    (IN_ENVELOPE, True),
    (UNKNOWN, True),            # the common case for months on a new lane
])
def test_IT_DOES_NOT_ACT_OTHERWISE(state, evidence):
    """`None`, not a default action, so a caller cannot stand a lane down on an UNKNOWN or on a
    single bad window. UNKNOWN especially: `Envelope.min_live_sessions` guarantees it is where
    every newly registered lane sits, and standing those down would quarantine every new lane."""
    a = Assessment(state, ("r",), evidence_sufficient=evidence)
    assert a.acts is False
    assert a.action is None


def test_action_AND_acts_CANNOT_DISAGREE():
    """Two derivations of one fact are a detector, and here they must always agree — `action` is
    defined FROM `acts` rather than beside it. If they are ever computed separately, this fails."""
    for state in (OUT_OF_ENVELOPE, IN_ENVELOPE, UNKNOWN):
        for evidence in (True, False):
            a = Assessment(state, ("r",), evidence_sufficient=evidence)
            assert (a.action is not None) == a.acts, (state, evidence)


# -- the type separation, which was a deliberate departure from the brief ------------------------

def test_STAND_DOWN_IS_NOT_A_MarketAction():
    """It was asked for as a third member of `MarketAction`. It should not be one.

    `MarketAction` answers "what do I do when MY MARKET is bad"; `SelfAction` answers "what do I do
    when I AM bad". One enum spanning both makes `MarketViewConfig(action=STAND_DOWN)`
    constructible — a market view that quarantines, which means nothing — and that would then need
    a runtime guard for something a type prevents outright.
    """
    assert "STAND_DOWN" not in {m.name for m in MarketAction}
    assert {m.name for m in MarketAction} == {"EXIT_ONLY", "LIQUIDATE"}


def test_A_MARKET_VIEW_CANNOT_BE_CONFIGURED_TO_STAND_A_LANE_DOWN():
    """The consequence of the separation, asserted rather than assumed. A market view declaring
    STAND_DOWN would be a market signal delivering a verdict about the lane's own edge."""
    with pytest.raises((ValueError, TypeError)):
        MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                         action=SelfAction.STAND_DOWN).blocks_entries  # type: ignore[arg-type]


# -- the three mechanisms are different claims ----------------------------------------------------

# NOTE — THREE TESTS WERE DELETED HERE, and the reason is worth keeping.
#
# They asserted that `SelfAction.STAND_DOWN.__doc__` distinguished STAND_DOWN from EXIT_ONLY and
# HALT, said exits keep working, and said it never self-clears. Python does not build those
# strings: a string literal after an enum member assignment is source text, not `__doc__`, so
# `STAND_DOWN.__doc__` returns the CLASS docstring and the assertions were checking prose that
# does not exist at runtime.
#
# Rewriting them to read the source would make them substring-assertions against source text,
# which this repo guards against for good reason. So they are gone: they only ever asserted that
# I had written certain words, and the distinction they were guarding is documentation, not
# behaviour. When something CONSUMES `SelfAction` there will be behaviour to assert instead — that
# exits still run under STAND_DOWN, that it does not clear itself — and that is where it belongs.


# -- the naming collision, decided ------------------------------------------------------------------

def test_IT_IS_NOT_CALLED_QUARANTINE():
    """kumo-trading-platform already uses "quarantine" for the #79 quarantine PLANE — broker activity it did
    not originate, rendered as "UNCLAIMED · foreign strategy". That is about positions whose OWNER
    IS UNKNOWN; this is a lane reporting on ITSELF, with a known owner.

    One word for two unrelated things across two repos is a collision that costs somebody a day,
    and the repo that has not shipped the word is the one that should move. This one had not.
    """
    names = {m.name for m in SelfAction}
    assert "QUARANTINE" not in names, (
        "STAND_DOWN was renamed to QUARANTINE, which collides with cockpit's #79 plane for foreign "
        "positions — a different subject entirely")
    assert "STAND_DOWN" in names
