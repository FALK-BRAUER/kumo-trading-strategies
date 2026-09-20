"""The emergency exit acts through a CALLBACK, and cannot report success it did not achieve (#147).

`MarketAware.emergency_exit` was a Protocol stub with nothing able to trigger it. The only
emergency signal in the system is `daily_loss` -> `Action.HALT`, which stops the lane DECIDING and
leaves positions held. Cockpit has a `LIQUIDATING` state with no programmatic setter and one
symbol-level path that sells AT THE NEXT SESSION. An emergency that must act now was expressible
in neither repo.

WHY A CALLBACK AND NOT AN EVENT — the design was nearly the other way round:

    AN EMERGENCY THAT IS EMITTED AND NOT RECEIVED IS INDISTINGUISHABLE FROM ONE THAT DID NOT
    HAPPEN.

A subscription adds a silent-failure mode to the single path that must never fail silently. The
precedent is `daily_loss.enforce(..., on_halt=...)`: strategies decides, the owner of the state
acts, and strategies never touches what it does not own.
"""

from __future__ import annotations

import asyncio

import pytest

from kumo_strategies.strategies.emergency import (
    ACCEPTED, REFUSED, UNREACHABLE, EmergencyOutcome, call_emergency_exit)


def _run(coro):
    return asyncio.run(coro)


def _verdict():
    from kumo_strategies.strategies.market_view import YES, Verdict
    return Verdict(YES, ("index fell 9% in a session",))


# -- the three answers, and two of them are not success ------------------------------------------

def test_AN_ACCEPTED_OUTCOME_CARRIES_THE_COUNT():
    """"The lane called for liquidation and the owner accepted N positions" must be ONE durable
    fact, not two systems each holding half."""
    async def hook(*, lane, verdict, reasons):
        return EmergencyOutcome(ACCEPTED, positions=7)

    out = _run(call_emergency_exit(hook, lane="TECHIVOL-005", verdict=_verdict()))
    assert out.acted is True and out.positions == 7


def test_ZERO_POSITIONS_IS_STILL_ACCEPTED():
    """The lane held nothing. That is a completed emergency exit, not a failure — and conflating
    them would make an empty book look like a broken hook."""
    async def hook(*, lane, verdict, reasons):
        return EmergencyOutcome(ACCEPTED, positions=0)

    assert _run(call_emergency_exit(hook, lane="L", verdict=_verdict())).acted is True


def test_A_REFUSAL_IS_NOT_SUCCESS_AND_MUST_SAY_WHY():
    """A hook that cannot express refusal will have refusal rendered as success by the first
    caller who forgets."""
    async def hook(*, lane, verdict, reasons):
        return EmergencyOutcome(REFUSED, reason="lifecycle is PAUSED; operator must clear it")

    out = _run(call_emergency_exit(hook, lane="L", verdict=_verdict()))
    assert out.acted is False and out.status == REFUSED and "PAUSED" in out.reason


def test_REFUSED_AND_UNREACHABLE_ARE_DIFFERENT_ANSWERS():
    """A refusal is a DECISION BY A SYSTEM THAT IS WORKING. An unreachable owner is a system that
    is not. They demand different responses, and collapsing them loses the distinction exactly
    when it matters most."""
    assert REFUSED != UNREACHABLE


def test_A_MISSING_HOOK_IS_UNREACHABLE_NOT_A_NO_OP():
    """`daily_loss`'s `on_halt` defaults to local-only because a local halt is still a halt. There
    is NO local liquidation, so there is no safe default — a lane whose owner wired nothing cannot
    liquidate, and reporting that as "nothing to do" is how an emergency exit becomes a function
    that returns quietly."""
    out = _run(call_emergency_exit(None, lane="L", verdict=_verdict()))
    assert out.status == UNREACHABLE and out.acted is False
    assert "nothing can liquidate" in out.reason


def test_A_HOOK_THAT_RAISES_IS_UNREACHABLE_NOT_REFUSED():
    async def hook(*, lane, verdict, reasons):
        raise ConnectionError("cockpit api is down")

    out = _run(call_emergency_exit(hook, lane="L", verdict=_verdict()))
    assert out.status == UNREACHABLE
    assert "ConnectionError" in out.reason and "cockpit api is down" in out.reason


def test_A_HOOK_RETURNING_NONE_IS_UNREACHABLE():
    """THE ONE EVERY PYTHON FUNCTION DOES BY ACCIDENT. A hook that forgets to return must not read
    as a completed liquidation — which is precisely what `if not result: ...` in the caller would
    have made of it."""
    async def hook(*, lane, verdict, reasons):
        return None

    out = _run(call_emergency_exit(hook, lane="L", verdict=_verdict()))
    assert out.status == UNREACHABLE and out.acted is False
    assert "None" in out.reason


def test_A_SYNCHRONOUS_HOOK_WORKS_TOO():
    """Cockpit's implementation may not be async. A shape mismatch must not become an emergency
    that silently did not happen."""
    def hook(*, lane, verdict, reasons):
        return EmergencyOutcome(ACCEPTED, positions=3)

    assert _run(call_emergency_exit(hook, lane="L", verdict=_verdict())).positions == 3


# -- the outcome cannot be constructed into a lie --------------------------------------------------

def test_AN_UNRECOGNISED_STATUS_IS_REFUSED_AT_CONSTRUCTION():
    """It would be treated as success by anything testing against ACCEPTED, which is the one
    direction this must never fail in."""
    with pytest.raises(ValueError, match="not one of"):
        EmergencyOutcome("probably_fine")


@pytest.mark.parametrize("status", [REFUSED, UNREACHABLE])
def test_A_NON_SUCCESS_OUTCOME_MUST_CARRY_A_REASON(status):
    """An emergency exit that did not happen, with no record of why, is what an operator reads at
    3am."""
    with pytest.raises(ValueError, match="must carry a reason"):
        EmergencyOutcome(status)


def test_acted_IS_THE_ONLY_THING_TO_BRANCH_ON():
    """`status == ACCEPTED` invites a caller to add `or UNREACHABLE` when the hook is flaky, which
    is how an unreachable owner comes to read as a completed liquidation. One derivation."""
    assert EmergencyOutcome(ACCEPTED).acted is True
    assert EmergencyOutcome(REFUSED, reason="no").acted is False
    assert EmergencyOutcome(UNREACHABLE, reason="no").acted is False


def test_NEGATIVE_POSITIONS_ARE_REFUSED():
    with pytest.raises(ValueError, match="negative"):
        EmergencyOutcome(ACCEPTED, positions=-1)


# -- the hook is passed what it needs and nothing it must guess at ---------------------------------

def test_THE_HOOK_RECEIVES_LANE_VERDICT_AND_REASONS():
    """Reasons IN FULL, not summarised: the reason a lane called an emergency is the first thing
    asked afterwards and the first thing lost. QC27's inverted valve went unexplained for months
    because it recorded that it fired and not why."""
    seen = {}

    async def hook(*, lane, verdict, reasons):
        seen.update(lane=lane, verdict=verdict, reasons=reasons)
        return EmergencyOutcome(ACCEPTED)

    v = _verdict()
    _run(call_emergency_exit(hook, lane="QC345-003", verdict=v,
                             reasons=("index -9% in a session", "breadth 4%")))
    assert seen["lane"] == "QC345-003"
    assert seen["verdict"] is v
    assert seen["reasons"] == ("index -9% in a session", "breadth 4%")
