"""The emergency exit: the lane decides, the OWNER of the positions acts (#147, platform issue 873).

THIS IS THE ONE MECHANISM THAT EXISTED NOWHERE. `MarketAware.emergency_exit` was a Protocol stub,
and nothing could trigger it. The only emergency signal in the system is `daily_loss`, which
produces `Action.HALT` — the lane stops DECIDING and its positions stay held. Cockpit has a
`LIQUIDATING` lifecycle state with no programmatic setter and one symbol-level path that sells AT
THE NEXT SESSION (verified by the coordinator on cockpit main). **An emergency that must act now
was expressible in neither repo.**

A CALLBACK FOR THE ACT, AN EVENT FOR THE NOTICE, ON SEPARATE PATHS
-------------------------------------------------------------------
An event alone was the obvious design and it is wrong: AN EMERGENCY THAT IS EMITTED AND NOT
RECEIVED IS INDISTINGUISHABLE FROM ONE THAT DID NOT HAPPEN. A subscription adds a silent-failure
mode to the single path that must never fail silently — the lane emits, believes it acted,
continues; nothing liquidates; every surface reads normal. That is the shape of the two detectors
that sat inert for two deploys because `Notifier.send` took an Alert and got an f-string, where
"no alarms" read as "nothing wrong".

The precedent is `daily_loss.enforce(..., on_halt=...)` — "an async hook the OWNER of the durable
state supplies". Strategies DECIDES; whoever owns the positions ACTS. That respects the venue rule
completely: this module never touches a broker, it calls a hook whose implementation belongs to
someone who can.

THE NOTICE MUST NEVER COST THE ACT, and this is the rule that cost two defects in one PR earlier
today — an observability addition destroying the thing it observes, twice, by different routes.
Here it would cost money rather than a journal row, so: act first, emit after, guard the emit, and
never let its failure propagate.

REFUSAL AND UNREACHABILITY ARE DIFFERENT ANSWERS AND NEITHER IS SUCCESS
------------------------------------------------------------------------
Three states, for the fourth time in this package. A hook that cannot express refusal will have
refusal rendered as success by the first caller who forgets — and "cockpit said no" and "cockpit
did not answer" demand different responses: the first is a decision by a system that is working,
the second is a system that is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EmergencyOutcome", "ACCEPTED", "PARTIAL", "REFUSED", "UNREACHABLE",
           "call_emergency_exit"]

#: The owner acted on the WHOLE book. `positions` says on how many — 0 is a legitimate ACCEPTED
#: (the lane held nothing) and is not the same as a refusal.
ACCEPTED = "accepted"
#: The owner acted on SOME of it and something remains. A fourth status rather than an ACCEPTED
#: carrying a `failed` list, because a caller must not have to REMEMBER to look: `acted` is False
#: here, so a partial emergency exit cannot be read as a handled one.
#:
#: Raised by kumo-trading-platform, whose `liquidate_lane` can submit some symbols and fail others. Their
#: proposal was ACCEPTED with `detail.failed`, which is honest and has the defect this module
#: exists to remove — the lane reads `acted=True` while positions remain.
PARTIAL = "partial"
#: The owner was reached and DECLINED. A working system said no, and it must say why.
REFUSED = "refused"
#: The owner could not be reached, or answered in a shape that cannot be read. NOT a refusal: a
#: refusal is evidence the other side is alive.
UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class EmergencyOutcome:
    """What the owner of the positions did about it.

    FROZEN and carrying the count, because "the lane called for liquidation and the owner accepted
    N positions" must be ONE durable fact rather than two systems each holding half of it.
    """

    status: str
    positions: int = 0
    """How many positions the owner SUBMITTED ORDERS FOR — not how many are flat.

    SUBMIT IS NOT HELD, and cockpit's response says submitted. `nautilus/broker.py` documents the
    same distinction for the ordinary path: acceptance into a local lifecycle is not the venue
    doing anything. A lane that read this as "I am flat" would stand down believing an exposure
    is gone while the orders are still resting.
    """
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in (ACCEPTED, PARTIAL, REFUSED, UNREACHABLE):
            raise ValueError(
                f"status {self.status!r} is not one of "
                f"{ACCEPTED}/{PARTIAL}/{REFUSED}/{UNREACHABLE}. An "
                f"unrecognised status would be treated as success by anything testing against "
                f"{ACCEPTED!r}, which is the one direction this must never fail in.")
        if self.status in (PARTIAL, REFUSED, UNREACHABLE) and not self.reason:
            raise ValueError(
                f"a {self.status} outcome must carry a reason. An emergency exit that did not "
                f"happen, with no record of why, is the failure this whole module exists to make "
                f"impossible — and it is what an operator reads at 3am.")
        if self.positions < 0:
            raise ValueError(f"positions cannot be negative ({self.positions})")

    @property
    def acted(self) -> bool:
        """THE WHOLE BOOK WAS HANDLED. The one thing a caller should branch on.

        `status == ACCEPTED` invites a caller to add `or UNREACHABLE` when the hook is flaky, which
        is how an unreachable owner comes to read as a completed liquidation. PARTIAL is False for
        the same family of reason: something remains, and a caller must not have to remember to
        check a `failed` list to discover that.
        """
        return self.status == ACCEPTED

    def summary(self) -> str:
        if self.status == ACCEPTED:
            return (f"emergency exit ACCEPTED by the position owner — orders SUBMITTED for "
                    f"{self.positions} position(s); submitted is not filled")
        if self.status == PARTIAL:
            return (f"emergency exit PARTIAL — submitted for {self.positions}, SOMETHING REMAINS: "
                    f"{self.reason}")
        return f"emergency exit {self.status.upper()} — {self.reason}"


async def call_emergency_exit(hook, *, lane: str, verdict, reasons: tuple[str, ...] = ()
                              ) -> EmergencyOutcome:
    """Invoke the owner's hook and normalise whatever comes back into an outcome that cannot lie.

    `hook(lane=..., verdict=..., reasons=...)` is supplied by whoever owns the positions, exactly
    as `daily_loss`'s `on_halt` is. It should return an `EmergencyOutcome`; anything else is
    normalised here rather than trusted, because a hook returning `None` — which every Python
    function does by accident — must NOT read as success.

    NO HOOK IS `UNREACHABLE`, NOT A NO-OP. A lane whose owner never supplied one has no way to
    liquidate, and reporting that as "nothing to do" is how an emergency exit comes to be a
    function that returns quietly. `daily_loss`'s `on_halt` defaults to local-only because a local
    halt is still a halt; there is no local liquidation, so there is no safe default here.
    """
    if hook is None:
        return EmergencyOutcome(
            UNREACHABLE, reason=(
                f"{lane}: no emergency-exit hook is wired, so nothing can liquidate. The lane "
                f"decided to exit and no owner was listening — this is not a no-op."))
    try:
        got = hook(lane=lane, verdict=verdict, reasons=tuple(reasons))
        if hasattr(got, "__await__"):
            got = await got
    except Exception as exc:                                            # noqa: BLE001
        # THE OWNER RAISING IS UNREACHABLE, NOT REFUSED. A refusal is a decision by a system that
        # is working; an exception is a system that is not, and the two demand different
        # responses. Swallowed here so the CALLER can journal and escalate rather than dying in
        # whatever dispatch invoked it — but never swallowed into a success.
        return EmergencyOutcome(
            UNREACHABLE, reason=f"{lane}: the emergency-exit hook raised {type(exc).__name__}: {exc}")
    if isinstance(got, EmergencyOutcome):
        return got
    return EmergencyOutcome(
        UNREACHABLE, reason=(
            f"{lane}: the emergency-exit hook returned {type(got).__name__} rather than an "
            f"EmergencyOutcome, so what it did cannot be known. A hook that returns None — which "
            f"every Python function does by accident — must not read as a completed liquidation."))
