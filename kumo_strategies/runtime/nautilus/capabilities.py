"""What a session runner can EXECUTE, declared by the runner and checked at registration (#177).

A LANE AND ITS RUNNER CAN DISAGREE ABOUT WHAT A DECISION IS, and nothing detects it. Every lane in
this package until now decides in `enter` and `exit` lists, and every runner consumes them. SMHGLD
does not: its contract is a TARGET and a DELTA — `order_plan()` emits `target_qty - held_qty`, an
entry is a delta from zero and an exit is a target of zero, and the words `enter` and `exit` never
appear.

Against an enter/exit runner that lane REGISTERS, WARMS, DECIDES EVERY SESSION, WRITES A JOURNAL ROW
EVERY SESSION — AND EXECUTES NOTHING, while every surface reads healthy.

THIS IS THE STRUCTURAL MEMBER OF A FAMILY WE HAVE PAID FOR FOUR TIMES THIS MONTH. The others were
two sides disagreeing about a VALUE: a config flag read nowhere (#26), a slot reservation written
nowhere (#172), an account handler subscribed nowhere (#174). Each of those is findable by asking
"who reads this". This one is not, because BOTH SIDES ARE INDIVIDUALLY CORRECT — the disagreement is
about what a decision IS, and code review of either half shows nothing wrong.

THREE RULES, AND EACH ONE IS A DEFECT WE HAVE ALREADY SHIPPED
--------------------------------------------------------------
DECLARED BY THE RUNNER, NOT BY THE LANE. A flag the lane sets about itself says what it WANTS. The
question is what the runner OFFERS, and only the runner can answer it. `ORDER_PATH_COMPLETE` is the
lane-side twin of this and is deliberately about the lane's OWN order path, not its runner's.

ABSENCE IS A REFUSAL. A runner that declares nothing cannot execute deltas. Reading silence as
permission is the failure mode this package has a standing rule about, and here it produces a lane
that looks alive and is not.

IDENTITY, NEVER A NAME. `getattr(runner, "executes_deltas", False)` is satisfied by any object
carrying a truthy attribute of that spelling. kumo-trading-platform issue 955 is exactly that shape: two hand-rolled
attribute names, both plausible, meant market-aware journal rows were written on NO lane on EITHER
instance — and nothing failed, because a name is satisfied by anything that spells it. So a
capability is a TOKEN a runner must IMPORT from this module. A string that reads the same, the
token's own name as a string, a bare `True`, or somebody else's sentinel do not match it. Importing
a symbol is a deliberate act; spelling one is not.
"""

from __future__ import annotations

__all__ = ["Capability", "EXECUTES_DELTAS", "offers", "require"]


class Capability:
    """A named singleton whose IDENTITY is the declaration.

    Not an enum member and not a string, deliberately. An enum can be reconstructed from its value
    (`Capability("execute_deltas")`) and a string can be typed, so either would let a runner declare
    this without importing anything — which is the accident the gate exists to prevent.

    `__slots__` and no `__eq__`: identity is the only comparison, so a look-alike carrying the same
    text cannot compare equal to this by defining its own.
    """

    __slots__ = ("name", "means")

    def __init__(self, name: str, means: str) -> None:
        self.name = name
        self.means = means

    def __repr__(self) -> str:                                    # appears in the refusal message
        return f"<Capability {self.name}>"


#: The runner applies each order as a SIGNED CHANGE to the current holding, and sells before buys.
#:
#: THE ORDERING IS PART OF THE CAPABILITY, not a hint. `order_plan` returns sells first because the
#: sells fund the buys; a runner that executes deltas in a different order asks a fully-invested
#: sleeve for margin it does not have. A runner declaring this is declaring both.
EXECUTES_DELTAS = Capability(
    "execute_deltas",
    "applies each order as a signed change to the current holding, sells before buys")


def offers(runner, capability: Capability) -> bool:
    """Has `runner` DECLARED `capability`?

    Reads `runner.EXECUTES`, a tuple of tokens. Absent, empty, or not containing this exact object
    is False — including when it contains something that merely looks like it.
    """
    if runner is None:
        return False
    declared = getattr(runner, "EXECUTES", ())
    try:
        return any(d is capability for d in declared)
    except TypeError:                                   # not iterable: a declaration we cannot read
        return False


def require(runner, capability: Capability, *, lane: str, needs: str) -> None:
    """Refuse to build `lane` against a runner that has not declared `capability`.

    RAISES AT CONSTRUCTION, which is the whole point: a lane that cannot act must not exist. The
    alternative — refusing later, per session — is a lane that registers, arms, decides and journals
    while executing nothing, which is the state this is here to make impossible.

    The message names BOTH HALVES because an operator reading it at 3am needs both: what the lane
    requires, and what this particular runner actually offers. "Incompatible runner" is not a
    message.
    """
    if offers(runner, capability):
        return
    who = type(runner).__name__ if runner is not None else "no runner (session_runner=None)"
    declared = getattr(runner, "EXECUTES", None) if runner is not None else None
    offered = ("declares nothing" if not declared
               else f"offers {tuple(getattr(d, 'name', d) for d in declared)!r}")
    raise ValueError(
        f"{lane} cannot register against {who}: it {offered}, and this lane needs "
        f"{capability.name!r} — {capability.means}.\n\n"
        f"{needs}\n\n"
        f"A runner declares this by importing the token and listing it:\n"
        f"    from kumo_strategies.runtime.nautilus.capabilities import {capability.name.upper()}\n"
        f"    EXECUTES = ({capability.name.upper()},)\n"
        f"It is the OBJECT, not the name: a string that reads the same does not declare it, because "
        f"a name is satisfied by anything that spells it (kumo-trading-platform issue 955).\n\n"
        f"To build a lane that computes and publishes WITHOUT trading, pass shadow_only=True.")
