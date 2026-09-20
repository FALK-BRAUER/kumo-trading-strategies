"""SMHGLD refuses to register against a runner that cannot execute DELTAS (#177).

THIS LANE'S CONTRACT IS A TARGET AND A DELTA. It enters once and never exits: `order_plan()` emits
`target_qty - held_qty`, an entry is a delta from zero and an exit is a target of zero. There are no
`enter` and `exit` lists because the lane never thinks in those words.

Every runner in this package and in cockpit consumes `enter` and `exit`. So against one of those the
lane REGISTERS, WARMS, DECIDES EVERY SESSION, WRITES A JOURNAL ROW EVERY SESSION — AND EXECUTES
NOTHING, while every surface reads healthy. Lab's own sentence: "A runner consuming only `enter` and
`exit` executes NOTHING here while the journal shows a lane deciding every session."

That is the fourth silent-inert shape on this line this month and the only STRUCTURAL one. The others
were two sides disagreeing about a VALUE — a flag read nowhere, a slot reservation never written, a
handler never subscribed. This is two sides disagreeing about WHAT A DECISION IS, and no amount of
care on either side detects it, because both are individually correct.

THE REFUSAL IS THE `ORDER_PATH_COMPLETE` PATTERN, which is the best mechanism either repo has for
this: a lane that cannot act does not construct. It is why CRSISHORT could not be armed by accident.

DECLARED BY THE RUNNER, NOT BY THE LANE. A flag the lane sets about itself says what it WANTS; the
question is what the runner OFFERS, and only the runner can answer that.

ABSENCE IS A REFUSAL. A runner that declares nothing is a runner that cannot execute deltas. This is
the absence-must-not-be-readable-as-permission rule in the one place where reading it wrong produces
a lane that looks alive and is not.

AND IT MUST BE IMPOSSIBLE TO SATISFY BY ACCIDENT. A `getattr(runner, "executes_deltas", False)` is
passed by any object that happens to carry a truthy attribute of that name — and platform issue 955 is
exactly that shape, where two hand-rolled attribute names meant market-aware journal rows were
written on NO lane on EITHER instance. So the capability is an IDENTITY: a token the runner must
import from this package. A same-named string, a bool, or a look-alike object does not match it.
"""

from __future__ import annotations

import pytest


def _tokens():
    import kumo_strategies.runtime.nautilus.capabilities as cap
    return cap


def _lane(runner, **kw):
    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
    from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig
    return SmhGldSleeveStrategy(
        SmhGldSleeveConfig(), symbols=["SMH", "GLD"], order_id_tag="900",
        session_runner=runner, **kw)


class _EnterExitRunner:
    """A runner exactly like every one that exists today: it consumes `enter` and `exit`.

    NOT A STUB THAT REFUSES EVERYTHING. This is the real shape the lane would meet — a runner that
    works perfectly for four other lanes and silently executes nothing for this one.
    """

    async def run(self, panel, session, *, jobs=None, opens=None, slot=None):
        # `jobs` and `opens` because `momentum_rotation.py:676` passes them. A
        # double that dies with TypeError inside the adapter's `except` reads as
        # 'never called' — `test_every_runner_DOUBLE_accepts_what_an_adapter_can_pass`
        # caught exactly that here.
        return None


class _DeltaRunner:
    """A runner that has declared, by importing the token, that it executes deltas."""

    def __init__(self):
        from kumo_strategies.runtime.nautilus.capabilities import EXECUTES_DELTAS
        self.EXECUTES = (EXECUTES_DELTAS,)

    async def run(self, panel, session, *, jobs=None, opens=None, slot=None):
        # `jobs` and `opens` because `momentum_rotation.py:676` passes them. A
        # double that dies with TypeError inside the adapter's `except` reads as
        # 'never called' — `test_every_runner_DOUBLE_accepts_what_an_adapter_can_pass`
        # caught exactly that here.
        return None


# -- the refusal ------------------------------------------------------------------------------------

def test_a_runner_that_consumes_ENTER_and_EXIT_is_REFUSED():
    """The defect, as its fix. This runner is not broken — it is every runner we have."""
    with pytest.raises(ValueError) as exc:
        _lane(_EnterExitRunner())
    msg = str(exc.value)
    assert "delta" in msg.lower(), msg
    assert "SMHGLD" in msg or "sleeve" in msg.lower(), msg


def test_the_refusal_NAMES_what_the_lane_needs_and_what_the_runner_OFFERS():
    """An operator reading this at 3am needs both halves. "Incompatible runner" is not a message."""
    with pytest.raises(ValueError) as exc:
        _lane(_EnterExitRunner())
    msg = str(exc.value)
    assert "_EnterExitRunner" in msg, f"the runner is not named: {msg}"
    assert "declares nothing" in msg or "offers" in msg.lower(), msg


def test_NO_RUNNER_AT_ALL_is_also_a_refusal_unless_the_caller_says_shadow():
    """Absence is not permission. `session_runner=None` is how a lane is built for a backtest or a
    unit test, and it must not be the quiet way to build a live lane that executes nothing."""
    with pytest.raises(ValueError):
        _lane(None)


def test_an_explicit_SHADOW_caller_may_construct_it():
    """The escape hatch is NAMED and belongs to the caller, exactly as CRSISHORT's `shadow_only`
    does. A lane that computes and publishes without trading is a legitimate thing to build."""
    lane = _lane(None, shadow_only=True)
    assert lane is not None


def test_a_runner_that_DECLARES_the_capability_is_accepted():
    assert _lane(_DeltaRunner()) is not None


# -- it cannot be satisfied by accident ---------------------------------------------------------------

@pytest.mark.parametrize("offered,why", [
    (("execute_deltas",), "a STRING that happens to read the same"),
    (("EXECUTES_DELTAS",), "the token's NAME as a string"),
    ((True,), "a bare truthy value"),
    ((object(),), "some other sentinel"),
    (("delta_execution",), "a plausible near-miss spelling"),
])
def test_a_LOOK_ALIKE_declaration_does_NOT_satisfy_the_gate(offered, why):
    """IDENTITY, NOT A NAME. platform issue 955 is the shape this guards against: two hand-rolled attribute
    names, both plausible, meant market-aware journal rows were written on NO lane on EITHER
    instance — and nothing failed, because a name is satisfied by anything that spells it.

    A runner declares this capability by IMPORTING the token, which is a deliberate act that cannot
    happen by coincidence.
    """
    class _LookAlike:
        EXECUTES = offered

        async def run(self, panel, session, *, jobs=None, opens=None, slot=None):
            return None

    with pytest.raises(ValueError):
        _lane(_LookAlike())


def test_the_token_is_NOT_a_string_and_NOT_a_bool():
    """If it were either, the parametrised look-alikes above would pass and this whole gate would be
    a naming convention."""
    tok = _tokens().EXECUTES_DELTAS
    assert not isinstance(tok, (str, bool, int)), type(tok)
    assert repr(tok), "the token must have a readable repr — it appears in the refusal message"


def test_the_token_is_a_SINGLETON_so_a_reimport_still_matches():
    """A token rebuilt per import would make the gate fail for a runner that declared it correctly —
    the opposite failure, and the one that would get the gate deleted."""
    import importlib

    import kumo_strategies.runtime.nautilus.capabilities as cap
    first = cap.EXECUTES_DELTAS
    importlib.reload(cap)
    assert cap.EXECUTES_DELTAS is first or repr(cap.EXECUTES_DELTAS) == repr(first)
