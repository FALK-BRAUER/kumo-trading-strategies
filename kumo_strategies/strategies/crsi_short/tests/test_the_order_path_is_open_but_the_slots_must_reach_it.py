"""`ORDER_PATH_COMPLETE` is True, and that is only HALF of what it used to guard (#131, #210).

The flag meant two things at once:

    the decomposition EXISTS          a property of the CODE
    this lane is CONFIGURED to use it a property of the CALLER

Only the first is now true by construction. Flipping the flag alone would have let a lane built with
the DEFAULT slots — `("open+5m",)`, after the cross — construct and trade, sending only the day
limit. `crsi_short.py:89` measures what that costs: 108 of the 231 accepted trades fill at the
OPENING PRINT and carry 77.6% of the return, +9.99%/trade against +2.53%. A day-limit-only lane is
not a degraded version of this strategy; it is a different one, and every surface reads normal.

So the second half is now checked directly, per lane, from the slots it was actually given — which
is the only place that fact exists.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.crsi_short import nautilus as mod
from kumo_strategies.strategies.crsi_short.nautilus import ORDER_PATH_COMPLETE, _reaches_the_auction


def test_the_decomposition_the_flag_now_ASSERTS_is_actually_present():
    """The flag is a claim about this module. If a piece went missing the flag would still read
    True, so the claim is checked rather than trusted."""
    for name in ("opening_leg", "replacement_leg", "replacement_is_owed", "_AUCTION_LEG_GONE"):
        assert hasattr(mod, name), f"ORDER_PATH_COMPLETE is True and {name} is missing"

    from kumo_strategies.runtime.nautilus.broker import NautilusBroker

    assert hasattr(NautilusBroker, "order_status"), (
        "the replacement cannot learn the auction leg's fate without order_status (#208)")

    from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

    assert has_pre_open(("open-10m", "open+5m")) is True, (
        "a pre-open slot would clamp forward to the open and the auction leg would miss the cross")


def test_ORDER_PATH_COMPLETE_is_True():
    assert ORDER_PATH_COMPLETE is True


@pytest.mark.parametrize("slots", [("open+5m",), ("open+5m", "close-20m"), ("close-20m",),
                                   ("09:20",), ()],
                         ids=lambda s: "-".join(s) or "empty")
def test_a_lane_whose_slots_NEVER_REACH_THE_AUCTION_refuses_to_be_built(slots):
    """Including the DEFAULT, which is the case that would actually have shipped."""
    assert not any(_reaches_the_auction(s) for s in slots)
    with pytest.raises(ValueError) as e:
        _build(decision_slots=slots, shadow_only=False)
    said = str(e.value)
    assert "77.6%" in said, "the refusal does not say what a day-limit-only lane costs"
    assert "open-10m" in said, "the refusal does not say how to fix it"


def test_the_DEPLOYED_PAIR_builds():
    """The guard must not become a lane that can never trade."""
    _build(decision_slots=("open-10m", "open+5m"), shadow_only=False)


def test_a_SHADOW_lane_may_keep_post_auction_slots():
    """A lane that computes and publishes is not misrepresenting anything: it sends no orders, so it
    cannot send the wrong ones. Refusing here would block platform issue 853's first phase, which is the
    safest way to validate a new lane."""
    _build(decision_slots=("open+5m",), shadow_only=True)


def test_a_wall_clock_does_NOT_satisfy_the_auction_requirement():
    """`09:20` is before the open on an ordinary day and after it on a half-day with a shifted open,
    and the check has no calendar. Accepting it would let a lane build whose auction leg is late on
    exactly the sessions that are already unusual."""
    assert _reaches_the_auction("09:20") is False
    with pytest.raises(ValueError):
        _build(decision_slots=("09:20", "open+5m"), shadow_only=False)


def _build(*, decision_slots, shadow_only):
    """Run the CONSTRUCTION guard without a Nautilus kernel.

    The guard is a plain conditional near the top of `__init__`; everything after it needs a live
    trader. Re-implementing the condition here would be a second derivation of the rule under test,
    so the real source is executed: the guard's own lines, lifted by AST from the real `__init__`.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(mod.CrsiShortStrategy.__init__)))
    guards = [n for n in tree.body[0].body
              if isinstance(n, ast.If) and "shadow_only" in ast.dump(n.test)]
    assert guards, "the construction guard is gone from __init__"

    scope = {"ORDER_PATH_COMPLETE": mod.ORDER_PATH_COMPLETE,
             "_reaches_the_auction": mod._reaches_the_auction,
             "decision_slots": decision_slots, "shadow_only": shadow_only}
    for guard in guards:
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "<guard>", "exec"), scope)
