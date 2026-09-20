"""The side a lane holds is a STATED invariant, not an accident of how its exit path is written (#88).

THE DEFECT, found 2026-08-28 against four UNCLAIMED positions the UI labelled SHORT:

    qc345_rotation.py:679   Quantity.from_str(str(abs(position.quantity)))   + OrderSide.SELL
    penny_gap.py:214        Quantity.from_int(int(abs(pos.quantity)))        + OrderSide.SELL
    qc27_rotation.py:505    Quantity.from_str(str(abs(pos.quantity)))        + OrderSide.SELL
    momentum_rotation.py:840  Quantity.from_int(int(pos.quantity))           + OrderSide.SELL

`abs()` plus an unconditional SELL **doubles a short and reports success**. Momentum's raises inside
Nautilus instead, from a line that reads as an ordinary exit. Neither considers the sign, and
`abs()` is the dangerous one precisely because it succeeds.

Every rotation lane is BUY-to-open, and a short that appears in one came from reconciliation, a
manual order, or a phantom — none of which make it ours to double.

CRSISHORT (#123) IS THE FIRST LANE THAT HOLDS THE OTHER SIDE, and that is why these tests are about
a DECLARED side rather than about the word "long". A lane publishes `POSITION_SIDE`; its opening
path must use the order side that OPENS what it declares, and its exit path must consult
`closing_quantity` with that same side. A long position inside a short lane is exactly as
unmanageable as a short inside a long one, and `abs()` would be the same defect mirrored.

A lane that declares nothing is LONG, which is what every lane written before #123 is.

`reduce_only=True` on two of those calls LOOKS like it already covers this. On IBKR it does not:
that adapter drops the flag, so on the venue that is now live the refusal is the only protection.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from kumo_strategies.runtime.nautilus.sides import (
    LONG, SHORT, closing_quantity, closing_order_side, refusal_reason)


def _declared_side(cls):
    """A lane that declares nothing is LONG — every lane written before #123."""
    return getattr(cls, "POSITION_SIDE", LONG)


# ==================================================================================================
# THE RULE
# ==================================================================================================

@pytest.mark.parametrize("qty,expected", [
    (10, 10),
    (1, 1),
    (0, None),
    (-1, None),
    (-93, None),          # the HSBC phantom's size (kumo-trading-platform issue 197 B8)
    (None, None),
    ("", None),
])
def test_only_a_positive_quantity_yields_an_exit_size(qty, expected):
    assert closing_quantity(qty, side=LONG) == expected


@pytest.mark.parametrize("qty,expected", [
    (-10, 10),
    (-1, 1),
    (0, None),
    (1, None),
    (93, None),
    (None, None),
])
def test_on_the_SHORT_side_only_a_negative_quantity_yields_a_cover_size(qty, expected):
    """The mirror, and always POSITIVE: a cover is a BUY of that many shares. Returning the signed
    quantity here would put a negative into `Quantity.from_int`, which is momentum's failure mode
    with the sign reversed."""
    assert closing_quantity(qty, side=SHORT) == expected


def test_the_closing_side_is_looked_up_rather_than_decided_per_call_site():
    """Four call sites each deciding this for themselves, and one of them wrapping the quantity in
    `abs()` first, is the whole of #88."""
    assert closing_order_side(LONG) == "SELL"
    assert closing_order_side(SHORT) == "BUY"
    with pytest.raises(ValueError, match="side must be"):
        closing_order_side("flat")


def test_a_SHORT_is_refused_with_a_reason_an_operator_can_act_on():
    """"Refused" alone sends someone to read the source. The message has to say what the position is,
    why selling is wrong, and that it cannot have originated here."""
    why = refusal_reason(-93, side=LONG)
    assert why and "SHORT 93" in why
    assert "DOUBLE" in why, "the message does not say what selling would actually do"
    assert "long-only" in why
    mirrored = refusal_reason(93, side=SHORT)
    assert mirrored and "LONG 93" in mirrored and "short-only" in mirrored


def test_a_FLAT_position_is_silent():
    """A flat position is the ordinary end of a holding. A line per flat symbol per session would
    bury the one that matters — which is the failure mode `missed_sessions` already paid for at 115
    false alarms against ~20 real ones."""
    assert refusal_reason(0, side=LONG) is None
    assert refusal_reason(0, side=SHORT) is None


def test_the_refusal_is_not_abs():
    """THE HEADLINE, as a property rather than an example. `abs()` is what turns "I do not understand
    this position" into a confident order in the wrong direction."""
    assert closing_quantity(-50, side=LONG) != 50
    assert closing_quantity(-50, side=LONG) is None
    assert closing_quantity(50, side=SHORT) is None


# ==================================================================================================
# EVERY ADAPTER, DISCOVERED — not the four I happened to open
# ==================================================================================================

def _adapters():
    """Every Nautilus Strategy in this package, lane or research."""

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    out = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ == m.__name__ and issubclass(obj, Strategy) and obj is not Strategy:
                out.append(obj)
    return sorted(set(out), key=lambda c: c.__name__)


def _exit_methods(cls):
    """Every method on `cls` that submits a SELL, found by reading the tree rather than by name.

    Named methods would miss the next one. A SELL is the thing that can be wrong here, so a SELL is
    what gets looked for.
    """
    out = []
    for name, fn in inspect.getmembers(cls, inspect.isfunction):
        if fn.__qualname__.split(".")[0] != cls.__name__:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except (OSError, TypeError, SyntaxError):
            continue
        sells = any(isinstance(n, ast.Attribute) and n.attr == "SELL" for n in ast.walk(tree))
        if sells:
            out.append((name, tree))
    return out


@pytest.mark.parametrize("cls", _adapters(), ids=lambda c: c.__name__)
def test_no_exit_path_takes_the_ABSOLUTE_value_of_a_position(cls):
    """`abs()` on a position quantity, on a path that then SELLs, doubles a short.

    AST-bound: a grep is satisfied by a comment naming `abs` — and this file's own source names it
    several times, which is exactly the trap a substring assertion falls into.
    """
    offenders = []
    for name, tree in _exit_methods(cls):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "abs"):
                arg = ast.unparse(node.args[0]) if node.args else "?"
                if "quantity" in arg or "qty" in arg:
                    offenders.append(f"{cls.__name__}.{name} -> abs({arg})")
    assert not offenders, (
        f"an exit path takes the absolute value of a position quantity and then SELLs it, which "
        f"DOUBLES a short instead of closing it: {offenders}. Use `sides.closing_quantity`, which "
        f"refuses rather than guessing a direction.")


@pytest.mark.parametrize("cls", _adapters(), ids=lambda c: c.__name__)
def test_every_exit_path_goes_through_the_shared_refusal(cls):
    """Discovered rather than listed, so a fifth adapter inherits this by existing.

    The check is not "does it mention closing_quantity" but "does every SELL-submitting method
    consult it", because the defect was four separate exit paths each deciding for itself.
    """
    methods = _exit_methods(cls)
    if not methods:
        # HONEST SKIP. `_exit_methods` only reads methods this class DEFINES, so a subclass that
        # inherits its exit path skips here — and its exit is still covered, on the parent. Saying
        # "submits no SELL" would be false for BCTROT, which inherits momentum's `_close`; the
        # distinction is named so nobody reads this skip as "nothing to check".
        inherited = [b.__name__ for b in cls.__mro__[1:] if _exit_methods(b)]
        pytest.skip(f"{cls.__name__} defines no SELL of its own"
                    + (f"; its exits are inherited from {inherited[0]} and checked there"
                       if inherited else " and inherits none"))
    for name, tree in methods:
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "closing_quantity" in called, (
            f"{cls.__name__}.{name} submits a SELL without asking `sides.closing_quantity` what the "
            f"position actually is. The side a lane holds is an invariant of this package and every "
            f"exit has to assert it, not assume it.")


def test_no_lane_can_OPEN_a_position_on_a_side_IT_DOES_NOT_DECLARE():
    """The other half of the invariant. Refusing a wrong-sided position is only the right answer
    because nothing in the lane could have created one — so the opening path has to be checked
    against what the lane says it holds.

    A long lane that opens with SELL, or a short lane that opens with BUY, breaks the reasoning in
    this whole file and must fail here first.
    """
    from kumo_strategies.runtime.nautilus import sides  # noqa: F401  (documents where the rule lives)

    for cls in _adapters():
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if fn.__qualname__.split(".")[0] != cls.__name__ or not name.startswith("_open"):
                continue
            src = textwrap.dedent(inspect.getsource(fn))
            sides_used = {n.attr for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.Attribute) and n.attr in ("BUY", "SELL")}
            opens_with = "BUY" if _declared_side(cls) == LONG else "SELL"
            wrong = {"BUY", "SELL"} - {opens_with}
            assert not (sides_used & wrong), (
                f"{cls.__name__}.{name} declares POSITION_SIDE={_declared_side(cls)!r} but can open "
                f"with {sorted(sides_used & wrong)}, so this lane CAN create a position on the side "
                f"`sides.py` refuses to manage — which would strand it.")
            assert sides_used, f"{cls.__name__}.{name} submits no side at all"


# ==================================================================================================
# THE EXECUTOR HALF — a short must be unmanageable AND MENTIONED
# ==================================================================================================

def test_own_ceiling_floors_a_short_to_zero():
    """The floor is correct and stays: a negative size would flip a sell into a buy.

    Pinned so the reporting below is understood as covering a gap the floor CREATES, not as a
    disagreement with it.
    """
    from kumo_strategies.strategies.momentum_rotation.runner import own_ceiling

    assert own_ceiling(-93, 100, 0) == 0, "a short must not produce a sellable size"
    assert own_ceiling(100, 100, 0) == 100, "the ordinary case still sells"


def test_a_short_is_REPORTED_rather_than_silently_skipped():
    """THE GAP THE FLOOR CREATES. Zero-from-short and zero-from-not-held are indistinguishable in
    `held_qty`, so a short is skipped by give-back, stall, forced exits AND liquidating with no
    trace. kumo-trading-platform issue 197 B8 was a permanent -93 HSBC short that nothing looked for.

    Asserted on the SOURCE's parse tree of the method that builds `held_qty`, because driving a full
    session for this needs the whole fixture and the property is structural: the account dict is
    inspected for negatives and something is written when there are any.
    """
    import ast
    import inspect
    import textwrap

    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    src = textwrap.dedent(inspect.getsource(PgSessionRunner.run))
    tree = ast.parse(src)
    unparsed = ast.unparse(tree)
    assert "< 0" in unparsed and "account.items()" in unparsed, (
        "nothing in `run` inspects the account for negative quantities, so a short is invisible")
    assert "shorts" in unparsed, "the negative case is computed but not named"
    # And it must REACH the journal, not just be computed — the built-never-executed shape.
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "_j" for n in ast.walk(tree)), (
        "a short is detected and never journalled, which is the same as not detecting it")
