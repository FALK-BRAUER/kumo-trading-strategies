"""`completes` must be told which side the lane holds, the way `sides.py` is (#88, #123).

`order_provenance.completes` mapped `enter -> BUY` and `exit -> SELL`. Its own comment says why that
is not a property of trading:

    "ENTER MEANS BUY" IS A PROPERTY OF THIS LONG-ONLY SHOP, NOT OF TRADING. [...] A future
    short-entry strategy makes this mapping wrong for it, and the coverage test will at least force
    whoever writes it to arrive in this file rather than discover it in production.

CRSISHORT (#123) is that strategy, and this is that arrival. The failure is not that the check
refuses a good fill — it is worse than that. For a short lane the mapping is INVERTED, so the entry
SELL that actually filled is refused as "cannot complete an enter", while the covering BUY is
accepted as if it were the entry. The lane then believes it is short a name it has just closed.

So the side becomes a REQUIRED, KEYWORD-ONLY argument, exactly as it did in `sides.py`: there is no
default to inherit and no call site that fails to state what its lane holds. A default of LONG would
have been the more polite change and the worse one — every short lane written afterwards would get
the long mapping for free and be wrong in a way that reads correctly at the call site.

The call-site assertion is on the AST, not on the source text: a grep for "side=" is satisfied by a
comment and broken by a rename.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from kumo_strategies.strategies import _layout
from kumo_strategies.runtime.nautilus.order_provenance import completes
from kumo_strategies.runtime.nautilus.sides import LONG, SHORT



def _fill(side: str | None):
    return SimpleNamespace(order_side=SimpleNamespace(name=side) if side else None,
                           client_order_id="MOMENTUM-002-X")


@pytest.mark.parametrize("intent,side,expected", [
    # A long lane is unchanged — four deployed lanes depend on exactly this.
    ("enter", "BUY", True), ("enter", "SELL", False),
    ("exit", "SELL", True), ("exit", "BUY", False),
])
def test_a_long_lane_keeps_todays_mapping(intent, side, expected):
    assert completes(intent, _fill(side), side=LONG) is expected


@pytest.mark.parametrize("intent,side,expected", [
    # A short lane is the mirror: it OPENS by selling and CLOSES by buying.
    ("enter", "SELL", True), ("enter", "BUY", False),
    ("exit", "BUY", True), ("exit", "SELL", False),
])
def test_a_short_lane_is_the_mirror_not_the_same(intent, side, expected):
    """Under the long mapping every one of these four answers is the opposite of the truth, which is
    why this cannot be left to a default: the lane would read its entry fill as foreign noise and its
    cover as the entry completing."""
    assert completes(intent, _fill(side), side=SHORT) is expected


def test_an_unreadable_side_still_answers_no_on_both_sides():
    """None is a third state, not a default. A shape the handler cannot read confirms nothing, and
    moving book state on it is how a strategy comes to believe it holds something nobody bought."""
    for side in (LONG, SHORT):
        for intent in ("enter", "exit", None):
            assert completes(intent, _fill(None), side=side) is False


def test_the_side_is_required_and_keyword_only():
    """No default, so no lane inherits a side it never stated — `sides.py`'s rule, same reason."""
    p = inspect.signature(completes).parameters
    assert "side" in p, "completes() does not take a side at all"
    assert p["side"].kind is inspect.Parameter.KEYWORD_ONLY, "side can be passed positionally"
    assert p["side"].default is inspect.Parameter.empty, (
        "side has a default — a short lane would silently get the long mapping")


def test_an_unknown_side_raises_rather_than_picking_one():
    with pytest.raises(ValueError):
        completes("enter", _fill("BUY"), side="sideways")


def test_every_adapter_passes_a_DECLARED_side_to_completes():
    """AST, not text. This asserts on the actual call node: `completes(...)` must carry a `side=`
    keyword whose value is not a bare literal guess. A source-substring test here would pass on a
    comment mentioning `side=` and fail on a rename."""
    offenders = []
    for path in _layout.nautilus_sources():   # shared adapter layer + every lane (ks#211)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "completes":
                continue
            kw = {k.arg for k in node.keywords}
            if "side" not in kw:
                offenders.append(f"{path.name}:{node.lineno} calls completes() with no side=")
    assert not offenders, "\n".join(offenders)
