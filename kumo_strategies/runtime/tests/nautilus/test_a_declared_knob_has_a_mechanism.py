"""The constructor argument is the DECLARATION; the method is the MECHANISM. Nothing pinned them (#197).

THE LIVE FAILURE. ibkr-paper booted at 16:07:28Z on `strategies ea03cfb` and five of six lanes armed:

    SMHGLD-007: UNARMED — the calendar answered but arming FAILED:
    AttributeError("'SmhGldSleeveStrategy' object has no attribute '_reread_slots'").
    This is not a network problem and will not be retried.

`smhgld_sleeve` accepted `read_slots`, stored it, and called `self._reread_slots()` unconditionally
inside `_arm` — a method it never defined. There is no caller-side workaround: passing
`read_slots=None` does not help, because the `if self._read_slots is None: return` guard lives inside
the method that is missing.

IT WAS TWO LANES, NOT ONE. `template_rotation` had the identical hole and nobody had hit it because
nothing had armed it yet. It was reported as SMHGLD alone; the sibling was found by asking every
adapter the same question rather than only the one that failed — the first death hides identical
siblings, which this package has filed before.

SO THE TEST ENUMERATES RATHER THAN LISTS. A named list is satisfied by the lanes that were known
when it was typed, which is exactly how the second one stayed invisible.
"""

from __future__ import annotations

import inspect

import pytest
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout


def _adapters():
    """Every `Strategy` subclass this package ships, discovered at runtime."""
    found = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for name, obj in vars(m).items():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                found[obj.__name__] = obj
    return found


ADAPTERS = _adapters()

#: A constructor argument and the method that consumes it. Adding a row here is how the next
#: declared-knob-with-no-mechanism is caught before it reaches a venue.
DECLARED_KNOBS = [("read_slots", "_reread_slots")]


def test_the_discovery_actually_found_the_adapters():
    """A detector that recognises its subject by a property the defect destroys can never fire. If
    the import sweep silently returned nothing, every test below would pass vacuously."""
    assert len(ADAPTERS) >= 7, sorted(ADAPTERS)
    assert "SmhGldSleeveStrategy" in ADAPTERS and "TemplateRotationStrategy" in ADAPTERS


@pytest.mark.parametrize("argument,method", DECLARED_KNOBS)
@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_a_declared_constructor_knob_has_the_METHOD_that_consumes_it(name, argument, method):
    cls = ADAPTERS[name]
    try:
        takes = argument in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):                                    # Cython __init__
        pytest.skip(f"{name}.__init__ has no readable signature")
    if not takes:
        return
    assert hasattr(cls, method), (
        f"{name}.__init__ accepts {argument!r} and the class has no {method!r}. The argument is the "
        f"declaration and the method is the mechanism; a lane that has one without the other raises "
        f"at ARM time, which reads as a lane that decided to hold rather than one that is broken.")


@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_any_adapter_that_CALLS_reread_slots_can_resolve_it(name):
    """The call is what kills the lane, so the call is what is bound. AST over every method the
    class actually runs, `ast.Attribute` callees included — the hole
    `test_foreign_order_events.py` once shipped green with."""
    import ast
    import textwrap

    cls = ADAPTERS[name]
    calls_it = False
    for klass in cls.__mro__:
        for attr, fn in vars(klass).items():
            if not inspect.isfunction(fn):
                continue
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            except (OSError, SyntaxError):                              # noqa: PERF203
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_reread_slots"):
                    calls_it = True
    if calls_it:
        assert hasattr(cls, "_reread_slots"), (
            f"{name} calls self._reread_slots() and cannot resolve it — this is the SMHGLD-007 "
            f"failure verbatim: AttributeError inside _arm, not retried, lane never decides")
