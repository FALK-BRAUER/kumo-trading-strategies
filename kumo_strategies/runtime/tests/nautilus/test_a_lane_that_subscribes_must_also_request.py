"""A lane that subscribes to bars and never requests them can never warm inside a session (#200).

THE LIVE FAILURE. SMHGLD-007 armed correctly at 16:19Z on `dd1b04a` (ks#197), fired its first rung
at 16:25Z, and wrote NOTHING. No journal row, no log line. Every rung until the lane accrued
`_need` sessions of live bars would have done the same.

    smhgld_sleeve.on_start     subscribe_bars(...)          <- FUTURE bars only
                               request_bars                 <- absent from the class entirely
    smhgld_sleeve              if fired is None or not self.warm: return   <- SILENT

SUBSCRIBING DOES NOT BACKFILL, and that is the fact the whole defect rests on. Read off the
installed Nautilus rather than assumed — `Strategy.subscribe_bars(bar_type, client_id,
update_catalog, params)` carries no start, no limit and no lookback of any kind, while
`request_bars(bar_type, start, ...)` takes one. So on a 1-DAY feed an unrequested lane accrues ONE
BAR PER NAME PER SESSION, and `warm` needs `_need` of them on every leg.

THREE LANES HAD THE HOLE, not one: `smhgld_sleeve`, `template_rotation` and
`momentum_rotation_intraday`. And `crsi_short.py` already NAMES the template as where its own
omission came from — a template that omits the hard part teaches that the hard part is optional, and
it taught SMHGLD.

WHY THE TEST IS SHAPED THIS WAY. It binds SUBSCRIBING to REQUESTING, because those two are the pair
that must travel together, and it discovers adapters at runtime rather than listing them — a named
list is satisfied by the lanes known when it was typed, which is how the second and third holes
stayed invisible while the first was being fixed.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout


def _adapters():
    found = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for name, obj in vars(m).items():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                found[obj.__name__] = obj
    return found


ADAPTERS = _adapters()


def _calls_in(cls, method):
    """Attribute-call names in `cls`'s own `method`, or the nearest ancestor that defines it."""
    for klass in cls.__mro__:
        fn = klass.__dict__.get(method)
        if fn is None:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except (OSError, SyntaxError):
            return set()
        return {n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    return set()


def test_the_EXEMPTIONS_are_exactly_the_ones_we_believe_them_to_be():
    """A declared exemption is only safe while somebody is counting them. If a second lane declares
    one, this fails and a human decides whether that lane really warms differently."""
    exempt = sorted(n for n, c in ADAPTERS.items() if hasattr(c, "NO_HISTORY_REQUEST"))
    assert exempt == ["IntradayMomentumRotation"], exempt


def test_the_discovery_found_the_adapters():
    """Vacuity guard: an empty sweep would make every test below pass without asserting anything."""
    assert len(ADAPTERS) >= 7, sorted(ADAPTERS)
    assert {"SmhGldSleeveStrategy", "TemplateRotationStrategy"} <= set(ADAPTERS)


def test_SUBSCRIBING_DOES_NOT_BACKFILL_which_is_what_the_rest_of_this_file_rests_on():
    """Taken from the installed Nautilus on every run rather than quoted from a docstring. If a
    future version gives `subscribe_bars` a lookback, this test says so and the rule can be
    revisited — instead of the rule quietly becoming folklore."""
    sub = inspect.signature(Strategy.subscribe_bars).parameters
    req = inspect.signature(Strategy.request_bars).parameters

    assert "start" not in sub and "limit" not in sub, (
        f"subscribe_bars now takes {sorted(sub)} — it may backfill, so the premise of #200 needs "
        f"re-checking rather than assuming")
    assert "start" in req, sorted(req)


@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_a_lane_that_SUBSCRIBES_to_bars_also_REQUESTS_history(name):
    """The pair that must travel together. Subscribing is the future; requesting is the past; a lane
    with only the first is inert for `_need` sessions and looks exactly like one that decided to
    hold."""
    cls = ADAPTERS[name]
    calls = _calls_in(cls, "on_start")
    if "subscribe_bars" not in calls:
        return
    if hasattr(cls, "NO_HISTORY_REQUEST"):
        # Exempt BY DECLARATION, never by omission — the `NO_LANE_BOOK` pattern. A lane that does
        # not need history must SAY so with a reason, or "does not need it" and "was forgotten"
        # are the same silence, which is how three lanes carried this hole at once.
        assert isinstance(cls.NO_HISTORY_REQUEST, str) and len(cls.NO_HISTORY_REQUEST) > 40, (
            f"{name}.NO_HISTORY_REQUEST must carry a real reason, not a bare flag")
        return
    assert "request_bars" in calls, (
        f"{name}.on_start subscribes to bars and never requests any. `subscribe_bars` does not "
        f"backfill, so on a daily feed this lane accrues one bar per name per SESSION and returns "
        f"silently at its warmth gate until it has enough — indistinguishable from a lane that "
        f"decided to hold (#200, SMHGLD-007 on 2026-09-11)")


@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_a_lane_that_REQUESTS_history_ACCEPTS_the_lookback_it_needs(name):
    """The declaration-and-mechanism rule from #197, one layer up. `request_bars` needs a `start`,
    which comes from `history_days` — and cockpit passes that ONLY to adapters whose signature names
    it (#514). A lane that requests history without naming the parameter gets None, guards the call
    off, and is inert exactly as before while looking fixed."""
    cls = ADAPTERS[name]
    if "request_bars" not in _calls_in(cls, "on_start"):
        return
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        pytest.skip(f"{name}.__init__ has no readable signature")
    assert "history_days" in params, (
        f"{name}.on_start calls request_bars but __init__ does not NAME history_days, so cockpit "
        f"cannot pass it (#514) and the request is guarded off forever")


# NOT ASSERTED HERE, DELIBERATELY. The "say so when history_days is absent" announcement is a
# convention this change introduces; `smhgld_sleeve`, `template_rotation` and `crsi_short` carry it
# and the momentum family does not. Asserting it now would fail four deployed lanes for a rule they
# predate, in a hotfix. Filed as a follow-up instead — a test that fails on arrival teaches people
# to skip tests.
