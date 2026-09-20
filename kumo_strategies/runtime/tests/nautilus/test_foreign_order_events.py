"""EVERY strategy that handles order events must ignore the ones it did not cause (kumo-trading-platform issue 748).

Nautilus routes order events by the ORDER's strategy_id, so once cockpit stamps each protective stop
with the lane whose shares it covers, every rotation strategy here starts receiving `PROT-*` fills
and cancels it never placed. All five handled them identically and wrongly: `_pending.pop(sym)` by
SYMBOL alone, so a protective SELL was read as that lane's entry completing.

FIXING THE ONE WE WERE LOOKING AT WOULD HAVE LEFT THE CLASS ALIVE. `momentum_rotation` was the lane
under discussion; `qc345_rotation` and `qc27_rotation` back LIVE lanes, and `template_rotation` was
the worst of them — its `else` branch adds to `_held` for ANY non-exit fill, so a foreign sell with
no pending intent marks a symbol held out of nothing.

SO THIS ASSERTS COVERAGE, NOT EXISTENCE. A test that found SOME guarded strategy would pass while
four were naked. It enumerates every module defining `on_order_filled` and requires each to route
through the shared helper — which is also why the helper is shared rather than copied five times:
five derivations of one rule drift, and this file is what fails when a sixth strategy is written
without it.
"""

from __future__ import annotations

import ast

import pytest

from kumo_strategies.strategies import _layout

_HANDLERS = ("on_order_filled", "on_order_denied", "on_order_rejected", "on_order_canceled")

#: The GENERIC idiom. No strategy here uses it today — but cockpit's own `UiFeedStrategy` does, so it
#: is a natural thing for the next author to reach for, and a sweep keyed only on the specific names
#: would silently stop covering them. Asserted absent rather than assumed absent.
_GENERIC_HANDLERS = ("on_order_event", "on_event")


def _modules_with_handlers() -> dict[str, ast.Module]:
    out = {}
    # Everything runtime/ used to hold, wherever it lives now: the shared layer recursively, plus
    # every strategy's lane and runner (ks#211). Keyed by package-relative path, since every lane
    # is a `nautilus.py` now.
    for path in _layout.all_runtime_sources():
        tree = ast.parse(path.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name in _HANDLERS for n in ast.walk(tree)):
            out[_layout.rel(path)] = tree
    return out


def test_the_sweep_actually_finds_the_strategies_it_is_meant_to_cover():
    """FIXTURE PROPERTY FIRST. A sweep that matched nothing would pass every assertion below while
    checking nothing — the vacuity that let an AST scan report a class closed three times in the
    sibling repo. Every known handler must be found."""
    found = set(_modules_with_handlers())
    # ENUMERATED, not "the known set was fixed": the list once started at five — the strategies
    # found by grepping for the defect's exact shape — and the sweep immediately turned up a sixth
    # (a since-removed lane) that handled fills through a different idiom and was missed. That is
    # the whole reason the check enumerates.
    expected = {
        "strategies/momentum_rotation/nautilus.py", "strategies/momentum_rotation/nautilus_intraday.py",
        "strategies/qc345_rotation/nautilus.py",
        "strategies/qc27_tech_inverse_vol/nautilus.py", "strategies/template/nautilus.py",
    }
    missing = expected - found
    assert not missing, f"the sweep no longer sees {missing} — it is blind, not clean"
    assert len(found) >= len(expected)


def test_no_strategy_handles_order_events_through_the_GENERIC_idiom():
    """The sweep keys on the specific handler names, so a strategy using `on_order_event`/`on_event`
    would be invisible to every assertion in this file. None does today; this is what fails when one
    starts, rather than the coverage quietly shrinking."""
    offenders = []
    for path in _layout.all_runtime_sources():
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in _GENERIC_HANDLERS:
                offenders.append(f"{_layout.rel(path)}::{n.name}")
    assert not offenders, (
        f"{offenders} handle order events through the generic idiom, which this file's sweep does "
        f"not see — extend the sweep, or route them through `order_provenance` explicitly"
    )


@pytest.mark.parametrize("module", sorted(_modules_with_handlers()))
def test_every_guarded_module_IMPORTS_the_shared_helper(module):
    """A local `def is_foreign(e): return False` satisfies a call-name check while disarming the
    guard entirely — the same evasion a stubbed-out function used to slip an AST scan in the sibling
    repo, where a `# TODO: stub` comment passed a substring check while the test did live I/O.

    So the import is asserted too: the name must come from `order_provenance`, not from anywhere the
    module could define it itself."""
    tree = _modules_with_handlers()[module]
    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("order_provenance")
        for alias in n.names
    }
    assert "is_foreign" in imported, (
        f"{module} does not import `is_foreign` from order_provenance — a locally defined one would "
        f"satisfy the call check while disarming the guard"
    )
    locally_defined = {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name in ("is_foreign", "completes")}
    assert not locally_defined, f"{module} redefines {locally_defined} locally, shadowing the shared guard"


@pytest.mark.parametrize("module", sorted(_modules_with_handlers()))
def test_every_order_event_handler_ignores_FOREIGN_orders(module):
    """Each handler must reach `is_foreign` before it touches book state."""
    tree = _modules_with_handlers()[module]
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in _HANDLERS]:
        calls = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "is_foreign" in calls, (
            f"{module}::{fn.name} does not check `is_foreign`, so a protective stop's event moves "
            f"this strategy's book state — a SELL read as an entry completing, or a foreign cancel "
            f"clearing a pending intent for an order that is still working"
        )


@pytest.mark.parametrize("module", sorted(_modules_with_handlers()))
def test_every_FILL_handler_checks_the_side_can_complete_the_intent(module):
    """The second guard, and it must be independently present.

    `is_foreign` depends on a client-order-id convention owned by ANOTHER REPO and unimportable
    across the seam. `completes` needs no shared vocabulary at all — a SELL cannot complete an entry,
    whoever sent it — so it is what still holds when that convention drifts. A strategy carrying only
    one of the two is one prefix change away from the original defect.
    """
    tree = _modules_with_handlers()[module]
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "on_order_filled"]:
        calls = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        # EITHER WAY OF ESTABLISHING THE SIDE COUNTS. Most strategies here run an enter/exit intent
        # and use `completes`; a handler that compares `order_side` directly answers the same
        # question. Demanding one spelling would force a worse fit on a narrower handler.
        #
        # BOUND TO THE AST, NOT TO SOURCE TEXT. A substring check here is satisfied by writing the
        # word in a comment and broken by deleting one — this repo has a test that fails any such
        # assertion, and it caught this one.
        reads_side = any(
            isinstance(n, ast.Attribute) and n.attr == "order_side" for n in ast.walk(fn)
        )
        assert "completes" in calls or reads_side, (
            f"{module}::on_order_filled never establishes the fill's SIDE, so it relies entirely on "
            f"a client-order-id convention owned by another repo"
        )


@pytest.mark.parametrize("module", sorted(_modules_with_handlers()))
def test_no_strategy_pops_the_pending_intent_BEFORE_deciding_the_event_is_its_own(module):
    """The original defect was the POP, not the read. `_pending.pop(sym, None)` consumed the intent
    on the way through, so even a handler that later declined to move `_held` had already destroyed
    the in-flight marker — unblocking a second decision on a symbol with a live order out."""
    tree = _modules_with_handlers()[module]
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "on_order_filled"]:
        # ORDERING BY AST LINE NUMBER, not by string position: a comment mentioning either name
        # would move a substring search and prove nothing.
        pops = [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "pop"
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "_pending"]
        # `completes` SPECIFICALLY, not any guard. An earlier `is_foreign` satisfied a looser
        # version of this check while the pop still ran before the side was established — a mutation
        # moving the pop back above `completes` passed, which is how the weaker assertion was found.
        guards = [n.lineno for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "completes"]
        if not pops:
            continue
        assert guards and min(guards) < min(pops), (
            f"{module}::on_order_filled pops the pending intent before establishing the event is "
            f"its own — the marker is destroyed even when the fill is refused"
        )
