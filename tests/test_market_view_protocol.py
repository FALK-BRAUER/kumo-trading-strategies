"""The protocol every lane's market view must satisfy (kumo-trading-platform issue 873, epic #147).

THE CONFORMANCE TEST IS THE DELIVERABLE, NOT THE INTERFACE. QC27 already shipped a de-risking valve
that is INVERTED in production — mean cash weight 0.01 while the book is >30% below its high, against
0.44 while within 5% of it. Fully invested at the bottom, in cash near the top. It was designed and
reviewed. If five lanes each invent a view, that is five chances to rebuild it, and an interface
alone does not stop one.

So the rules below are properties, not conventions:

  1. A view's input is a LEVEL or a STATE, never a COUNT of the lane's own selection scores. A count
     made of the falling things cannot tell you they are falling.
  2. The view is DECLARED ON THE STRATEGY CONFIG, not passed in by a runner. `rebalance_period` is
     the precedent: a research-only knob where a sweep reported weekly while both production call
     sites stayed hardwired monthly, and nothing failed to say so.
  3. BLOCKING ENTRIES AND LIQUIDATING ARE DIFFERENT ACTIONS and a lane must say which it means.
     Every number published for the market view measures LIQUIDATE; EXIT_ONLY is unmeasured and
     plausibly much cheaper, since it neither crystallises losses nor pays a round trip per flip.
  4. UNKNOWN never acts. A view we could not compute is not evidence the market is bad.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from kumo_strategies.strategies import market_view as mv
from kumo_strategies.strategies.market_view import (
    MarketAction, MarketSignal, MarketState, MarketViewConfig, RISK_OFF, RISK_ON, UNKNOWN)

STRATEGIES = pathlib.Path(mv.__file__).parent


# -- rule 3: the two actions are distinct, and a lane must choose ------------------------------------

def test_a_view_must_declare_what_it_DOES_when_risk_off():
    """No default. Every published figure measures LIQUIDATE, so defaulting to EXIT_ONLY would ship
    an unmeasured behaviour, and defaulting to LIQUIDATE would make the more destructive action the
    one you get by not thinking. This repo's answer to that choice is to refuse to make it — the
    same rule as `POSITION_SIDE`, `price_adjustment` and `strategy_id`."""
    with pytest.raises(ValueError, match="action"):
        MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50)


def test_EXIT_ONLY_blocks_entries_and_keeps_the_book():
    cfg = MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                           action=MarketAction.EXIT_ONLY)
    off = MarketState(RISK_OFF, ("index below its average",), action=cfg.action)
    assert off.blocks_entries is True
    assert off.liquidates is False, "EXIT_ONLY sold the book — that is the other action"


def test_LIQUIDATE_does_both():
    """It is a superset: a lane being flattened is not also opening positions."""
    cfg = MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                           action=MarketAction.LIQUIDATE)
    off = MarketState(RISK_OFF, ("index below its average",), action=cfg.action)
    assert off.liquidates is True
    assert off.blocks_entries is True


@pytest.mark.parametrize("action", list(MarketAction))
@pytest.mark.parametrize("state", [RISK_ON, UNKNOWN])
def test_nothing_acts_unless_the_state_is_RISK_OFF(action, state):
    """Rule 4. UNKNOWN in particular: a data gap must not become a de-risking event, whichever
    action the lane declared."""
    s = MarketState(state, ("because",), action=action)
    assert s.blocks_entries is False and s.liquidates is False


# -- rule 1: a level or a state, never a count of the lane's own scores -------------------------------

def test_every_signal_is_declared_as_a_LEVEL_or_a_STATE():
    """The rule that would have caught QC27's inverted valve before it shipped.

    Each `MarketSignal` member must declare its kind. A signal whose input is the lane's own
    selection scores — a COUNT of how many names still rank well — is the shape that fails: after a
    run-up, ten names still qualify while the index is falling.
    """
    undeclared = [s.name for s in MarketSignal
                  if s is not MarketSignal.NONE and s not in mv.SIGNAL_KIND]
    assert not undeclared, (
        f"these signals do not declare whether they are a LEVEL or a STATE: {undeclared}. A view "
        f"built from a count of the lane's own scores cannot tell you the market is falling.")
    for signal, kind in mv.SIGNAL_KIND.items():
        assert kind in ("level", "state"), (
            f"{signal.name} is declared {kind!r}; a market view may only be a level or a state")


def test_the_kind_map_is_not_a_HAND_WRITTEN_list_that_forgets():
    """#138: a hand-written list dropped BCTROT because it inherits its handler. The same failure
    here would be a signal added to the enum and not to the map, which is why the test above
    enumerates the ENUM and looks the member up, rather than iterating the map."""
    assert set(mv.SIGNAL_KIND) | {MarketSignal.NONE} == set(MarketSignal)


# -- rule 2: the declaration travels with the strategy ------------------------------------------------

def _configs_that_declare_a_view():
    """Every strategy config class that carries a `market_view` field, found by walking the package
    rather than naming lanes."""
    out = []
    for path in sorted(STRATEGIES.rglob("config.py")) + sorted(STRATEGIES.rglob("live.py")):
        src = path.read_text()
        if "market_view" in src:
            out.append(path)
    return out


def test_a_lane_that_uses_a_view_DECLARES_it_on_its_config():
    """Not passed in by a runner. `rebalance_period` is the precedent: a sweep reported weekly while
    both production call sites stayed hardwired monthly, and nothing failed to say so.

    The check is on the AST: a `market_view=` keyword inside a `MarketViewConfig(...)` construction
    that sits in the lane's own config or live module. A runner passing `market_view=` at a call
    site is what this rule exists to prevent, and it is caught by the companion test below.
    """
    declaring = _configs_that_declare_a_view()
    assert declaring, (
        "no lane declares a market view on its config — if that is true the guard below is vacuous, "
        "and if it is false this walk has stopped finding them")


def test_no_RUNNER_hardwires_a_view_of_its_own():
    """A runner may ACCEPT `market_view=` as an override; it may not CONSTRUCT one. The moment a
    runner builds its own `MarketViewConfig`, the lane's declaration and the measured behaviour have
    two sources and the `rebalance_period` failure is back."""
    offenders = []
    runners = pathlib.Path(mv.__file__).parent.parent / "backtesting"
    for path in sorted(runners.glob("runner*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "MarketViewConfig" and node.args or
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "MarketViewConfig" and node.keywords):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"a runner constructs a market view of its own: {offenders}. It may accept one as an "
        f"override; declaring it belongs on the strategy.")


# -- the interface the platform calls ------------------------------------------------------------------

def test_the_protocol_names_all_three_hooks():
    """#873's three, and they are three because they answer different questions: may I open, must I
    close, and how am I doing. Collapsing any two is how `blocks_entries` came to liquidate."""
    for hook in ("entries_blocked", "emergency_exit", "self_assessment"):
        assert hasattr(mv.MarketAware, hook), f"the protocol does not declare {hook}"


def test_every_hook_returns_a_REASON():
    """#873 needs one for the UI, and an operator being de-risked deserves to know why. A bare bool
    is what makes a valve unexplainable after the fact."""
    src = inspect.getsource(mv.MarketAware)
    tree = ast.parse(src)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if fn.name.startswith("_"):
            continue
        assert fn.returns is not None, f"{fn.name} declares no return type"
        rendered = ast.unparse(fn.returns)
        assert "Verdict" in rendered or "Assessment" in rendered, (
            f"{fn.name} returns {rendered}, which carries no reason — #873 needs one for the UI")
