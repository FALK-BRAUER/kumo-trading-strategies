"""EXIT_ONLY must reach the book, or one of the protocol's two actions is unmeasurable (#170).

MEASURED on the ablation gate's first real use (`research/market-view/`, 2026-09-11): both
EXIT_ONLY arms reproduced the no-view control BYTE FOR BYTE — same return, same worst drawdown,
same 29,285 fills. `runner_qc27_verified` read `.liquidates` at lines 186 and 259 and NEVER read
`.blocks_entries`.

WORSE THAN A MISSING FEATURE. The protocol records in several places that EXIT_ONLY is "unmeasured
and plausibly much cheaper", with a plan to measure it before choosing which action to arm. Anyone
running that measurement here would have received a clean, plausible ZERO delta and quoted it. Not
a failed arm: a dead mechanism reported as a harmless one.

WHY THE RULE IS TESTED DIRECTLY AND NOT THROUGH A PANEL. Three successive fixtures failed to
separate the arms, each for a real reason, and each time a GUARD caught it rather than a false
green:

  1. no rotation      the lane already held its two names and never tried to open one while
                      risk-off, so there was no entry to block
  2. full rotation    exits are ALLOWED under EXIT_ONLY, so a fully rotating book drains to
                      nothing anyway and both arms ended flat — the mutation making EXIT_ONLY an
                      alias for LIQUIDATE then killed nothing
  3. cash proxy       every spare dollar went to GLD, so the control had no cash to enter a tech
                      name — and the guard PASSED on the GLD purchase, because it counted all buys
                      rather than tech buys

The rule itself is four lines. Coaxing a panel into exhibiting it was producing fixture
archaeology, not evidence, so it is extracted as `market_view.target_weights_under` and asserted
directly. The end-to-end evidence that it was broken is the recorded research run.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from kumo_strategies.backtesting.families import (
    monthly as qc27,  # the monthly family of the ONE runner (#270)
)
from kumo_strategies.strategies.market_view import (
    RISK_OFF,
    RISK_ON,
    UNKNOWN,
    MarketAction,
    MarketState,
    target_weights_under,
)

WEIGHTS = {"EEE": 0.4, "DDD": 0.6}
HELD = {"EEE": 10, "CCC": 5}


def _view(state, action):
    return MarketState(state, ("because",), action=action)


def test_LIQUIDATE_EMPTIES_THE_TARGET():
    assert target_weights_under(_view(RISK_OFF, MarketAction.LIQUIDATE), WEIGHTS, HELD) == {}


def test_EXIT_ONLY_KEEPS_WHAT_IS_HELD_AND_OPENS_NOTHING():
    """THE DEFECT. EEE is wanted and already held, so it stays. DDD is wanted and NOT held, so it
    is declined — that is the entry being blocked."""
    got = target_weights_under(_view(RISK_OFF, MarketAction.EXIT_ONLY), WEIGHTS, HELD)
    assert got == {"EEE": 0.4}, got


def test_EXIT_ONLY_IS_NOT_LIQUIDATE():
    """Rule 3: they are different actions. A fix that made one an alias for the other would
    satisfy "EXIT_ONLY differs from the control" and destroy the distinction the protocol exists
    to keep — which is exactly what an earlier fixture failed to catch."""
    eo = target_weights_under(_view(RISK_OFF, MarketAction.EXIT_ONLY), WEIGHTS, HELD)
    liq = target_weights_under(_view(RISK_OFF, MarketAction.LIQUIDATE), WEIGHTS, HELD)
    assert eo != liq and eo, "EXIT_ONLY flattened the book — that is LIQUIDATE's action"


def test_EXIT_ONLY_IS_NOT_THE_CONTROL():
    eo = target_weights_under(_view(RISK_OFF, MarketAction.EXIT_ONLY), WEIGHTS, HELD)
    assert eo != dict(WEIGHTS), "EXIT_ONLY changed nothing — the defect"


def test_ALL_THREE_ANSWERS_DIFFER():
    """Three actions, three answers. TWO was the bug."""
    answers = [
        tuple(sorted(target_weights_under(None, WEIGHTS, HELD).items())),
        tuple(sorted(target_weights_under(
            _view(RISK_OFF, MarketAction.EXIT_ONLY), WEIGHTS, HELD).items())),
        tuple(sorted(target_weights_under(
            _view(RISK_OFF, MarketAction.LIQUIDATE), WEIGHTS, HELD).items())),
    ]
    assert len(set(answers)) == 3, f"indistinguishable answers: {answers}"


def test_A_HELD_NAME_THAT_LEFT_THE_RANKING_IS_STILL_EXITED():
    """"Exit only" means exits are NOT blocked. CCC is held and no longer wanted, so it must be
    absent from the target and get sold by the ordinary rule. A version that kept every holding
    would be "hold everything", which is a third action nobody declared."""
    got = target_weights_under(_view(RISK_OFF, MarketAction.EXIT_ONLY), WEIGHTS, HELD)
    assert "CCC" not in got


@pytest.mark.parametrize("state", [RISK_ON, UNKNOWN])
@pytest.mark.parametrize("action", list(MarketAction))
def test_NOTHING_IS_WITHHELD_UNLESS_THE_STATE_IS_RISK_OFF(state, action):
    """Rule 4. UNKNOWN especially: a view we could not compute is not evidence the market is bad,
    and turning a data gap into a blocked entry is the same error as turning it into a loss."""
    assert target_weights_under(_view(state, action), WEIGHTS, HELD) == dict(WEIGHTS)


def test_NO_VIEW_IS_THE_LANES_OWN_WEIGHTS():
    assert target_weights_under(None, WEIGHTS, HELD) == dict(WEIGHTS)


@pytest.mark.parametrize("view", [
    None,
    _view(RISK_ON, MarketAction.EXIT_ONLY),
    _view(UNKNOWN, MarketAction.LIQUIDATE),
], ids=["no-view", "risk-on", "unknown"])
def test_IT_RETURNS_A_COPY_NOT_THE_CALLERS_DICT(view):
    """The runner mutates the target afterwards to add the cash proxy, so returning the lane's own
    weights dict would push that mutation into the decision record.

    EVERY PASS-THROUGH BRANCH, because there are three of them and a mutation on one killed
    NOTHING while the test only exercised another. That read as an unearned test and was an
    insufficient mutation — the first hypothesis when a bite goes green, not the second.
    """
    out = target_weights_under(view, WEIGHTS, HELD)
    out["INJECTED"] = 1.0
    assert "INJECTED" not in WEIGHTS, "the caller's weights dict was handed back and mutated"


# -- the runner must actually delegate ------------------------------------------------------------

def test_THE_RUNNER_USES_THE_SHARED_RULE_RATHER_THAN_ITS_OWN():
    """Bound to the AST. The whole defect was a driver implementing half the rule inline; a second
    driver doing the same is the `exits.py` failure this repo already has a name for."""
    tree = ast.parse(inspect.getsource(qc27))
    called = {c.func.id for c in ast.walk(tree)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "target_weights_under" in called, (
        "runner_qc27_verified no longer delegates to the shared rule — it is deciding the target "
        "book itself again, which is how EXIT_ONLY went unimplemented")
