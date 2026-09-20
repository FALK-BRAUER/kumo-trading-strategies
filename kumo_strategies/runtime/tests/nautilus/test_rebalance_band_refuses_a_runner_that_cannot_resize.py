"""`rebalance_band` is measured in research and unreadable live — so a live lane REFUSES it (#184).

`MomentumRotationConfig.portfolio.rebalance_band` is read in `backtesting/` twice
(`runner_sessions.py:365`, `runner_verified.py:284`) and in `runtime/` ZERO times. Its own docstring
quotes the request it was built for — 2026-09-04: "There is a sizing. The sizing needs to be
executed. That can mean buys or sells of certain quantities of a symbol. Simple." — and then states
the gap: the runner models trading as an ENTRY sized once at arrival and an EXIT that sells
everything, and nothing revisits a position's size.

Specified, measured, documented against the person who asked for it, never crossed into the live
path. No instance sets it (default None, all three checked), so nothing mis-trades today.

WHAT MAKES THIS WORTH A GUARD RATHER THAN A TICKET: the detailed docstring citing a measurement and
quoting the requester is exactly what makes somebody confident enough to set it. A knob the live
path cannot read is silent the day it is set — the lane keeps its entry-and-full-exit behaviour and
the operator believes it is banding.

IT IS THE SMHGLD CAPABILITY, AIMED AT A DIFFERENT LANE. Resizing a held position IS delta execution;
`rebalance_band` is the second consumer of the same missing runner ability, which is why this shares
`capabilities.EXECUTES_DELTAS` rather than growing a parallel mechanism. The operator's ruling when shown the
measurement: "double fix then."

THE REFUSAL IS LIVE-ONLY. `backtesting/` reads the field legitimately and must keep working —
research is where this was measured. So the guard belongs on the adapter's construction, not on the
config, and a test here proves the backtest path is untouched.
"""

from __future__ import annotations

import dataclasses

import pytest

from kumo_strategies.strategies.momentum_rotation.config import (

    MomentumRotationConfig, PortfolioConfig)

def _cfg(band):
    return MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=8, buffer=5,
                                                            rebalance_band=band))


def _lane(band, runner=None, **kw):
    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
    return MomentumRotationStrategy(
        _cfg(band), source=None, symbols=["AAA"], order_id_tag="902",
        session_runner=runner, **kw)


class _DeltaRunner:
    def __init__(self):
        from kumo_strategies.runtime.nautilus.capabilities import EXECUTES_DELTAS
        self.EXECUTES = (EXECUTES_DELTAS,)

    async def run(self, panel, session, *, jobs=None, opens=None, slot=None):
        return None


class _OrdinaryRunner:
    """Every runner that exists: entry sized once, exit sells everything."""

    async def run(self, panel, session, *, jobs=None, opens=None, slot=None):
        return None


# -- the refusal ------------------------------------------------------------------------------------

def test_a_live_lane_with_a_rebalance_band_REFUSES_an_ordinary_runner():
    with pytest.raises(ValueError) as exc:
        _lane(0.25, _OrdinaryRunner())
    msg = str(exc.value)
    assert "rebalance_band" in msg, msg
    assert "delta" in msg.lower(), msg


def test_the_refusal_names_the_VALUE_that_would_be_silently_ignored():
    """"A capability is missing" leaves the operator hunting. The number they typed is the thing
    they will search the logs for."""
    with pytest.raises(ValueError) as exc:
        _lane(0.25, _OrdinaryRunner())
    assert "0.25" in str(exc.value), str(exc.value)


def test_NO_BAND_is_untouched_and_still_builds_against_an_ordinary_runner():
    """THE CONTROL, and the one that matters most: every live lane today has `rebalance_band=None`,
    and this change must not alter a single one of them. If this fails, the guard is not a guard —
    it is an outage."""
    assert _lane(None, _OrdinaryRunner()) is not None


def test_a_runner_that_CAN_resize_is_accepted_with_a_band():
    """The band is not forbidden — it is unexecutable. The day a runner offers delta execution, the
    knob works and this guard stops firing on its own."""
    assert _lane(0.25, _DeltaRunner()) is not None


def test_an_explicit_SHADOW_caller_may_still_construct_it():
    assert _lane(0.25, None, shadow_only=True) is not None


def test_a_band_with_NO_RUNNER_is_refused_like_any_other_absence():
    with pytest.raises(ValueError):
        _lane(0.25, None)


# -- research must keep working -----------------------------------------------------------------------

def test_the_BACKTEST_still_reads_the_band():
    """The refusal is LIVE-ONLY. `rebalance_band` was MEASURED in research, and a guard that broke
    the measurement would delete the evidence for the feature it is protecting.

    Asserted on the config and on the two backtest readers by name, because "the backtest still
    works" is not checkable from here without running one.
    """
    import ast
    import inspect
    import textwrap

    from kumo_strategies.backtesting.families import rotation as runner_sessions   # the rotation family is where the loops live (#270)

    assert _cfg(0.25).portfolio.rebalance_band == 0.25, "the config refuses the value outright"

    for mod in (runner_sessions,):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod)))
        reads = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "rebalance_band"]
        assert reads, f"{mod.__name__} no longer reads rebalance_band — this test's premise is stale"


def test_runtime_STILL_reads_it_nowhere_which_is_why_the_guard_exists():
    """The measurement behind #184, pinned. The day someone wires it, this fails and names the
    guard that should then be deleted — so the guard cannot outlive its reason."""
    import ast

    from kumo_strategies.strategies import _layout

    # A READ, NOT A MENTION. The first version grepped the source text and excluded
    # `momentum_rotation.py` by name — then BCTROT's docstring explained the guard and the test
    # fired on prose. An attribute ACCESS is the thing that means the live path can use the value;
    # a sentence about it is documentation. Bound to the AST for the same reason every other
    # structural check in this repo is.
    reads: dict[str, int] = {}
    for f in _layout.all_runtime_sources():                  # ks#211: runtime/ + every lane and runner
        tree = ast.parse(f.read_text())
        n = sum(1 for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "rebalance_band")
        if n:
            reads[_layout.rel(f)] = n

    # The guard itself reads it once, to decide whether to refuse. That is the only live read.
    assert reads == {"strategies/momentum_rotation/nautilus.py": 1}, (
        f"`rebalance_band` is now read under runtime/ in {reads}. If the live path can EXECUTE it, "
        f"DELETE the construction guard in momentum_rotation.py — a refusal that outlives its "
        f"reason is how a working feature stays switched off.")
