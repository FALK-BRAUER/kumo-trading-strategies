"""Adding a market view to an engine must not move a single existing number (#147, #120).

`run_sessions` carries phase 1's acceptance gate — MOMENTUM reproduced `run_cadence(signal_lag=1)`
to **0.0000 on every KPI**, with the tolerance stated before the run. Every recorded result for
MOMENTUM-002 and BCTROT-004 comes from this engine.

So wiring a view into it has one hard requirement before any measurement is worth reading: WITH NO
VIEW CONFIGURED, THE ENGINE MUST BE BYTE-IDENTICAL. Not "within tolerance" — identical. A view that
shifted the default would silently move every number this engine has ever produced, and the shift
would be invisible precisely because both sides of any comparison would carry it.

This is the port gate's rule applied to the engine itself: here, IDENTICAL IS THE PASS.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from kumo_strategies.backtesting import runner_sessions
from kumo_strategies.backtesting.families import rotation  # the intraday loop lives here (#270)
from kumo_strategies.strategies.market_view import MarketSignal, MarketViewConfig


def _is_market_panel(t) -> bool:
    """`market_panel = …` (a local) or `self.market_panel = …` (the family holds it)."""
    return (isinstance(t, ast.Name) and t.id == "market_panel") or \
           (isinstance(t, ast.Attribute) and t.attr == "market_panel")


def _fn(func):
    """The function's own AST. `textwrap.dedent` because a method's source is indented."""
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def test_THE_DEFAULT_VIEW_IS_SIGNAL_NONE():
    """The whole inertness argument rests on this. `MarketViewConfig()` must default to a signal
    that yields RISK_ON with a reason, so `target_weights_under` passes the decision through."""
    assert MarketViewConfig().signal is MarketSignal.NONE


def test_A_NONE_SIGNAL_BUILDS_NO_MARKET_PANEL():
    """The pivot is skipped entirely when no view is configured, so the added code costs nothing —
    not even time — on the path every existing result was produced by.

    BOUND TO THE AST, not to source text. This repo has a standing guard against substring
    assertions and it failed me here (the second time today): such a check is satisfied by writing
    the line in a COMMENT and broken by reformatting it. The property is structural — a
    `pivot_table` call reachable only under a test of `mv.signal` — so that is what is asserted.
    """
    fn = _fn(rotation.IntradayRotation)

    def assigns_market_panel(node):
        return any(_is_market_panel(t)
                   for a in ast.walk(node) if isinstance(a, ast.Assign) for t in a.targets)

    # The engine pivots other frames for its own reasons; only the MARKET panel is at issue.
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If) and "signal" in ast.unparse(n.test)
              and assigns_market_panel(n)]
    assert guards, (
        "the market panel is not assigned inside a test of mv.signal — a lane with no view builds "
        "one anyway, and the added code is no longer free on the path every existing result came "
        "from")
    assigns = [a for a in ast.walk(fn) if isinstance(a, ast.Assign)
               and any(_is_market_panel(t) for t in a.targets)]
    inside = [a for a in assigns if any(a in list(ast.walk(g)) for g in guards)]
    outside = [a for a in assigns if a not in inside]
    # One assignment outside the guard is the `market_panel = None` initialiser, which is the
    # thing that makes the no-view path inert. More than that means a real build escaped it.
    assert all(isinstance(a.value, ast.Constant) and a.value.value is None for a in outside), (
        "a market panel is BUILT outside the signal guard")


def test_market_view_DEFAULTS_TO_NONE_ON_THE_SIGNATURE():
    """An engine argument that defaulted to a live view would arm every caller that did not think
    about it. `rebalance_period` is the precedent this repo already paid for."""
    params = inspect.signature(runner_sessions.run_sessions).parameters
    assert "market_view" in params, "run_sessions cannot express a view at all"
    assert params["market_view"].default is None


def test_THE_VIEW_IS_TAKEN_FROM_THE_CONFIG_WHEN_THE_ARGUMENT_IS_ABSENT():
    """Rule 2 of the protocol: the declaration travels with the STRATEGY. An engine that only
    honoured its own argument would make the view a research-only knob the live lane never sees —
    which is exactly the `rebalance_period` failure, where a sweep reported weekly while both
    production call sites stayed hardwired monthly."""
    fn = _fn(rotation.IntradayRotation)
    # ATTRIBUTE ACCESS, not `getattr(cfg, "market_view", ...)`. The string form was invisible to
    # static analysis — the repo's config-group reachability guard reported the field as read by
    # NOTHING, correctly, because a string lookup is not a read anything can see. Same shape as
    # forwarding through `**kwargs`: the call works and the seam is undiscoverable.
    reads_config = [n for n in ast.walk(fn)
                    if isinstance(n, ast.Attribute) and n.attr == "market_view"
                    and isinstance(n.value, ast.Name) and n.value.id == "cfg"]
    assert reads_config, (
        "run_sessions never reads `cfg.market_view` — a view declared on the strategy would not "
        "travel, which is the `rebalance_period` failure exactly")
    assert not [c for c in ast.walk(fn)
                if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "getattr"
                and any(isinstance(a, ast.Constant) and a.value == "market_view" for a in c.args)], (
        "the config is read by string lookup — invisible to the reachability guard")


def test_IT_USES_THE_SHARED_RULE_NOT_ITS_OWN():
    """#170: `runner_qc27_verified` implemented half of rule 3 inline and EXIT_ONLY measured as a
    clean zero for months. A second engine doing the same is the `exits.py` failure again."""
    fn = _fn(rotation.IntradayRotation)
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    missing = {"liquidates", "blocks_entries"} - attrs
    assert not missing, (
        f"run_sessions never reads {sorted(missing)} — that action is unmeasurable on this engine "
        f"and will measure as a clean zero rather than as an error (#170)")


# -- runtime, and the pair is the point ---------------------------------------------------------

import importlib.util as _il
import pathlib as _pl

# The sibling test module holds the panel/config/instruments fixtures. Loaded by PATH rather than
# by bare name: `tests/` is not a package, so a bare import depends on pytest's sys.path insertion
# and breaks the moment this file is run any other way.
_spec = _il.spec_from_file_location(
    "_rs_fixtures", _pl.Path(__file__).parent / "test_runner_sessions.py")
_T = _il.module_from_spec(_spec)
_spec.loader.exec_module(_T)


@pytest.fixture(scope="module")
def run_kw(tmp_path_factory):
    return _T.run_kw.__wrapped__(tmp_path_factory)


def test_AN_EXPLICITLY_EMPTY_VIEW_CHANGES_NOTHING(run_kw):
    """IDENTICAL IS THE PASS here. Verified once against the PRE-CHANGE engine as well — the
    module loaded from `git show HEAD:...` produced the same 77 fills and 119 equity points — which
    is the check that actually proves the wiring cost nothing. That comparison cannot live in the
    suite (it needs a copy of the old file), so this is its durable half.
    """
    bars, cfg = _T._bars(), _T._cfg()
    a = runner_sessions.run_sessions(bars, _T._AllEligible(), cfg=cfg, **run_kw)
    b = runner_sessions.run_sessions(bars, _T._AllEligible(), cfg=cfg,
                                     market_view=MarketViewConfig(), **run_kw)
    assert a.report.fills.equals(b.report.fills)
    assert a.report.equity.equals(b.report.equity)


def test_A_REAL_VIEW_DOES_MOVE_IT(run_kw):
    """VERIFY BY DISAGREEMENT, and without this the test above passes for the wrong reason: if a
    CONFIGURED view also produced identical fills, the wiring would be inert and "nothing changed"
    would be a report about dead code rather than about a safe default.

    This is the same pairing the ablation gate enforces — identical where it must be, different
    where it must differ — and #170 is why it is not optional.
    """
    from kumo_strategies.strategies.market_view import MarketAction

    bars, cfg = _T._bars(), _T._cfg()
    none = runner_sessions.run_sessions(bars, _T._AllEligible(), cfg=cfg, **run_kw)
    viewed = runner_sessions.run_sessions(
        bars, _T._AllEligible(), cfg=cfg, **run_kw,
        market_view=MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=20,
                                     action=MarketAction.LIQUIDATE))
    assert not none.report.fills.equals(viewed.report.fills), (
        "a configured LIQUIDATE view changed nothing — the wiring is inert, and the inertness "
        "test above is then a report about dead code")


def test_EXIT_ONLY_DIFFERS_FROM_NO_VIEW_ON_THIS_ENGINE(run_kw):
    """THE TEST THAT WAS MISSING, and its absence was exposed by a bite that killed nothing.

    The companion below compares EXIT_ONLY against LIQUIDATE — and those still differ when
    EXIT_ONLY is a NO-OP, because LIQUIDATE keeps acting. So it passed with `blocks_entries`
    ignored, which is #170 reproduced on the new engine and invisible to the test written for it.

    EXIT_ONLY must differ from the CONTROL. That is the assertion the defect actually violates.
    """
    from kumo_strategies.strategies.market_view import MarketAction

    bars, cfg = _T._bars(), _T._cfg()
    none = runner_sessions.run_sessions(bars, _T._AllEligible(), cfg=cfg, **run_kw)
    eo = runner_sessions.run_sessions(
        bars, _T._AllEligible(), cfg=cfg, **run_kw,
        market_view=MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=20,
                                     action=MarketAction.EXIT_ONLY))
    assert not none.report.fills.equals(eo.report.fills), (
        "EXIT_ONLY produced the control's fills — `blocks_entries` is not reaching the book, and "
        "the arm would measure as a clean zero (#170 on this engine)")


def test_EXIT_ONLY_AND_LIQUIDATE_DIFFER_ON_THIS_ENGINE_TOO(run_kw):
    """#170 on the other engine, pre-emptively. `runner_qc27_verified` accepted configuration for
    both actions and implemented one; a second engine doing the same would make EXIT_ONLY
    unmeasurable here as well, and it would measure as a clean zero rather than as an error."""
    from kumo_strategies.strategies.market_view import MarketAction

    bars, cfg = _T._bars(), _T._cfg()

    def arm(action):
        return runner_sessions.run_sessions(
            bars, _T._AllEligible(), cfg=cfg, **run_kw,
            market_view=MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=20,
                                         action=action)).report.fills

    eo, liq = arm(MarketAction.EXIT_ONLY), arm(MarketAction.LIQUIDATE)
    assert not eo.equals(liq), "EXIT_ONLY and LIQUIDATE are indistinguishable on run_sessions"
