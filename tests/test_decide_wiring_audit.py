"""#26 item 3 — which `PortfolioConfig` fields actually reach an observable effect, per call site.

THE CLASS, NOT THE INSTANCE
---------------------------
`decide()` takes `corr` and `vol` as OPTIONAL arguments and degrades silently when they are absent:
`_diversified` returns the plain ranking when `corr is None`, and `_weights` returns equal weight
when `vol is None`. Neither logs, neither raises. So every field consumed inside `decide()` is inert
in any runtime that does not pass the inputs it needs — and the default is to pass nothing.

That is not a bug in one runner. It is a shape: the caller decides whether the callee's config has
any effect, and nothing checks the two agree. `max_correlation` and `inverse_vol_sizing` were both
swept as if live while doing nothing, and were found by accident, twice.

This test enumerates the call sites from the source and pins what each one passes. It is not a
correctness test — several rows below record a defect rather than a guarantee. It exists so that the
audit cannot silently rot, and so that adding a fifth call site that forgets `vol` fails here rather
than in a research result six months later.

WHAT THE AUDIT FOUND
--------------------
    call site                     corr  vol  overlay  consumes dec.weights
    nautilus intraday adapter      yes  yes     yes    yes
    nautilus daily adapter         yes  yes      NO    yes
    pgrunner (THE LIVE RUNNER)     yes  yes      NO    yes

`overlay` is unwired in all three daily drivers by design: they decide once per session and have no
intraday bars to build an overlay from. `IntradayConfig` is therefore reachable only from the
intraday adapter, which is why the cadence question still cannot be asked on the verified harness.

`max_correlation` and `inverse_vol_sizing` WERE inert in production — the live runner passed neither
input and sized at a flat budget, so either flag could be set in a live config, typecheck, deploy,
and change nothing. Now fixed: corr/vol come from the shared `engine.panel_stats` (one
implementation for both drivers, same reasoning as #197 P2 for the exits), and entries size from
`d.weights` as a multiple of the equal-weight slot, so a config that does not ask for inverse-vol
sizing is byte-identical to before.

STILL OPEN: `ExitConfig` in both Nautilus adapters. Neither calls `evaluate_exits`. That is not an
oversight left for later — the exit rules need a durable `TrailState` (entry, peak, sessions since
high), and these adapters have no store. #197 B1 is precisely what happens when that trail is not
durable: give-back went dead in production because the peak did not survive a restart. Adding an
in-memory trail here would reproduce that bug with a green test suite. It needs a persistence
decision first.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]


SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies"

#: module path -> ({optional decide() kwarg -> passed?}, why)
#:
#: A row here is a STATEMENT OF FACT about the code, not an aspiration. `False` entries are the
#: audit's findings. Changing behaviour means changing the row, deliberately, in the same commit.
#:
#: `overlay` is audited alongside `corr`/`vol` because it gates a whole config group the same way:
#: with `overlay=None`, `blended` collapses to `ranked` and every `IntradayConfig` field except
#: `entry_only` stops mattering. Same silent-degradation shape, larger surface.
EXPECTED: dict[str, tuple[dict[str, bool], str]] = {
    "backtesting/families/rotation.py": (
        {"corr": True, "vol": True, "overlay": False},
        "THE backtest engine (2026-09-08): N slots a session from cfg.execution.decision_slots, "
        "ranking on the prior completed session at every slot exactly as pgrunner does, fills at "
        "the next bar, gap dead band applied. corr/vol from the config. `overlay` is not passed — "
        "live does not pass it either, and a backtest that exercised a signal live never sees "
        "would measure a strategy that does not trade."),
    "strategies/momentum_rotation/nautilus_intraday.py": (
        {"corr": True, "vol": True, "overlay": True},
        "the only caller that passes all three; also sizes from d.weights at :242"),
    "strategies/momentum_rotation/nautilus.py": (
        {"corr": True, "vol": True, "overlay": False},
        "FIXED: derives corr/vol from the config via engine.panel_stats and sizes _open() by a "
        "multiple of the equal-weight slot. `overlay` stays unwired — this is a daily-decision "
        "adapter with no intraday bars to build one from."),
    "strategies/momentum_rotation/runner.py": (
        {"corr": True, "vol": True, "overlay": False},
        "FIXED: the live runner now derives corr/vol from the config via the shared "
        "engine.panel_stats, and sizes entries from d.weights as a multiple of the equal-weight "
        "slot. `overlay` remains unwired — the live runner is a daily-decision driver and has no "
        "intraday bars to build one from, so IntradayConfig is still inert here."),
}

#: Optional `decide()` arguments that silently disable a config group when omitted.
GATING_KWARGS = ("corr", "vol", "overlay")


#: The four decision drivers, and which shared rule modules each one actually invokes.
#:
#: `decide()` is not the only place a config group can go dead. `ExitConfig` reaches behaviour only
#: through `evaluate_exits`, and `GateConfig` only through `apply_gates`. A driver that does not call
#: one of those ignores that whole group, exactly as silently.
DRIVERS: dict[str, tuple[dict[str, bool], str]] = {
    "backtesting/families/rotation.py": (
        {"apply_gates": True, "evaluate_exits": True},
        "the single engine: gates on the completed daily panel, the shared exit evaluator at every "
        "slot on prices as of that instant — the same two calls the live runner makes"),
    "strategies/momentum_rotation/runner.py": (
        {"apply_gates": True, "evaluate_exits": True},
        "the live runner: gates and exits wired — this is what #197 P2 unified"),
    "strategies/momentum_rotation/nautilus.py": (
        {"apply_gates": True, "evaluate_exits": True},
        "FIXED: calls evaluate_exits with a trail RECONSTRUCTED from bars rather than remembered, "
        "so it needs no store and cannot repeat #197 B1 across a restart."),
    "strategies/qc345_rotation/nautilus.py": (
        {"apply_gates": False, "evaluate_exits": True},
        "QC345 has its own point-in-time feature builder rather than momentum's apply_gates, but "
        "the shared exit evaluator still has to reach the adapter or ExitConfig goes inert again."),
    "strategies/momentum_rotation/nautilus_intraday.py": (
        {"apply_gates": True, "evaluate_exits": True},
        "FIXED the same way. stop_atr_mult / flat_by_close remain constructor arguments and remain "
        "unsweepable — an intra-session mechanism alongside the session-level ExitConfig rules, not "
        "a substitute for them."),
}


def _calls_named(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text())
    return any(isinstance(n, ast.Call)
               and ((isinstance(n.func, ast.Name) and n.func.id == name)
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == name))
               for n in ast.walk(tree))


#: Drivers that reach `evaluate_exits` through a helper, and the helper that must be INVOKED.
#:
#: `_calls_named` only proves a call EXISTS somewhere in the module. It cannot tell a live call from
#: one sitting in a method nobody invokes — a mutation that replaced `forced = self._trail_exits()`
#: with `forced = set()` left the whole audit green, because the helper still contained the call.
#: Structural checks see presence, not reachability, and dead code is presence.
EXIT_HELPERS = {
    "strategies/momentum_rotation/nautilus.py": "_trail_exits",
    "strategies/qc345_rotation/nautilus.py": "_trail_exits",
    "strategies/momentum_rotation/nautilus_intraday.py": "_trail_exits",
}


@pytest.mark.parametrize("module", sorted(EXIT_HELPERS))
def test_the_exit_helper_is_actually_invoked_not_merely_defined(module):
    """Closes the dead-code hole in the audit above.

    Asserts the helper is CALLED, not just that it exists and internally calls the evaluator. Both
    are needed: the helper proves the rules are consulted, the call proves the consultation happens.
    """
    helper = EXIT_HELPERS[module]
    tree = ast.parse((SRC / module).read_text())
    called = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == helper for n in ast.walk(tree))
    defined = any(isinstance(n, ast.FunctionDef) and n.name == helper for n in ast.walk(tree))
    assert defined, f"{module}: `{helper}` is gone — update EXIT_HELPERS with what replaced it"
    assert called, (
        f"{module}: `{helper}` is defined but never invoked, so ExitConfig is inert again while "
        "every structural check still passes. This is the exact shape of #26.")


@pytest.mark.parametrize("module", sorted(DRIVERS))
def test_shared_rule_modules_reach_each_driver_as_audited(module):
    want, why = DRIVERS[module]
    for fn, expected in want.items():
        assert _calls_named(SRC / module, fn) is expected, f"{module}: `{fn}` wiring changed — {why}"


def test_the_legacy_intraday_runner_is_gone():
    """`runner_intraday` charged no execution cost, used synthetic instruments, did not compound and
    emitted no Report. It had zero production callers and one research caller, and the claim it
    produced — "cadence measured identical at 30/60/120/390 minutes" — was retracted. Deleted
    2026-09-08 with the engine consolidation; a runner that cannot be compared to the baseline must
    not exist to be reached for."""
    assert not (SRC / "backtesting/runner_intraday.py").exists(), (
        "runner_intraday is back. It charges no spread and cannot be compared to any recorded "
        "result — use runner_sessions.")


def _decide_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == "decide")
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == "decide"))]


def _imports_momentum_decide(tree: ast.Module) -> bool:
    """Does this module import `decide` from the MOMENTUM engine specifically?

    Matching on the bare name `decide` is not enough. `qc345_rotation` has its own engine with its
    own `decide(features, cfg, held)` — a different function with a different signature — and the
    audit flagged its runner as an unwired momentum driver. A false positive in a guard is worse
    than none: it trains everyone to add rows to silence it, and the next real omission gets the
    same treatment.
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module \
                and n.module.endswith("momentum_rotation.engine"):
            if any(a.name == "decide" for a in n.names):
                return True
    return False


def _call_sites() -> dict[str, list[ast.Call]]:
    out = {}
    for p in _src_only(sorted(SRC.rglob("*.py"))):
        tree = ast.parse(p.read_text())
        if not _imports_momentum_decide(tree):
            continue
        calls = _decide_calls(p)
        if calls:
            out[str(p.relative_to(SRC))] = calls
    return out


def test_every_decide_call_site_is_audited():
    """A new caller must be added to the table above — with its wiring stated — or this fails.

    This is the guard that makes the audit durable. Without it the fifth call site gets written,
    forgets `vol`, and the next size-varying variant quietly measures nothing.
    """
    found = set(_call_sites())
    assert found == set(EXPECTED), (
        f"decide() call sites changed.\n  new/unaudited: {sorted(found - set(EXPECTED))}\n"
        f"  gone: {sorted(set(EXPECTED) - found)}\n"
        "Add the new site to EXPECTED with its corr/vol wiring stated, or remove the stale row.")


@pytest.mark.parametrize("module", sorted(EXPECTED))
def test_call_site_wiring_matches_the_audit(module):
    calls = _call_sites()[module]
    kwargs = {k.arg for c in calls for k in c.keywords}
    want, why = EXPECTED[module]
    for arg in GATING_KWARGS:
        assert (arg in kwargs) is want[arg], f"{module}: `{arg}` wiring changed — {why}"


def test_the_verified_backtest_still_cannot_measure_the_intraday_overlay():
    """The finding that matters most for the #24 cadence variant, named so it reads as one.

    `runner_verified` is the harness behind every recorded result — real instruments, measured
    per-symbol half-spreads, deflated Sharpe. It does not pass `overlay`, so it cannot exercise
    `IntradayConfig` at all. The only runner that can is `runner_intraday`, which drives the
    Nautilus adapter and therefore does pass it — but that runner uses `TestInstrumentProvider`
    synthetic XNAS equities, charges NO execution cost, sizes at a fixed `equity_per_position`
    without compounding, and returns raw Nautilus reports rather than a `Report`, so it produces no
    Sharpe, no drawdown and no deflated Sharpe.

    That matters beyond bookkeeping. Deciding more often costs turnover, and turnover is paid in
    spread — so a zero-cost harness is structurally blind to the main cost of the thing a cadence
    variant varies. Any "cadence made no difference" result measured there is unsafe in the
    direction that flatters intraday trading.

    If this test fails because someone wired `overlay` into `runner_sessions`, that is good news:
    delete it, update the table, and re-run the cadence question on the one runner.
    """
    want, _ = EXPECTED["backtesting/families/rotation.py"]
    assert want["overlay"] is False, (
        "runner_sessions now passes `overlay` — the cadence and intraday-weight questions can "
        "finally be asked with real instruments and measured costs. Update EXPECTED, and re-check "
        "any recorded intraday finding that predates this.")


def test_the_live_runner_consumes_the_weights_it_asks_for():
    """Replaces the old known-gap test, which said to delete it once live was wired. It is wired.

    The replacement is not the mirror image of the old assertion. Passing `vol` alone is exactly the
    HALF-FIX that left `inverse_vol_sizing` inert in the backtest for a second reason: the weights
    were computed and then discarded by a flat sizing formula. So this checks the second half —
    that the live runner actually reads `weights` in its submit path — rather than trusting that
    wiring the inputs was enough.
    """
    src = (SRC / "strategies/momentum_rotation/runner.py").read_text()
    assert "weights[s2] * len(weights)" in src, (
        "pgrunner passes vol to decide() but no longer sizes from the resulting weights — that is "
        "the half-fix that made inverse_vol_sizing inert twice in the backtest")
