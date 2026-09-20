"""A config field the BACKTEST enforces must reach the LIVE runner, or be declared dead (#714).

    TECHIVOL-005 has traded on paper since 2026-08-04 WITHOUT the 2% portfolio stop it was
    verified with. `stop_loss_portfolio_frac` is enforced at
    `backtesting/runner_qc27_verified.py:257` and read by NOTHING under `runtime/`.

    **The live strategy is not the strategy that was verified, and the difference is its stop.**

WHY A DETECTOR RATHER THAN A FIX FOR THAT ONE FIELD. This is the third time a config field has been
"typechecked, backtested and silently ignored live" — #197 B12 was `stall_days`, `max_hold_days` and
`off_peak_pct`; #26 was `inverse_vol_sizing`; this is the risk control. Fixing the instance leaves
the mechanism, and the mechanism is that nothing compares the two sides.

TWO WAYS A FIELD IS BACKTEST-ONLY, and the second is what a naive walk misses:

    DIRECT       the backtest reads `cfg.x` and nothing under `runtime/` does.
    VIA A PURE   a function in `strategies/` reads `cfg.x`, and only `backtesting/` calls it.
    FUNCTION     `filter_tech_universe` reads `sector_value` and `require_us_country`
                 (`engine.py:77-78`) and is called from `runner_qc27_verified.py:86` and
                 `runner_qc27_nautilus.py:70` — nowhere else. So the verified strategy trades a
                 sector-filtered US common-stock universe and the live lane trades whatever symbol
                 list cockpit passes.

WHAT THIS DETECTOR CANNOT SEE, stated rather than implied — an undeclared limit is how three of
these survived:

  * Reads are matched by ATTRIBUTE NAME across all configs, not per class. A same-named field read
    live for another lane counts as live here; `test_names_shared_across_configs_are_AMBIGUOUS`
    enumerates which names that currently affects.
  * Call-following is ONE HOP, and only for calls made by bare name.
  * A pure function called from BOTH sides marks its fields live, even if the live caller passes a
    config that never sets them.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib
import pkgutil

import pytest

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]



from kumo_strategies.strategies import _layout


def _files(root: str) -> list[pathlib.Path]:
    return _src_only(sorted(pathlib.Path(root).rglob("*.py")))


#: THE THREE SIDES, as file lists. A strategy's lane and session runner live in its own folder now
#: (ks#211), so "live" is no longer a directory: it is everything under runtime/ plus every
#: `strategies/<name>/{nautilus,nautilus_intraday,runner}.py`, and "pure" is the rest of strategies/.
#: Walking strategies/ whole would count a lane's `run` as a pure function the backtest reaches.
_RUNTIME = _layout.all_runtime_sources()
_BACKTEST = _files("kumo_strategies/backtesting")
_PURE = [f for f in _files("kumo_strategies/strategies") if f.stem not in _layout.RUNTIME_STEMS]


def _all_configs() -> dict[str, type]:
    """Every dataclass config this package ships, DISCOVERED.

    Hand-listing three of them is how a fourth lane joins without anybody comparing its sides —
    the failure this file is about, one level up.
    """
    import kumo_strategies.strategies as pkg

    out = {}
    for mod in pkgutil.walk_packages(pkg.__path__, f"{pkg.__name__}."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:                                              # noqa: BLE001
            continue
        for name in dir(m):
            obj = getattr(m, name)
            if isinstance(obj, type) and dataclasses.is_dataclass(obj) and name.endswith("Config"):
                out[name] = obj
    return out


def _cfg_reads(tree) -> set[str]:
    """Config fields genuinely USED in this tree — refusal guards excluded.

    READING A FIELD TO REFUSE IT IS NOT IMPLEMENTING IT. A guard that rejects a config the live path
    cannot honour must not make the field look supported, or adding the guard closes the very gap it
    announces.

    THE GUARD TEST IS "the body raises AT ALL", not "the body is entirely Raise". The stricter form
    was foolable in the worst direction: one log line before the raise reclassified the field as
    read-live, the divergence vanished, and the equality assertion below then instructed the
    maintainer to DELETE the ledger entry — green, with the control still unimplemented.
    """
    def cfg_attr(node):
        base = node.value
        name = (base.id if isinstance(base, ast.Name)
                else base.attr if isinstance(base, ast.Attribute) else "")
        return node.attr if ("cfg" in name.lower() or "config" in name.lower()) else None

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(isinstance(b, ast.Raise) for b in ast.walk(node)):
            for t in ast.walk(node):
                guarded.add(id(t))

    used = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            attr = cfg_attr(node)
            if attr:
                used.add(attr)
    return used


def _reads_under(files: list[pathlib.Path]) -> set[str]:
    """PER FILE, then unioned. Subtracting a per-file refusal set from a cumulative one made the
    answer depend on directory traversal order."""
    out: set[str] = set()
    for f in files:
        try:
            out |= _cfg_reads(ast.parse(f.read_text()))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return out


def _pure_function_reads() -> dict[str, set[str]]:
    """{function name: config fields it reads} for everything under `strategies/`."""
    out: dict[str, set[str]] = {}
    for f in _PURE:
        try:
            tree = ast.parse(f.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                reads = _cfg_reads(fn)
                if reads:
                    out.setdefault(fn.name, set()).update(reads)
    return out


def _called_names_under(files: list[pathlib.Path]) -> set[str]:
    out: set[str] = set()
    for f in files:
        try:
            tree = ast.parse(f.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            # BOTH callee shapes. A bare `filter_asset_universe(...)` is an `ast.Name`; the same
            # call through a module alias — `QC345.filter_asset_universe(...)`, how the one runner's
            # monthly family spells it (#270) — is an `ast.Attribute`, and collecting names only
            # made that read invisible: the field flipped to "appears read live" the day the call
            # was namespaced, with nothing about the code's behaviour changed. Same hole
            # `test_foreign_order_events.py` once had with `self.protective_close(...)`.
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    out.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    out.add(n.func.attr)
    return out


def _side_reads(files: list[pathlib.Path]) -> set[str]:
    """Direct reads in `files`, plus those reached through a pure function they call."""
    pure = _pure_function_reads()
    calls = _called_names_under(files)
    reached = [v for k, v in pure.items() if k in calls]
    via_pure = set().union(*reached) if reached else set()
    return _reads_under(files) | via_pure


#: KNOWN divergences, each with the reason it is accepted. EQUALITY-asserted: adding one fails, and
#: closing one fails until its entry is deleted. A stale exemption is how these survive.
_BACKTEST_ONLY = {
    # CLOSED 2026-09-11: `signal`, `window`, `dwell`, `action` and the `market_view` group itself
    # were all recorded here, each entry naming the same closing condition — "closes when #873's
    # `entries_blocked`/`emergency_exit` hooks land and the adapter calls `market_state`." They have
    # landed: `runtime/nautilus/market_aware.py` implements all three hooks on all five lanes and
    # calls `market_state` from `entries_blocked`.
    #
    # DELETED ON EVIDENCE, NOT ON RECLASSIFICATION — this detector's own message warns that the two
    # look identical. `window` and `dwell` reach the live path only THROUGH `market_state`, so being
    # "read" was shown rather than asserted: same lane, same bars, two settings, two different live
    # answers (`test_DWELL_changes_the_live_answer_so_it_is_read_rather_than_merely_present` and
    # `test_WINDOW_changes_the_live_answer_too`). `signal` and `action` are read directly in the
    # hooks, and TECHIVOL's declared 50d/EXIT_ONLY view is exercised end to end in both regimes.
    #
    # WHAT IS STILL TRUE AND IS NOT THIS DETECTOR'S QUESTION: the lanes now ANSWER, and they do not
    # yet ACT. Cockpit's poller owns the acting half (#873), so a lane whose `entries_blocked` says
    # YES still opens positions until the platform stops it. "Declared and answered" is not
    # "armed", and nothing here should be read as saying the view is live in production.
    "stop_loss_portfolio_frac": (
        "TECHIVOL-005's 2% portfolio stop. Enforced at `runner_qc27_verified.py:257`, read by "
        "nothing under `runtime/`. Live since 2026-08-04 without it, so the live lane is not the "
        "strategy that was verified. NOT fixable by a constructor refusal demanding 0.0: in the "
        "verifier `threshold = 0.0` makes `drawdown_value > threshold` fire for ANY losing "
        "position, so 0.0 means 'no stop' live and 'stop out everything' in research, and the "
        "honest config becomes un-re-verifiable. Needs a `None` sentinel reconciled on both sides, "
        "and cockpit's QC27_STOP_LOSS_PORTFOLIO_FRAC override wired or retired, before any "
        "refusal. kumo-trading-platform issue 714."),
    "cash_proxy_symbol": (
        "GLD, read live by nothing. NOT inert — an earlier version of this entry claimed it was, on "
        "the grounds that `cash_proxy_weight` measured 0.0 on 301 of 301 sessions. That covers only "
        "the decision-weight path at `runner_qc27_verified.py:166`. The RESIDUAL-CASH SWEEP at "
        "`:236-250` fires whenever leftover cash exists, independent of that weight, and carries "
        "its own `gld_sweep_buys` diagnostic and its own test. The verified equity curve holds GLD "
        "on residual cash; the live lane leaves it idle. A real economic divergence."),
    "sector_value": (
        "Enforced inside `filter_tech_universe` (`engine.py:77`), called only from "
        "`runner_qc27_verified.py:86` and `runner_qc27_nautilus.py:70`. The verified strategy "
        "trades a sector-filtered universe; the live lane trades whatever symbol list cockpit "
        "passes, with no sector enforcement anywhere on that path."),
    "require_us_country": (
        "Same function, same two callers (`engine.py:78`). The verified QC27 universe is US common "
        "stock; the live one is whatever cockpit passes, with no country enforcement on that path "
        "at all — so a non-US listing entering the pool would be traded by the live lane and "
        "excluded from the run that justified it."),
    "asset_universe_mode": (
        "QC345's, and found by this detector rather than by review. Defaults to "
        "'fundamental_like', and is enforced inside `filter_asset_universe` "
        "(`qc345_rotation/engine.py:156`) — called ONLY from the one runner's monthly family (`backtesting/monthly.py`). So "
        "the verified QC345 trades an asset-filtered universe and QC345-003 live trades the raw "
        "symbol list. Exactly the QC27 sector/country shape on a second lane, which is the "
        "argument for this being a detector and not four fixes."),
}


def test_the_walk_finds_reads_on_both_sides():
    """VACUITY GUARD. If the receiver rule stopped matching, both sets would empty, their difference
    would empty, and this file would report green while comparing nothing."""
    live, backtest = _side_reads(_RUNTIME), _side_reads(_BACKTEST)
    assert len(live) > 10, f"only {len(live)} live reads found; the walk has stopped working"
    assert len(backtest) >= 5, f"only {len(backtest)} backtest reads found"
    assert live & backtest, "the two sides share no field at all, so they measure different things"


def test_the_ONE_HOP_call_following_resolves_something():
    """The pure-function hop is what catches `sector_value` and `require_us_country`. If it resolved
    nothing it would degrade silently to the naive walk that missed both."""
    pure = _pure_function_reads()
    assert pure, "no config-reading functions found under strategies/"
    assert {k for k in pure if k in _called_names_under(_BACKTEST)}, (
        "no pure function is reached from backtesting/; the hop resolves nothing and the detector "
        "is back to direct reads only")


def test_every_backtest_enforced_field_is_read_live_or_DECLARED_dead():
    """THE HEADLINE. Equality, not subset.

    A `gone` entry does NOT by itself mean the gap closed: a refusal guard that gained a log line,
    or a pure function that gained a live caller, reclassifies a field without implementing it.
    """
    fields: set[str] = set()
    for cls in _all_configs().values():
        fields |= {f.name for f in dataclasses.fields(cls)}
    found = (fields & _side_reads(_BACKTEST)) - _side_reads(_RUNTIME)
    recorded = set(_BACKTEST_ONLY)

    new = found - recorded
    assert not new, (
        f"enforced in the BACKTEST and read by NOTHING live: {sorted(new)}. The measured strategy "
        f"and the trading one differ by exactly these. Implement them live, or record each here "
        f"with the reason it is dead.")
    gone = recorded - found
    assert not gone, (
        f"recorded as backtest-only but now appear read live: {sorted(gone)}. Before deleting the "
        f"entries, CHECK the field is genuinely USED — reclassification is not implementation.")


@pytest.mark.parametrize("field", sorted(_BACKTEST_ONLY))
def test_each_recorded_gap_says_WHY(field):
    """A name in a list is an exemption. A name with a reason is a decision, and the next person can
    judge whether it still holds."""
    assert len(_BACKTEST_ONLY[field]) > 120, f"{field} is exempted without an actionable reason"


def test_names_shared_across_configs_are_AMBIGUOUS():
    """THE DETECTOR'S OWN LIMIT, enumerated rather than implied.

    Reads match by attribute NAME across every config, so a field of the same name read live for
    another lane counts as live here. Naming the overlap keeps a divergence hiding behind one a
    KNOWN risk instead of an invisible one.
    """
    seen: dict[str, list[str]] = {}
    configs = _all_configs()
    assert configs, "no configs discovered; the ambiguity report describes nothing"
    for cls_name, cls in configs.items():
        for f in dataclasses.fields(cls):
            seen.setdefault(f.name, []).append(cls_name)
    shared = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    for field in _BACKTEST_ONLY:
        if field not in shared:
            continue
        # A SHARED NAME IS ALLOWED ONLY IF THE REASON OWNS THE AMBIGUITY. The detector matches by
        # attribute name, so a field of the same name read live for ANOTHER lane would count as
        # live here — the entry could be true for one lane and false for the one it describes.
        #
        # `market_view` is the case that forced this: it is now declared on all three configs, and
        # the claim "read by nothing under runtime/" happens to hold for every one of them. That
        # is a fact to be STATED, not a reason to drop the check. So the reason must name each
        # config sharing the name, and an entry that quietly acquires a fourth sharer fails until
        # someone re-checks it.
        reason = _BACKTEST_ONLY[field]
        unnamed = [c for c in shared[field] if c not in reason]
        assert not unnamed, (
            f"{field} is recorded as backtest-only and its name is shared with {shared[field]}, "
            f"but the reason does not name {unnamed}. The detector matches by NAME, so the "
            f"evidence for this entry may be coming from another lane entirely — either name "
            f"every sharer and state that the claim holds for each, or split the entry.")


def test_the_0_point_0_sentinel_would_NOT_mean_no_stop():
    """WHY THE OBVIOUS FIX IS WRONG, pinned so nobody re-proposes it — I did, and shipped it briefly.

    A constructor refusal demanding `stop_loss_portfolio_frac=0.0` is the same shape as
    `momentum_price_field`'s, and it is wrong here. Asserted as ARITHMETIC rather than by grepping
    the verifier: a source-substring assertion is satisfied by deleting the line it looks for, and
    this repo has a test that forbids exactly that.
    """
    equity = 100_000.0
    losing_position_drawdown = 1.0          # one dollar under water; the mildest loss there is

    disabled = 0.0 * equity                 # what the "acknowledgement" value produces
    real_stop = 0.02 * equity               # what the verified config produces

    assert losing_position_drawdown > disabled, (
        "0.0 does NOT disable the stop in the verifier — it makes the threshold zero, so ANY "
        "position under water is stopped out. The value meaning 'no stop' live means 'stop out "
        "everything' in research, and the honest config can never be re-verified.")
    assert not losing_position_drawdown > real_stop, (
        "the 2% threshold no longer tolerates a small drawdown; re-check this reasoning")


def test_the_refusal_exclusion_actually_excludes():
    """`_cfg_reads` must not count a field read ONLY to reject it. Exercised on a synthetic tree,
    because no such guard currently exists in the package — a mechanism nothing exercises is one
    nobody knows is broken, and the first version of it cancelled itself out."""
    import ast as _ast

    refusing = _ast.parse(
        "def f(cfg):\n"
        "    if cfg.doomed:\n"
        "        log('nope')\n"
        "        raise ValueError(f'doomed={cfg.doomed}')\n")
    assert "doomed" not in _cfg_reads(refusing), (
        "a field read only inside a guard that raises counts as implemented, so adding the guard "
        "closes the very gap it announces")

    using = _ast.parse("def f(cfg):\n    return cfg.doomed * 2\n")
    assert "doomed" in _cfg_reads(using), "a real use is no longer counted at all"

    both = _ast.parse(
        "def f(cfg):\n"
        "    if cfg.doomed:\n"
        "        raise ValueError('no')\n"
        "def g(cfg):\n"
        "    return cfg.doomed\n")
    assert "doomed" in _cfg_reads(both), (
        "a field refused in one place and USED in another must count as used — refusal excludes "
        "only where refusal is all that happens")
