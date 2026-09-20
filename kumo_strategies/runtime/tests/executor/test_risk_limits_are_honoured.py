"""A `RiskLimits` field a runner ACCEPTS and never reads is an inert risk limit.

`RiskLimits` is shared: `PgSessionRunner` and `QC27SessionRunner` both declare `limits: RiskLimits`,
so both accept all ten fields, both render them in any config dump, and an operator setting one on
either lane gets no error. Only one of them enforces them.

Found 2026-08-24, following kumo-trading-platform's observation that `max_position_notional` has exactly two
mentions in `src/` — its definition, and `pgrunner.py:1227`. Walking the rest of the fields the same
way showed the cap was not the exception:

    pgrunner      reads 10 of 10
    qc27_runner   reads  1 of 10  (allocated_equity, and only since be244d9 gave it a second branch)

The nine TECHIVOL-005 does not enforce include every automatic stop this repo has:

    daily_loss_frac        `momentum_rotation.broker_equity`'s own docstring calls the daily-loss
                           limit "the only automatic stop this strategy has". TECHIVOL has none.
    max_positions          no position cap
    max_position_notional  no per-name ceiling — the finding that started this
    max_deployed_frac      no gross bound from the runner
    max_price_age_seconds  no staleness bound on the price it divides by
    max_source_age_hours / min_bar_coverage / min_rankable_frac / book_size

Some of those have an ANALOGUE inside qc27's own engine — `portfolio_size` bounds the book and
`gross_weight` bounds gross — but an analogue is not the field. Setting `max_positions=4` on
TECHIVOL's limits changes nothing and reports nothing, which is the defect: the knob turns and the
mechanism is not attached to it. This file does not demand qc27 grow nine enforcement paths; it
demands the gap be EXPLICIT, so that adding a limit to `RiskLimits` cannot silently be inert on a lane
that accepts it.

BOUND TO THE AST, NOT TO THE SOURCE TEXT. A grep-the-source assertion here would be satisfied by
deleting a comment that mentions a field name and broken by writing one — backwards in both
directions. The AST answers the only question that matters: does this runner ever evaluate
`self.limits.<field>`?
"""

from __future__ import annotations

import ast
import dataclasses

import pytest

from kumo_strategies.runtime.executor.runner import RiskLimits

from kumo_strategies.strategies import _layout

#: Every session runner, by package-relative path (ks#211: each lives in its strategy's folder).
_RUNNERS = {_layout.rel(p): p for p in _layout.runner_files()}

#: What each runner is KNOWN not to enforce, recorded so the gap is explicit rather than discovered
#: again. Shrinking one of these sets is a fix; growing one silently is the thing under test.
_UNENFORCED = {
    "strategies/qc27_tech_inverse_vol/runner.py": {
        "book_size", "max_positions", "max_position_notional", "max_deployed_frac",
        "max_source_age_hours", "max_price_age_seconds",
        "min_bar_coverage", "min_rankable_frac",
    },
    "strategies/momentum_rotation/runner.py": set(),
}


def _limit_reads(filename: str) -> set[str]:
    """Every `<...>.limits.<field>` this module actually evaluates."""
    tree = ast.parse(_RUNNERS[filename].read_text())
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Attribute) and n.value.attr == "limits"}


def _fields() -> set[str]:
    return {f.name for f in dataclasses.fields(RiskLimits)}


@pytest.mark.parametrize("filename", sorted(_UNENFORCED))
def test_the_set_of_UNENFORCED_limits_is_exactly_what_is_recorded(filename):
    """Both directions matter, which is why this is equality and not a subset check.

    A field that BECOMES inert (someone deletes the read, or adds a field to `RiskLimits` that a
    runner never wires) fails here instead of shipping as a knob attached to nothing. A field that
    becomes enforced also fails, forcing the record above to be corrected rather than left describing
    a gap that has closed.
    """
    read = _limit_reads(filename)
    assert read, f"the AST walk found no limit reads at all in {filename} — it proves nothing"

    inert = _fields() - read
    assert inert == _UNENFORCED[filename], (
        f"{filename} enforces a different set than recorded.\n"
        f"  newly inert (accepted, never read): {sorted(inert - _UNENFORCED[filename])}\n"
        f"  newly enforced (update the record): {sorted(_UNENFORCED[filename] - inert)}")


def test_pgrunner_enforces_EVERY_field_so_the_comparison_has_a_reference():
    """The control. Without a runner that reads all ten, "qc27 reads one" is a number with nothing to
    mean anything against — and a field added to `RiskLimits` and wired NOWHERE would satisfy every
    other assertion in this file."""
    assert _limit_reads("strategies/momentum_rotation/runner.py") >= _fields(), (
        f"pgrunner stopped enforcing {sorted(_fields() - _limit_reads('strategies/momentum_rotation/runner.py'))} — the "
        f"reference runner has a gap, so every other assertion here is measured against nothing")


def test_EVERY_runner_enforces_the_only_automatic_stop_this_repo_has():
    """The inverse of the test this replaces, which recorded the gap while it was open.

    Until kumo-trading-platform issue 548, `daily_loss_frac` was enforced in `pgrunner` and read nowhere in
    `qc27_runner`, so TECHIVOL-005 had no automatic loss stop at all while carrying a `RiskLimits`
    that advertised one with a 5% default. `momentum_rotation.broker_equity` calls this limit "the
    only automatic stop this strategy has".

    ASSERTED OVER EVERY RUNNER, not the two that exist. A third runner added later inherits this by
    existing, which is the whole difference between fixing the lane that was reported and fixing the
    class — QC27 was never the lane anyone reported, and it was the one with nothing.
    """
    runners = sorted(_RUNNERS)
    assert len(runners) >= 2, f"discovery found only {runners}; this test would be vacuous"
    # SCOPED BY HAVING `limits` AT ALL, not by reading some of it. The earlier version said
    # `and _limit_reads(f)` — which exempts any runner that reads ZERO limits fields, and reading
    # zero of ten IS the TECHIVOL shape this test exists to commemorate. The exemption excused
    # precisely the defect.
    def _carries_limits(name):
        tree = ast.parse(_RUNNERS[name].read_text())
        return any(isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                   and n.target.id == "limits" for c in ast.walk(tree)
                   if isinstance(c, ast.ClassDef) for n in c.body)

    missing = [f for f in runners
               if _carries_limits(f) and "daily_loss_frac" not in _limit_reads(f)]
    assert not missing, (
        f"{missing} accept a RiskLimits advertising a daily-loss stop and never read it, so those "
        f"lanes cannot halt themselves on risk under any condition. Worse, they leave no trace: "
        f"'had no reason to halt' and 'has no ability to halt' produce the same journal — nothing "
        f"(kumo-trading-platform issue 548).")


def test_the_stop_is_not_merely_READ_but_reaches_a_HALT():
    """Reading the field is not enforcing it. A runner could evaluate `limits.daily_loss_frac` into
    a log line and satisfy the test above while never halting.

    Bound to the call, not to a substring: every runner that reads the field must reach
    `daily_loss.enforce`, which is the only thing in this repo that calls `lifecycle.halt` for a
    loss. A grep for "halt" is satisfied by the docstrings explaining this defect, including these.
    """
    for name, filename in sorted(_RUNNERS.items()):
        if "daily_loss_frac" not in _limit_reads(name):
            continue
        tree = ast.parse(filename.read_text())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "enforce" in called, (
            f"{name} reads daily_loss_frac without calling `daily_loss.enforce` — it has "
            f"the knob and not the stop. A second derivation of this rule is how the two halves of "
            f"'a usable equity' diverged (#111).")
