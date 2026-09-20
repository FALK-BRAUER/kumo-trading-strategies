"""The overnight gap dead band: one rule, honoured on both sides, fired only where it is defined.

Three separate failure modes are covered, because this rule sits exactly where the repo's expensive
defects live:

  the rule is wrong          -> behaviour tests on `gap_declines_entry`
  the rule is INERT live     -> the AST test binding both drivers to the shared predicate (#197 B12,
                                where three of four exit rules existed and never fired)
  the rule fires where it
  was never measured         -> the slot gate, and the config test that a named slot is real
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

from kumo_strategies.strategies.momentum_rotation.config import (
    ExecutionConfig, MomentumRotationConfig)
from kumo_strategies.strategies.momentum_rotation.engine import (
    filter_entries_by_gap, gap_declines_entry)

SRC = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies"
ARMED = MomentumRotationConfig(execution=ExecutionConfig(min_abs_gap_pct=0.015))


@pytest.mark.parametrize("fill,declined,why", [
    (100.0, True, "no gap at all is the middle of the dead band"),
    (101.4, True, "+1.4% is inside 1.5%"),
    (98.6, True, "-1.4% is inside 1.5% — the band is symmetric"),
    # NOT the exact edge: 101.5/100 - 1 == 0.014999999999999902 in binary floating point, so
    # "is 1.5% inside a 1.5% band" is a question about float representation rather than about the
    # rule. The edge is deliberately unspecified; what matters is that either side of it is right.
    (101.6, False, "just outside the band is admitted"),
    (98.4, False, "just outside on the downside is admitted too"),
    (102.0, False, "a real gap UP is the tail this rule exists to keep"),
    (98.0, False, "a real gap DOWN is equally the tail — magnitude, not direction"),
])
def test_the_dead_band_declines_the_flat_middle_and_keeps_BOTH_tails(fill, declined, why):
    """The bucket study found the shape is MAGNITUDE: big gaps pay in both directions and the flat
    middle is where the return disappears. A rule that kept only gap-ups would be the DIRECTIONAL
    hypothesis, which the research tested and did not support."""
    assert gap_declines_entry(fill, 100.0, ARMED) is declined, why


def test_the_SAME_verdict_at_every_slot_because_the_gap_is_a_property_of_the_DAY():
    """`today's open / yesterday's close` is fixed once the market opens, so a lane deciding at
    09:35, 12:00 and 15:40 must reach the same answer at all three with nothing remembered between
    them.

    An earlier version took the FILL price and gated the rule to a slot list. At 09:35 the fill is
    ~the open so it agreed; by midday `fill / prior close` is the day's move so far, a quantity
    nothing has measured, and the slot list existed to hide that. The operator caught it: the gap does not
    change during the day, so the slots cannot disagree.
    """
    for _slot in ("open+5m", "open+150m", "close-20m"):
        assert gap_declines_entry(100.2, 100.0, ARMED) is True
        assert gap_declines_entry(103.0, 100.0, ARMED) is False


def test_an_unset_rule_never_declines():
    assert gap_declines_entry(100.0, 100.0, MomentumRotationConfig()) is False


@pytest.mark.parametrize("fill,prior", [
    (None, 100.0), (100.0, None), (float("nan"), 100.0), (100.0, float("nan")),
    (0.0, 100.0), (100.0, 0.0), (-1.0, 100.0),
])
def test_missing_or_impossible_prices_ADMIT_the_entry(fill, prior):
    """Fails open, like `_diversified` on a missing correlation. Declining on absent data lets a feed
    hiccup quietly shrink the book, and a NaN would otherwise survive a `> 0` comparison and decline
    EVERY entry — the `or`-default shape that disarmed a halt in #111."""
    assert gap_declines_entry(fill, prior, ARMED) is False


def test_the_prior_close_is_supplied_BY_THE_CALLER_not_looked_up():
    """`_prior_closes` is gone. It searched the runner's panel for the newest close before a date —
    which was the right answer to the wrong question, because that panel's newest row IS yesterday
    and the rule needs yesterday's close against TODAY's open, not the day-before's.

    The prior close is now `last`, the newest close in the trimmed panel, handed straight in. A name
    missing from it simply has no prior close and `gap_declines_entry` admits, which is the same
    fail-open behaviour the lookup had.
    """
    tree = ast.parse((SRC / "strategies/momentum_rotation/runner.py").read_text())
    defs = [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert "_prior_closes" not in defs, (
        "_prior_closes is back — the prior close must come from `last`, not from a panel search")


def _calls_the_shared_predicate(relative: str) -> bool:
    tree = ast.parse((SRC / relative).read_text())
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in {"gap_declines_entry", "filter_entries_by_gap"}
               for n in ast.walk(tree))


def test_the_filter_removes_the_declined_names_and_reports_them():
    """Behavioural, on the function the live runner assigns from."""
    survivors, declined = filter_entries_by_gap(
        ("FLAT", "GAPUP", "GAPDOWN"),
        {"FLAT": 100.2, "GAPUP": 103.0, "GAPDOWN": 97.0},
        {"FLAT": 100.0, "GAPUP": 100.0, "GAPDOWN": 100.0},
        ARMED)
    assert survivors == ("GAPUP", "GAPDOWN")
    assert set(declined) == {"FLAT"} and declined["FLAT"] == pytest.approx(0.002)


def test_the_filter_is_a_NO_OP_when_the_rule_is_unset():
    enters = ("FLAT", "ALSOFLAT")
    prices = {"FLAT": 100.0, "ALSOFLAT": 100.0}
    assert filter_entries_by_gap(enters, prices, prices,
                                 MomentumRotationConfig()) == (enters, {})


def test_the_live_runner_ASSIGNS_enters_from_the_gap_filter():
    """Stronger than "does it call the function". An earlier version of this test asserted only that
    the Call node existed, and a mutation neutering the call to `if False and ...` passed it — a
    structural test certifying a rule that never fires.

    Binding the ASSIGNMENT is what closes that: the live path must compute `enters` FROM the filter,
    so there is no guarded branch to switch off and still look right.
    """
    tree = ast.parse((SRC / "strategies/momentum_rotation/runner.py").read_text())
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name)
               and n.value.func.id == "filter_entries_by_gap"
               and any("enters" in {e.id for e in ast.walk(t) if isinstance(e, ast.Name)}
                       for t in n.targets)]
    assert assigns, (
        "pgrunner does not assign `enters` from filter_entries_by_gap, so min_abs_gap_pct cannot "
        "remove an entry live — the config would set a rule that typechecks, deploys and does "
        "nothing (#26, #197 B12)")


@pytest.mark.parametrize("driver", ["strategies/momentum_rotation/runner.py",
                                    "backtesting/families/rotation.py"])
def test_BOTH_drivers_reach_the_SAME_shared_rule_so_it_cannot_drift(driver):
    """The defect this repo keeps paying for: a rule written twice fires in the backtest and not in
    the thing holding real positions. `give_back_frac` and friends shipped that way (#197 B12), and
    `max_correlation` shipped as a flag that typechecked and did nothing (#26).

    Bound to the AST rather than to a substring: a grep test passes when someone deletes a warning
    comment and fails when someone writes one.
    """
    assert _calls_the_shared_predicate(driver), (
        f"{driver} reaches neither gap_declines_entry nor filter_entries_by_gap, so "
        f"min_abs_gap_pct is inert in it — a config can set the rule, typecheck, deploy, and "
        f"change nothing on that side")


def test_the_live_runner_takes_TODAYS_OPEN_FROM_THE_CALLER():
    """WEAK BY CONSTRUCTION, and kept only as a tripwire. This binds the SHAPE of the call; it
    cannot see which DATE the open belongs to, which is exactly how the shipped defect passed every
    test in this file while computing the previous session's gap.

    `test_gap_uses_todays_open.py` is the real guard — it drives `PgSessionRunner.run` and was seen
    red against the defective implementation. If this test and that one ever disagree, believe that
    one."""
    src = (SRC / "strategies/momentum_rotation/runner.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "filter_entries_by_gap"]
    assert calls, "pgrunner does not call filter_entries_by_gap"
    second = ast.dump(calls[0].args[1])
    assert "opens" in second, (
        f"today's open must come from the caller's `opens`, not from the panel — the panel's newest "
        f"row is YESTERDAY: {second}")
