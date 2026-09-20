"""Sizing divides by the book that EXISTS, not the one that was planned — everywhere or nowhere.

`min_abs_gap_pct` declines entries, and a declined entry used to leave its slot's capital in cash.
This is the rule that redistributes it, and the three ways it can go wrong are the three this file
covers:

  the arithmetic is wrong        -> behaviour tests on `sizing_denominator`
  it is inert where it matters   -> the AST tests binding all three sizing sites, live included
  it double-counts on resume     -> the set-arithmetic test, which is the one a len()-based
                                    implementation passes everywhere except the path that runs
                                    after a partial failure
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from kumo_strategies.strategies.momentum_rotation.config import (
    MomentumRotationConfig, PortfolioConfig)
from kumo_strategies.strategies.momentum_rotation.engine import sizing_denominator

SRC = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies"
OFF = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=8, buffer=5))
ON = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=8, buffer=5, size_to_book=True))


def test_off_is_the_planned_book_whatever_is_actually_held():
    """Every result recorded before 2026-09-04 was measured with this off; it must be a no-op."""
    for book in (0, 1, 3, 8, 12):
        assert sizing_denominator(book, 8, OFF) == 8


def test_on_divides_by_the_book_that_exists():
    assert sizing_denominator(3, 8, ON) == 3
    assert sizing_denominator(8, 8, ON) == 8


def test_a_full_book_sizes_IDENTICALLY_with_the_rule_on_or_off():
    """The rule may only touch a THIN book. If it changed a full one it would be a leverage change
    wearing a cash-drag fix's name."""
    assert sizing_denominator(8, 8, ON) == sizing_denominator(8, 8, OFF)


@pytest.mark.parametrize("book", [0, -1, None, 3.0, "3"])
def test_an_unusable_book_count_falls_back_to_the_plan(book):
    """Never divide by zero, and never put the whole book into one name because a decision arrived
    empty. `3.0` and `"3"` are rejected too: a float denominator would silently change sizing on a
    path that only ever meant to pass an int."""
    assert sizing_denominator(book, 8, ON) == 8


def _divides_by_the_shared_rule(relative: str) -> bool:
    tree = ast.parse((SRC / relative).read_text())
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
               and isinstance(n.right, ast.Call) and isinstance(n.right.func, ast.Name)
               and n.right.func.id == "sizing_denominator"
               for n in ast.walk(tree))


@pytest.mark.parametrize("driver", ["strategies/momentum_rotation/runner.py",
                                    "backtesting/families/rotation.py"])
def test_EVERY_sizing_site_divides_by_the_shared_rule(driver):
    """pgrunner is the one that matters: a config field honoured only in the backtest is the defect
    this repo keeps paying for (#26, #197 B12) — measured well, inert where real positions are.

    Bound to the DIVISION, not to the call. A file could call `sizing_denominator`, ignore the
    result and keep its old denominator, and an "is it called" assertion would pass.
    """
    assert _divides_by_the_shared_rule(driver), (
        f"{driver} does not divide by sizing_denominator, so size_to_book cannot change sizing "
        f"there — the config would set a rule that typechecks, deploys and does nothing")


def test_the_live_book_count_is_SET_arithmetic_not_len_arithmetic():
    """THE RESUME PATH. `_submit` runs again after a partial failure with only the entries not yet
    sent, and some of those may already sit in `held_qty`. `len(held) - len(exits) + len(enters)`
    double-counts them and sizes every position too small, on precisely the path that runs when
    something has already gone wrong. A union counts each name once.
    """
    src = (SRC / "strategies/momentum_rotation/runner.py").read_text()
    tree = ast.parse(src)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "book_after" for t in n.targets)]
    assert assigns, "pgrunner no longer derives `book_after`"
    expr = ast.dump(assigns[0].value)
    assert "BinOp" in expr and "BitOr" in expr and "Sub" in expr, (
        "`book_after` is not set arithmetic — a len()-based count double-counts a name that is "
        "both already held and still queued to enter, which is the resume path")
    assert "Add" not in expr, "`book_after` adds counts; that is the double-count this test exists for"


def test_the_backtest_counts_the_POST_decision_book():
    """`dec.hold` is survivors plus entries with exits already removed. Counting the pre-decision
    book would include names being sold this session and size every entry too small."""
    src = (SRC / "backtesting/families/rotation.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "sizing_denominator"]
    assert len(calls) >= 2, "runner_sessions sizes on both paths (intraday and daily) — expected two calls"
    # The post-decision book has two spellings in the one runner: `dec.hold` on the daily path
    # (survivors + entries, exits removed) and `book_after` on the intraday path (the same set,
    # built per slot). Either is the book that EXISTS after the decision; anything else is not.
    for call in calls:
        first = ast.dump(call.args[0])
        assert "hold" in first or "book_after" in first, \
            f"first argument is not the post-decision book: {first}"
