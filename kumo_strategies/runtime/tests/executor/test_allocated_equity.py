"""Sizing must respect the strategy's allocation, not the whole account (#32).

MOMENTUM-002 carries a 20k target on a ~100k account. `slot` was
`broker.equity() * max_deployed_frac / max_positions`, so it sized every entry off 100k — about 10k
a name — and the budget gate rejected the order at submission. The failure mode is not "buys
smaller" or "buys fewer": it is "buys the same size and gets refused", and once the book does fit
under target it rebuilds as TWO oversized positions where the strategy is built for eight.

That is the deployment risk for BCTROT specifically: the backtest measured an 8-name equally
weighted book of the strategy's own capital. A 2-name book sized off the account is a different
portfolio with roughly 4x the single-name concentration.
"""

from __future__ import annotations

from kumo_strategies.runtime.executor.runner import RiskLimits


def _slot(limits: RiskLimits, account_equity: float) -> float:
    """The sizing expression under test, isolated from the async runner.

    Divides by `book_size`, NOT `max_positions`. This helper is a second copy of the production
    formula and drifted from it the moment the two were split — it kept dividing by the cap, so it
    reported a position size no code path would ever produce. A duplicated formula is a duplicated
    bug; if this needs changing again, make the runner expose the expression instead."""
    alloc = limits.allocated_equity
    equity = account_equity if alloc is None else float(alloc)
    return equity * limits.max_deployed_frac / max(limits.book_size, 1)


def test_without_an_allocation_it_sizes_off_the_account_exactly_as_before():
    """Every running config leaves this unset. The default must not move a live book."""
    assert _slot(RiskLimits(), 100_000.0) == 100_000.0 * 0.80 / 8


def test_with_an_allocation_it_sizes_off_THAT():
    limits = RiskLimits(allocated_equity=20_000.0)
    assert _slot(limits, 100_000.0) == 20_000.0 * 0.80 / 8


def test_the_allocation_is_what_makes_the_book_the_RIGHT_SHAPE():
    """The point, stated as the number that matters. On a 20k target with 8 slots, sizing off the
    account fills the allocation with 2 names; sizing off the allocation fills it with 8."""
    account, target = 100_000.0, 20_000.0
    deployable = target * RiskLimits().max_deployed_frac      # what the strategy may actually put on

    unallocated = _slot(RiskLimits(), account)                # 100k * 0.8 / 8 = 12_500... no: 10_000
    allocated = _slot(RiskLimits(allocated_equity=target), account)

    assert int(deployable // unallocated) == 1, (
        "sizing off the account fills the whole deployable allocation with ONE name — the strategy "
        "is designed for eight")
    assert int(deployable // allocated) == 8, "sizing off the allocation gives the designed eight"


def test_a_zero_allocation_SIZES_TO_ZERO_and_does_not_reach_the_account():
    """REVERSED 2026-08-24. This asserted the opposite, and the reasoning was wrong in consequence.

    It read: "`or` treats 0.0 as unset deliberately. A strategy allocated zero should be stopped by
    the gate, not silently submit zero-quantity orders that look like decisions." But the fallback
    does not stop such a lane, it ARMS it — 0.0 is falsy, so a lane granted nothing sized off the
    whole account. kumo-trading-platform measured the exposure on kumo-staging: BCTROT-004 allocated 100000,
    MANUAL-001 / MOMENTUM-002 / QC345-003 / TECHIVOL-005 all allocated 0, account equity 999,215.43.
    Any of those four deciding would have sized off ~1M of BCTROT's capital. What prevented it was
    three accidents and no guard: two lanes disabled by flag, one with no lifecycle row, and an
    `exec_action_log` on that stack that is still empty.

    The gate was never the backstop it was assumed to be either: `budget_gate` defaults to None in
    `QC27SessionRunner`, so a lane wired without cockpit's `may_submit` has nothing between the size
    and the broker.

    Zero-quantity orders are not the alternative risk: both runners already refuse them
    (`notional <= 0` skips the symbol), so zero allocation degrades to INACTION, which is what a
    lane granted nothing should do."""
    assert _slot(RiskLimits(allocated_equity=0.0), 100_000.0) == 0.0


def test_an_UNSET_allocation_still_falls_back_to_the_account():
    """The half that must NOT move. `None` means cockpit configured no allocation — every running
    config passes it, and falling through to the account is today's behaviour. The zero fix narrowed
    the fallback; if this goes red it deleted it."""
    assert _slot(RiskLimits(), 100_000.0) == 100_000.0 * 0.80 / 8


# -- the cap and the sizing divisor are two different numbers ---------------------------------------
def test_sizing_uses_the_BOOK_SIZE_and_the_cap_is_separate():
    """MOMENTUM-002 could not rotate for two weeks because ONE number did TWO jobs.

    `max_positions` was both the position cap and the sizing divisor
    (`slot = equity * max_deployed_frac / max_positions`). MOMENTUM's researched book is n_hold=8 and
    its limits are a bare `RiskLimits()`, so the cap was also 8 — target book size equal to the cap,
    zero rotation headroom. It sold N and had all N replacements refused by the names it had just
    sold. 19 entries lost across seven sessions from 2026-08-06.

    Raising the cap was not available as a fix precisely BECAUSE of the conflation: it would have
    silently halved every position. Splitting them is what makes the cap adjustable at all.

    `book_size` is the researched number of positions and drives SIZING. `max_positions` is the hard
    cap and must exceed it.
    """
    limits = RiskLimits()
    assert limits.book_size == 8, "the researched book size moved"
    assert limits.max_positions > limits.book_size, (
        f"cap {limits.max_positions} does not exceed book size {limits.book_size} — a full book can "
        f"sell but never re-enter")


def test_the_split_does_NOT_change_what_anything_is_sized_at():
    """THE WHOLE POINT. Position size is the researched quantity; only the cap was wrong. If this
    moves, the split has changed the strategy rather than fixing its headroom."""
    assert _slot(RiskLimits(), 100_000.0) == 100_000.0 * 0.80 / 8
    assert _slot(RiskLimits(allocated_equity=20_000.0), 100_000.0) == 20_000.0 * 0.80 / 8


def test_a_cap_that_does_not_exceed_the_book_size_is_REFUSED():
    """The defect was silent — two numbers in two repositories that merely happened to be equal, with
    nothing asserting they must not be. Configuring it again must fail loudly at construction rather
    than as a strategy that quietly stops rotating."""
    import pytest

    with pytest.raises(ValueError, match="rotate"):
        RiskLimits(book_size=8, max_positions=8)
    with pytest.raises(ValueError, match="rotate"):
        RiskLimits(book_size=10, max_positions=4)


def test_an_explicit_matching_pair_is_still_allowed_to_be_tight():
    """Headroom of one is enough to rotate a single name; the guard must not demand a round number."""
    assert RiskLimits(book_size=8, max_positions=9).max_positions == 9
