"""The four live invariants a PARTIAL SELL disturbs, checked against this lane.

`momentum_rotation`'s `rebalance_band` is backtest-only for exactly this reason: the runner models
trading as an entry sized once and an exit that sells everything, and a trim is neither. Its
docstring names what breaks — trail state, order identity, round trips, claims ledger.

This lane's ONLY trade is a partial sell paired with a partial buy, about 93 times a year, so it
cannot adopt the same deferral. Each invariant is either satisfied by construction here, and these
tests pin that construction so it cannot quietly change, or it is not, and the test says so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kumo_strategies.strategies.smhgld_sleeve import (
    Decision, SmhGldSleeveConfig, live_config, order_plan)

CFG = live_config()
PRICES = {"SMH": 560.0, "GLD": 396.0}


def plan(held: dict[str, float], equity: float | None = None,
         regime: str = "rebalance") -> tuple:
    equity = equity if equity is not None else sum(held[s] * PRICES[s] for s in held)
    return order_plan(Decision(regime=regime, weights=CFG.target_weights),
                      held, PRICES, equity)


# -- invariant 1: trail state -------------------------------------------------------------------
def test_the_lane_carries_no_trail_for_a_partial_sell_to_corrupt():
    """A trimmed position stays OPEN, so give-back peak tracking must survive the partial sale or
    the remaining shares get a fresh and wrong trail.

    Satisfied by CONSTRUCTION: this lane has no give-back rule, no trailing stop and no per-
    position peak state. It holds two legs forever at fixed weights. If a trail is ever added, this
    test is where the interaction has to be thought about first.

    THIS GUARD WAS SATISFIED BY ABSENCE and is fixed here (e70suark's review of #177). It looked at
    `SmhGldSleeveConfig.__dataclass_fields__` alone and matched five literal spellings, so it could
    not fail in the two ways it actually would:

      - TRAIL STATE LIVES ON THE ADAPTER, not the config. A `_peaks` dict or a `_trail` attribute on
        `SmhGldSleeveStrategy` never touches `__dataclass_fields__` — and the adapter is exactly
        where every lane in this repo that HAS a trail keeps it (`crsi_short.py::_trail`).
      - FIVE SPELLINGS. `giveback_pct`, `trailing_stop`, `peak_days` all passed.

    A guard aimed one object away from where the defect would live. The docstring claimed "the test
    fails if such a field is ever added", which was true for five names in one class and false
    everywhere else.
    """
    forbidden = {"giveback", "trail", "peak", "stoploss", "atr", "highwater", "drawdown"}

    def _hits(names) -> set[str]:
        # UNDERSCORES STRIPPED FROM BOTH SIDES. A mutation bite caught this: `giveback_pct` does not
        # contain `give_back`, so widening from exact names to substrings still missed the very
        # spelling the review named. Normalising is what makes the match about the WORD rather than
        # about where somebody put the underscores — `giveback_pct`, `give_back_frac`,
        # `trailing_stop` and `peak_days` all hit now.
        def norm(x):
            return x.lower().replace("_", "")
        return {n for n in names if any(f in norm(n) for f in forbidden)}

    # HALF ONE, AND IT IS THE HALF USUALLY SKIPPED: prove the vocabulary is REAL before asserting
    # its absence. A typo in all seven names gives a permanently green test that proves nothing —
    # and that is indistinguishable from the lane genuinely having no trail. So the same words must
    # HIT on a lane that does have one.
    from kumo_strategies.strategies.momentum_rotation.config import ExitConfig
    proof = _hits(ExitConfig.__dataclass_fields__)
    assert proof, (
        f"none of {sorted(forbidden)} appears in ExitConfig, so this guard is matching a vocabulary "
        f"that does not exist in this repo and would stay green whatever SMHGLD grew")

    # HALF TWO: the subject is the CONFIG *and* the CONSTRUCTED ADAPTER, not a field list.
    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy

    lane = SmhGldSleeveStrategy(SmhGldSleeveConfig(), symbols=["SMH", "GLD"],
                                order_id_tag="901", shadow_only=True)
    subjects = {
        "SmhGldSleeveConfig": set(SmhGldSleeveConfig.__dataclass_fields__),
        "SmhGldSleeveStrategy instance": set(vars(lane)),
        "SmhGldSleeveStrategy slots": set(getattr(SmhGldSleeveStrategy, "__slots__", ()) or ()),
    }
    for where, names in subjects.items():
        hit = _hits(names)
        assert not hit, (
            f"this lane grew trailing state on {where}: {sorted(hit)}. A partial sell now has peak "
            f"tracking to corrupt and invariant 1 is no longer free — think about the interaction "
            f"here before shipping it.")


# -- invariant 2: order identity ----------------------------------------------------------------
def test_one_side_per_symbol_per_session():
    """`client_order_id` is keyed (session, symbol, side, attempt), so a trim and a later full exit
    of the same symbol in one session collide and Nautilus denies the second locally — #51's replay
    guard firing on a legitimate order.

    Satisfied by CONSTRUCTION: a target/delta plan emits at most ONE order per symbol, because a
    position has one target and therefore one delta. There is no path that trims and then exits.
    """
    for held in ({"SMH": 70.0, "GLD": 150.0}, {"SMH": 10.0, "GLD": 400.0}, {}):
        orders = plan(held, equity=100_000.0, regime="opening" if not held else "rebalance")
        symbols = [order.symbol for order in orders]
        assert len(symbols) == len(set(symbols)), f"two orders for one symbol from {held}"


def test_sells_are_submitted_before_buys():
    """A fully-invested sleeve funds its buys with its sells. Submitting the buy first asks for
    margin this lane does not have and the broker rejects on a cash account."""
    orders = plan({"SMH": 70.0, "GLD": 150.0})
    deltas = [order.delta for order in orders]
    assert deltas == sorted(deltas), "sells must precede buys"
    assert deltas[0] < 0 < deltas[-1], "a rebalance of a drifted sleeve is one sell and one buy"


# -- invariant 3: round trips -------------------------------------------------------------------
def test_a_rebalance_never_closes_a_position():
    """FIFO round-trip accounting assumes whole exits, and every KPI reads those.

    This lane NEVER closes a leg, so it opens no round trip and closes none: the position-level
    ledger is the only correct reading of it. A plan that took a leg to zero would silently create
    a round trip whose entry was a rebalance months earlier.
    """
    # Even a violent drift leaves both legs held: the targets are 32% and 68%, never zero.
    for held in ({"SMH": 300.0, "GLD": 5.0}, {"SMH": 1.0, "GLD": 900.0}):
        for order in plan(held):
            remaining = order.held_qty + order.delta
            assert remaining > 0, (
                f"{order.symbol} would be flattened to {remaining}; that is a round-trip close, "
                f"not a rebalance, and every KPI would read it as one")


def test_targets_are_never_zero_for_either_leg():
    """The structural reason invariant 3 holds. Stated separately so that if someone ever adds a
    weight of 0 or 1 to this config, the failure names the consequence rather than the arithmetic."""
    assert all(weight > 0 for weight in CFG.target_weights.values())


# -- invariant 4: claims ledger -----------------------------------------------------------------
def test_the_plan_carries_the_post_trade_quantity_the_claim_is_set_to():
    """Reconciliation tracks whole positions, and a partial sell moves the number rather than
    opening or closing the claim.

    The claim is SET ABSOLUTELY to the post-trade book quantity via `sync_claim_to`, never adjusted
    by a delta. Absolute is idempotent and self-correcting — a missed, duplicated or reordered call
    still converges on the book — where a delta accumulates, and one dropped call leaves the claim
    permanently wrong with nothing to compare against. This lane's own sequence is the
    discriminating case: 100, 93, 88, 88 with no trade, 0. A delta API applies the 88 twice and
    takes the claim to zero on a live position.
    """
    for order in plan({"SMH": 70.0, "GLD": 150.0}):
        assert order.held_qty >= 0
        assert order.target_qty > 0
        assert order.post_trade_qty == pytest.approx(order.held_qty + order.delta)


def test_the_claim_is_set_to_the_rounded_fill_not_the_unrounded_target():
    """`post_trade_qty` is held + delta, and the delta is rounded to a whole lot while the target
    is not. A claim set to `target_qty` would be wrong by up to a lot on every one of ~93 resizes a
    year, permanently, with nothing in the system that would ever disagree with it."""
    orders = plan({"SMH": 70.4, "GLD": 150.6})
    assert orders, "this holding must drift far enough to trade"

    for order in orders:
        assert order.post_trade_qty == pytest.approx(order.held_qty + order.delta)
        assert float(order.post_trade_qty).is_integer() == float(order.held_qty).is_integer(), (
            "the post-trade quantity must inherit the book's lot alignment, not the target's")
        if abs(order.target_qty - order.held_qty) % 1.0:
            assert order.post_trade_qty != pytest.approx(order.target_qty), (
                "claim would be set to an unrounded target the broker never fills")


def _rounded(delta: float, lot: float = 1.0) -> float:
    return (abs(delta) // lot) * lot * (1.0 if delta > 0 else -1.0)


# -- the plan must not act when the decision did not ---------------------------------------------
def test_no_orders_when_the_sleeve_is_on_target_or_unknown():
    """`order_plan` reads the REGIME, not the arithmetic. A plan that recomputed drift for itself
    would be a second derivation of the decision, free to disagree with the one journalled."""
    for regime in ("on_target", "unknown"):
        assert plan({"SMH": 70.0, "GLD": 150.0}, regime=regime) == ()


def test_rounding_never_overshoots_the_target():
    """Rounding away from the holding would push the position past target and invite the opposite
    trade next session — churn manufactured by arithmetic rather than by drift."""
    for held in ({"SMH": 56.4, "GLD": 169.1}, {"SMH": 55.6, "GLD": 170.2}):
        for order in plan(held, equity=98_600.0):
            moved = abs(order.delta)
            wanted = abs(order.target_qty - order.held_qty)
            assert moved <= wanted + 1e-9, f"{order.symbol} overshot: moved {moved} of {wanted}"
