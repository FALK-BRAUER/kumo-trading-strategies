"""The live config is a CONTRACT, so it is pinned (#33, #63).

Every assertion here corresponds to a measurement. A change that flips one of these is not a tweak —
it ships a different strategy under the same name — so it has to break a test and be argued for.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import (
    EXTERNAL_ID, STRATEGY_NAME, QC27RotationStrategy,
)
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    OPEN_OFFSET_MINUTES, ORDER_ID_TAG, WIRE_ID, QC27TechInverseVolConfig, live_config, live_notes,
)


def test_the_live_config_can_actually_construct_the_adapter():
    """THE POINT OF THE FILE. The adapter refuses `close_adj` at construction because a Nautilus bar
    cannot carry it. A live config that cannot build the strategy is not a live config, and the
    failure would otherwise surface at deploy time in front of whoever is deploying."""
    s = QC27RotationStrategy(cfg=live_config(), instrument_ids=[], order_id_tag=ORDER_ID_TAG)
    assert str(s.id) == WIRE_ID


def test_the_tag_is_not_one_already_allocated():
    """MANUAL 001, MOMENTUM 002, QC345 003, BCTROT 004 are live. A duplicate does not degrade —
    Nautilus raises at `Trader.add_strategy` and the node does not boot, taking the other strategies
    with it."""
    assert ORDER_ID_TAG not in {"001", "002", "003", "004"}
    assert WIRE_ID == f"{STRATEGY_NAME}-{ORDER_ID_TAG}"
    assert EXTERNAL_ID == "QC27", "QC27 stays provenance; the wire id is the strategy name"


def test_live_runs_daily_not_the_monthly_it_was_built_with():
    """Operator decision (2026-08-21), taken on drawdown: daily is best of the three on risk
    (-21.9% maxDD against weekly -26.6% and monthly -33.5%) and is the cadence that can actually
    leave a sustained decline (-0.6% through Jun-Aug 26 against monthly's -21.8%).

    Deliberately NOT weekly, which was the analytic pick on split-half stability. Recorded here so
    the trade-off is visible to whoever changes it next."""
    assert live_config().rebalance_period == "D"
    assert QC27TechInverseVolConfig().rebalance_period == "M", \
        "the package default moved; the live config no longer states a deliberate choice"


def test_live_uses_the_only_price_field_a_bar_can_supply():
    assert live_config().momentum_price_field == "close"


def test_the_fill_offset_is_midday_not_the_open():
    """09:35 is the widest spread of the session: median charged spread falls 11.09 -> 3.80 bps/leg
    moving to 12:00. Deliberately NOT the close — the 12:00 -> 15:55 step has no cost difference at
    all, so its apparent gain is unbacked drift on one window."""
    assert OPEN_OFFSET_MINUTES == 150
    assert OPEN_OFFSET_MINUTES != 5, "still filling at the open"


def test_the_live_config_differs_from_defaults_only_where_intended():
    """Guards against a field being changed here without the reasoning that belongs with it.

    `market_view` joined 2026-09-11 (kumo-trading-platform issue 873, the operator approved). It is a DECLARATION and not
    yet an action — no poller reads it, so the live lane trades without the view while research
    measures it with one, and `test_backtest_enforced_config_reaches_the_live_runner` carries that
    gap by name. Adding it here rather than leaving it defaulted is the point: a view supplied only
    as a runner argument is the `rebalance_period` shape, where a sweep reported one cadence and
    both production call sites ran another with nothing failing to say so."""
    d, live = QC27TechInverseVolConfig(), live_config()
    changed = {f for f in vars(d) if getattr(d, f) != getattr(live, f)}
    expected = {"momentum_price_field", "rebalance_period", "market_view"}
    assert changed == expected, f"undocumented departure from defaults: {changed ^ expected}"


@pytest.mark.parametrize("key", ["cash_proxy_never_opens", "universe_not_point_in_time",
                                 "cost_coverage", "one_window_no_holdout"])
def test_the_operator_caveats_ship_with_the_config(key):
    """These travel in code rather than a document because the thing that ships is what gets read."""
    assert key in live_notes() and len(live_notes()[key]) > 40
