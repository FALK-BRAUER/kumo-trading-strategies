from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kumo_strategies.contracts import OrderIntent, OrderSide, OrderType, StrategyIdentity


def test_order_intent_requires_cycle_identity() -> None:
    identity = StrategyIdentity(
        account_id="paper",
        client_id="alpaca-paper",
        instrument_id="AAPL.NASDAQ",
        strategy_id="MANUAL-001",
    )

    with pytest.raises(ValueError, match="cycle_id is required"):
        OrderIntent(
            identity=identity,
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )


def test_limit_order_requires_limit_price() -> None:
    identity = StrategyIdentity(
        account_id="paper",
        client_id="alpaca-paper",
        instrument_id="AAPL.NASDAQ",
        strategy_id="MANUAL-001",
        cycle_id="cycle-1",
    )

    with pytest.raises(ValueError, match="limit_price"):
        OrderIntent(
            identity=identity,
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
        )
