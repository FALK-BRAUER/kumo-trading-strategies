from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from kumo_strategies.contracts import FillEvent, OrderSide, StrategyIdentity
from kumo_strategies.evaluation import RealizedPnlLedger


def _identity() -> StrategyIdentity:
    return StrategyIdentity(
        account_id="paper",
        client_id="alpaca-paper",
        instrument_id="AAPL.NASDAQ",
        strategy_id="MOMENTUM-001",
        cycle_id="cycle-1",
    )


def test_long_round_trip_realized_pnl_net_of_fees() -> None:
    identity = _identity()
    ledger = RealizedPnlLedger(identity)

    ledger.record_fill(
        FillEvent(
            identity=identity,
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            fee=Decimal("1"),
        )
    )
    snapshot = ledger.record_fill(
        FillEvent(
            identity=identity,
            timestamp=datetime(2026, 7, 13, 1, tzinfo=UTC),
            side=OrderSide.SELL,
            quantity=Decimal("10"),
            price=Decimal("105"),
            fee=Decimal("1"),
        )
    )

    assert snapshot.quantity == Decimal("0")
    assert snapshot.realized_pnl == Decimal("48")
    assert snapshot.fees == Decimal("2")


def test_reversal_resets_average_price_to_reversal_fill() -> None:
    identity = _identity()
    ledger = RealizedPnlLedger(identity)

    ledger.record_fill(
        FillEvent(identity, datetime(2026, 7, 13, tzinfo=UTC), OrderSide.BUY, Decimal("10"), Decimal("100"))
    )
    snapshot = ledger.record_fill(
        FillEvent(identity, datetime(2026, 7, 13, 1, tzinfo=UTC), OrderSide.SELL, Decimal("15"), Decimal("90"))
    )

    assert snapshot.quantity == Decimal("-5")
    assert snapshot.average_price == Decimal("90")
    assert snapshot.realized_pnl == Decimal("-100")
