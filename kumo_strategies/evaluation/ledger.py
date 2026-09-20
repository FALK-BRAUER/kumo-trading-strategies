"""Simple average-cost realized-PnL ledger for replay verification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kumo_strategies.contracts import FillEvent, OrderSide, PositionSnapshot, StrategyIdentity


@dataclass(slots=True)
class RealizedPnlLedger:
    identity: StrategyIdentity
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    trade_count: int = 0

    def record_fill(self, fill: FillEvent) -> PositionSnapshot:
        if fill.identity != self.identity:
            raise ValueError("fill identity does not match ledger identity")

        signed_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        self.fees += fill.fee
        self.trade_count += 1

        if self.quantity == 0 or _same_direction(self.quantity, signed_quantity):
            self._increase_position(signed_quantity, fill.price)
            return self.snapshot(fill)

        self._close_or_reverse(signed_quantity, fill.price)
        return self.snapshot(fill)

    def snapshot(self, fill: FillEvent) -> PositionSnapshot:
        return PositionSnapshot(
            identity=self.identity,
            timestamp=fill.timestamp,
            quantity=self.quantity,
            average_price=self.average_price,
            realized_pnl=self.realized_pnl - self.fees,
            fees=self.fees,
        )

    def _increase_position(self, signed_quantity: Decimal, price: Decimal) -> None:
        total_quantity = self.quantity + signed_quantity
        if total_quantity == 0:
            self.average_price = Decimal("0")
            self.quantity = Decimal("0")
            return

        total_cost = (abs(self.quantity) * self.average_price) + (abs(signed_quantity) * price)
        self.quantity = total_quantity
        self.average_price = total_cost / abs(total_quantity)

    def _close_or_reverse(self, signed_quantity: Decimal, price: Decimal) -> None:
        closing_quantity = min(abs(self.quantity), abs(signed_quantity))
        if self.quantity > 0:
            self.realized_pnl += (price - self.average_price) * closing_quantity
        else:
            self.realized_pnl += (self.average_price - price) * closing_quantity

        remaining_quantity = self.quantity + signed_quantity
        if remaining_quantity == 0:
            self.quantity = Decimal("0")
            self.average_price = Decimal("0")
            return

        if _same_direction(self.quantity, remaining_quantity):
            self.quantity = remaining_quantity
            return

        self.quantity = remaining_quantity
        self.average_price = price


def _same_direction(left: Decimal, right: Decimal) -> bool:
    return (left > 0 and right > 0) or (left < 0 and right < 0)
