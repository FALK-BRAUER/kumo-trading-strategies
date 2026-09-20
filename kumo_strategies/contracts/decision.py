"""Runtime-agnostic strategy decision, order, fill, and performance models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from kumo_strategies.contracts.identity import StrategyIdentity


class DecisionAction(StrEnum):
    WATCH = "WATCH"
    ARM_ENTRY = "ARM_ENTRY"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    REJECT = "REJECT"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    identity: StrategyIdentity
    timestamp: datetime
    features: Mapping[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


@dataclass(frozen=True, slots=True)
class DecisionRow:
    identity: StrategyIdentity
    timestamp: datetime
    action: DecisionAction
    score: float | None = None
    confidence: float | None = None
    reason: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class OrderIntent:
    identity: StrategyIdentity
    timestamp: datetime
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.identity.require_cycle()
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and self.stop_price is None:
            raise ValueError("stop_price is required for stop orders")


@dataclass(frozen=True, slots=True)
class FillEvent:
    identity: StrategyIdentity
    timestamp: datetime
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")
    venue_order_id: str | None = None

    def __post_init__(self) -> None:
        self.identity.require_cycle()
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.fee < 0:
            raise ValueError("fee must be non-negative")


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    identity: StrategyIdentity
    timestamp: datetime
    quantity: Decimal
    average_price: Decimal
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    strategy_id: str
    cycle_count: int
    trade_count: int
    realized_pnl: Decimal
    fees: Decimal
    max_drawdown: Decimal | None = None
    win_rate: float | None = None

    def __post_init__(self) -> None:
        if self.cycle_count < 0:
            raise ValueError("cycle_count must be non-negative")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if self.win_rate is not None and not 0 <= self.win_rate <= 1:
            raise ValueError("win_rate must be between 0 and 1")
