"""Stable strategy and cycle identity contracts."""

from __future__ import annotations

from dataclasses import dataclass


def _require_text(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


@dataclass(frozen=True, slots=True)
class StrategyIdentity:
    """Canonical Cockpit/engine identity for strategy-owned decisions and orders."""

    account_id: str
    client_id: str
    instrument_id: str
    strategy_id: str
    cycle_id: str | None = None
    manager_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _require_text("account_id", self.account_id))
        object.__setattr__(self, "client_id", _require_text("client_id", self.client_id))
        object.__setattr__(self, "instrument_id", _require_text("instrument_id", self.instrument_id))
        object.__setattr__(self, "strategy_id", _require_text("strategy_id", self.strategy_id))
        if self.cycle_id is not None:
            object.__setattr__(self, "cycle_id", _require_text("cycle_id", self.cycle_id))
        if self.manager_id is not None:
            object.__setattr__(self, "manager_id", _require_text("manager_id", self.manager_id))

    @property
    def strategy_name(self) -> str:
        """Cockpit-facing strategy name from Nautilus-style `NAME-tag` ids."""
        return self.strategy_id.rsplit("-", 1)[0]

    def with_cycle(self, cycle_id: str, manager_id: str | None = None) -> StrategyIdentity:
        """Return the same identity bound to a durable cycle id."""
        return StrategyIdentity(
            account_id=self.account_id,
            client_id=self.client_id,
            instrument_id=self.instrument_id,
            strategy_id=self.strategy_id,
            cycle_id=cycle_id,
            manager_id=self.manager_id if manager_id is None else manager_id,
        )

    def require_cycle(self) -> StrategyIdentity:
        """Return self, raising if the identity is not yet cycle-scoped."""
        if self.cycle_id is None:
            raise ValueError("cycle_id is required for order, fill, and position attribution")
        return self
