"""Ports used to keep strategy logic independent from runtime APIs."""

from __future__ import annotations

from typing import Protocol

from kumo_strategies.contracts import CandidateSnapshot, DecisionRow, OrderIntent, StrategyIdentity


class CandidateSource(Protocol):
    def latest_candidate(self, identity: StrategyIdentity) -> CandidateSnapshot | None:
        """Return the latest candidate snapshot for a strategy/instrument identity."""


class DecisionPublisher(Protocol):
    def publish_decision(self, decision: DecisionRow) -> None:
        """Publish a strategy decision to the runtime or evidence log."""


class ExecutionPort(Protocol):
    def submit_order(self, intent: OrderIntent) -> str:
        """Submit an order intent and return the runtime's order identifier."""


class DecisionEngine(Protocol):
    def decide(self, candidate: CandidateSnapshot) -> DecisionRow:
        """Convert a candidate snapshot into a deterministic strategy decision."""
