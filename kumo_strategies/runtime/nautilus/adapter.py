"""Nautilus adapter placeholder.

The pure strategy layer is usable before this adapter is implemented. Real Nautilus code should be added only
after checking current NautilusTrader docs with Context7 and validating against Cockpit's ADRs.
"""

from __future__ import annotations

from dataclasses import dataclass

from kumo_strategies.runtime.ports import DecisionEngine


@dataclass(frozen=True, slots=True)
class NautilusStrategyAdapter:
    decision_engine: DecisionEngine

    def build_strategy(self) -> object:
        """Build the runtime strategy object once Nautilus integration is implemented."""
        raise NotImplementedError(
            "Implement after binding to nautilus_trader StrategyConfig/Strategy APIs and Cockpit identity rules."
        )
