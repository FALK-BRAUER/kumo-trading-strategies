"""What every path of the one runner returns (#270)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.backtesting.report import Report


@dataclass
class BacktestResult:
    report: Report
    decisions: pd.DataFrame
    #: Per-family counters and caveats (the monthly family records e.g. cash-shortfall skips,
    #: stop triggers, delisting buckets); empty for the rotation family.
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ShortBacktestResult(BacktestResult):
    """The short family's result: the covers with the rule that produced each (the exit MIX is an
    acceptance number — #123: flat 48 %, reversal 48 %, no-data 4 %), the borrow carry (NOT in
    `report.fills`: `total_cost_usd` there is spread and commission only) and the refused signals."""

    covers: pd.DataFrame = field(default_factory=pd.DataFrame)
    borrow_paid: float = 0.0
    refused: pd.DataFrame = field(default_factory=pd.DataFrame)

    def exit_mix(self) -> dict[str, float]:
        if not len(self.covers):
            return {}
        return self.covers["kind"].value_counts(normalize=True).to_dict()
