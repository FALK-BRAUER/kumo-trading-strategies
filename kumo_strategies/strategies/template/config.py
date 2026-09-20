"""Configuration for the TEMPLATE lane. Copy this directory to start a new strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kumo_strategies.strategies.market_view import MarketViewConfig

#: A live Nautilus `Bar` carries OHLCV and NOTHING ELSE. `close_adj` cannot come off the wire, so a
#: config defaulting to it produces a strategy that backtests fine and raises at its FIRST REBALANCE,
#: weeks later. QC345 shipped that (`KeyError: 'eligible'`) and QC27 nearly did.
PriceField = Literal["close", "close_adj"]
RebalancePeriod = Literal["M", "W", "D"]


@dataclass(frozen=True)
class TemplateConfig:
    """FROZEN, so a runtime layer cannot quietly mutate what the backtest validated.

    Every field a runner or adapter needs must live HERE, not as an argument on one call path. QC27's
    cadence was a `rebalance_period` argument on the backtest runner while BOTH production call sites
    called `rebalance_dates()` with no period — so a sweep could report daily and the live strategy
    still rebalanced monthly, with nothing failing to say so.
    """

    lookback_sessions: int = 20
    warmup_sessions: int = 30
    portfolio_size: int = 5
    price_floor: float = 5.0

    #: `close` for anything live. See PriceField.
    price_field: PriceField = "close_adj"

    #: Cadence lives in the CONFIG so research and production cannot disagree about it.
    rebalance_period: RebalancePeriod = "M"

    #: The lane's declared view of its own market. REQUIRED, NO DEFAULT (#212; 2026-09-13:
    #: "no strategy can opt out. Those are emergency rules."). A lane built from this template
    #: cannot construct its config without stating a view — before `MarketAwareMixin.register`
    #: ever runs, which refuses an undeclared one at the node.
    #:
    #: NO DEFAULT BECAUSE THERE IS NO NUMBER TO DEFAULT TO. `market_view.py`: a window is MEASURED
    #: PER LANE, NEVER INHERITED — TECHIVOL's 50 is TECHIVOL's number, SMHGLD's 50 is the lab's for
    #: SMHGLD. A copied constant is exactly what the register refusal cannot catch (a declared wrong
    #: number), so the template refuses to supply one. Measure the window and the action's cost
    #: (the #212 table shape: none / EXIT_ONLY / LIQUIDATE, CAGR / Sharpe / maxDD, current regime
    #: first), then write `MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=<yours>,
    #: action=MarketAction.EXIT_ONLY)` here.
    market_view: MarketViewConfig = field(kw_only=True)

    def tag(self) -> str:
        """Artifact identity. Every swept field belongs here or two runs overwrite each other."""
        return (f"lb-{self.lookback_sessions}__top-{self.portfolio_size}"
                f"__cad-{self.rebalance_period}__px-{self.price_field}")

    def __post_init__(self) -> None:
        """Refuse a value that is not the type this config declares.

        Settings overrides arrive UNCOERCED — cockpit tests `is_dataclass(f.type)`, and
        `from __future__ import annotations` makes `f.type` a STRING on every field, so
        that test is unconditionally False and raw values pass through. A string "10"
        for an int then fails later, somewhere that does not name the config.
        """
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)

