"""Configuration for QC #27 Tech Momentum with Inverse Volatility Allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kumo_strategies.strategies.market_view import MarketViewConfig


MomentumPriceField = Literal["close", "close_adj"]
RebalancePeriod = Literal["M", "W", "D"]


@dataclass(frozen=True)
class QC27TechInverseVolConfig:
    lookback_sessions: int = 63
    liquidity_window: int = 21
    min_liquidity_history: int = 10
    realized_vol_window: int = 20
    min_realized_vol_history: int = 15
    warmup_sessions: int = 100

    price_floor: float = 5.0
    liquidity_filter_size: int = 100
    portfolio_size: int = 10
    stop_loss_portfolio_frac: float = 0.02

    rebalance_period: RebalancePeriod = "M"
    """Cadence, and it lives HERE rather than as a runner argument because it was a research-only
    knob: `runner_qc27_verified.run()` took a `rebalance_period`, while BOTH production call sites --
    the Nautilus adapter's `_is_rebalance` and the executor gateway -- called `rebalance_dates()`
    with no period and were therefore hardwired monthly. A cadence sweep could report weekly and the
    live strategy would still rebalance monthly, with nothing failing to say so.

    Same shape as `warmup_sessions` on this branch: declared, swept, never read by the thing that
    trades. `M` monthly, `W` weekly, `D` every session."""

    market_view: MarketViewConfig = field(default_factory=MarketViewConfig)
    """The lane's own read of its market (kumo-trading-platform issue 873).

    IT LIVES ON THE CONFIG for the same reason `rebalance_period` above does. A view passed as a
    runner argument is a research-only knob by construction: the thing that trades never sees it,
    a sweep can report one rule while production runs another, and nothing fails to say so. That
    exact shape is documented two fields up and it has already happened twice on this lane.

    Defaults to `MarketViewConfig()` — signal NONE — so a lane gets a view only by declaring one.

    STRATEGY DECLARES, PLATFORM RUNS. This field is the declaration and nothing more: it blocks no
    entry and sells no position on its own. `runner_qc27_verified` reads it so the rule can be
    MEASURED, and platform issue 873's poller will read it so the rule can be ACTED ON. Until that poller
    exists, `test_backtest_enforced_config_reaches_the_live_runner` records the gap, which is the
    honest state rather than a silent one."""

    momentum_price_field: MomentumPriceField = "close_adj"
    cash_proxy_symbol: str = "GLD"
    sector_value: str = "Technology"
    require_us_country: bool = True

    def tag(self) -> str:
        return (
            f"mom-{self.lookback_sessions}"
            f"__vol-{self.realized_vol_window}"
            f"__top-{self.portfolio_size}"
            f"__liq-{self.liquidity_filter_size}"
            f"__stop-{self.stop_loss_portfolio_frac:g}"
            f"__cash-{self.cash_proxy_symbol}"
            # Cadence belongs in the tag: two runs that differ only in cadence would otherwise write
            # to the same artifact path and silently overwrite each other.
            f"__cad-{self.rebalance_period}"
        )

    def __post_init__(self) -> None:
        """Refuse a value that is not the type this config declares.

        Settings overrides arrive UNCOERCED — cockpit tests `is_dataclass(f.type)`, and
        `from __future__ import annotations` makes `f.type` a STRING on every field, so
        that test is unconditionally False and raw values pass through. A string "10"
        for an int then fails later, somewhere that does not name the config.
        """
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)

