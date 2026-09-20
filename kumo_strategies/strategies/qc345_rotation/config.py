"""Configuration for the QC #345 monthly momentum rotation."""

from __future__ import annotations

from kumo_strategies.strategies.market_view import MarketViewConfig
from dataclasses import dataclass, field, fields
from typing import Literal

from kumo_strategies.strategies.momentum_rotation.config import ExitConfig


AssetUniverseMode = Literal["all", "fundamental_like"]
MarketCapMode = Literal["dv_proxy", "price_x_dv", "skip"]
MomentumPriceField = Literal["close", "close_adj", "close_split", "close_split_dividend"]


@dataclass(frozen=True)
class QC345RotationConfig:
    lookback_sessions: int = 252
    liquidity_window: int = 21
    min_liquidity_history: int = 10
    realized_vol_window: int = 20
    min_realized_vol_history: int = 10
    corporate_action_move: float = 0.40
    corporate_action_window: int = 41

    price_floor: float = 5.0
    liquidity_filter_size: int = 100
    universe_size: int = 50
    portfolio_size: int = 5

    asset_universe_mode: AssetUniverseMode = "fundamental_like"
    market_cap_mode: MarketCapMode = "price_x_dv"
    momentum_price_field: MomentumPriceField = "close_split_dividend"

    mania_momentum_threshold: float | None = 3.0
    mania_volatility_threshold: float | None = 0.05
    exits: ExitConfig = field(default_factory=ExitConfig)

    forced_rebalance_dates: tuple[str, ...] = ()

    market_view: "MarketViewConfig" = field(default_factory=lambda: MarketViewConfig())
    """NO VIEW, for the same reason as the rotation lanes and with this lane's own numbers still
    to be measured.

    QC345-003's arm has not been run — its runner could not express a view until 2026-09-11 and
    the config sheet for the paper instance it runs on is not frozen yet. A lane must not carry a
    window nobody measured for it: TECHIVOL's 50 is TECHIVOL's answer on TECHIVOL's data, and
    inheriting it is the error that made one fitted constant look like a platform default.

    `signal=NONE` rather than a view marked unmeasured, deliberately. The inert state is LOUD BY
    CONSTRUCTION — it yields RISK_ON with the reason "no market view configured", and that reason
    travels to the notification payload — whereas a view with an unmeasured window reads as a view
    to every surface downstream and hides the truth in a comment.

    REVISIT when this lane's arm has been run on its own panel with its own window swept.
    """
    """Operator override, live-runtime only: dates ('YYYY-MM-DD') to treat as a rebalance session
    even though `rebalance_dates()` (the pure monthly rule) says they are not -- e.g. a strategy
    enabled mid-month that would otherwise sit idle until the next natural month-start.

    Read once by `QC345RotationStrategy.__init__`, not consulted anywhere in this module or in
    `engine.py`/`decide()` -- a backtest run has no notion of "the operator forced a date", and this
    field existing on the settings-domain dataclass must never let one leak into the pure decision
    path research and production share. Deliberately excluded from `tag()` below for the same
    reason: it is an operational knob, not a research parameter, and mixing it into a backtest's
    config fingerprint would make two runs of the identical strategy hash differently for a reason
    that has nothing to do with the result."""

    def tag(self) -> str:
        parts = [
            f"assets-{self.asset_universe_mode}",
            f"mcap-{self.market_cap_mode}",
            f"mom-{self.momentum_price_field}",
        ]
        if self.mania_momentum_threshold is not None and self.mania_volatility_threshold is not None:
            parts.append(
                "mania"
                f"-mom{self.mania_momentum_threshold:g}"
                f"-vol{self.mania_volatility_threshold:g}"
            )
        exit_parts = [
            f"{f.name}-{getattr(self.exits, f.name):g}"
            if isinstance(getattr(self.exits, f.name), float)
            else f"{f.name}-{getattr(self.exits, f.name)}"
            for f in fields(self.exits)
            if getattr(self.exits, f.name) is not None
        ]
        if exit_parts:
            parts.append("exits-" + "-".join(exit_parts))
        return "__".join(parts)

    def __post_init__(self) -> None:
        """Refuse a value that is not the type this config declares.

        Settings overrides arrive UNCOERCED — cockpit tests `is_dataclass(f.type)`, and
        `from __future__ import annotations` makes `f.type` a STRING on every field, so
        that test is unconditionally False and raw values pass through. A string "10"
        for an int then fails later, somewhere that does not name the config.
        """
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)

