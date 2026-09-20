"""The PURE decision layer. No Nautilus, no database, no broker, no clock.

WHY PURE. Everything here must be testable and backtestable with no engine present, and the SAME
function must serve the backtest and the live path. Two implementations of one decision disagree — the
whole point of the trade-for-trade comparison harnesses in `backtesting/`.

WHAT MUST NEVER APPEAR HERE: order submission, position lookups, wall-clock reads, or a broker. A
decision that touches those cannot be replayed, and a decision that cannot be replayed cannot be
explained six weeks later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.strategies.template.config import TemplateConfig


@dataclass(frozen=True)
class Decision:
    """WHAT the strategy wants, never HOW it is executed.

    `enter`/`exit`/`hold` are the shape every gateway and both slot detectors read. A strategy that
    invents its own decision vocabulary is invisible to cross-strategy monitoring — three gateways
    currently write three incompatible decision schemas, and the one detector that would have caught
    every failure this week could not be written across them.
    """

    hold: tuple[str, ...] = ()
    enter: tuple[str, ...] = ()
    exit: tuple[str, ...] = ()
    scores: dict = field(default_factory=dict)


def rebalance_dates(dates, period: str = "M") -> list[pd.Timestamp]:
    """The cadence rule, ONCE. Adapters and gateways call this rather than re-deriving it."""
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Index(dates).unique())}).sort_values("date")
    if period.upper() == "D":
        return frame["date"].tolist()
    return (frame.assign(bucket=lambda d: d["date"].dt.to_period(period))
            .drop_duplicates("bucket", keep="first")["date"].tolist())


def build_feature_panel(bars: pd.DataFrame, cfg: TemplateConfig) -> pd.DataFrame:
    """Features as of the PRIOR close.

    EVERY feature is `.shift(1)`. The decision for session `d` must be computable from data through
    `d-1` only — otherwise the backtest reads a close that had not happened, and the live path cannot
    reproduce it. A look-ahead here does not fail; it inflates the result and ships.
    """
    if cfg.price_field not in bars.columns:
        raise ValueError(f"bars missing price field {cfg.price_field!r}")
    panel = bars.sort_values(["ticker", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    g = panel.groupby("ticker", sort=False)

    panel["asof_close"] = g["close"].shift(1)
    panel["momentum"] = g[cfg.price_field].transform(
        lambda s: s.shift(1) / s.shift(cfg.lookback_sessions + 1) - 1.0)
    # WARMUP IS A UNIVERSE PROPERTY for a cross-sectional strategy. Ranking three warm names out of
    # fifty is not an early answer, it is a different and wrong strategy.
    panel["sessions_seen"] = g.cumcount()
    panel["eligible"] = (
        (panel["sessions_seen"] >= cfg.warmup_sessions)
        & panel["asof_close"].gt(cfg.price_floor)
        & panel["momentum"].notna())
    return panel


def decide(day: pd.DataFrame, cfg: TemplateConfig, held: set[str]) -> Decision:
    """One session's decision. `held` is what THIS STRATEGY owns — never the account's book.

    Passing the account's positions here is how TECHIVOL-005 proposed exiting eight names belonging to
    two other strategies on its first live session. The caller must pass `strategy_positions()`, or
    the native `positions_open(strategy_id=...)` it wraps.
    """
    eligible = day.loc[day["eligible"]]
    if eligible.empty:
        return Decision(hold=tuple(sorted(held)), exit=())
    ranked = eligible.sort_values(["momentum", "ticker"], ascending=[False, True])
    want = tuple(ranked.head(cfg.portfolio_size)["ticker"])
    return Decision(
        hold=want,
        enter=tuple(s for s in want if s not in held),
        exit=tuple(sorted(s for s in held if s not in want)),
        scores={s: float(m) for s, m in zip(ranked["ticker"], ranked["momentum"])},
    )
