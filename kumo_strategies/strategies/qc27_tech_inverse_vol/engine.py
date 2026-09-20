"""Pure selection logic for QC #27 Tech Momentum with Inverse Volatility Allocation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import QC27TechInverseVolConfig

_NON_EQUITY_NAME_MARKERS = (
    " ETF",
    " ETN",
    " TRUST",
    " FUND",
    " ADR ETF",
)


@dataclass(frozen=True)
class SelectionDiagnostics:
    sector_snapshot_symbols: int
    tech_snapshot_symbols: int
    eligible_name_days: int
    positive_name_days: int


@dataclass(frozen=True)
class QC27Decision:
    date: pd.Timestamp
    hold: tuple[str, ...]
    enter: tuple[str, ...]
    exit: tuple[str, ...]
    scores: dict[str, float]
    weights: dict[str, float]
    cash_proxy_weight: float


def _name_text(name: object) -> str | None:
    """See `qc345_rotation.engine._name_text` — the same defect, the same line, in two engines.

    `(name or "")` does not guard a NaN, because **NaN is truthy**: `float("nan") or ""` is the NaN
    and `.upper()` on it raises `AttributeError: 'float' object has no attribute 'upper'`. A missing
    name arrives as NaN rather than None whenever the frame came through pandas.

    NOT imported from qc345: these two strategies are deliberately independent modules and a shared
    import would couple their universes. Duplicated with both sites named in each docstring, so a
    future change to one is visibly a change to two.
    """
    if not isinstance(name, str):
        return None
    text = name.strip().upper()
    return text or None


def _is_common_stock_like(name: str | None) -> bool:
    text = _name_text(name)
    if text is None:
        # MISSING MEANS EXCLUDE. `not any(marker in "")` is True, so coercing a missing name to ""
        # ADMITS the instrument to the tech universe. A filter asking "is this a common stock"
        # cannot answer yes about an instrument whose name it does not have.
        return False
    return not any(marker in text for marker in _NON_EQUITY_NAME_MARKERS)


def filter_tech_universe(
    bars: pd.DataFrame,
    sectors: pd.DataFrame,
    cfg: QC27TechInverseVolConfig,
) -> tuple[pd.DataFrame, SelectionDiagnostics]:
    required = {"symbol", "sector", "name"}
    missing = required - set(sectors.columns)
    if missing:
        raise ValueError(f"sectors missing required columns: {sorted(missing)}")

    snap = sectors.copy()
    allowed = snap["sector"].astype(str).str.upper().eq(cfg.sector_value.upper())
    if cfg.require_us_country and "country" in snap.columns:
        allowed &= snap["country"].astype(str).str.upper().eq("UNITED STATES")
    allowed &= snap["name"].map(_is_common_stock_like)
    tech_symbols = set(snap.loc[allowed, "symbol"].astype(str))

    filtered = bars[bars["ticker"].isin(tech_symbols)].copy()
    diag = SelectionDiagnostics(
        sector_snapshot_symbols=int(snap["symbol"].astype(str).nunique()),
        tech_snapshot_symbols=len(tech_symbols),
        eligible_name_days=0,
        positive_name_days=0,
    )
    return filtered, diag


def rebalance_dates(dates: pd.Index | pd.Series, period: str = "M") -> list[pd.Timestamp]:
    """First session of each `period`. Default "M" is QC27's specified monthly cadence, and every
    existing caller gets exactly the behaviour it had.

    `period` exists so cadence can be MEASURED rather than assumed -- "W" weekly, "D" every session.
    It is a pandas period alias, and the rule is unchanged in shape: take the first session in each
    bucket. Keeping this as one function means research and production cannot disagree about what a
    rebalance day is, which is the failure mode this repo keeps paying for."""
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Index(dates).unique())}).sort_values("date")
    if period.upper() == "D":
        return frame["date"].tolist()
    return (
        frame.assign(bucket=lambda d: d["date"].dt.to_period(period))
        .drop_duplicates("bucket", keep="first")["date"]
        .tolist()
    )


def build_feature_panel(
    bars: pd.DataFrame,
    cfg: QC27TechInverseVolConfig,
    snapshot_diag: SelectionDiagnostics | None = None,
) -> tuple[pd.DataFrame, SelectionDiagnostics]:
    if cfg.momentum_price_field not in bars.columns:
        raise ValueError(f"bars missing momentum price field {cfg.momentum_price_field!r}")

    panel = bars.sort_values(["ticker", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    g = panel.groupby("ticker", sort=False)

    panel["raw_dollar_volume"] = panel["close"] * panel["volume"]
    panel["asof_close"] = g["close"].shift(1)
    panel["liquidity_proxy"] = g["raw_dollar_volume"].transform(
        lambda s: s.rolling(cfg.liquidity_window, min_periods=cfg.min_liquidity_history).median().shift(1)
    )
    panel["momentum"] = g[cfg.momentum_price_field].transform(
        lambda s: s.shift(1) / s.shift(cfg.lookback_sessions + 1) - 1.0
    )
    panel["realized_volatility"] = g[cfg.momentum_price_field].transform(
        lambda s: s.pct_change().rolling(
            cfg.realized_vol_window,
            min_periods=cfg.min_realized_vol_history,
        ).std(ddof=0).shift(1)
    )
    # WARMUP, ENFORCED. `warmup_sessions` carries QC27's own stated 100-session warmup and was a
    # DECLARED-BUT-UNREAD field: `grep -rn warmup_sessions` across this package and the verified
    # runner returned exactly one hit, its declaration in config.py. Nothing consulted it, so the
    # backtest began trading as soon as `momentum` stopped being NaN (~lookback+2 = 65 sessions)
    # and rebalanced 2025-05-01 on roughly 82 sessions -- a month early, on a third less history
    # than the strategy specifies.
    #
    # Found by running the Nautilus adapter against this runner: the adapter DOES enforce it (its
    # `warmup_bars_needed` takes the max including `warmup_sessions`), the two disagreed on exactly
    # one date, and the disagreement was the defect. Code review had not found it in either repo.
    #
    # Enforced HERE, in the pure layer, so every driver inherits it rather than each remembering.
    panel["sessions_seen"] = g.cumcount()
    panel["warm"] = panel["sessions_seen"] >= cfg.warmup_sessions
    panel["positive_momentum"] = panel["momentum"] > 0.0
    panel["eligible"] = (
        panel["warm"]
        & panel["asof_close"].gt(cfg.price_floor)
        & panel["liquidity_proxy"].notna()
        & panel["momentum"].notna()
        & panel["realized_volatility"].notna()
        & panel["realized_volatility"].gt(0.0)
    )

    base = snapshot_diag or SelectionDiagnostics(0, 0, 0, 0)
    diag = SelectionDiagnostics(
        sector_snapshot_symbols=base.sector_snapshot_symbols,
        tech_snapshot_symbols=base.tech_snapshot_symbols,
        eligible_name_days=int(panel["eligible"].sum()),
        positive_name_days=int((panel["eligible"] & panel["positive_momentum"]).sum()),
    )
    return panel, diag


def _top(df: pd.DataFrame, size: int, column: str) -> pd.DataFrame:
    if size <= 0 or df.empty:
        return df.iloc[0:0].copy()
    return df.sort_values([column, "ticker"], ascending=[False, True]).head(size)


def select_portfolio(
    panel_for_day: pd.DataFrame,
    cfg: QC27TechInverseVolConfig,
) -> tuple[list[str], dict[str, float], dict[str, float], float]:
    eligible = panel_for_day.loc[panel_for_day["eligible"]].copy()
    if eligible.empty:
        return [], {}, {}, 1.0

    liquid = _top(eligible, cfg.liquidity_filter_size, "liquidity_proxy")
    ranked = _top(liquid.loc[liquid["positive_momentum"]], cfg.portfolio_size, "momentum")
    if ranked.empty:
        return [], {}, {}, 1.0

    inv_vol = 1.0 / ranked["realized_volatility"]
    norm = inv_vol / inv_vol.sum()
    gross_weight = min(1.0, len(ranked) / cfg.portfolio_size)
    weights = {sym: float(w * gross_weight) for sym, w in zip(ranked["ticker"], norm)}
    scores = {sym: float(m) for sym, m in zip(ranked["ticker"], ranked["momentum"])}
    return ranked["ticker"].tolist(), scores, weights, float(1.0 - gross_weight)


def decide(panel_for_day: pd.DataFrame, cfg: QC27TechInverseVolConfig, held: set[str]) -> QC27Decision:
    date = pd.Timestamp(panel_for_day["date"].iloc[0])
    target_names, scores, weights, cash_proxy_weight = select_portfolio(panel_for_day, cfg)
    target_set = set(target_names)
    return QC27Decision(
        date=date,
        hold=tuple(target_names),
        enter=tuple(sym for sym in target_names if sym not in held),
        exit=tuple(sorted(held - target_set)),
        scores=scores,
        weights=weights,
        cash_proxy_weight=cash_proxy_weight,
    )
