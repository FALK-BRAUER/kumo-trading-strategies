"""Pure selection logic for the QC #345 monthly momentum rotation."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from .config import QC345RotationConfig

_FUND_LIKE_NAME_MARKERS = (
    " ETF",
    " ETF ",
    " ETN",
    " ETN ",
    " FUND",
    " TRUST",
    "ISHARES",
    "PROSHARES",
    "DIREXION",
    "VANECK",
    "SPDR",
    "INVESCO",
    "WISDOMTREE",
    "YIELDMAX",
    "ROUNDHILL",
    "KRANESHARES",
    "GLOBAL X",
    "FIRST TRUST",
    "TIDAL TRUST",
    "SIMPLIFY",
    "PIMCO",
    "GRANITESHARES",
    "BLACKROCK CORE BD TR",
    "MUNICIPAL INCOME",
    "TAX-FREE",
    "STRATEGIC MUNICIPALS",
    "STRATEGIC TOTAL RETURN",
    "CORPORATE INVESTORS",
    "PARTICIPATION INVESTORS",
    "HEALTHCARE INVESTORS",
    "LIFE SCIENCES INVESTORS",
    "BDC",
    "BUSINESS DEVELOPMENT",
    "DIRECT LENDING",
    "SPECIALTY LENDING",
    "OXFORD LANE CAPITAL",
    "ARES CAPITAL CORPORATION",
    "BLACKROCK TCP CAPITAL",
    "GREAT ELM CAPITAL",
    "FS KKR CAPITAL",
    "MAIN STREET CAPITAL",
    "PREFERRED",
    "PREFERENCE",
    " CAPITAL OBLIGATION",
    " FIXED-INCOME",
    "SYNTHETIC FIXED INCOME",
    "FIXED INCOME SECURITIES",
    "BENEFICIAL UNIT CERTIFICATES",
    "OPTION NOTE UNIT",
    " SUBORDINATED NOTES",
    " SUBORDINATED DEBENTURES",
    " NOTES",
    " DEBENTURES",
    "NOTES DUE",
    "NOTE DUE",
    "WARRANT",
    " ACQUISITION CORP. UNIT",
    " ACQUISITION CORP UNIT",
    " ACQUISITION CORPORATION UNIT",
    " ACQUISITION CORP. RIGHT",
    " ACQUISITION CORP RIGHT",
    " ACQUISITION CORPORATION RIGHT",
    " UNITS",
    " RIGHTS",
)
_SPAC_PRODUCT_NAME_RE = re.compile(r"\b(?:UNIT|UNITS|RIGHT|RIGHTS)\b")
_PRODUCT_SUFFIX_NAME_RE = re.compile(r"\b(?:UNIT|RIGHT)\.?$")
_PRODUCT_SUFFIX_ISSUER_MARKERS = (
    " ACQUISITION",
    " CORP",
    " CORPORATION",
    " COMPANY",
    " CO.",
    " INC",
    " HOLDINGS",
    " GROUP",
    " LTD",
    " SPAC",
)


@dataclass(frozen=True)
class SelectionDiagnostics:
    split_blocked_name_months: int
    mania_blocked_name_months: int


@dataclass(frozen=True)
class QC345Decision:
    date: pd.Timestamp
    hold: tuple[str, ...]
    enter: tuple[str, ...]
    exit: tuple[str, ...]
    scores: dict[str, float]


def _name_text(name: object) -> str | None:
    """The instrument's name as UPPERCASE text, or None when there ISN'T one.

    `(name or "")` was the guard here and it does not hold, because **NaN is truthy**:
    `float("nan") or ""` is the NaN, and `.upper()` on it raises `AttributeError: 'float' object has
    no attribute 'upper'`. A missing name arrives as NaN rather than None whenever the frame came
    through pandas — an unmatched left join, an empty CSV cell, or a symbol the name source does not
    carry. kumo-trading-platform hit exactly that on 2026-08-24 feeding QC345 the 107-name momentum pool:
    BLLLN, GTLAB, IQVIA, JEPO and OVVI are not names Alpaca knows.

    Returns None for anything that is not usable text, so callers must DECIDE what a missing name
    means rather than silently getting "" — which reads as "a name with no disqualifying markers in
    it" and admits the instrument.
    """
    if not isinstance(name, str):
        return None
    text = name.strip().upper()
    return text or None


def is_fundamental_like_asset(name: str | None, exchange: str | None) -> bool:
    text = _name_text(name)
    if text is None:
        # MISSING MEANS EXCLUDE. `True` from this predicate means KEEP, and an empty string contains
        # none of the rejection markers, so coercing a missing name to "" falls through every check
        # to `return True` and ADMITS an instrument whose name we do not have. The AttributeError was
        # hiding that path rather than causing it.
        return False
    venue = _name_text(exchange) or ""
    if venue == "ARCA":
        return False
    if any(marker in text for marker in _FUND_LIKE_NAME_MARKERS):
        return False
    if (
        _SPAC_PRODUCT_NAME_RE.search(text)
        and _PRODUCT_SUFFIX_NAME_RE.search(text)
        and any(marker in text for marker in _PRODUCT_SUFFIX_ISSUER_MARKERS)
    ):
        return False
    return True


def filter_asset_universe(
    bars: pd.DataFrame,
    assets: pd.DataFrame | None,
    cfg: QC345RotationConfig,
) -> pd.DataFrame:
    if cfg.asset_universe_mode == "all":
        return bars.copy()
    if assets is None:
        raise ValueError("assets are required for asset_universe_mode='fundamental_like'")
    required = {"symbol", "name", "exchange"}
    missing = required - set(assets.columns)
    if missing:
        raise ValueError(f"assets missing required columns: {sorted(missing)}")
    allowed = assets.loc[
        assets.apply(lambda row: is_fundamental_like_asset(row["name"], row["exchange"]), axis=1),
        "symbol",
    ]
    return bars[bars["ticker"].isin(set(allowed))].copy()


def terminal_buckets(bars: pd.DataFrame, assets: pd.DataFrame | None) -> dict[str, str]:
    """Best-effort terminal classification for names that stop printing before the panel ends."""
    if assets is None or "status" not in assets.columns:
        return {ticker: "active" for ticker in bars["ticker"].drop_duplicates()}
    inactive = set(assets.loc[assets["status"] == "inactive", "symbol"])
    rows = []
    panel_end = pd.Timestamp(bars["date"].max()) if not bars.empty else pd.NaT
    for ticker, grp in bars.groupby("ticker", sort=False):
        tail = grp.tail(20)
        last_close = float(tail["close"].iloc[-1])
        tail_high = float(tail["close"].max())
        mean_close = float(tail["close"].mean()) if len(tail) else float("nan")
        tail_cv = float(tail["close"].std(ddof=0) / mean_close) if mean_close else float("nan")
        bucket = "active"
        stopped_before_panel_end = pd.Timestamp(grp["date"].iloc[-1]) < panel_end
        if ticker in inactive and stopped_before_panel_end:
            if (
                ticker.endswith("Q")
                or last_close <= 1.0
                or (tail_high > 0 and last_close / tail_high <= 0.25)
            ):
                bucket = "bankruptcy"
            elif pd.notna(tail_cv) and tail_cv <= 0.03 and last_close >= 5.0:
                bucket = "acquisition"
            else:
                bucket = "unclassified"
        rows.append((ticker, bucket))
    return dict(rows)


def rebalance_dates(dates: pd.Index | pd.Series) -> list[pd.Timestamp]:
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Index(dates).unique())}).sort_values("date")
    return (
        frame.assign(month=lambda d: d["date"].dt.to_period("M"))
        .drop_duplicates("month", keep="first")["date"]
        .tolist()
    )


def build_feature_panel(
    bars: pd.DataFrame,
    cfg: QC345RotationConfig,
) -> tuple[pd.DataFrame, SelectionDiagnostics]:
    if cfg.momentum_price_field not in bars.columns:
        raise ValueError(f"bars missing momentum price field {cfg.momentum_price_field!r}")
    if cfg.momentum_price_field == "close" and cfg.corporate_action_window < cfg.lookback_sessions:
        raise ValueError(
            "momentum_price_field='close' requires corporate_action_window >= lookback_sessions "
            "or an adjusted momentum field"
        )

    panel = bars.sort_values(["ticker", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    g = panel.groupby("ticker", sort=False)

    panel["raw_dollar_volume"] = panel["close"] * panel["volume"]
    panel["asof_close"] = g["close"].shift(1)
    panel["liquidity_proxy"] = g["raw_dollar_volume"].transform(
        lambda s: (
            s.rolling(cfg.liquidity_window, min_periods=cfg.min_liquidity_history).median().shift(1)
        )
    )
    panel["market_cap_proxy"] = panel["liquidity_proxy"]
    panel["market_cap_proxy_price_x_dv"] = panel["liquidity_proxy"] * panel["asof_close"]
    panel["momentum"] = g[cfg.momentum_price_field].transform(
        lambda s: s.shift(1) / s.shift(cfg.lookback_sessions + 1) - 1.0
    )
    panel["realized_volatility"] = g["close"].transform(
        lambda s: (
            s.pct_change()
            .rolling(
                cfg.realized_vol_window,
                min_periods=cfg.min_realized_vol_history,
            )
            .std(ddof=0)
            .shift(1)
        )
    )

    raw_ret = g["close"].pct_change()
    panel["split_flag"] = raw_ret.abs() > cfg.corporate_action_move
    panel["split_blocked"] = g["split_flag"].transform(
        lambda s: (
            s.astype(int)
            .rolling(cfg.corporate_action_window, min_periods=1)
            .max()
            .shift(1)
            .fillna(0)
            .astype(bool)
        )
    )

    panel["eligible"] = (
        panel["asof_close"].gt(cfg.price_floor)
        & panel["liquidity_proxy"].notna()
        & panel["momentum"].notna()
        & ~panel["split_blocked"]
    )
    panel["mania_blocked"] = False
    if cfg.mania_momentum_threshold is not None and cfg.mania_volatility_threshold is not None:
        panel["mania_blocked"] = panel["momentum"].gt(cfg.mania_momentum_threshold) & panel[
            "realized_volatility"
        ].gt(cfg.mania_volatility_threshold)
        panel["eligible"] &= ~panel["mania_blocked"]

    split_blocked_name_months = int(
        panel.loc[panel["split_blocked"], ["ticker", "date"]]
        .assign(month=lambda d: d["date"].dt.to_period("M"))
        .drop_duplicates(["ticker", "month"])
        .shape[0]
    )
    mania_blocked_name_months = int(
        panel.loc[panel["mania_blocked"], ["ticker", "date"]]
        .assign(month=lambda d: d["date"].dt.to_period("M"))
        .drop_duplicates(["ticker", "month"])
        .shape[0]
    )
    return panel, SelectionDiagnostics(
        split_blocked_name_months=split_blocked_name_months,
        mania_blocked_name_months=mania_blocked_name_months,
    )


def _top(df: pd.DataFrame, size: int, column: str) -> pd.DataFrame:
    if size <= 0 or df.empty:
        return df.iloc[0:0].copy()
    return df.sort_values([column, "ticker"], ascending=[False, True]).head(size)


def select_universe(
    panel_for_day: pd.DataFrame,
    cfg: QC345RotationConfig,
) -> list[str]:
    universe = panel_for_day.loc[
        panel_for_day["asof_close"].notna() & panel_for_day["liquidity_proxy"].notna()
    ].copy()
    if universe.empty:
        return []

    liquid = _top(universe, cfg.liquidity_filter_size, "liquidity_proxy")
    if cfg.market_cap_mode == "dv_proxy":
        scoped = liquid.loc[liquid["market_cap_proxy"].notna()]
        ranked = _top(scoped, cfg.universe_size, "market_cap_proxy")
    elif cfg.market_cap_mode == "price_x_dv":
        scoped = liquid.loc[liquid["market_cap_proxy_price_x_dv"].notna()]
        ranked = _top(scoped, cfg.universe_size, "market_cap_proxy_price_x_dv")
    elif cfg.market_cap_mode == "skip":
        ranked = liquid
    else:
        raise ValueError(f"unsupported market_cap_mode: {cfg.market_cap_mode}")
    return ranked["ticker"].tolist()


def preselection_bounds(panel: pd.DataFrame, cfg: QC345RotationConfig) -> pd.DataFrame:
    """Per-rebalance bounds needed to reproduce `select_universe` from a broader panel.

    This is not a live scanner by itself. It is the audit surface for deciding how wide the live
    preselection fetch must be: compute features over the full candidate substrate, then measure how
    narrow a price/liquidity/price-x-dollar-volume prefilter could have been without losing the
    researched top `universe_size`.
    """
    rows: list[dict[str, object]] = []
    for day in rebalance_dates(panel["date"]):
        frame = panel.loc[panel["date"] == day].copy()
        rankable = frame.loc[frame["asof_close"].notna() & frame["liquidity_proxy"].notna()].copy()
        liquid = _top(rankable, cfg.liquidity_filter_size, "liquidity_proxy")
        selected = select_universe(frame, cfg)
        selected_frame = frame.set_index("ticker").loc[selected] if selected else frame.iloc[0:0]
        rows.append(
            {
                "date": pd.Timestamp(day),
                "rankable_names": int(len(rankable)),
                "liquidity_candidates": int(len(liquid)),
                "selected_names": int(len(selected)),
                "min_selected_price": (
                    float(selected_frame["asof_close"].min()) if selected else None
                ),
                "min_selected_liquidity_proxy": (
                    float(selected_frame["liquidity_proxy"].min()) if selected else None
                ),
                "min_selected_market_cap_proxy_price_x_dv": (
                    float(selected_frame["market_cap_proxy_price_x_dv"].min())
                    if selected and "market_cap_proxy_price_x_dv" in selected_frame
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def preserving_preselection_thresholds(bounds: pd.DataFrame) -> dict[str, float]:
    """Tightest global observed floors that preserve every non-empty selected universe."""
    nonempty = bounds.loc[bounds["selected_names"].gt(0)].copy()
    if nonempty.empty:
        return {}
    return {
        "price_floor": float(nonempty["min_selected_price"].min()),
        "liquidity_proxy_floor": float(nonempty["min_selected_liquidity_proxy"].min()),
        "market_cap_proxy_price_x_dv_floor": float(
            nonempty["min_selected_market_cap_proxy_price_x_dv"].min()
        ),
    }


def select_portfolio(
    panel_for_day: pd.DataFrame,
    cfg: QC345RotationConfig,
    *,
    universe: set[str] | None = None,
) -> tuple[list[str], dict[str, float]]:
    scoped = panel_for_day
    if universe is not None:
        scoped = scoped.loc[scoped["ticker"].isin(universe)].copy()
    eligible = scoped.loc[scoped["eligible"]].copy()
    if eligible.empty:
        return [], {}

    ranked = _top(eligible, cfg.portfolio_size, "momentum")
    names = ranked["ticker"].tolist()
    scores = dict(zip(ranked["ticker"], ranked["momentum"]))
    return names, scores


def decide(
    panel_for_day: pd.DataFrame,
    cfg: QC345RotationConfig,
    held: set[str],
    *,
    universe: set[str] | None = None,
) -> QC345Decision:
    date = pd.Timestamp(panel_for_day["date"].iloc[0])
    selection_universe = None if universe is None else set(universe) | set(held)
    target_names, scores = select_portfolio(panel_for_day, cfg, universe=selection_universe)
    target_set = set(target_names)
    return QC345Decision(
        date=date,
        hold=tuple(target_names),
        enter=tuple(
            sym for sym in target_names if sym not in held and (universe is None or sym in universe)
        ),
        exit=tuple(sorted(held - target_set)),
        scores=scores,
    )
