"""Load raw daily bars into Nautilus objects for a backtest.

Reads the same panel the research used, so a backtest result and a research result are comparable
by construction. Prices are RAW and unadjusted by policy — corporate actions are gated in the
strategy's own layer, not silently repaired here. An exporter that "fixed" a split would be applying
exactly the adjustment the policy forbids.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

BAR_SUFFIX = "-1-DAY-LAST-EXTERNAL"
ET = ZoneInfo("America/New_York")


def load_panel(path: str | Path, symbols: list[str] | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    if symbols:
        d = d[d.ticker.isin(set(symbols))]
    if start:
        d = d[d.date >= pd.Timestamp(start)]
    if end:
        d = d[d.date <= pd.Timestamp(end)]
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def instruments_for(symbols: list[str], venue: str = "XNAS") -> dict[str, Equity]:
    return {s: TestInstrumentProvider.equity(symbol=s, venue=venue) for s in sorted(set(symbols))}


def bars_for(panel: pd.DataFrame, instrument: Equity) -> list[Bar]:
    """One symbol's rows -> Nautilus Bars at the instrument's precision.

    A daily bar is stamped at its own session's CLOSE, which is the first moment it could exist.
    Stamping it at the session's midnight -- as this did -- puts a full trading day of prices on the
    bus roughly fourteen hours before the market that produced them opened. Nothing caught it while
    decisions were triggered by bar arrival, because the strategy only ever compared bars to each
    other. Against a clock, it is the difference between filling at the session's open and filling at
    the next one, and only in backtest.

    ET, not UTC, so the stamp tracks the DST shift the exchange actually observes.
    """
    bt = BarType.from_str(f"{instrument.id}{BAR_SUFFIX}")
    pp, sp = instrument.price_precision, instrument.size_precision
    out = []
    for r in panel.itertuples(index=False):
        close_et = pd.Timestamp(r.date).normalize().tz_localize(ET) + pd.Timedelta(hours=16)
        ts = close_et.tz_convert("UTC").value
        out.append(Bar(bt, Price(r.open, pp), Price(r.high, pp), Price(r.low, pp),
                       Price(r.close, pp), Quantity(max(r.volume, 0), sp), ts, ts))
    return out
