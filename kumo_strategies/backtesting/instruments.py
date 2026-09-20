"""Real Nautilus instruments from Alpaca's asset definitions.

Backtests here used `TestInstrumentProvider.equity(symbol=s, venue="XNAS")` — every name stamped
NASDAQ with invented metadata. Of the 220 names this strategy trades, 142 are NYSE and only 70 are
NASDAQ, so the instrument ids in a backtest did not match the ids a live fill would carry, and no
reconciliation between the two was possible.

Definitions come from Cockpit's `scripts/export_instruments.py` (Alpaca `/v2/assets`), which owns
the provider connection and the canonical `TICKER.MIC` identity.
"""

from __future__ import annotations

import json
from pathlib import Path

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity


def load_definitions(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())


def build(defn: dict) -> Equity:
    """One exported definition -> a Nautilus Equity carrying its real venue and tick size."""
    inc = defn.get("price_increment") or "0.01"
    return Equity(
        instrument_id=InstrumentId(Symbol(defn["symbol"]), Venue(defn["mic"])),
        raw_symbol=Symbol(defn["symbol"]),
        currency=USD,
        price_precision=abs(Price.from_str(str(inc)).precision),
        price_increment=Price.from_str(str(inc)),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
        info={k: defn[k] for k in ("name", "exchange", "shortable", "fractionable") if k in defn},
    )


def build_all(path: str | Path, symbols: list[str] | None = None) -> dict[str, Equity]:
    defs = load_definitions(path)
    keep = set(symbols) if symbols else set(defs)
    return {s: build(d) for s, d in defs.items() if s in keep}


def venues(instruments: dict[str, Equity]) -> list[str]:
    """Distinct venues — a backtest must add every one of them, not just XNAS."""
    return sorted({i.id.venue.value for i in instruments.values()})
