"""Validate the pool against the broker's asset list. Spec follow-up to #186.

Agreed split with ledger-tool: I hold the ground truth (Alpaca's tradable universe), so I validate and
report failures back; they resolve them into their alias map or investigate the source frames. They
deliberately shipped NO fuzzy misread detection — an edit-distance pass flagged 13 symbols of which
11 were real tickers (ABEV/ABBV, CTRA/CTVA, CORT/COST, CYRX/CPRX), and a false flag skips a name the
trader actually holds.

A symbol we cannot trade must not sit silently in the pool: it can be ranked, chosen, and then fail
at submission — a slot consumed by an order that can never fill.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

_TRADING = "https://paper-api.alpaca.markets"


@dataclass
class TradableUniverse:
    """Alpaca's active, tradable US equities. Cached — it changes slowly."""

    key: str = field(default_factory=lambda: os.environ.get("APCA_API_KEY_ID", ""))
    secret: str = field(default_factory=lambda: os.environ.get("APCA_API_SECRET_KEY", ""))
    base: str = _TRADING
    ttl_hours: float = 12.0
    _symbols: set[str] = field(default_factory=set, repr=False)
    _exchanges: dict[str, str] = field(default_factory=dict, repr=False)
    _fetched: datetime | None = field(default=None, repr=False)

    def _fresh(self) -> bool:
        return bool(self._symbols) and self._fetched is not None and (
            datetime.now(timezone.utc) - self._fetched < timedelta(hours=self.ttl_hours))

    def symbols(self) -> set[str]:
        if self._fresh():
            return self._symbols
        if not self.key or not self.secret:
            raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set")
        req = urllib.request.Request(
            f"{self.base}/v2/assets?status=active&asset_class=us_equity",
            headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret})
        with urllib.request.urlopen(req, timeout=60) as r:
            assets = json.loads(r.read())
        tradable = [a for a in assets if a.get("tradable")]
        self._symbols = {a["symbol"] for a in tradable}
        # The listing venue rides along on the SAME response, so keep it rather than making a second
        # round trip. Callers that build Nautilus instrument ids need it: Alpaca is multi-venue, and
        # an id assembled with a guessed MIC matches no instrument in the cache -- no bars, no fills,
        # and nothing that looks like an error.
        self._exchanges = {a["symbol"]: a.get("exchange", "") for a in tradable}
        self._fetched = datetime.now(timezone.utc)
        return self._symbols

    def exchanges(self) -> dict[str, str]:
        """{symbol: Alpaca exchange} for every tradable asset, e.g. {"AAPL": "NASDAQ", "BAC": "NYSE"}.

        Raw Alpaca strings, deliberately not MICs: the exchange->MIC map belongs to whoever is
        building venue-specific ids, not here.
        """
        self.symbols()
        return dict(self._exchanges)

    def asset(self, symbol: str) -> dict | None:
        """Authoritative single-symbol lookup. None means the broker has never had it.

        The BULK list is not sufficient for classification: `asset_class=us_equity` omitted JWN,
        CTRA, CPRX and NGD, all of which this endpoint reports as inactive. Classifying from the
        bulk list would have told ledger-tool that Nordstrom and Coterra never existed.
        """
        try:
            req = urllib.request.Request(
                f"{self.base}/v2/assets/{symbol}",
                headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def check(self, pool: set[str], classify: bool = True) -> dict:
        """Split a pool into what the broker will accept and what it will not.

        With `classify`, each failure is looked up individually so the two very different causes are
        separated: a real company that no longer trades (nothing to fix — correct history) versus a
        symbol the broker has never had (the misread signature).
        """
        ok = self.symbols()
        bad = sorted(pool - ok)
        out = {"checked": len(pool), "tradable": len(pool) - len(bad), "not_tradable": bad,
               "coverage": round((len(pool) - len(bad)) / max(len(pool), 1), 4)}
        if classify and bad:
            detail = []
            for s in bad:
                a = self.asset(s)
                detail.append({
                    "symbol": s, "alpaca_status": (a or {}).get("status"),
                    "alpaca_name": (a or {}).get("name"),
                    "cause": ("delisted_or_inactive" if a and a.get("status") != "active"
                              else "known_but_untradable" if a else "unknown_to_broker")})
            out["detail"] = detail
            out["unknown_to_broker"] = [d["symbol"] for d in detail
                                        if d["cause"] == "unknown_to_broker"]
        return out
