"""Universe sources for QC345.

The source answers the SCANNER question: which names are in the monthly universe on date `d`?
Ranking and exits are a separate concern and stay in `engine.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from kumo_strategies.strategies.momentum_rotation.candidates import CandidateSource, register

from .config import QC345RotationConfig
from .engine import (
    build_feature_panel,
    filter_asset_universe,
    preselection_bounds,
    rebalance_dates,
    select_universe,
    terminal_buckets,
)

if TYPE_CHECKING:
    from .engine import SelectionDiagnostics


@register
class QC345ComputedSource(CandidateSource):
    """Point-in-time monthly scanner reconstructed from QC345's selection stages.

    The universe is rescanned only on QC345 rebalance sessions. Between rebalances the latest scan is
    carried forward unchanged: leaving the universe mid-month does not itself exit a held position.
    """

    name = "qc345_computed"

    def __init__(
        self,
        bars: pd.DataFrame,
        assets: pd.DataFrame | None,
        cfg: QC345RotationConfig,
    ) -> None:
        filtered = (
            filter_asset_universe(bars, assets, cfg)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
        filtered["date"] = pd.to_datetime(filtered["date"])

        self.filtered_bars = filtered
        self.cfg = cfg
        self.panel, self.selection_diagnostics = build_feature_panel(filtered, cfg)
        self.rebalance_sessions = tuple(rebalance_dates(self.panel["date"]))
        self._last_session_by_symbol = {
            str(symbol): pd.Timestamp(day)
            for symbol, day in filtered.groupby("ticker", sort=False)["date"].max().items()
        }
        self._terminal_bucket_by_symbol = terminal_buckets(filtered, assets)

        current: frozenset[str] = frozenset()
        scans = set(self.rebalance_sessions)
        eligible_by_date: dict[pd.Timestamp, frozenset[str]] = {}
        for day, frame in self.panel.groupby("date", sort=False):
            session = pd.Timestamp(day)
            if session in scans:
                current = frozenset(select_universe(frame, cfg))
            eligible_by_date[session] = current

        self._eligible_by_date = eligible_by_date
        self._sessions = pd.Index(sorted(eligible_by_date))

    @property
    def diagnostics(self) -> "SelectionDiagnostics":
        return self.selection_diagnostics

    def preselection_bounds(self) -> pd.DataFrame:
        return preselection_bounds(self.panel, self.cfg)

    def eligible(self, d: pd.Timestamp) -> set[str]:
        ts = pd.Timestamp(d).normalize()
        if ts in self._eligible_by_date:
            return set(self._eligible_by_date[ts])
        pos = self._sessions.searchsorted(ts, side="right") - 1
        if pos < 0:
            return set()
        return set(self._eligible_by_date[pd.Timestamp(self._sessions[pos])])

    def terminal_symbols(
        self,
        d: pd.Timestamp,
        symbols: set[str] | None = None,
    ) -> dict[str, str]:
        """Held names known to have stopped printing before `d`, by terminal bucket."""
        ts = pd.Timestamp(d).normalize()
        names = symbols if symbols is not None else set(self._terminal_bucket_by_symbol)
        out: dict[str, str] = {}
        for symbol in names:
            bucket = self._terminal_bucket_by_symbol.get(symbol, "active")
            last_session = self._last_session_by_symbol.get(symbol)
            if bucket != "active" and last_session is not None and last_session < ts:
                out[symbol] = bucket
        return out
