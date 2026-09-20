"""CRSISHORT — short overbought, high-volatility, liquid US equities (#123).

The pure layer is importable without Nautilus. Ported from the lab's frozen spec in
`research-lab/qc-strategy-review/replication/`; the backtest runner and the Nautilus adapter follow.

NOT QuantConnect #411. What survives from that strategy is the ENTRY CONDITION, measured independently
of any portfolio (9,492 signals, forward return -1.93% at 5 sessions and -5.13% at 10, negative in
every year and every liquidity bucket). Its published exit returns -99.8% here; the exits are ours.
"""

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import (
    LOCATABLE,
    ADJUSTED, ShortDecision, apply_gates, build_feature_panel, decide, panel_signature)
from kumo_strategies.strategies.crsi_short.screen import Screen, ScreenRefused, screen_universe
from kumo_strategies.strategies.crsi_short.exits import (
    FLAT, REVERSAL, WEAKNESS, SessionBar, ShortExitPlan, ShortTrailState, evaluate_short_exits,
    open_state)

__all__ = [
    "LOCATABLE", "Screen", "ScreenRefused", "screen_universe", "CrsiShortConfig", "ShortDecision", "ADJUSTED", "build_feature_panel", "decide",
           "apply_gates", "panel_signature",
           "SessionBar", "ShortTrailState", "ShortExitPlan", "evaluate_short_exits", "open_state",
           "FLAT", "REVERSAL", "WEAKNESS"]
