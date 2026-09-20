"""The universe CRSISHORT trades is a SCREEN, refreshed daily — not a list.

The backtest scores every US equity and lets three gates decide who is in: price above the floor,
prior-session dollar volume above the entry floor, 100 warm sessions of indicators. That is ~770
names on a typical 2026 session and ~2,500 distinct names over the year. The first live deployment
handed the lane the 130 names the 2025-26 acceptance run had TRADED — an output of the backtest,
survivorship-shaped — and on 2026-09-08 only 34 of them were still eligible while 575 eligible
names were invisible to the lane. A rule measured on the screen and run on its own past trade list
is a different strategy with the same name.

This module is the screen as a pure function, so the platform's refresh job and the backtest ask
the SAME gates the same way (`apply_gates` is the one derivation; this only asks it a question):

    screen_universe(bars, cfg, adjustment=ADJUSTED)  ->  Screen(symbols, eligible_asof, ...)

WHY A LOOKBACK, NOT "ELIGIBLE TODAY". A name that crosses the dollar-volume floor tomorrow needs
100 sessions of bars already loaded to signal the day after, and a lane learns its symbols at
boot. So the universe the lane WATCHES is every name eligible at least once in the last
`lookback_sessions` (20 → ~1,240 names on 2026-09-08); the lane's own gates decide each session
who may signal. Watching is cheap on a venue without a line cap; on one with a cap the number
here is the number that has to fit, and it is stated in the diagnostics rather than discovered.

REFUSALS, NEVER AN EMPTY LIST. Too little history to warm anyone, a raw (unadjusted) frame, or a
screen that comes back empty each raise and say why. A refresh that quietly wrote `[]` would make
the lane boot with nothing to watch and every surface read green.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.engine import build_feature_panel


class ScreenRefused(ValueError):
    """The screen cannot honestly answer; the message says what was missing."""


@dataclass(frozen=True)
class Screen:
    asof: pd.Timestamp
    """The last session the screen looked at — the prior close the lane will decide from."""
    symbols: tuple[str, ...]
    """Names eligible at least once in the last `lookback_sessions` — what the lane WATCHES."""
    eligible_asof: tuple[str, ...]
    """Names eligible on `asof` itself — the subset that may signal on the next session."""
    lookback_sessions: int
    diagnostics: dict = field(default_factory=dict)


def screen_universe(bars: pd.DataFrame, cfg: CrsiShortConfig, *, adjustment: str,
                    lookback_sessions: int = 20, asof: pd.Timestamp | str | None = None) -> Screen:
    """Screen `bars` (ticker/date/open/high/low/close/volume, DAILY, split-adjusted) for the lane.

    `adjustment` is forwarded to `build_feature_panel`, which refuses anything but the adjusted
    series — a reverse split on raw prices is a −1000% short in the backtest and a universe entry
    here. `asof` defaults to the last session in `bars`; a later `asof` than the data holds is
    refused rather than silently clamped.
    """
    if lookback_sessions < 1:
        raise ScreenRefused(f"lookback_sessions must be >= 1, got {lookback_sessions!r}")
    panel = build_feature_panel(bars, cfg, adjustment=adjustment)
    sessions = sorted(panel["date"].unique())
    if not sessions:
        raise ScreenRefused("no sessions in the frame")
    last = pd.Timestamp(sessions[-1])
    asof_ts = last if asof is None else pd.Timestamp(asof)
    if asof_ts > last:
        raise ScreenRefused(f"asof {asof_ts.date()} is after the last session in the frame ({last.date()})")
    if asof_ts not in set(pd.Timestamp(s) for s in sessions):
        raise ScreenRefused(f"asof {asof_ts.date()} is not a session in the frame")
    # Warmth is per NAME, but a frame shorter than the warmup cannot warm anyone: say so rather
    # than return the empty screen such a frame produces.
    upto = [pd.Timestamp(s) for s in sessions if pd.Timestamp(s) <= asof_ts]
    if len(upto) < cfg.warmup_sessions:
        raise ScreenRefused(
            f"{len(upto)} sessions up to {asof_ts.date()}; the indicators need {cfg.warmup_sessions} "
            f"to warm — nothing can be eligible on this frame")
    window = upto[-lookback_sessions:]
    p = panel[panel["date"].isin(window)]
    watched = tuple(sorted(p.loc[p["eligible"], "ticker"].astype(str).unique()))
    today = p[p["date"] == asof_ts]
    eligible_asof = tuple(sorted(today.loc[today["eligible"], "ticker"].astype(str).unique()))
    if not watched:
        raise ScreenRefused(
            f"no name was eligible in the {len(window)} sessions to {asof_ts.date()} — "
            f"{p['ticker'].nunique()} names seen, {int(p['manageable'].sum())} manageable rows. "
            f"A screen that finds nothing is a data problem, not a universe.")
    diag = dict(
        asof=str(asof_ts.date()), lookback_sessions=lookback_sessions, sessions_in_window=len(window),
        names_seen=int(panel["ticker"].nunique()), names_in_window=int(p["ticker"].nunique()),
        watched=len(watched), eligible_asof=len(eligible_asof),
        eligible_per_session=dict((str(pd.Timestamp(d).date()), int(v))
                                  for d, v in p.groupby("date")["eligible"].sum().items()),
        gates=dict(price_floor=cfg.price_floor, min_dollar_volume=cfg.min_dollar_volume,
                   warmup_sessions=cfg.warmup_sessions,
                   note="the volatility gate is applied at SIGNAL time, not here — the lane must watch "
                        "a name before its vol crosses the bar"),
    )
    return Screen(asof=asof_ts, symbols=watched, eligible_asof=eligible_asof,
                  lookback_sessions=lookback_sessions, diagnostics=diag)
