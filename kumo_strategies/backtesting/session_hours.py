"""What "the regular session" means, in one place, for filtering vendor panels.

WHY THIS IS IN THE PACKAGE AND NOT IN A RESEARCH SCRIPT
-------------------------------------------------------
It started as two module-level strings in `research/envelopes/build_panel.py`, and within a day a
second research script imported them across directories with `sys.path.insert` and a bare
`from build_panel import ...`. There are THREE files named `build_panel.py` in `research/`, and
`sys.modules` is consulted before `sys.path` — so once any of them has been imported under that
bare name, the others resolve to it. Reproduced: the import raises `ImportError`, the caller falls
back to a local copy of the constants, and the duplicate definition the import existed to prevent is
silently reinstated.

A shared definition needs a stable import path. `from kumo_strategies.backtesting.session_hours
import ...` cannot resolve to the wrong file.

THE TIMEZONE IS THE WHOLE BUG, SO IT IS NOT OPTIONAL HERE
----------------------------------------------------------
research-lab shipped, then caught by running it, a gate that compared these EASTERN bounds against UTC
bar timestamps: `"09:30" <= ts[11:16] <= "15:55"` on UTC data selects roughly 04:30-10:55 Eastern
(05:30-11:55 under EDT) — a window that opens in the pre-market and closes before lunch. It flagged a
real ticker-day as broken because the FIRST bar it admitted was the 04:30 pre-market print. A gate
written to catch session/timezone confusion, containing session/timezone confusion.

Note the shape, because it is why review does not catch these: the filter was not too NARROW, which
returns nothing and is obvious. It was too WIDE — it still returned bars, and only the first one was
wrong.

WHY A REAL ZONE AND NOT AN OFFSET, which is the strongest reason this module exists and was not
originally the reason it gave. From research-lab, after their own arithmetic caught them out (they
asserted 13:30Z was 09:30 Eastern on a March date, and it is 08:30, because March 2 is still EST):

    a fix written by someone reasoning about it would plausibly have hardcoded a UTC offset,
    and been wrong for about four months a year

`tz_convert` on a real zone is not a tidiness preference over `hour - 5`. It is the difference
between correct and correct-in-summer — and a hardcoded offset passes every test written in March.

Two vendors, two conventions, in this repo today:

    research-lab/data/raw/massive/intraday/   tz-NAIVE, already Eastern
    Alpaca /v2/stocks/bars                tz-AWARE, UTC

So `regular_hours_mask` REQUIRES the caller to say what their timestamps mean. There is no default,
for the same reason `POSITION_SIDE`, `price_adjustment`, `strategy_id` and `MarketAction` have none:
the wrong answer here is silent, plausible, and moves every downstream number.

THESE BOUNDS DO NOT RESOLVE SLOTS. `run_sessions` derives its session bounds from the BARS on
purpose -- "the tape is what the venue printed, so a half day is short here without anyone having to
know it is a half day". Hardcoding 15:55 into that path would break exactly the half-day handling it
was written for. This module is for FILTERING A VENDOR PANEL down to the regular session BEFORE the
runner sees it; `momentum_rotation.slots` remains the only thing that decides when a lane acts.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = ["ET", "RTH_LAST", "RTH_LAST_STR", "RTH_OPEN", "RTH_OPEN_STR", "regular_hours_mask"]

#: The exchange's timezone. US equities regular hours are defined in Eastern and shift against UTC
#: twice a year, so a UTC-only comparison is wrong for part of every year rather than always — which
#: is worse, because it works in testing.
ET = ZoneInfo("America/New_York")

#: The first and last 5-minute BAR OPENS of a regular session. 15:55 is the last bar's open, and it
#: covers to 16:00 — not a typo, and the reason a naive `< 16:00` filter drops the closing bar.
RTH_OPEN = time(9, 30)
RTH_LAST = time(15, 55)

#: String forms, for callers filtering with `.strftime` or comparing wire values. Derived from the
#: `time` objects rather than written twice, so the two cannot disagree.
RTH_OPEN_STR = RTH_OPEN.strftime("%H:%M")
RTH_LAST_STR = RTH_LAST.strftime("%H:%M")


def regular_hours_mask(ts: pd.Series, *, tz: str | ZoneInfo | None) -> pd.Series:
    """Boolean mask selecting the regular session from `ts`.

    `tz` says WHAT THE TIMESTAMPS MEAN and has no default:

      * tz-aware input  -- `tz` must be None; the series carries its own zone and is converted.
      * tz-naive input  -- `tz` names the zone the clock readings are already in, e.g. "UTC" for
        Alpaca-shaped data stripped of its offset, or `ET` for lab's per-session store.

    Passing `tz=None` with naive timestamps RAISES rather than assuming Eastern. Assuming is how
    "09:30-15:55" came to select a pre-market-to-lunchtime window in a gate built to catch exactly
    that, while still returning enough bars to look right.
    """
    s = pd.to_datetime(ts)
    aware = s.dt.tz is not None
    if aware:
        if tz is not None:
            raise ValueError(
                f"timestamps already carry {s.dt.tz}; passing tz={tz!r} as well is two claims about "
                f"one fact. Pass tz=None for tz-aware input.")
        local = s.dt.tz_convert(ET)
    else:
        if tz is None:
            raise ValueError(
                "tz-naive timestamps and tz=None: refusing to guess. Lab's per-session intraday "
                "store is already Eastern (pass ET); Alpaca's wire format is UTC (pass 'UTC'). "
                "Guessing wrong selects roughly 04:30-10:55 Eastern — it opens in the pre-market "
                "and closes before lunch, still returns bars, and looks like an ordinary session.")
        local = s.dt.tz_localize(tz).dt.tz_convert(ET)
    t = local.dt.time
    return (t >= RTH_OPEN) & (t <= RTH_LAST)
