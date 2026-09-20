"""One definition of the regular session, and it refuses to guess a timezone.

TWO REAL DEFECTS, ONE DAY APART, BOTH IN CODE WRITTEN TO CATCH THE OTHER:

  * kumo-trading-strategies fitted MOMENTUM-002's envelope on lab's intraday store, which carries
    extended-hours bars. `run_sessions` derives session bounds FROM THE BARS, so `open+5m` resolved
    to 04:05 in the PRE-MARKET. Found by disagreement — 474 fills against 217 over the same dates.
  * research-lab then wrote a gate to assert its own panel was the regular session, and the gate
    compared EASTERN bounds against UTC timestamps: `"09:30" <= ts[11:16] <= "15:55"` on UTC data
    selects roughly 04:30-10:55 Eastern (05:30-11:55 under EDT) — a window that starts in the
    pre-market and ends before lunch. It flagged a real ticker-day as broken against a pre-market
    print, because the first bar it admitted was the 04:30 one.

The second is the one this module exists to make unrepresentable, because a comment saying "convert
first" is what the first version already had.
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from kumo_strategies.backtesting.session_hours import (
    ET,
    RTH_LAST,
    RTH_LAST_STR,
    RTH_OPEN,
    RTH_OPEN_STR,
    regular_hours_mask,
)


def _naive(*stamps: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stamps)))


# -- the refusal, which is the whole point -----------------------------------------------------------

def test_NAIVE_TIMESTAMPS_WITHOUT_A_TZ_ARE_REFUSED():
    """The defect this module exists for. Assuming Eastern on UTC data silently selects roughly
    04:30-10:55 Eastern — still returns bars, still looks like an ordinary session."""
    with pytest.raises(ValueError, match="refusing to guess"):
        regular_hours_mask(_naive("2026-03-02 09:30:00"), tz=None)


def test_the_refusal_NAMES_BOTH_CONVENTIONS_IN_THIS_REPO():
    """"Pass a timezone" is not actionable when the caller does not know which one their vendor
    uses. Both live sources are named so the answer is in the error."""
    with pytest.raises(ValueError) as e:
        regular_hours_mask(_naive("2026-03-02 09:30:00"), tz=None)
    msg = str(e.value)
    assert "Eastern" in msg and "UTC" in msg
    assert "pre-market" in msg, "the error does not say what going wrong looks like"


def test_AWARE_TIMESTAMPS_PLUS_AN_EXPLICIT_TZ_ARE_REFUSED():
    """Two claims about one fact. If they disagree, one silently wins."""
    s = pd.Series(pd.to_datetime(["2026-03-02 14:30:00+00:00"]))
    with pytest.raises(ValueError, match="two claims about one fact"):
        regular_hours_mask(s, tz="UTC")


# -- the two vendor conventions, both of which appear in this repo today -------------------------------

def test_UTC_AWARE_BARS_SELECT_THE_EASTERN_SESSION():
    """Alpaca's wire format. In early March (still EST, UTC-5) 14:30Z is the 09:30 open and 20:55Z
    the 15:55 last bar."""
    s = pd.Series(pd.to_datetime([
        "2026-03-02 09:30:00+00:00",   # 04:30 ET — pre-market
        "2026-03-02 14:30:00+00:00",   # 09:30 ET — the open
        "2026-03-02 20:55:00+00:00",   # 15:55 ET — the last bar
        "2026-03-02 21:00:00+00:00",   # 16:00 ET — after the close
    ]))
    assert list(regular_hours_mask(s, tz=None)) == [False, True, True, False]


def test_the_NAIVE_UTC_STRING_COMPARE_THAT_SHIPPED_TAKES_THE_PRE_MARKET_BAR_AS_THE_OPEN():
    """VERIFY BY DISAGREEMENT, as a test — and the disagreement is more specific than "it differs".

    Lab's gate took the FIRST bar passing its filter and compared the daily open against it. On UTC
    data the naive compare admits BOTH the 09:30Z bar (04:30 Eastern, pre-market) and the real
    14:30Z open, and 09:30Z sorts first — so `rth[0]` was the PRE-MARKET PRINT. That is why a
    correct panel was flagged as broken: not because the filter selected nothing, but because it
    selected too much and the first element was wrong.

    A filter that is merely too wide is the harder version to spot, because it still returns bars.
    """
    s = pd.Series(pd.to_datetime(["2026-03-02 09:30:00+00:00",    # 04:30 ET — pre-market
                                  "2026-03-02 14:30:00+00:00"]))  # 09:30 ET — the true open
    theirs = [RTH_OPEN_STR <= str(t)[11:16] <= RTH_LAST_STR for t in s]
    ours = list(regular_hours_mask(s, tz=None))
    assert theirs == [True, True], f"{theirs} — the naive compare admits both, which is the defect"
    assert ours == [False, True], ours
    assert theirs != ours, "the naive compare now agrees — this test no longer probes the defect"
    # The consequence, stated as the assertion that matters: their first selected bar is the wrong one.
    assert [t for t, keep in zip(s, theirs) if keep][0] != [t for t, keep in zip(s, ours) if keep][0]


def test_NAIVE_EASTERN_BARS_ARE_LABS_PER_SESSION_STORE():
    """`research-lab/data/raw/massive/intraday/` is tz-naive and already Eastern — 04:00 to 19:55."""
    s = _naive("2026-03-02 04:00:00", "2026-03-02 09:30:00",
               "2026-03-02 15:55:00", "2026-03-02 16:00:00", "2026-03-02 19:55:00")
    assert list(regular_hours_mask(s, tz=ET)) == [False, True, True, False, False]


def test_NAIVE_UTC_BARS_CONVERT_RATHER_THAN_COMPARE():
    s = _naive("2026-03-02 14:30:00", "2026-03-02 09:30:00")
    assert list(regular_hours_mask(s, tz="UTC")) == [True, False]


# -- the boundary that is not a typo ------------------------------------------------------------------

def test_THE_LAST_BAR_OPENS_AT_1555_AND_IS_INCLUDED():
    """15:55 is the last bar's OPEN and it covers to 16:00. A `< 16:00` filter written by someone
    who read 15:55 as the session end drops the closing bar — which is the one every close-anchored
    slot and every daily `close=last` depends on."""
    assert RTH_LAST == time(15, 55)
    s = _naive("2026-03-02 15:55:00")
    assert list(regular_hours_mask(s, tz=ET)) == [True]


def test_THE_OPENING_BAR_IS_INCLUDED():
    assert RTH_OPEN == time(9, 30)
    assert list(regular_hours_mask(_naive("2026-03-02 09:30:00"), tz=ET)) == [True]


def test_DST_IS_HANDLED_BECAUSE_THE_SESSION_IS_DEFINED_IN_EASTERN():
    """The reason a UTC-only rule is wrong for PART of every year rather than always — which is
    worse, because it works in testing. 13:30Z is the open in summer, 14:30Z in winter."""
    summer = pd.Series(pd.to_datetime(["2026-07-01 13:30:00+00:00"]))
    winter = pd.Series(pd.to_datetime(["2026-01-05 14:30:00+00:00"]))
    assert list(regular_hours_mask(summer, tz=None)) == [True]
    assert list(regular_hours_mask(winter, tz=None)) == [True]
    # and the same wall-clock UTC reading is NOT the open in the other half of the year
    assert list(regular_hours_mask(pd.Series(pd.to_datetime(["2026-01-05 13:30:00+00:00"])),
                                   tz=None)) == [False]


# -- one definition ------------------------------------------------------------------------------------

def test_THE_STRING_FORMS_ARE_DERIVED_NOT_WRITTEN_TWICE():
    assert RTH_OPEN_STR == RTH_OPEN.strftime("%H:%M")
    assert RTH_LAST_STR == RTH_LAST.strftime("%H:%M")


def test_THE_IMPORT_PATH_IS_UNAMBIGUOUS():
    """There were THREE files named `build_panel.py` under `research/`, and `sys.modules` is
    consulted before `sys.path` — so a bare `from build_panel import RTH_OPEN` after a
    `sys.path.insert` resolved to whichever was imported first. Reproduced: it raised ImportError
    and the caller fell back to a local copy, silently reinstating the duplicate the import existed
    to prevent.

    The studies left the tree (ks#211: a study is a variation of a set or it is private), so the
    collision is gone with them; what stays is the rule — this module is imported by its PACKAGE
    path, which cannot resolve to the wrong file, and no module under `kumo_strategies` carries a
    bare-name twin of it.
    """
    import pathlib

    import kumo_strategies.backtesting.session_hours as mod
    assert mod.__name__ == "kumo_strategies.backtesting.session_hours"
    pkg = pathlib.Path(mod.__file__).parents[1]
    twins = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("session_hours.py"))
    assert twins == ["backtesting/session_hours.py"], twins


def test_IT_DOES_NOT_RESOLVE_SLOTS():
    """`run_sessions` derives session bounds from the BARS on purpose, so a half day is short
    without anyone having to know it is a half day. Hardcoding 15:55 there would break exactly that.
    This module filters a vendor panel BEFORE the runner sees it; `momentum_rotation.slots` stays
    the only thing that decides when a lane acts."""
    import ast
    import inspect

    from kumo_strategies.backtesting import runner_sessions
    src = inspect.getsource(runner_sessions)
    imported = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert not any("session_hours" in (m or "") for m in imported), (
        "run_sessions imports the RTH bounds — its session bounds must come from the tape, or half "
        "days silently become full ones")
