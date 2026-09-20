"""`decide()` takes ONE session, the parameter has always been called `day`, and nothing made it true (#205).

TWO FAILURES, ONE MISSING CONTRACT, AND ONLY ONE OF THEM IS LOUD:

  NOT FEATURIZED   `KeyError: 'eligible'` — this killed every SMHGLD-007 rung on 2026-09-11. The
                   caller passed `panel()` straight through, and `eligible`/`asof_close` exist only
                   after `build_feature_panel`.
  NOT DAY-SLICED   SILENT, and worse. A featurized frame holding the whole window HAS `eligible`, so
                   a presence check passes — and `.iloc[0]` then decides on the OLDEST session in
                   the window: a plausible trade at month-old prices with nothing saying so.

A GUARD ON `eligible` ALONE REPAIRS THE LOUD HALF AND LEAVES THE HALF THAT TRADES. That is the
August lesson: QC345-003 died of this on its own first live rebalance (2026-08-20 13:35Z,
post-mortem at `qc345_rotation.py:585-598`), and it survives review because BOTH FRAMES ARE CALLED
`panel` and the featurization is a side effect of constructing something else.

THE FIXTURE MOVES THE LEGS IN OPPOSITE DIRECTIONS — SMH 100 -> 250 while GLD 400 -> 300 — so a
decision taken on the wrong ROW is DETECTABLE rather than merely wrong. Design taken from the
coordinator's cockpit test; the drift on the oldest session and on the newest point opposite ways,
so reading the wrong one cannot coincidentally agree.
"""

from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace

import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig
from kumo_strategies.strategies.smhgld_sleeve.engine import build_feature_panel, decide

SESSIONS = [pd.Timestamp(f"2026-09-{d:02d}") for d in (1, 2, 3, 4, 7)]
#: OPPOSITE directions, so a wrong-row decision cannot accidentally agree with the right one.
PRICES = {"SMH": [100.0, 130.0, 170.0, 210.0, 250.0],
          "GLD": [400.0, 375.0, 350.0, 325.0, 300.0]}


def _bar(symbol, close, ts):
    bt = BarType.from_str(f"{symbol}.XNAS-1-DAY-LAST-EXTERNAL")
    p = Price.from_str(f"{close:.2f}")
    return Bar(bar_type=bt, open=p, high=p, low=p, close=p,
               volume=Quantity.from_int(100), ts_event=ts, ts_init=ts)


class _Host:
    """A narrow host carrying the REAL adapter functions.

    Not `SmhGldSleeveStrategy.__new__`: `log` and `id` are read-only Cython attributes on Nautilus's
    `Actor`, so a constructed-but-uninitialised instance cannot be given a log to record into. Every
    method below is the attribute itself, so renaming or deleting one fails these tests rather than
    silently exercising a copy — the same construction the rest of this package's host doubles use.
    """

    on_historical_data = SmhGldSleeveStrategy.on_historical_data
    _ingest = SmhGldSleeveStrategy._ingest
    panel = SmhGldSleeveStrategy.panel
    _day_for = SmhGldSleeveStrategy._day_for
    _warmth = SmhGldSleeveStrategy._warmth

    def __init__(self):
        self.id = "SMHGLD-007"
        self._cfg = SmhGldSleeveConfig()
        self._need = 3
        self._bars = defaultdict(lambda: deque(maxlen=64))
        self.said: list[tuple[str, str]] = []
        self.log = SimpleNamespace(
            info=lambda m: self.said.append(("info", m)),
            warning=lambda m: self.said.append(("warning", m)),
            error=lambda m: self.said.append(("error", m)))


def _lane():
    """The real ingest path — bars arrive through `on_historical_data` (ks#202) and `panel()` is the
    real method, so this exercises the seam rather than a copy."""
    lane = _Host()
    for leg, series in PRICES.items():
        for ts, close in zip(SESSIONS, series):
            lane.on_historical_data(_bar(leg, close, int(ts.value)))
    return lane, lane.said


# -- the adapter hands over ONE day ---------------------------------------------------------------

def test_the_adapter_returns_ONLY_the_session_asked_for():
    lane, _ = _lane()
    day = lane._day_for(SESSIONS[-1], lane.panel())

    assert day is not None and not day.empty
    assert list(pd.Series(day["date"]).unique()) == [SESSIONS[-1]], (
        "the adapter handed over more than one session")
    assert {"eligible", "asof_close"} <= set(day.columns), "the adapter did not featurize"


def test_the_slice_is_BY_SESSION_and_not_by_the_newest_row():
    """A session with no rows must yield a HOLD, not a decision on whatever is newest. `max()` looks
    like the safe version of this slice and is not — on a stale feed it decides silently."""
    lane, said = _lane()
    missing = pd.Timestamp("2026-09-08")                    # no bars for this session

    assert lane._day_for(missing, lane.panel()) is None
    warned = [m for lvl, m in said if lvl == "warning"]
    assert warned and "no feature rows for 2026-09-08" in warned[0], said
    assert "2026-09-07" in warned[0], "the hold did not name the newest row it declined to use"


def test_a_panel_that_cannot_be_FEATURIZED_holds_rather_than_raising_into_the_rung():
    lane, said = _lane()
    got = lane._day_for(SESSIONS[-1], pd.DataFrame({"nonsense": [1]}))
    assert got is None
    assert any(lvl == "error" for lvl, _ in said)


# -- and decide() enforces it ---------------------------------------------------------------------

def test_the_RAW_panel_is_refused_by_name_rather_than_KeyError():
    """The 2026-09-11 failure, as its own test. `KeyError: 'eligible'` says nothing about what the
    caller should have done."""
    lane, _ = _lane()
    with pytest.raises(ValueError, match="build_feature_panel"):
        decide(lane.panel(), lane._cfg, set())


def test_a_FEATURIZED_BUT_UNSLICED_frame_is_refused_TOO_and_this_is_the_silent_half():
    """It has `eligible`, so every presence check passes — and `.iloc[0]` would decide on the
    OLDEST session. This is the case a guard on column presence cannot see."""
    lane, _ = _lane()
    featurized = build_feature_panel(lane.panel(), lane._cfg)

    with pytest.raises(ValueError) as e:
        decide(featurized, lane._cfg, set())
    said = str(e.value)
    assert "ONE session" in said and "OLDEST" in said
    assert "2026-09-01" in said, "the refusal did not name the row it would have decided on"
    assert "date.max()" in said or "max()" in said, "the refusal did not warn against the max() slice"


def test_the_CORRECTLY_SLICED_frame_still_decides():
    """The guard must not become a lane that cannot trade."""
    lane, _ = _lane()
    day = lane._day_for(SESSIONS[-1], lane.panel())
    assert decide(day, lane._cfg, set()).regime == "opening"


def test_an_EMPTY_but_well_formed_frame_is_allowed_through_and_holds():
    """A lane with no rows for its session is a legitimate state the caller reports as a hold;
    `decide` answers it by holding. Refusing here would turn a hold into a crash."""
    lane, _ = _lane()
    empty = build_feature_panel(lane.panel(), lane._cfg).iloc[0:0]
    assert decide(empty, lane._cfg, {"SMH"}).regime == "unknown"


def test_None_is_refused_because_a_missing_frame_is_not_an_empty_one():
    lane, _ = _lane()
    with pytest.raises(ValueError, match="not an empty one"):
        decide(None, lane._cfg, set())


# -- the wrong row would have been a real, opposite trade -----------------------------------------

def test_deciding_on_the_OLDEST_row_would_have_been_a_DIFFERENT_TRADE():
    """Why the silent half matters. Prices on the two legs move in opposite directions, so the
    decision taken on the first session and the one taken on the last are not the same decision —
    the unsliced frame does not merely lose precision, it inverts the drift."""
    lane, _ = _lane()
    featurized = build_feature_panel(lane.panel(), lane._cfg)
    first = featurized.loc[featurized["date"] == SESSIONS[1]]
    last = featurized.loc[featurized["date"] == SESSIONS[-1]]

    held = {"SMH", "GLD"}
    shares = {"SMH": 100.0, "GLD": 25.0}
    early = decide(first, lane._cfg, held, shares=shares)
    late = decide(last, lane._cfg, held, shares=shares)

    assert early.weights == late.weights                      # same target, different distance to it
    assert (early.regime, early.reasons) != (late.regime, late.reasons), (
        "the fixture cannot tell the two rows apart, so the unsliced case would be undetectable "
        "here — the prices must move in opposite directions for this to be a real check")
