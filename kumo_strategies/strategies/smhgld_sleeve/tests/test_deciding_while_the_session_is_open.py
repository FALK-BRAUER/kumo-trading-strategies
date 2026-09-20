"""A daily bar cannot exist at its own open, so the session's own row is absent intraday (#207).

MEASURED, NOT THEORISED. `SmhGldSleeveConfig.rebalance_period == 'D'`, and `rebalance_dates` with
'D' returns EVERY date in the panel — so `_is_rebalance` reduces to "is the session in the panel".
SMHGLD-007 was warm on 20 bars a leg on 2026-09-11 and still held at 19:40Z, because that session
was simply not in its panel. IB probed directly with the market shut: the newest daily bar is
`20260911`. The venue serves a session's daily bar once that session COMPLETES.

    Monday 16:25Z   rung fires while Monday is OPEN
                    newest daily bar = Friday
                    `== session` looks for Monday  ->  EMPTY  ->  hold, every rung, forever

THE OBVIOUS REPAIR COSTS A SECOND DAY, SILENTLY. Slicing to the newest session strictly before the
decision (`panel.date < due`) looks equivalent and is not: `build_feature_panel` sets
`asof_close = shift(1)` and `decide` reads prices from `asof_close` ALONE, so the PRIOR row carries
the close BEFORE prior. Two sessions stale, with nothing saying so.

A PLACEHOLDER ROW INVENTS NO PRICE. `close=NaN` for the session asserts "deciding for D, no bar for
D yet" — literally true — and `shift(1)` then makes `asof_close` the last COMPLETED close, which is
exactly what the engine was designed to receive.

AND THE COUPLING THAT MAKES THIS TWO CHANGES RATHER THAN ONE. Before this, a stale feed produced an
empty slice and a named hold — loud and safe. With a placeholder the slice ALWAYS finds rows, so
that signal is GONE. `max_stale_days` is therefore REQUIRED by this change, not an improvement
alongside it: without it the lane trades a loud uselessness for a silent wrongness, which is
strictly worse.
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

#: Friday 11th is the newest COMPLETED session; Monday 14th is the one being decided, still open.
COMPLETED = [pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09"),
             pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-11")]
MONDAY = pd.Timestamp("2026-09-14")
PRICES = {"SMH": [100.0, 140.0, 180.0, 220.0], "GLD": [400.0, 375.0, 350.0, 325.0]}


def _bar(symbol, close, ts):
    bt = BarType.from_str(f"{symbol}.XNAS-1-DAY-LAST-EXTERNAL")
    p = Price.from_str(f"{close:.2f}")
    return Bar(bar_type=bt, open=p, high=p, low=p, close=p,
               volume=Quantity.from_int(100), ts_event=ts, ts_init=ts)


class _Host:
    """The REAL adapter functions on a host that can hold a log — `Actor.log` is read-only."""

    on_historical_data = SmhGldSleeveStrategy.on_historical_data
    _ingest = SmhGldSleeveStrategy._ingest
    panel = SmhGldSleeveStrategy.panel
    _day_for = SmhGldSleeveStrategy._day_for
    _is_rebalance = SmhGldSleeveStrategy._is_rebalance
    _with_a_row_for = SmhGldSleeveStrategy._with_a_row_for
    _refuse_a_stale_feed = SmhGldSleeveStrategy._refuse_a_stale_feed

    def __init__(self, sessions=COMPLETED, prices=None):
        self.id = "SMHGLD-007"
        self._cfg = SmhGldSleeveConfig()
        self._need = 3
        self._bars = defaultdict(lambda: deque(maxlen=64))
        self.missed_sessions: list[pd.Timestamp] = []
        self.said: list[tuple[str, str]] = []
        self.log = SimpleNamespace(
            info=lambda m: self.said.append(("info", m)),
            warning=lambda m: self.said.append(("warning", m)),
            error=lambda m: self.said.append(("error", m)))
        for leg, series in (prices or PRICES).items():
            # zip() truncates to the shorter side, so a fixture adding a session without adding a
            # price would silently NOT serve that bar — which is the very state under test here.
            assert len(series) == len(sessions), "fixture: one price per session per leg"
            for ts, close in zip(sessions, series):
                self.on_historical_data(_bar(leg, close, int(ts.value)))


# -- the Monday-intraday case nobody could test live ----------------------------------------------

def test_WITHOUT_a_session_row_the_lane_holds_every_rung_forever():
    """The defect, stated as the state it produced. Kept as the control for everything below."""
    lane = _Host()
    raw = lane.panel()

    assert MONDAY not in set(pd.to_datetime(raw["date"])), "the fixture is not the intraday case"
    assert lane._day_for(MONDAY, raw) is None, "the raw panel somehow yielded a decision"


def test_the_session_row_lets_a_decision_happen_WHILE_the_session_is_open():
    lane = _Host()
    panel = lane._with_a_row_for(MONDAY, lane.panel())

    assert lane._is_rebalance(MONDAY, panel), (
        "`_is_rebalance` reads the panel's dates and runs BEFORE the slice — without the row the "
        "lane returns before it ever reaches a decision")
    day = lane._day_for(MONDAY, panel)
    assert day is not None and list(pd.Series(day["date"]).unique()) == [MONDAY]


def test_asof_close_is_the_LAST_COMPLETED_CLOSE_and_not_the_one_before_it():
    """The whole reason for a placeholder rather than a different slice. Slicing to the prior ROW
    would give the close BEFORE Friday — two sessions stale, silently."""
    lane = _Host()
    day = lane._day_for(MONDAY, lane._with_a_row_for(MONDAY, lane.panel()))

    got = {r["ticker"]: r["asof_close"] for _, r in day.iterrows()}
    assert got["SMH"] == 220.0, f"SMH asof_close is {got['SMH']}, expected Friday's close 220.0"
    assert got["GLD"] == 325.0, f"GLD asof_close is {got['GLD']}, expected Friday's close 325.0"

    # And the trap made explicit: the PRIOR row carries the close before Friday.
    featurized = build_feature_panel(lane.panel(), lane._cfg)
    prior = featurized.loc[featurized["date"] == COMPLETED[-1]]
    assert float(prior.loc[prior["ticker"] == "SMH", "asof_close"].iloc[0]) == 180.0, (
        "the prior-row slice does not carry a staler price, so this test proves nothing")


def test_the_placeholder_INVENTS_NO_PRICE_IN_ANY_FIELD():
    """EVERY price field, not just `close`.

    `cfg.price_field` is configurable — `build_feature_panel` reads `grouped[cfg.price_field]` and
    the config validates against `_BAR_FIELDS`. So a placeholder carrying a plausible `open` is not
    a harmless filler: on a lane configured with `price_field='open'` it becomes a REAL INVENTED
    PRICE that `asof_close` would carry into a live decision. Asserting only `close` leaves that
    reachable, and a bite setting open/high to 1.0 survived until this named every field.
    """
    lane = _Host()
    panel = lane._with_a_row_for(MONDAY, lane.panel())
    added = panel.loc[pd.to_datetime(panel["date"]) == MONDAY]

    assert len(added) == 2
    for field in ("open", "high", "low", "close"):
        assert added[field].isna().all(), (
            f"the placeholder carried a made-up {field}; with price_field={field!r} that would "
            f"become the price a live decision is taken on")


def test_every_configurable_PRICE_FIELD_yields_the_last_completed_close():
    """The property the test above protects, exercised rather than argued: whichever field the lane
    is configured to price on, the placeholder must leave `asof_close` pointing at the last
    COMPLETED session's value of that field."""
    for field in ("open", "high", "low", "close"):
        lane = _Host()
        lane._cfg = SmhGldSleeveConfig(price_field=field)
        day = lane._day_for(MONDAY, lane._with_a_row_for(MONDAY, lane.panel()))
        assert day is not None, f"price_field={field!r} produced no decision row"
        got = {r["ticker"]: r["asof_close"] for _, r in day.iterrows()}
        assert got["SMH"] == 220.0 and got["GLD"] == 325.0, (
            f"price_field={field!r} gave {got}, not Friday's completed values")


def test_a_session_the_venue_HAS_served_is_not_duplicated():
    lane = _Host(sessions=COMPLETED + [MONDAY],
                 prices={"SMH": PRICES["SMH"] + [260.0], "GLD": PRICES["GLD"] + [300.0]})
    panel = lane._with_a_row_for(MONDAY, lane.panel())

    assert len(panel.loc[pd.to_datetime(panel["date"]) == MONDAY]) == 2, (
        "a real session row was duplicated by a placeholder")
    assert not panel.loc[pd.to_datetime(panel["date"]) == MONDAY, "close"].isna().any()


def test_the_decision_actually_RUNS_end_to_end_on_the_open_session():
    lane = _Host()
    day = lane._day_for(MONDAY, lane._with_a_row_for(MONDAY, lane.panel()))
    assert decide(day, lane._cfg, set()).regime == "opening"


# -- and the guard that replaces the signal the placeholder removed -------------------------------

def test_a_STALE_FEED_is_refused_because_the_empty_slice_no_longer_warns():
    """Before #207 a stale feed produced an empty slice and a named hold. The placeholder means the
    slice always finds rows, so without this the lane would decide on a week-old close in silence."""
    lane = _Host()
    far = MONDAY + pd.Timedelta(days=30)

    assert lane._refuse_a_stale_feed(far, lane.panel()) is True
    assert any(lvl == "error" and "max_stale_days" in m for lvl, m in lane.said), lane.said
    assert lane.missed_sessions == [far], "the refused session was not counted"


def test_a_WEEKEND_GAP_is_not_stale():
    """Friday to Monday is three calendar days and is the normal case. A guard that fired on it
    would be disabled within a week, and then absent for the real outage."""
    lane = _Host()
    assert lane._refuse_a_stale_feed(MONDAY, lane.panel()) is False
    assert lane.missed_sessions == []


def test_the_staleness_guard_reads_REAL_BARS_and_not_its_own_placeholder():
    """A guard that counted the invented row would measure a gap of zero, always, and report health
    it manufactured — the circularity rule, applied to a freshness check."""
    lane = _Host()
    far = MONDAY + pd.Timedelta(days=30)
    with_row = lane._with_a_row_for(far, lane.panel())

    assert lane._refuse_a_stale_feed(far, with_row) is False, (
        "reading a panel that already contains the placeholder makes the gap vanish — this is why "
        "the guard must run BEFORE the row is added, asserted here so the ORDER cannot drift")


def test_the_guard_runs_BEFORE_the_placeholder_is_added():
    """The ordering the test above makes dangerous. Bound to the source so it cannot be rearranged."""
    import inspect

    src = inspect.getsource(SmhGldSleeveStrategy._on_session_alert)
    assert src.index("_refuse_a_stale_feed") < src.index("_with_a_row_for"), (
        "the staleness guard would read a panel containing its own placeholder and never fire")


def test_an_empty_panel_is_not_reported_as_stale():
    """Nothing to be stale ABOUT, and the warmth gate already refuses. Reporting it here would fire
    a data-outage alarm on every cold boot."""
    lane = _Host(sessions=[], prices={"SMH": [], "GLD": []})
    assert lane._refuse_a_stale_feed(MONDAY, lane.panel()) is False
    assert lane.missed_sessions == []
