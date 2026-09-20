"""SMHGLD declares a market view, and the view changes NOTHING it trades (#212).

2026-09-13: "no strategy can opt out. Those are emergency rules." The platform view is a
GOVERNANCE contract — cockpit must be able to read this lane's state and, in an emergency, act on
it — and `signal=NONE` answers `not_asked`, which makes the lane invisible to the poller.

The lab measured what a view costs this lane (research-lab `research/workshop/smhgld_market_view.py`,
platform view = the lane's own equal-weight SMH+GLD vs its 50-day MA, one-day shift, 2 bps):

    action on risk-off          2018-26 CAGR / Sharpe / maxDD    2022      2005-26
    none (before #212)          22.1% / 1.25 / -23%              -11.0%    14.7% / 0.93 / -32%
    EXIT_ONLY                   identical -- both legs always held, rebalance buys/sells continue
    LIQUIDATE (emergency)       14.2% / 1.04 / -18%              -4.2%      7.4% / 0.65 / -28%

EXIT_ONLY is free for THIS lane because it never opens a name it does not already hold; that is
pinned below by driving the engine under a risk-off view and asserting the orders are byte-identical.
LIQUIDATE stays the cockpit's emergency act, priced above, never the lane's decision.

Every test here was seen red on b575db1 before the fix, for the reason its docstring names.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from kumo_strategies.strategies.smhgld_sleeve import nautilus as adapter_module
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.strategies.market_view import (
    NO, UNKNOWN, YES, MarketAction, MarketSignal, MarketViewConfig)
from kumo_strategies.strategies.smhgld_sleeve import (
    SmhGldSleeveConfig, build_feature_panel, decide, live_config, order_plan)
from kumo_strategies.strategies.smhgld_sleeve.tests.test_smhgld_sleeve_engine import (
    bars, last_day, prices_from, shares_at)


# ------------------------------------------------------------------ the declaration ------------

def test_the_live_config_declares_the_LABS_MEASURED_view():
    """The numbers are the lab's, not TECHIVOL's: INDEX_VS_MA over the lane's own SMH+GLD index,
    50 sessions, EXIT_ONLY. `market_view.py` says a window is measured per lane, never inherited —
    this one was (the table in the module docstring)."""
    view = live_config().market_view
    assert view.signal is MarketSignal.INDEX_VS_MA
    assert view.window == 50
    assert view.action is MarketAction.EXIT_ONLY


def test_the_DEFAULT_config_and_the_live_config_declare_the_SAME_view():
    """`live_config()` returns `SmhGldSleeveConfig()` on purpose ("this lane has exactly one form"),
    so the declaration must live on the default, not be patched in by the live builder."""
    assert SmhGldSleeveConfig().market_view == live_config().market_view


# ------------------------------------------------------------------ the hooks, visible ---------

def _bar(symbol: str, close: float, ts: int) -> Bar:
    bt = BarType.from_str(f"{symbol}.XNAS-1-DAY-LAST-EXTERNAL")
    px = Price.from_str(f"{close:.2f}")
    return Bar(bar_type=bt, open=px, high=px, low=px, close=px, volume=Quantity.from_int(100),
               ts_event=ts, ts_init=ts)


class _Host(MarketAwareMixin):
    """The shape the poller meets: a lane with its config and whatever bars it has accumulated.

    BARS ARRIVE THROUGH THE ADAPTER'S OWN `_ingest`, bound here unchanged — so the row shape
    `_view_prices` pivots on is the one production writes, not one this test typed. A key rename in
    `_ingest` reaches this test the way it would reach the poller (coverage review, #212)."""

    EXTERNAL_ID = "SMHGLD"
    _ingest = adapter_module.SmhGldSleeveStrategy._ingest

    def __init__(self, cfg, sessions: int = 0, drift: float = 1.002):
        from collections import deque

        self._cfg = cfg
        self._bars = {sym: deque() for sym in ("SMH", "GLD")} if sessions else {}
        if sessions:
            frame = bars(sessions=sessions, drift=drift)
            for r in frame.itertuples():
                ts = int(pd.Timestamp(r.date, tz="UTC").value)
                self._ingest(_bar(r.ticker, float(r.close), ts))


def test_with_NO_bars_entries_blocked_is_UNKNOWN_and_NAMES_the_window():
    """On b575db1 this answered NO with "declares signal=NONE" — the opt-out. UNKNOWN is the honest
    answer for a declared view with no history yet, and it is VISIBLE: the poller reads a lane that
    is watching and cannot yet see, not a lane that declined to look."""
    verdict = _Host(live_config()).entries_blocked()
    assert verdict.state == UNKNOWN, verdict
    assert any("50" in r for r in verdict.reasons), verdict.reasons
    assert not any("NONE" in r for r in verdict.reasons), verdict.reasons


def test_with_a_FALLING_index_entries_blocked_is_YES_under_EXIT_ONLY():
    """The view computes from the lane's own bars: 60 sessions of both legs falling puts the
    equal-weight index under its 50-session average → YES, action=exit_only."""
    verdict = _Host(live_config(), sessions=60, drift=0.995).entries_blocked()
    assert verdict.state == YES, verdict
    assert any("exit_only" in r for r in verdict.reasons), verdict.reasons


def test_with_a_RISING_index_entries_blocked_is_NO():
    verdict = _Host(live_config(), sessions=60, drift=1.005).entries_blocked()
    assert verdict.state == NO, verdict


def _view_37():
    """A window the default cannot produce. Every other test here uses 50, which is BOTH the lab's
    number for this lane AND `MarketViewConfig`'s default AND TECHIVOL's — so `"50" in reason`
    cannot tell a config read from a literal. Agreement at the default is the shape that hid
    platform issue 1029 (coverage review, #212)."""
    return SmhGldSleeveConfig(market_view=MarketViewConfig(
        signal=MarketSignal.INDEX_VS_MA, window=37, action=MarketAction.EXIT_ONLY))


def test_the_hooks_read_the_WINDOW_from_the_config_not_a_literal():
    unknown = _Host(_view_37()).entries_blocked()
    assert unknown.state == UNKNOWN and any("37" in r for r in unknown.reasons), unknown

    yes = _Host(_view_37(), sessions=45, drift=0.995).entries_blocked()
    assert yes.state == YES, yes                 # 45 sessions clear a 37-window; they would not clear 50


def _real_lane(cfg=None):
    """THE ADAPTER ITSELF, not a host: `Strategy.__init__` runs without a kernel, so the deque the
    production `__init__` builds — with its `maxlen` — is the one the bars land in."""
    return adapter_module.SmhGldSleeveStrategy(
        cfg or live_config(), symbols=["SMH", "GLD"], order_id_tag="007", shadow_only=True)


def test_the_REAL_adapter_keeps_enough_bars_for_its_view_and_answers_from_them():
    """BLOCKER found by the coverage review, measured in the deployed package: `_bars` is
    `deque(maxlen=self._need + 5)` and `_need` is 3, so the lane retained 8 bars per leg against a
    50-session average that needs 51 — the declared view would have read UNKNOWN FOREVER, on any
    `history_days`, while every surface said "declared". The `_Host` above fills plain deques and
    could not represent the cap; this drives 60 real bars through the real `__init__`'s container."""
    lane = _real_lane()
    frame = bars(sessions=60, drift=0.995)
    for r in frame.itertuples():
        lane._ingest(_bar(r.ticker, float(r.close), int(pd.Timestamp(r.date, tz="UTC").value)))

    view = live_config().market_view
    assert lane._bars["SMH"].maxlen >= view.window + view.dwell, (
        f"the adapter retains {lane._bars['SMH'].maxlen} bars per leg; the declared "
        f"{view.window}-session view needs {view.window + view.dwell} — UNKNOWN forever")
    verdict = lane.entries_blocked()
    assert verdict.state == YES, verdict


def test_the_cap_is_DERIVED_from_the_view_not_a_literal():
    """A 37-window lane retains at least 38; a lane with no view keeps exactly what it kept
    before #212 (`_need + 5`), so the four owed lanes are byte-for-byte unchanged by the helper."""
    from kumo_strategies.runtime.nautilus.market_aware import bars_to_keep

    assert _real_lane(_view_37())._bars["SMH"].maxlen >= 38
    assert bars_to_keep(3, SmhGldSleeveConfig(market_view=MarketViewConfig())) == 8
    assert bars_to_keep(100, live_config()) == 105     # warmup already covers the window: unchanged


def test_emergency_exit_stays_NO_because_EXIT_ONLY_never_escalates():
    """A bear detector is not an emergency. The lane's declared action stops it OPENING and never
    asks for the book to be flattened; LIQUIDATE is the cockpit's act, priced in the docstring."""
    verdict = _Host(live_config(), sessions=60, drift=0.995).emergency_exit()
    assert verdict.state == NO, verdict
    assert any("exit_only" in r for r in verdict.reasons), verdict.reasons


# ------------------------------------------------------------------ zero behaviour change ------

def _risk_off_day(cfg: SmhGldSleeveConfig) -> pd.DataFrame:
    """A session whose index sits well under its 50-session average — the state in which a view
    with teeth would change what a lane does."""
    frame = bars(sessions=60, drift=0.995)
    return last_day(frame, cfg)


def test_FIXTURE_the_invariance_test_compares_a_DECLARED_view_against_NONE():
    """The precondition, on its own, so the invariance test below is red for exactly one reason.
    Red today = #212 itself (the live config declares NONE)."""
    without = SmhGldSleeveConfig(market_view=MarketViewConfig())
    assert without.market_view.signal is MarketSignal.NONE
    assert live_config().market_view.signal is not MarketSignal.NONE


@pytest.mark.parametrize("book", ["opening", "drifted", "on_target"])
def test_a_rebalance_session_places_IDENTICAL_orders_with_and_without_the_view_on_a_RISK_OFF_day(book):
    """THE PR'S CLAIM, DRIVEN: zero day-to-day behaviour change. The same day, the same book, one
    config with signal=NONE (b575db1) and one with the declared view — `decide` and `order_plan`
    must agree to the share. A future engine that reads the view to skip a rebalance, or to skip
    the BUY leg of one, turns this red; that is the sleeve becoming the flatten-SMH-only arm the
    lab priced at -3pp CAGR and never shipped."""
    without = SmhGldSleeveConfig(market_view=MarketViewConfig())
    with_view = live_config()
    day = _risk_off_day(with_view)
    prices = prices_from(day)
    held = set(with_view.universe)
    shares = {"opening": {}, "drifted": shares_at({"SMH": 0.62, "GLD": 0.38}, prices),
              "on_target": shares_at(with_view.target_weights, prices)}[book]
    if book == "opening":
        held = set()

    a = decide(day, without, held, shares)
    b = decide(day, with_view, held, shares)

    assert a == b, f"the view changed the decision on a risk-off day: {a} != {b}"
    plan = order_plan(a, shares, prices, 100_000.0)
    assert plan == order_plan(b, shares, prices, 100_000.0)
    if book in ("opening", "drifted"):
        # A fixture on which nothing trades proves the view changed nothing about nothing.
        assert a.regime != "hold" and plan, f"{book}: nothing traded — the invariance is vacuous"


def test_the_adapter_does_not_consult_its_OWN_view_on_the_decision_path():
    """The view is the PLATFORM's to act on (#873's poller), not the lane's. The adapter's session
    path must not read `entries_blocked` / `emergency_exit` / `blocks_entries` — a lane that
    gated its own rebalance on its view would be the -8pp arm nobody chose, arriving by accident."""
    src = textwrap.dedent(inspect.getsource(adapter_module.SmhGldSleeveStrategy))
    tree = ast.parse(src)
    hooks = {"entries_blocked", "emergency_exit", "blocks_entries", "liquidates", "market_state"}
    reads = sorted({
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in hooks})
    assert reads == [], f"the adapter reads its own view on the decision path: {reads}"
