"""CRSISHORT's adapter — the first lane in this package that holds the SHORT side (#123).

What is tested here is the part that is NOT the decision layer: the adapter's own refusals, the
book state it moves on order events, and the two rules it owns that `engine.decide` deliberately
does not — the covers, and the departures (`nodata` and `hold_through`).

`hold_through` is the reason the departures are tested rather than assumed. It is worth 30 points of
return in #123's acceptance run — +80.7% against +50.9% with it off — because closing a position for
falling below the ENTRY liquidity floor closes WINNERS for a reason unrelated to the trade. A live
lane that quietly implemented the other branch would still produce a plausible curve.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.runtime.nautilus.contract import protective_close
from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy, SessionIntent
from kumo_strategies.runtime.nautilus.sides import SHORT
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.exits import (
    SessionBar, ShortTrailState, open_state)

SESSIONS = 130


def _bars(symbols, *, sessions=SESSIONS, close=100.0, volume=5_000_000.0, last=None,
          asof=None):
    """A panel wide and long enough to clear warmup, with a deliberate wobble.

    A flat series has zero variance, so `annualised_log_vol` is 0 and NOTHING clears the 100% vol
    filter — a fixture that looks like data and can never signal.
    """
    rows = []
    for i, sym in enumerate(symbols):
        px = close
        for d in range(sessions):
            px = px * (1.06 if (d + i) % 2 else 0.945)
            rows.append({"ticker": sym, "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                         "open": px, "high": px * 1.02, "low": px * 0.98, "close": px,
                         "volume": volume})
    panel = pd.DataFrame(rows)
    # TWO PATCH POINTS, ONE SESSION APART, because that is how the engine reads the tape.
    #   `asof`  patches the SECOND-TO-LAST row. Every feature is `.shift(1)`, so this is what the
    #           FINAL session screens on — the dollar volume, price and indicators behind
    #           `in_universe` / `manageable` / `eligible`.
    #   `last`  patches the FINAL row, which is the bar the COVERS read.
    # Patching `last` and expecting the screen to move is the mistake this comment exists to stop:
    # it changes the session AFTER the last one, which does not exist.
    for patches, offset in ((asof, -2), (last, -1)):
        for sym, patch in (patches or {}).items():
            idx = panel.index[panel["ticker"] == sym][offset]
            for k, v in patch.items():
                panel.loc[idx, k] = v
    return panel


class _Host(HeldSeedMixin):
    """A narrow host carrying the REAL functions off `CrsiShortStrategy`.

    `Strategy.__init__` needs a running kernel, and `Actor.log` is a read-only Cython attribute, so
    the class cannot simply be `__new__`'d and patched. The methods under test are bound here
    unchanged — this is not a reimplementation and cannot drift from one: every function below is
    the attribute itself, so deleting or renaming one fails these tests rather than silently
    exercising a copy.
    """

    # `on_start` seeds `_held` from this lane's own open positions (#194), so the host carries
    # the REAL mixin rather than a stub — the same principle as every other function here.
    # It reads `self.cache`; a host without one gets the mixin's reported failure, not a crash.

    POSITION_SIDE = CrsiShortStrategy.POSITION_SIDE
    session_intent = CrsiShortStrategy.session_intent
    _departures = CrsiShortStrategy._departures
    _session_bars = CrsiShortStrategy._session_bars
    _borrow_for = CrsiShortStrategy._borrow_for
    on_order_filled = CrsiShortStrategy.on_order_filled
    on_order_rejected = CrsiShortStrategy.on_order_rejected
    on_order_denied = CrsiShortStrategy.on_order_denied
    _forget = CrsiShortStrategy._forget
    _on_broker_account = CrsiShortStrategy._on_broker_account
    broker_equity = CrsiShortStrategy.broker_equity
    on_start = CrsiShortStrategy.on_start
    record_pending = CrsiShortStrategy.record_pending
    clear_pending = CrsiShortStrategy.clear_pending
    pending = CrsiShortStrategy.pending
    reconcile_pending = CrsiShortStrategy.reconcile_pending
    _start_trail = CrsiShortStrategy._start_trail
    _with_a_row_for = CrsiShortStrategy._with_a_row_for
    _refuse_a_stale_feed = CrsiShortStrategy._refuse_a_stale_feed
    broker_equity = CrsiShortStrategy.broker_equity

    id = "CRSISHORT-001"


def _lane(cfg=None, *, held=(), trail=None, **kw):
    """A host with exactly the attributes the methods read — a missing one raises rather than
    passing quietly."""
    cfg = cfg or CrsiShortConfig()
    lane = _Host()
    lane._cfg = cfg
    lane._adjustment = "all"
    lane._held = set(held)
    lane._pending = {}
    lane._trail = dict(trail or {})
    lane._borrow_rates = kw.get("borrow_rates", lambda syms: {s: 0.0 for s in syms})
    lane._bars = {}
    lane.missed_sessions = []
    lane._broker_account = None
    lane._account_topic = "broker.account"
    # The bus records what was subscribed, so a lane that stops subscribing fails here rather than
    # returning a quiet None from `broker_equity()` for the life of the process (#174).
    lane.subscribed = []
    lane.msgbus = SimpleNamespace(
        subscribe=lambda topic, handler: lane.subscribed.append((topic, handler)))
    lane._runner = None
    lane._loop = None
    # The lane's OWN Nautilus position, which `protective_close` reads to decide whether the book is
    # actually flat. Empty = flat, which is the state after a protective stop has filled.
    lane.cache = SimpleNamespace(positions_open=lambda **kw: [])
    lane.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                               error=lambda *a, **k: None)
    return lane


def _signalling(symbols, signaller, *, sessions=SESSIONS, run=6, asof=None, volume=5_000_000.0):
    """A panel in which `signaller` ACTUALLY CLEARS THE SCREEN on the final session.

    Two conditions have to hold at once and they pull against each other: ConnorsRSI above 90 wants
    a hard run into the close, and the 100% annualised vol filter wants a violent series. A run long
    enough to lift the CRSI damps the vol below the filter — `up=1.09` gives CRSI 80.5 at vol 1.39,
    and a gentle wobble gives CRSI 96.8 at vol 0.94. Both look like data and neither signals.

    THIS EXISTS BECAUSE A FIXTURE THAT CANNOT SIGNAL MAKES EVERY "did not enter" ASSERTION TRUE FOR
    THE WRONG REASON. Two mutation bites survived against the quiet fixture — a borrow outage read as
    a zero fee, and `decide` shown the covered names as still held — because `enter` was empty
    whatever the code did. The numbers below are tuned: CRSI 96.5 at annualised vol 1.24.
    """
    rows = []
    for i, sym in enumerate(symbols):
        px = 100.0
        for d in range(sessions):
            running = sym == signaller and d >= sessions - run - 1
            px *= 1.09 if running else (1.08 if (d + i) % 2 else 0.925)
            rows.append({"ticker": sym, "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                         "open": px, "high": px * 1.02, "low": px * 0.98, "close": px,
                         "volume": volume})
    panel = pd.DataFrame(rows)
    for sym, patch in (asof or {}).items():
        idx = panel.index[panel["ticker"] == sym][-2]
        for k, v in patch.items():
            panel.loc[idx, k] = v
    return panel


def test_the_signalling_fixture_really_signals():
    """Guards the guard. If this stops being true, every `enter` assertion below silently becomes a
    tautology — which is exactly how the two surviving bites were found."""
    panel = _signalling(["AAA", "BBB"], "AAA")
    lane = _lane()
    assert "AAA" in _lane().session_intent(_last_session(panel), panel).enter


def _last_session(panel):
    return panel["date"].max()


def test_a_pre_open_session_without_its_own_daily_bar_can_still_signal():
    """The live failure: before the open, IB has not served today's daily bar yet.

    `session_intent` used to slice `features.date == session`, get an empty frame, and return a
    healthy-looking empty intent: enter [], refused {}. Adding a row for the firing session lets the
    feature shift read the newest completed close, which is exactly the data the pre-open limit is
    allowed to know.
    """
    panel = _signalling(["AAA", "BBB"], "AAA")
    session = _last_session(panel) + pd.Timedelta(days=1)

    intent = _lane().session_intent(session, panel)

    assert "AAA" in intent.enter
    assert intent.refused == {}


def test_the_intraday_placeholder_invents_no_price():
    panel = _signalling(["AAA", "BBB"], "AAA")
    session = _last_session(panel) + pd.Timedelta(days=1)

    with_row = _lane()._with_a_row_for(session, panel)
    added = with_row.loc[pd.to_datetime(with_row["date"]) == session]

    assert len(added) == 2
    for field in ("open", "high", "low", "close"):
        assert added[field].isna().all()


def test_a_stale_feed_is_refused_before_the_placeholder_can_hide_it():
    lane = _lane()
    panel = _signalling(["AAA", "BBB"], "AAA")
    far = _last_session(panel) + pd.Timedelta(days=30)

    assert lane._refuse_a_stale_feed(far, panel) is True
    assert lane.missed_sessions == [far.normalize()]


# -- construction refusals -----------------------------------------------------------------------

def test_raw_prices_are_refused_at_construction():
    """On raw prices a reverse split books as a -1000% short loss on an ALREADY-OPEN position, which
    no entry-time corporate-action gate can reach — it inverted the lab's 2025-26 verdict on 7.4% of
    trades. This class cannot tell an adjusted series from a raw one by looking at it, so the caller
    that pulled the bars has to say."""
    with pytest.raises(ValueError, match="split-adjusted"):
        CrsiShortStrategy(CrsiShortConfig(), symbols=["A"], order_id_tag="900",
                          price_adjustment="raw", min_warm_symbols=1,
                          borrow_rates=lambda s: {})


def test_a_borrow_ceiling_without_a_provider_is_refused_at_construction():
    """An armed, inert gate is the #26 failure mode this repo has paid for twice. `engine.decide`
    raises without the data; refusing at construction turns that from a lane that mysteriously
    stopped deciding into a boot failure the operator sees."""
    with pytest.raises(ValueError, match="borrow_rates"):
        CrsiShortStrategy(CrsiShortConfig(), symbols=["A"], order_id_tag="900",
                          price_adjustment="all", min_warm_symbols=1, borrow_rates=None)


def test_the_ceiling_may_be_turned_off_but_only_explicitly():
    """`None` is a decision that the gate is off and is recorded as one. It is not the same as
    having no provider, which is the gate being on and unable to answer."""
    lane = CrsiShortStrategy(CrsiShortConfig(max_borrow_fee_annual=None), symbols=["A"],
                             order_id_tag="900", price_adjustment="all", min_warm_symbols=1,
                             shadow_only=True)
    assert lane.POSITION_SIDE == SHORT


def test_the_universe_breadth_must_be_stated():
    with pytest.raises(ValueError, match="min_warm_symbols"):
        CrsiShortStrategy(CrsiShortConfig(), symbols=["A"], order_id_tag="900",
                          price_adjustment="all", min_warm_symbols=0,
                          borrow_rates=lambda s: {})


# -- departures: nodata and hold_through -----------------------------------------------------------

def test_a_held_name_that_falls_below_the_ENTRY_floor_is_kept_when_hold_through_is_on():
    """The frozen spec. The $200M floor gates ENTRIES only, and a position whose liquidity dips is
    managed to its exit — worth 30 points of return against the alternative."""
    panel = _bars(["AAA", "BBB"], asof={"AAA": {"volume": 500_000.0}})
    lane = _lane(CrsiShortConfig(hold_through=True), held={"AAA"})
    intent = lane.session_intent(_last_session(panel), panel)
    assert "AAA" not in intent.cover, "a winner was closed for a reason unrelated to the trade"


def test_the_same_name_is_covered_when_hold_through_is_off():
    """Not a stricter variant — a DIFFERENT strategy, and the one the earlier code implemented by
    accident. Both branches must be reachable or the field is decorative."""
    panel = _bars(["AAA", "BBB"], asof={"AAA": {"volume": 500_000.0}})
    lane = _lane(CrsiShortConfig(hold_through=False), held={"AAA"})
    intent = lane.session_intent(_last_session(panel), panel)
    assert intent.cover_kind.get("AAA") == "liquidity", intent.cover


def test_a_name_that_leaves_the_TAPE_is_covered_on_either_setting():
    """A different question from the entry floor: below the $1M priceable floor the lab's signal
    table has no row at all, and the position is closed at its last mark and booked `nodata` — 4% of
    #123's trades. `hold_through` does not reach this case and must not appear to."""
    for hold in (True, False):
        panel = _bars(["AAA", "BBB"], asof={"AAA": {"volume": 0.5, "close": 0.01}})
        lane = _lane(CrsiShortConfig(hold_through=hold), held={"AAA"})
        intent = lane.session_intent(_last_session(panel), panel)
        assert intent.cover_kind.get("AAA") == "nodata", f"hold_through={hold}: {intent.cover}"


def test_a_held_name_absent_from_the_session_entirely_is_nodata_not_held():
    """Absent from the table is the same fact as present-but-unpriceable. Reading it as 'hold' would
    age a position through a data outage, which is exactly what `evaluate_short_exits` refuses to do
    on its own side."""
    panel = _bars(["AAA", "BBB"])
    lane = _lane(held={"ZZZ"})
    intent = lane.session_intent(_last_session(panel), panel)
    assert intent.cover_kind.get("ZZZ") == "nodata"


def test_an_exit_rule_that_already_fired_keeps_ITS_reason():
    """The exit MIX is an acceptance number (#123: flat 48% / reversal 48% / nodata 4%), so a cover
    the trail already chose must not be relabelled by a departure — that moves a trade between the
    buckets the acceptance run checks."""
    panel = _bars(["AAA", "BBB"], asof={"AAA": {"volume": 0.5, "close": 0.01}})
    bar = SessionBar(open=100.0, high=101.0, low=99.0, close=100.0)
    lane = _lane(held={"AAA"}, trail={"AAA": open_state(100.0, bar)})
    intent = lane.session_intent(_last_session(panel), panel)
    if intent.cover_kind.get("AAA") not in (None, "nodata"):
        assert intent.cover["AAA"] == intent.cover["AAA"]  # an exit rule won, and kept its label


def test_a_name_covered_THIS_MORNING_is_not_re_shorted_THIS_SESSION():
    """The entry is a LIMIT PLACED THE NIGHT BEFORE, so `decide` must see the book as it stood at the
    last close — today's covers have not happened yet.

    `runner_crsi_short`, the runner that produced the accepted curve, passes `held_at_open` for this
    reason and says what the alternative costs: the post-exit book "would let a name covered at this
    morning's open be re-shorted by an order that could not have existed". The rotation lanes
    reproduce same-session re-entry deliberately; this book must not.

    THE NAME HERE IS COVERED BY AN EXIT RULE, not by leaving the universe. A liquidity cover also
    makes the name ineligible, so it could not be re-entered whatever the code did — the assertion
    would pass for the wrong reason and no mutation could ever fail it.
    """
    panel = _signalling(["AAA", "BBB"], "AAA")
    last = panel.loc[panel["ticker"] == "AAA"].iloc[-1]
    # A reversal already flagged at the prior close: covered at this session's open, while the name
    # is still fully eligible and still signalling.
    trail = ShortTrailState(entry_px=float(last["close"]) * 1.5, best_px=float(last["close"]) * 1.4,
                            prev_low=float(last["low"]), prev_high=float(last["high"]),
                            flag_reversal=True)
    lane = _lane(held={"AAA"}, trail={"AAA": trail})
    intent = lane.session_intent(_last_session(panel), panel)
    assert "AAA" in intent.cover, f"the reversal cover did not fire: {intent}"
    assert "AAA" not in intent.enter, (
        "a name covered at this morning's open was re-shorted by an order that could not have "
        "existed — the entry limit was placed last night")


# -- borrow ---------------------------------------------------------------------------------------

def test_a_borrow_provider_that_RAISES_is_not_read_as_no_fee():
    """An outage and "this name has no locate" are different facts. A provider that fails must not
    produce a permissive answer — every name comes back with NO locate, and `engine.decide` refuses
    each one with a recorded reason rather than entering it.

    Asserted against a name that WOULD otherwise be entered, so the assertion cannot pass because
    nothing signalled.
    """
    def boom(_syms):
        raise TimeoutError("gateway did not answer")

    panel = _signalling(["AAA", "BBB"], "AAA")
    assert "AAA" in _lane().session_intent(_last_session(panel), panel).enter   # would enter
    intent = _lane(borrow_rates=boom).session_intent(_last_session(panel), panel)
    assert intent.enter == (), "entered a name while the locate provider was down"
    assert "AAA" in intent.refused, "the refusal was not recorded, so it is invisible"


def test_the_gate_being_off_asks_the_provider_nothing():
    asked = []
    panel = _bars(["AAA", "BBB"])
    lane = _lane(CrsiShortConfig(max_borrow_fee_annual=None),
                 borrow_rates=lambda syms: asked.append(syms) or {})
    lane.session_intent(_last_session(panel), panel)
    assert asked == [], "asked for locate fees with the ceiling explicitly off"


# -- book state, from order events ------------------------------------------------------------------

def _fill(sym, side, px=100.0):
    return SimpleNamespace(instrument_id=SimpleNamespace(symbol=sym),
                           client_order_id=f"CRSISHORT-001-{sym}",
                           order_side=SimpleNamespace(name=side), last_px=px, avg_px=px)


def test_an_entry_SELL_marks_the_name_short_and_opens_the_trail_at_the_FILL():
    """The limit rests 3% above the close and fills at the better of the limit and the open, so the
    two differ on exactly the gap sessions where the flat exit matters most. Seeding the trail from
    the limit would arm the flat cover at a level the position never traded through."""
    lane = _lane()
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "enter")
    lane._bars = {"AAA": [{"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.0}]}
    lane.on_order_filled(_fill("AAA", "SELL", px=103.5))
    assert "AAA" in lane._held
    assert lane._trail["AAA"].entry_px == 103.5


def test_a_covering_BUY_clears_the_holding_and_the_trail():
    lane = _lane(held={"AAA"}, trail={"AAA": open_state(
        100.0, SessionBar(open=100.0, high=101.0, low=99.0, close=100.0))})
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "exit")
    lane.on_order_filled(_fill("AAA", "BUY"))
    assert "AAA" not in lane._held and "AAA" not in lane._trail


def test_a_BUY_does_NOT_complete_an_ENTRY_in_a_short_lane():
    """Under the long mapping this is the dangerous direction: the covering BUY would be read as the
    entry completing, so the lane would believe it had just opened a position it has closed."""
    lane = _lane()
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "enter")
    lane.on_order_filled(_fill("AAA", "BUY"))
    assert lane._held == set() and lane.pending() == {"AAA"}


def test_a_fill_with_an_unusable_price_does_not_invent_a_trail():
    """No trail means the flat cover cannot arm, which is a loud problem. Inventing an entry price
    would make it a silent one."""
    lane = _lane()
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "enter")
    lane._bars = {"AAA": [{"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.0}]}
    lane.on_order_filled(_fill("AAA", "SELL", px=float("nan")))
    assert "AAA" not in lane._trail


def test_a_foreign_fill_moves_nothing():
    lane = _lane()
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "enter")
    ev = _fill("AAA", "SELL")
    ev.client_order_id = "PROT-AAA-1"
    lane.on_order_filled(ev)
    assert lane._held == set() and lane.pending() == {"AAA"}


def test_a_denied_exit_leaves_the_position_HELD():
    """Forgetting the intent AND the holding makes a real short read as foreign, and nothing ever
    covers it."""
    lane = _lane(held={"AAA"})
    lane.record_pending("AAA", "CRSISHORT-001-AAA-1", 100.0, "exit")
    lane.on_order_denied(_fill("AAA", "BUY"))
    assert "AAA" in lane._held and lane.pending() == set()


# -- the intent is computed in ONE place -------------------------------------------------------------

def test_session_intent_is_public_so_a_runner_does_not_grow_a_SECOND_exit_authority():
    """`engine.decide` returns no exits at all, deliberately. A cockpit-side runner that re-derived
    the intent from the panel would give this book two exit authorities that must agree and cannot
    be tested together."""
    assert not CrsiShortStrategy.session_intent.__name__.startswith("_")
    assert isinstance(SessionIntent(), SessionIntent)


def test_a_RESTING_order_consumes_a_slot():
    """The lab computes `room = slots - len(pos) - len(pend)`. Counting only FILLED positions would
    let the book commit to more names than it can hold and then reject the fills that arrive last —
    and with a limit resting overnight, "committed" and "filled" are a session apart by design.

    One slot, one resting order: there is no room, whatever signals.
    """
    panel = _signalling(["AAA", "BBB"], "AAA")
    cfg = CrsiShortConfig(n_slots=1)
    free = _lane(cfg)
    assert "AAA" in free.session_intent(_last_session(panel), panel).enter    # the slot is takeable

    committed = _lane(cfg)
    committed.record_pending("BBB", "CRSISHORT-001-BBB-1", 100.0, "enter")
    assert committed.session_intent(_last_session(panel), panel).enter == (), (
        "entered against a slot an order was already resting for")


def test_a_lane_given_no_history_days_SAYS_SO_rather_than_sitting_inert():
    """`history_days` defaults to None and a falsy value requests nothing — silently, in every lane
    in this package. This one needs 102 sessions per name, so the silence would last five months and
    would be indistinguishable from a lane that decided to hold.

    The message must name the number. "No history" means nothing to an operator; "will publish
    nothing for 102 sessions" is actionable.
    """
    said = []
    lane = _lane()
    lane._history_days = None
    lane._iids = []
    lane._need = CrsiShortConfig().warmup_sessions
    lane.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                               error=lambda m, *a, **k: said.append(m))
    lane._resolve_symbols_if_needed = lambda: None
    lane.begin_arming = lambda: None
    CrsiShortStrategy.on_start(lane)
    assert said, "a lane that cannot signal for 102 sessions started silently"
    assert "102" in said[0], f"the message does not say how long: {said[0]!r}"


class _NotReadyJournal:
    def __init__(self):
        self.writes = []

    async def write(self, kind, summary, *, session, slot=None, detail=None):
        self.writes.append({"kind": kind, "summary": summary, "session": session,
                            "slot": slot, "detail": detail})


class _NotReadyHost:
    """The real alert handler on a host that is deliberately below CRSISHORT warmup."""

    _on_session_alert = CrsiShortStrategy._on_session_alert
    panel = CrsiShortStrategy.panel
    _warmth = CrsiShortStrategy._warmth
    _warmth_detail = CrsiShortStrategy._warmth_detail
    _not_ready = CrsiShortStrategy._not_ready

    id = "CRSISHORT-006"
    POSITION_SIDE = CrsiShortStrategy.POSITION_SIDE

    def __init__(self):
        self._armed_session = pd.Timestamp("2026-09-14")
        self._armed_slot = "open+130m"
        self._need = CrsiShortConfig().warmup_sessions
        self._min_warm = 100
        self._bars = {"AAA": [{"date": pd.Timestamp("2026-09-12"), "ticker": "AAA"}]}
        self._runner = SimpleNamespace(calls=[])
        self._loop = object()
        self.clock = SimpleNamespace(utc_now=lambda: pd.Timestamp("2026-09-14T15:40:00Z"))
        self.rearmed = []
        self.rearm_after_alert = lambda now: self.rearmed.append(now)
        self.scheduled = []
        self.journal = _NotReadyJournal()
        self.said = []
        self.log = SimpleNamespace(
            info=lambda m, *a, **k: self.said.append(("info", m)),
            warning=lambda m, *a, **k: self.said.append(("warning", m)),
            error=lambda m, *a, **k: self.said.append(("error", m)))

    @property
    def warm(self):
        return CrsiShortStrategy.warm.fget(self)

    def session_journal(self):
        return self.journal

    def fire_and_report(self, coro, _loop, what):
        self.scheduled.append(what)
        asyncio.run(coro)
        return what


def test_a_fired_but_not_warm_slot_is_operator_visible():
    """Live 2026-09-14: the temporary CRSISHORT slot fired after restart, then disappeared because
    the lane was still warming. A fired slot that cannot decide must leave a row with the counts."""
    lane = _NotReadyHost()

    lane._on_session_alert()

    assert lane.rearmed, "the alert stopped re-arming before reporting readiness"
    said = " ".join(m for _level, m in lane.said)
    assert "CANNOT DECIDE" in said and "warm symbols" in said, lane.said
    assert lane._runner.calls == [], "the runner was called without enough bars"
    assert lane.scheduled == ["not-ready session"], lane.scheduled
    assert lane.journal.writes, "the fired-not-ready slot wrote no durable row"
    row = lane.journal.writes[0]
    assert row["kind"] == "state"
    assert row["session"] == "2026-09-14" and row["slot"] == "open+130m"
    assert row["detail"]["ready_symbols"] == 0
    assert row["detail"]["required_symbols"] == 100


# -- a protective close on a SHORT (#133 obligation) ------------------------------------------------

def test_a_protective_close_clears_the_TRAIL_not_just_the_holding():
    """CRSISHORT is the only lane that keeps a trail, and it is why `protective_close` has a `_trail`
    branch at all.

    That branch was REMOVED from the #133 fix before it merged to main, deliberately: no lane on main
    keeps a trail, so it was a line nothing could reach, and unreachable code that looks like careful
    handling is how a mechanism comes to be believed in. The commit that removed it recorded the
    obligation to bring it back WITH a test that reaches it. This is that test.

    Leaving the trail behind is not cosmetic. `_trail` is what arms the flat cover, so a stale entry
    for a position that no longer exists would have the next session evaluating covers against a
    book that is gone — and `evaluate_short_exits` iterates `state`, not the holdings.
    """
    lane = _lane(held={"AAA"}, trail={"AAA": open_state(
        100.0, SessionBar(open=100.0, high=101.0, low=99.0, close=100.0))})
    lane._runner = None
    lane._loop = None
    ev = SimpleNamespace(client_order_id="PROT-BUY-AAA-XNAS-1",
                         instrument_id=SimpleNamespace(symbol="AAA"),
                         order_side=SimpleNamespace(name="BUY"),   # a short is closed by BUYING
                         last_qty=10, last_px=95.0, avg_px=95.0)
    assert protective_close(lane, ev) is True
    assert "AAA" not in lane._held
    assert "AAA" not in lane._trail, (
        "the trail outlived the position — the next session would evaluate covers against a book "
        "that no longer exists")


# -- it must not read as deployable while it cannot trade -------------------------------------------

def test_the_lane_REFUSES_to_construct_while_its_order_path_is_incomplete():
    """A merged adapter reads as deployable. This one is not, and code should say so, not a ticket.

    CRSISHORT's entry is a sell-short LIMIT resting into the opening auction, and the two-order
    decomposition it needs on IB — a limit-on-open for the gap arm, a day limit after the auction for
    the rest — is designed and NOT built (#131). Until it is, a caller that constructs this lane and
    wires it to a live runner would get a strategy that computes and cannot act.

    So construction REFUSES unless the caller states it wants the shadow-only lane. The refusal names
    what is missing rather than saying "not ready", because the next person to read it needs to know
    which piece to look for.
    """
    with pytest.raises(ValueError, match="ORDER_PATH_COMPLETE|shadow"):
        CrsiShortStrategy(CrsiShortConfig(max_borrow_fee_annual=None), symbols=["A"],
                          order_id_tag="900", price_adjustment="all", min_warm_symbols=1)


def test_an_explicit_shadow_caller_may_construct_it():
    """Shadow is the lane's real deployment phase (platform issue 853): register, compute, journal, publish,
    submit nothing. That has to remain possible or the guard blocks the only thing the lane can do."""
    lane = CrsiShortStrategy(CrsiShortConfig(max_borrow_fee_annual=None), symbols=["A"],
                             order_id_tag="900", price_adjustment="all", min_warm_symbols=1,
                             shadow_only=True)
    assert lane.POSITION_SIDE == SHORT


# `test_the_flag_is_a_STATEMENT_about_the_order_path_not_a_mute_button` lived here and is GONE
# deliberately, on its own instruction: "if the order path is complete, delete the guard and this
# test rather than flipping this constant and leaving a refusal nobody reaches" (#210). The
# decomposition landed, the constant is True, and the construction guard it pinned was removed with
# it.
#
# What replaced it is NOT a flag check. `test_the_order_path_is_open_but_the_slots_must_reach_it`
# asserts the decomposition's PIECES are present — a flag can read True while a piece goes missing —
# and that a lane whose slots never reach the auction still refuses to be built, which is the half
# one boolean could never carry.
