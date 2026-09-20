"""`_pending` is READ everywhere and WRITTEN nowhere (#172, #131).

`CrsiShortStrategy._pending` is initialised empty at `crsi_short.py:197`, read at `:442` to size the
book (`decide(..., pending=...)`), and popped at `:539`, `:590`. **No line in the package ever puts
a key in it.** Every existing test that exercises the dict assigns it by hand in the fixture, so the
CONSUMER is covered and the PRODUCER does not exist — a green suite over a mechanism that cannot run.

That is the inert-code shape, and it has a second edge that is worse than a leaked slot.
`completes(intent=None, ...)` returns True (`order_provenance.py:107`, the fall-through): with the
dict permanently empty, EVERY fill reaches `on_order_filled`'s `else` branch and is recorded as an
ENTRY. A covering BUY that fills would add the symbol back to `_held` and open a fresh trail on it —
the lane covers a short and immediately believes it opened one. `completes`' own docstring names
this failure and fixes only the SIDE half of it; `intent=None` bypasses the mapping entirely.

These tests are RED before the seam lands, and each is named for the fact it asserts rather than for
the method it calls, because the method is what is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.exits import SessionBar, open_state


class _Host(HeldSeedMixin):
    """The real functions off `CrsiShortStrategy`, bound to a narrow host.

    Every entry is the attribute itself, so renaming or deleting one fails these tests rather than
    exercising a copy. The seam under test is listed here too — `getattr` on the class, so the file
    fails at COLLECTION with a clear name while the methods do not exist, which is the red this
    module is written to produce.
    """

    # `on_start` seeds `_held` from this lane's own open positions (#194), so the host carries
    # the REAL mixin rather than a stub — the same principle as every other function here.
    # It reads `self.cache`; a host without one gets the mixin's reported failure, not a crash.

    POSITION_SIDE = CrsiShortStrategy.POSITION_SIDE
    on_order_filled = CrsiShortStrategy.on_order_filled
    _start_trail = CrsiShortStrategy._start_trail
    _forget = CrsiShortStrategy._forget
    _on_broker_account = CrsiShortStrategy._on_broker_account
    clear_pending = CrsiShortStrategy.clear_pending
    record_pending = CrsiShortStrategy.record_pending
    pending = CrsiShortStrategy.pending
    reconcile_pending = CrsiShortStrategy.reconcile_pending
    on_start = CrsiShortStrategy.on_start
    session_intent = CrsiShortStrategy.session_intent
    _departures = CrsiShortStrategy._departures
    _session_bars = CrsiShortStrategy._session_bars
    _borrow_for = CrsiShortStrategy._borrow_for
    _with_a_row_for = CrsiShortStrategy._with_a_row_for

    id = "CRSISHORT-001"


def _lane(cfg=None, *, held=(), trail=None):
    lane = _Host()
    lane._cfg = cfg or CrsiShortConfig()
    lane._adjustment = "all"
    lane._held = set(held)
    lane._pending = {}
    lane._trail = dict(trail or {})
    lane._borrow_rates = lambda syms: {s: 0.0 for s in syms}
    lane._bars = {"AAA": [{"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.0}]}
    lane._broker_account = None
    lane._account_topic = "broker.account"
    # The bus records what was subscribed, so a lane that stops subscribing fails here rather than
    # returning a quiet None from `broker_equity()` for the life of the process (#174).
    lane.subscribed = []
    lane.msgbus = SimpleNamespace(
        subscribe=lambda topic, handler: lane.subscribed.append((topic, handler)))
    lane._runner = None
    lane._loop = None
    lane.cache = SimpleNamespace(positions_open=lambda **kw: [])
    lane.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                               error=lambda *a, **k: None)
    return lane


def _fill(sym, side, px=100.0):
    return SimpleNamespace(instrument_id=SimpleNamespace(symbol=sym),
                           client_order_id=f"CRSISHORT-001-{sym}",
                           order_side=SimpleNamespace(name=side), last_px=px, avg_px=px)


def _bind(name):
    """The seam method, or a skip-free failure naming what is missing."""
    fn = getattr(CrsiShortStrategy, name, None)
    if fn is None:
        pytest.fail(f"CrsiShortStrategy has no {name!r}: the _pending write-back seam does not "
                    f"exist, so the slot the decision layer reads can never be reserved (#172)")
    return fn


# -- the seam itself -----------------------------------------------------------------------------

def test_the_lane_can_RECORD_a_pending_intent():
    """The producer. Without it `_pending` is permanently empty and `decide` is handed `set()`."""
    lane = _lane()
    _bind("record_pending").__get__(lane)("AAA", "CRSISHORT-001-AAA-1", 103.5, "enter")
    assert _bind("pending").__get__(lane)() == {"AAA"}


def test_pending_returns_SYMBOLS_because_that_is_what_decide_consumes():
    """`session_intent` passes `pending=` straight into `decide`, which counts NAMES. Handing it the
    mapping would make `len()` right by accident and membership wrong."""
    lane = _lane()
    _bind("record_pending").__get__(lane)("AAA", "oid-1", 103.5, "enter")
    got = _bind("pending").__get__(lane)()
    assert isinstance(got, set) and got == {"AAA"}


def test_a_recorded_intent_REACHES_the_decision_layer():
    """The end the mechanism exists for: a reserved slot must be gone from the next decision.

    The existing `test_a_RESTING_order_consumes_a_slot` asserts this by ASSIGNING the dict, so it
    passes today against a producer that does not exist. This one goes through the seam.
    """
    import pandas as pd
    from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy as S  # noqa: F401
    from kumo_strategies.strategies.crsi_short.tests.test_crsi_short_adapter import _signalling, _last_session

    panel = _signalling(["AAA", "BBB"], "AAA")
    cfg = CrsiShortConfig(n_slots=1)
    free = _lane(cfg)
    assert "AAA" in free.session_intent(_last_session(panel), panel).enter

    committed = _lane(cfg)
    _bind("record_pending").__get__(committed)("BBB", "oid-1", 50.0, "enter")
    assert committed.session_intent(_last_session(panel), panel).enter == (), (
        "a slot reserved through record_pending did not reach decide()")


def test_clearing_a_slot_REQUIRES_a_reason():
    """A fill and a cancel both empty the slot and are not the same fact. #172 is what a slot whose
    state has no record of how it got there costs."""
    import inspect
    fn = _bind("clear_pending")
    params = inspect.signature(fn).parameters
    assert "reason" in params, "clear_pending takes no reason: a fill and a cancel read identically"
    assert params["reason"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "reason must be keyword-only, so a caller cannot pass it positionally as the symbol")
    assert params["reason"].default is inspect.Parameter.empty, (
        "reason must be REQUIRED: a default is how every caller comes to omit it")


def test_clearing_a_slot_frees_it():
    lane = _lane()
    _bind("record_pending").__get__(lane)("AAA", "oid-1", 103.5, "enter")
    _bind("clear_pending").__get__(lane)("AAA", reason="filled")
    assert _bind("pending").__get__(lane)() == set()


# -- the severe half: an UNRECORDED intent must not be read as an entry ---------------------------

def test_a_covering_BUY_with_NO_recorded_intent_does_NOT_open_a_position():
    """THE LIVE DEFECT, today, with `_pending` permanently empty.

    `completes(None, event, side=SHORT)` falls through to `return True`, so the BUY lands in
    `on_order_filled`'s `else` branch: the symbol is added to `_held` and a fresh trail opens at the
    COVER price. The lane closes a short and believes it opened one, then sizes its next session off
    that belief.

    The correct answer to "a fill arrived for an intent I never recorded" is to move nothing and say
    so — the same answer `_forget`'s docstring gives for a denied exit.
    """
    lane = _lane(held={"AAA"}, trail={"AAA": open_state(
        100.0, SessionBar(open=100.0, high=101.0, low=99.0, close=100.0))})
    assert lane._pending == {}
    lane.on_order_filled(_fill("AAA", "BUY", px=95.0))

    # THE TRAIL IS THE WITNESS, not `_held`. A cover implies the lane was holding, so `_held`
    # contains the symbol both when the handler correctly moves nothing AND when it wrongly
    # re-opens the position — `add` to a set that already holds it is invisible. The entry price
    # is what separates them: untouched leaves 100.0, re-opened seeds a fresh trail at the COVER
    # price of 95.0.
    assert lane._held == {"AAA"}, "an unreserved fill moved the holding"
    assert lane._trail["AAA"].entry_px == 100.0, (
        "a cover with no recorded intent was read as an ENTRY: a fresh trail was opened at the "
        "COVER price, so the lane believes it just shorted the name it has closed")


def test_an_entry_SELL_with_NO_recorded_intent_does_NOT_mark_the_lane_short():
    """The mirror. A SELL nobody recorded is an unattributed order, not this lane's entry — reading
    it as one makes the lane claim a position it did not open."""
    lane = _lane()
    assert lane._pending == {}
    lane.on_order_filled(_fill("AAA", "SELL", px=103.5))
    assert lane._held == set() and lane._trail == {}


# -- cancels and expiries free the slot ------------------------------------------------------------

@pytest.mark.parametrize("handler", ["on_order_canceled", "on_order_expired"])
def test_a_cancelled_or_expired_order_FREES_its_slot(handler):
    """The entry is a limit that rests overnight. A cancel — the venue's, or ours when the name
    leaves the universe — must return the slot, or the lane holds a reservation against an order
    that no longer exists and the slot is gone until the process restarts.

    Neither handler exists on the adapter today; `on_order_rejected` and `on_order_denied` do.
    """
    # `__dict__`, not `getattr`: Nautilus' Cython `Strategy` already defines both handlers, so
    # `getattr` finds a base method that does nothing with our slot AND type-checks its argument.
    # An inherited no-op is precisely the state this test exists to reject, and it is invisible to
    # `hasattr`.
    fn = CrsiShortStrategy.__dict__.get(handler)
    assert fn is not None, (
        f"{handler} is not OVERRIDDEN on the adapter — Nautilus' base handler runs instead and "
        f"moves nothing, so a resting entry the venue cancels leaks its slot forever and the "
        f"lane's capacity shrinks by one for the life of the process")
    lane = _lane()
    _bind("record_pending").__get__(lane)("AAA", "oid-1", 103.5, "enter")
    fn(lane, _fill("AAA", "SELL"))
    assert lane._pending == {}, f"{handler} left the slot reserved"
    assert lane._held == set(), f"{handler} moved the book"


# -- a restart must not lose the reservations ------------------------------------------------------

def test_an_UNKNOWN_kind_is_REFUSED_at_record_time():
    """`completes` returns True for any string it does not recognise (`order_provenance.py:107`).

    So a typo'd kind does not fail loudly at the fill — it makes EVERY fill for that symbol complete
    the intent, and the `else` branch records it as an entry. The refusal has to happen where the
    value enters, because the place it does damage cannot tell a typo from a decision.
    """
    lane = _lane()
    with pytest.raises(ValueError, match="enter|exit"):
        _bind("record_pending").__get__(lane)("AAA", "oid-1", 103.5, "entry")   # not "enter"


def test_pending_is_REBUILT_from_the_venue_on_restart():
    """`_pending` is process memory. A restart with a limit still resting at the venue would come
    back with an empty dict, re-signal the same name, and submit a SECOND order against one slot.

    Rebuilt from the CACHE rather than from a journalled copy of the dict, deliberately: a
    separately persisted `_pending` is a second record of one fact, and the two disagree silently
    the moment a cancel lands while the process is down. The venue's open orders are the fact.
    """
    fn = CrsiShortStrategy.__dict__.get("reconcile_pending")
    assert fn is not None, (
        "reconcile_pending is not implemented: after a restart the lane reads zero reserved slots "
        "with orders still resting, and re-submits against slots that are already taken")

    lane = _lane()
    resting = SimpleNamespace(
        instrument_id=SimpleNamespace(symbol="AAA"),
        client_order_id="CRSISHORT-001-AAA-1",
        order_side=SimpleNamespace(name="SELL"),
        price=103.5)
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [resting],
                                 orders_inflight=lambda **kw: [],
                                 positions_open=lambda **kw: [])
    fn(lane)
    assert _bind("pending").__get__(lane)() == {"AAA"}, (
        "a limit still resting at the venue did not reserve its slot after the restart")


def test_an_INFLIGHT_order_also_holds_its_slot_across_a_restart():
    """Submitted and not yet acknowledged is still committed. Counting only `orders_open` loses
    exactly the orders sent moments before the process died."""
    fn = CrsiShortStrategy.__dict__.get("reconcile_pending")
    assert fn is not None, "reconcile_pending is not implemented"
    lane = _lane()
    inflight = SimpleNamespace(
        instrument_id=SimpleNamespace(symbol="BBB"),
        client_order_id="CRSISHORT-001-BBB-1",
        order_side=SimpleNamespace(name="SELL"),
        price=50.0)
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [],
                                 orders_inflight=lambda **kw: [inflight],
                                 positions_open=lambda **kw: [])
    fn(lane)
    assert _bind("pending").__get__(lane)() == {"BBB"}


def test_reconciliation_reads_the_SIDE_to_tell_an_entry_from_a_cover():
    """A resting BUY on a short lane is a COVER, not an entry. Recording it as an entry makes the
    fill land in `on_order_filled`'s else branch and re-open the position it was closing — the same
    defect this module opens with, reintroduced through the restart path."""
    fn = CrsiShortStrategy.__dict__.get("reconcile_pending")
    assert fn is not None, "reconcile_pending is not implemented"
    lane = _lane(held={"AAA"})
    cover = SimpleNamespace(
        instrument_id=SimpleNamespace(symbol="AAA"),
        client_order_id="CRSISHORT-001-AAA-2",
        order_side=SimpleNamespace(name="BUY"),
        price=95.0)
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [cover],
                                 orders_inflight=lambda **kw: [],
                                 positions_open=lambda **kw: [])
    fn(lane)
    assert lane._pending["AAA"].kind == "exit", (
        "a resting BUY was reconciled as an ENTRY; its fill would re-open the short")


def test_on_start_ACTUALLY_RECONCILES_before_the_lane_can_decide():
    """The mechanism has to be CALLED, and a mutation bite proved it did not have to be.

    Deleting `self.reconcile_pending()` from `on_start` left the whole nautilus suite green: every
    other test here drives `reconcile_pending` directly, so the method could be perfectly correct and
    perfectly unreachable — the inert shape this module exists to close, reintroduced one level up.

    Driven through `on_start` rather than asserted on the source: the call has to have its EFFECT,
    and it has to happen before anything reads the slots.
    """
    lane = _lane()
    lane._history_days = 400
    lane._iids = []
    lane._need = CrsiShortConfig().warmup_sessions
    lane._resolve_symbols_if_needed = lambda: None
    lane.begin_arming = lambda: None
    resting = SimpleNamespace(
        instrument_id=SimpleNamespace(symbol="AAA"),
        client_order_id="CRSISHORT-001-AAA-1",
        order_side=SimpleNamespace(name="SELL"), price=103.5)
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [resting],
                                 orders_inflight=lambda **kw: [],
                                 positions_open=lambda **kw: [])

    CrsiShortStrategy.on_start(lane)

    assert lane.pending() == {"AAA"}, (
        "on_start did not reconcile: the lane came up reading zero reserved slots with a limit "
        "still resting at the venue, and its next session would submit a second order into it")
