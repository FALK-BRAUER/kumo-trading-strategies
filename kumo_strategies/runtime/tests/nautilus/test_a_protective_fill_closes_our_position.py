"""A protective stop that fills IS this lane's position closing, and every lane ignores it (#133).

Measured on an Alpaca paper instance 2026-09-10. Cockpit's trailing stop `PROT-SELL-TOST-XNYS-93bfd28c` — stamped
`TECHIVOL-005` since kumo-trading-platform issue 748 — triggered at 13:30:34 and filled 97 shares in partials.
Nautilus delivered six `OrderFilled` events to the lane, and `qc27_rotation.on_order_filled` returned
at `is_foreign(event)` for every one. No terminal row, `_held` still holding TOST, and
`exec_position_state` still claiming 97 against a book of 0 for the rest of the session.

THIS IS THE MIRROR OF THE DEFECT `is_foreign` WAS ADDED TO FIX, AND BOTH ARE TRUE AT ONCE:

  kumo-trading-platform issue 748   a foreign SELL must not be read as OUR ENTRY COMPLETING -> must not ADD to
                     `_held`. Correct today, and these tests keep it correct.
  #133               a foreign SELL that actually CLOSED our position must not leave us believing we
                     hold it -> must DROP from `_held`. Lost today.

One early return answers both questions with one word and gets one of them wrong. "Is this ours to
ACT on" and "did this CHANGE OUR BOOK" are different questions, and a fill can be the second without
being the first.

The fourth time this package has learned that lesson: `sides.py` (a position's side is declared, not
assumed), `completes()` (which fill completes which intent depends on the lane's side),
`broker.py:83` (a SELL is not always an exit), and now this.

WHAT MAKES THE FIXTURE HONEST. The events are shaped like the ones that actually arrived — a
`PROT-`-prefixed client order id, the lane's own strategy id, PARTIAL fills that only together reach
the held quantity — because a fixture that delivered one clean fill would not exercise the partial
path that produced the incident.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.contract import (
    _journal_protective_close, _sync_claim_to_book, protective_close)
from kumo_strategies.runtime.nautilus.order_provenance import is_foreign
from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import QC27RotationStrategy
from kumo_strategies.runtime.nautilus.sides import LONG, closing_order_side
from kumo_strategies.strategies import _layout



def _modules_with_a_fill_handler():
    """Every module in `runtime/nautilus/` that defines `on_order_filled`, found by reading the
    package rather than by naming files.

    THE LIST USED TO BE WRITTEN BY HAND AND IT WAS WRONG. `penny_gap` and
    `momentum_rotation_intraday` both define the handler and both were missing from it, so two lanes
    shipped without the #133 fix and every test here passed. This package already has a rule about
    that — `test_no_test_here_iterates_a_HAND_WRITTEN_list_of_strategies` — and this file broke it.
    """
    out = []
    for path in _layout.nautilus_sources():                  # ks#211: shared layer + every lane
        tree = ast.parse(path.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name == "on_order_filled"
               for n in ast.walk(tree)):
            out.append((path, tree))
    return out


def _lane_classes():
    """Every lane class, INCLUDING ones that inherit the handler rather than defining it.

    BCTROT was silently absent. It subclasses `MomentumRotationStrategy` and defines no
    `on_order_filled` of its own, so a scan that imports only modules DEFINING the handler never
    reached it — while the comment here claimed subclasses were covered. A comment asserting what
    the code does not do is worse than no comment: it stops the next reader checking.
    """
    seen, lanes = set(), []
    # Every lane module, so a lane that only INHERITS the handler is still found.
    for mod in _layout.lane_modules():
        for _n, cls in inspect.getmembers(mod, inspect.isclass):
            if (cls.__module__.startswith("kumo_strategies.strategies.")
                    and hasattr(cls, "on_order_filled") and hasattr(cls, "POSITION_SIDE")
                    and cls not in seen):
                seen.add(cls)
                lanes.append(cls)
    return lanes


LANES = _lane_classes()

#: The two lanes that subclass Nautilus's `Strategy` DIRECTLY and carry no `RegistrationMixin`, so
#: `session_journal` and `fire_and_report` do not exist on them. Discovered, not listed: a lane that
#: stops carrying the mixin tomorrow joins this set without an edit here.
NO_MIXIN = [c for c in LANES
            if not (hasattr(c, "session_journal") and hasattr(c, "fire_and_report"))]

#: The lanes that CAN write a durable row and re-sync a claim — the ones carrying
#: `RegistrationMixin`. The journal and claim assertions are parametrised over these, not over every
#: lane: asserting a row on a lane that structurally cannot write one would pass only while the
#: fixture pretended it could, which is the pretence this file exists to stop.
JOURNALLING = [c for c in LANES if c not in NO_MIXIN]



def test_every_module_with_a_fill_handler_calls_protective_close():
    """COVERAGE, not existence — the same rule `test_foreign_order_events` applies.

    Two lanes were missing when this list was written by hand. Asserting on the AST of every module
    that defines the handler is what makes a lane added tomorrow fail here instead of shipping
    without the fix.
    """
    missing = []
    for path, tree in _modules_with_a_fill_handler():
        # BOTH CALLEE SHAPES. `protective_close(self, event)` is an `ast.Name`;
        # `self.protective_close(event)` is an `ast.Attribute`. Collecting only one is precisely the
        # hole that let the lane's-own-exit regression ship green — `test_foreign_order_events`
        # gathered `ast.Name` callees only and never saw the method call at all.
        called = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if "protective_close" not in called:
            missing.append(path.name)
    assert not missing, (
        f"these define on_order_filled and never call protective_close, so a protective stop that "
        f"closes their position is invisible to them: {missing}")


def test_the_discovery_FINDS_every_lane_including_the_inherited_ones():
    """Guards the guard, against the COUNT THE PACKAGE ACTUALLY HAS rather than a number typed here.

    A fixed `>= 5` passed while BCTROT was missing — it only proves the scan is non-empty, not that
    it is complete, and "non-empty" is exactly what a broken discovery still looks like. The
    expectation is now derived the same way a reader would derive it: every class in the package
    that has a fill handler and declares a side.
    """
    expected = set()
    for mod in _layout.lane_modules():
        for _n, cls in inspect.getmembers(mod, inspect.isclass):
            if (cls.__module__.startswith("kumo_strategies.strategies.")
                    and hasattr(cls, "on_order_filled") and hasattr(cls, "POSITION_SIDE")):
                expected.add(cls.__name__)
    assert {c.__name__ for c in LANES} == expected
    assert any("BCT" in n for n in expected), (
        "BCTROT inherits its handler and must still be exercised — it went missing once already")


class _Symbol(str):
    """Nautilus's `Symbol`, in the two shapes the handlers actually use.

    `momentum_rotation:898` reads `event.instrument_id.symbol.value`; the others do
    `str(event.instrument_id.symbol)`. A plain string satisfies the second and raises on the first,
    so a double that used one would exercise three lanes and crash on two — which is exactly what
    happened the moment the `is_foreign` guard let the ordinary path run.
    """

    @property
    def value(self) -> str:
        return str(self)


def _prot_fill(symbol: str, qty: int, side: str, *, last_px: float = 36.5):
    """One of the six events that arrived: cockpit's order, this lane's strategy id."""
    return SimpleNamespace(
        client_order_id=f"PROT-SELL-{symbol}-XNYS-93bfd28c",
        instrument_id=SimpleNamespace(symbol=_Symbol(symbol)),
        order_side=SimpleNamespace(name=side),
        last_qty=qty, last_px=last_px, avg_px=last_px,
    )


def _ordinary_fill(symbol: str, qty: int, side: str, *, last_px: float = 36.5):
    return SimpleNamespace(
        client_order_id=f"O-{symbol}-20260915-000001",
        instrument_id=SimpleNamespace(symbol=_Symbol(symbol)),
        order_side=SimpleNamespace(name=side),
        last_qty=qty, last_px=last_px, avg_px=last_px,
    )


def test_the_fixture_really_is_foreign():
    """Guards the guard. If these events stopped being foreign, every assertion below would pass for
    the wrong reason — the handler would take its ordinary path and the #133 branch would never be
    exercised at all."""
    assert is_foreign(_prot_fill("TOST", 97, "SELL"))


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_protective_fill_that_closes_our_position_drops_the_holding(cls):
    """THE DEFECT. The lane holds TOST, a protective stop sells all 97, and the lane must stop
    believing it holds it.

    Failing today for every lane: the handler returns at `is_foreign` before touching `_held`.
    """
    lane = _lane(cls, held={"TOST"})
    # 97 -> 57 -> 17 -> 0. THE HOLDING SURVIVES THE PARTIALS and goes only when the book is flat:
    # a lane that forgets a position it still holds stops managing a real one, which is strictly
    # worse than the defect this fixes.
    for part, remaining in ((40, 57), (40, 17), (17, 0)):
        lane._book.take(part)
        cls.on_order_filled(lane, _prot_fill("TOST", part, closing_order_side(_side(cls))))
        if remaining:
            assert "TOST" in lane._held, (
                f"dropped the holding with {remaining} shares still on the book")
        # The claim sync needs `fire_and_report`, which two lanes do not have. Asserting it for
        # them would pass only while the fixture pretended they did. The BOOK, asserted above, is
        # the part that must hold for every lane.
        if cls in JOURNALLING:
            assert lane._runner.synced[-1] == ("TOST", remaining, 36.5), lane._runner.synced
    assert "TOST" not in lane._held, (
        "the lane still holds a position a protective stop closed — the claim stands, the slot is "
        "counted as occupied, and a later decision can try to exit shares that are gone")


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_protective_fill_still_does_NOT_complete_a_pending_ENTRY(cls):
    """kumo-trading-platform issue 748, unchanged, and the reason the fix cannot simply delete the guard. A foreign
    fill must never mark a symbol HELD out of nothing, whatever its side."""
    lane = _lane(cls, held=set())
    lane._pending = {"TOST": "enter"}
    cls.on_order_filled(lane, _prot_fill("TOST", 97, "BUY" if _side(cls) == LONG else "SELL"))
    assert "TOST" not in lane._held, "a foreign fill marked a symbol held out of nothing"


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_protective_fill_on_a_symbol_we_do_NOT_hold_changes_nothing(cls):
    """Protection on ANOTHER lane's shares reaches this handler too — that is the whole reason
    kumo-trading-platform issue 748 exists. Nothing about it is ours.

    ASSERTING ONLY ON `_held` IS NOT ENOUGH AND THAT IS WHY THE OTHER TWO ASSERTIONS ARE HERE:
    `set.discard` on an absent member is a no-op, so deleting the `sym not in held` guard entirely
    leaves `_held` correct while the lane journals another lane's close and re-syncs a claim it does
    not own. A mutation bite proved that — this test passed with the guard removed.
    """
    lane = _lane(cls, held={"AAPL"})
    cls.on_order_filled(lane, _prot_fill("TOST", 97, closing_order_side(_side(cls))))
    assert lane._held == {"AAPL"}
    assert lane._journal.rows == [], "journalled a close for a position this lane never held"
    assert lane._runner.synced == [], "re-synced a claim this lane does not own"


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_foreign_fill_on_the_OPENING_side_is_not_a_close(cls):
    """A foreign fill that moves the position the WRONG way is someone else opening something.

    For a long lane that is a foreign BUY while we hold: a book-repair leg, a transfer, another
    component's entry. It cannot have closed us, and treating it as a close would drop a holding
    that is still there — the lane then stops managing a real position, which is strictly worse than
    the defect this whole file is about.

    Held deliberately, because with `held=set()` the earlier guard returns first and the side check
    is never reached — which is exactly how this escaped the first round of bites.
    """
    lane = _lane(cls, held={"TOST"})
    opening = "BUY" if _side(cls) == LONG else "SELL"
    cls.on_order_filled(lane, _prot_fill("TOST", 97, opening))
    assert "TOST" in lane._held, "dropped a position on a fill that could not have closed it"
    assert lane._journal.rows == [] and lane._runner.synced == []


@pytest.mark.parametrize("cls", JOURNALLING, ids=lambda c: c.__name__)
def test_the_claim_is_RE_SYNCED_after_a_protective_close(cls):
    """The claims ledger is what over-claimed 97 TOST for a whole session.

    `sync_claim` exists on the runner and every terminal path has to reach it; otherwise an ordinary
    exit or a protective close can leave the durable claim where it was.
    """
    lane = _lane(cls, held={"TOST"})
    for part in (40, 40, 17):
        lane._book.take(part)
        cls.on_order_filled(lane, _prot_fill("TOST", part, closing_order_side(_side(cls))))
    assert lane._runner.synced, (
        f"{cls.__name__} closed a position by protection and never re-synced the claim")


def test_qc27_a_REJECTED_exit_re_syncs_the_claim_to_the_still_held_book():
    """TECHIVOL dropped its claim when Nautilus accepted a SELL submit.

    If the venue then rejected the exit, the lane still held the position but had no ownership row, so
    its next pass read the shares as foreign. The runner no longer drops on accepted submit, and this
    terminal path is the backstop that keeps the claim aligned with the cache state.
    """
    lane = _lane(QC27RotationStrategy, held={"TOST"})
    lane._pending = {"TOST": "exit"}

    QC27RotationStrategy.on_order_rejected(lane, _ordinary_fill("TOST", 97, "SELL"))

    assert lane._runner.synced[-1] == ("TOST", 97, None), lane._runner.synced


def test_qc27_a_FILLED_exit_re_syncs_the_claim_to_flat():
    """The release happens on the venue answer, not at submit acceptance."""
    lane = _lane(QC27RotationStrategy, held={"TOST"})
    lane._pending = {"TOST": "exit"}
    lane._book.take(97)

    QC27RotationStrategy.on_order_filled(lane, _ordinary_fill("TOST", 97, "SELL"))

    assert lane._runner.synced[-1] == ("TOST", 0, 36.5), lane._runner.synced


# -- the host ---------------------------------------------------------------------------------------

def _side(cls):
    return getattr(cls, "POSITION_SIDE", LONG)


class _Book:
    """The lane's own Nautilus position, DECREMENTING as the partials land.

    The first version of this double returned [] unconditionally, so the position read flat on the
    FIRST partial of 97 — `_held` was dropped before the book was flat, and the partial path the
    fix's docstring describes was never exercised at all. A mutation replacing `if qty == 0` with
    `if True` survived against it.

    97 -> 57 -> 17 -> 0, which is the fill sequence from the incident.
    """

    def __init__(self, held, start=97):
        self._qty = start if held else 0

    def take(self, n):
        self._qty = max(0, self._qty - n)

    def open(self):
        if self._qty == 0:
            return []
        return [SimpleNamespace(signed_qty=self._qty)]


class _Journal:
    """Records the durable rows. `write` is a coroutine because the real journal's is."""

    def __init__(self):
        self.rows = []

    async def write(self, kind, summary, *, session, detail=None, **kw):
        self.rows.append((kind, summary, session, detail))


class _Runner:
    """Records what the lane asked of it. `sync_claim`/`record_terminal` are coroutines because the
    real gateway's are, and the lanes hand them to `fire_and_report`."""

    def __init__(self):
        self.synced, self.terminals = [], []

    async def sync_claim(self, symbol, qty, px):
        self.synced.append((symbol, qty, px))

    async def record_terminal(self, session, symbol, ok, detail):
        self.terminals.append((session, symbol, ok, detail))


def _lane(cls, *, held):
    """The real handler bound to a narrow host.

    `Strategy.__init__` needs a running kernel and `Actor.log` is read-only, so the class cannot be
    constructed here. Every method under test is the REAL attribute, called unbound — a rename or a
    deletion fails these tests rather than exercising a copy.
    """
    runner, journal, book, said = _Runner(), _Journal(), _Book(held), []
    lane = SimpleNamespace(
        # NO `_trail`: not one lane in this package keeps one, so granting it would make the
        # `_trail` branch in `protective_close` look exercised while nothing reaches it. That branch
        # is removed in the same change. (CRSISHORT does keep a trail; it is not on this branch, and
        # its own commit re-adds the branch WITH a test that reaches it.)
        _held=set(held), _pending={}, _runner=runner, _loop=object(),
        _cfg=None, id=f"{cls.__name__}-001", POSITION_SIDE=_side(cls),
        log=SimpleNamespace(info=lambda *a, **k: None,
                            warning=lambda m="", *a, **k: said.append(str(m)),
                            error=lambda m="", *a, **k: said.append(str(m)),
                            debug=lambda *a, **k: None),
        cache=SimpleNamespace(positions_open=lambda **kw: book.open(), order=lambda coid: None),
        session_journal=lambda: journal,
        # `fire_and_report` schedules onto a live loop; here the coroutine is simply run to
        # completion so the call is OBSERVED rather than dropped.
        fire_and_report=lambda coro, loop, what: _drain(coro),
        _try_decide=lambda: None,
        # Attributes the ORDINARY exit paths of individual lanes read once the protection path
        # correctly declines them. They are here rather than stubbed away because the point of these
        # tests is that the ordinary path RUNS.
        _stops={}, _entry_px={}, _armed={}, _sent=set(), _exits={},
    )
    # CAPABILITIES ARE DERIVED FROM THE CLASS, never granted by the fixture.
    #
    # This is the structural fix for the failure that hid three defects in this work: a double more
    # capable than production passes, and the capability it invented IS the defect. So a method the
    # code under test may reach is attached only if the real class has it — `session_journal` and
    # `fire_and_report` are `RegistrationMixin`'s, and two lanes carry neither.
    #
    # Instance STATE (`_held`, `_pending`, `_loop`) is still supplied, because it is created in
    # `__init__` and cannot be read off the class. The split is the honest one: what a lane CAN DO
    # comes from the class, what it currently HOLDS comes from the test.
    for _cap in ("session_journal", "fire_and_report"):
        if not hasattr(cls, _cap):
            delattr(lane, _cap)

    # THE REAL FUNCTIONS, bound to this host — not stubs. `protective_close` is the code under
    # test, so a rename or a deletion has to fail here rather than be quietly re-implemented.
    # The REAL functions, bound to this host. They are module-level now (see contract.py) because
    # two lanes do not carry RegistrationMixin at all.
    lane.protective_close = lambda ev: protective_close(lane, ev)
    lane._sync_claim_to_book = lambda *a: _sync_claim_to_book(lane, *a)
    lane._journal_protective_close = lambda *a: _journal_protective_close(lane, *a)
    lane._journal = journal
    lane._book = book
    lane._said = said
    record = getattr(cls, "_record_terminal", None)
    if record is not None:
        lane._record_terminal = record.__get__(lane)
    forget = getattr(cls, "_forget", None)
    if forget is not None:
        lane._forget = forget.__get__(lane)
    return lane


def _drain(coro):
    try:
        coro.send(None)
    except StopIteration:
        pass
    return None


@pytest.mark.parametrize("cls", JOURNALLING, ids=lambda c: c.__name__)
def test_a_durable_row_says_the_position_left_and_who_closed_it(cls):
    """Zero journal rows for TOST is what made the incident invisible until the claims breach was
    read by hand.

    The row is written by the mixin and NOT through `_record_terminal`: that method recovers the
    session from the order's own `session:` tag, which only orders THIS adapter submitted carry, so
    for a protective stop it returns early and writes nothing. Calling it anyway would look like
    journalling and be silent — the kumo-trading-platform issue 549 / #383 failure exactly.
    """
    lane = _lane(cls, held={"TOST"})
    for part in (40, 40, 17):
        lane._book.take(part)
        cls.on_order_filled(lane, _prot_fill("TOST", part, closing_order_side(_side(cls))))
    rows = lane._journal.rows
    assert rows, f"{cls.__name__} wrote no row for a position that left the book"
    # ONE ROW PER PARTIAL, each stating the position AFTER it — the same cadence `record_terminal`
    # writes at, because each fill is a separate answer from the venue. The sequence is the
    # assertion that matters: a row claiming flat before the book is flat would be the fix
    # reintroducing the defect it fixes, one partial early.
    assert [d["position_after"] for _k, _s, _sess, d in rows] == [57, 17, 0], rows
    kind, summary, _session, detail = rows[-1]
    # KIND `order` + `detail.phase == "terminal"`, which is what `record_terminal` already writes and
    # what every cockpit reader matches on. A new KIND is written durably and is invisible in
    # /slots verdicts, the journal filters and the UI strategy plane — for a row whose whole job is
    # to make a silent close visible, that would be the same defect wearing a fix's clothes.
    assert kind == "order", f"a new journal kind is invisible where operators read: {kind!r}"
    assert detail["phase"] == "terminal"
    assert "protective" in summary.lower()
    assert detail["by"] == "protection" and detail["position_after"] == 0


def _own_fill(symbol: str, qty: int, side: str, *, last_px: float = 36.5):
    """THE LANE'S OWN EXIT — a client order id this adapter minted, no `PROT-` prefix."""
    return SimpleNamespace(
        client_order_id=f"O-20260910-{symbol}-001",
        instrument_id=SimpleNamespace(symbol=_Symbol(symbol)),
        order_side=SimpleNamespace(name=side),
        last_qty=qty, last_px=last_px, avg_px=last_px,
    )


def test_the_own_fill_fixture_is_NOT_foreign():
    """Guards the guard, the other way round. If this event were foreign, the tests below would pass
    for the wrong reason — they would be exercising the protection path they exist to keep OUT of."""
    assert not is_foreign(_own_fill("TOST", 97, "SELL"))


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_the_lanes_OWN_exit_is_not_captured_by_the_protection_path(cls):
    """THE REGRESSION 3311c6a INTRODUCED, and it is worse than the defect it fixed.

    `protective_close` runs BEFORE the `is_foreign` guard in the handler, and its own gates were only
    `sym in _held` and "the fill is on the closing side" — both true of the lane's OWN ordinary exit.
    So every exit, partial or full, on all five lanes, was captured by the protection path:

      - journalled as "closed by a protective stop — this lane did not place that order", which is
        false and is the row an operator would read during an incident;
      - `_record_terminal` never fires, so the venue's real answer is never recorded AND
        momentum/qc345's own `sync_claim` (momentum_rotation:978, qc345_rotation:799) is skipped;
      - qc27 never reaches `_try_decide`, so the session does not continue.

    It shipped green because `test_foreign_order_events.py` collected `ast.Name` callees only, and
    `self.protective_close(...)` is an `ast.Attribute` — invisible to it. That hole is closed in the
    same change.
    """
    lane = _lane(cls, held={"TOST"})
    lane._pending = {"TOST": "exit"}
    ev = _own_fill("TOST", 97, closing_order_side(_side(cls)))
    assert not is_foreign(ev)                       # the fixture property, asserted first
    lane._book.take(97)
    # THE UNIT UNDER TEST, called directly. Driving the whole handler is not possible for every lane
    # here — `PennyGapStrategy` and `IntradayMomentumRotation` do not carry `RegistrationMixin` and
    # their ordinary exit paths read lane-specific state — and `test_every_module_with_a_fill_handler
    # _calls_protective_close` is what proves each handler reaches this function at all.
    assert protective_close(lane, ev) is False, (
        "the lane's own exit was captured by the protection path")
    assert lane._journal.rows == [], (
        f"the lane's own exit was journalled as a protective close: {lane._journal.rows}")
    assert lane._runner.synced == []


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_the_lanes_own_PARTIAL_exit_is_not_captured_either(cls):
    """A partial is the same question with the book still non-empty — and it is where the mislabel
    would be most confusing, because the position is still there afterwards."""
    lane = _lane(cls, held={"TOST"})
    lane._pending = {"TOST": "exit"}
    lane._book.take(40)
    assert protective_close(lane, _own_fill("TOST", 40, closing_order_side(_side(cls)))) is False
    assert lane._journal.rows == []


def test_a_lane_that_does_NOT_declare_its_side_raises_rather_than_being_assumed_LONG():
    """The same rule `completes` and `sides.py` enforce, and it belongs here for the same reason.

    A defaulted LONG would hand a short lane the long mapping, and this function would then decline
    every protective close on it — the defect it exists to fix, inverted and silent. Every adapter in
    this package declares `POSITION_SIDE`; a new one that forgets must fail loudly.
    """
    lane = SimpleNamespace(
        _held={"TOST"}, _pending={}, _runner=_Runner(), _loop=object(),
        id="NOSIDE-001", log=SimpleNamespace(error=lambda *a, **k: None,
                                             warning=lambda *a, **k: None),
        cache=SimpleNamespace(positions_open=lambda **kw: []),
        session_journal=lambda: None, fire_and_report=lambda c, l, w: _drain(c))
    with pytest.raises(AttributeError, match="POSITION_SIDE"):
        protective_close(lane, _prot_fill("TOST", 97, "SELL"))


def test_a_lane_with_no_book_DECLARES_that_rather_than_being_silently_inert():
    """CALLING `protective_close` IS NOT THE SAME AS BEING COVERED BY IT.

    It returns False at its `_held` check, so a lane that keeps no `_held` calls it and gets
    nothing — while the coverage scan reports it covered. That is a green the lane has not earned,
    and it is this ticket's own shape: a mechanism wired, apparently armed, doing nothing.

    PENNY GAP is the real case — its book is `_entries`/`_stops` and its own trailing stop IS its
    protection, so a protective fill is its ORDINARY exit. It says so in `NO_LANE_BOOK`, and this
    test forces the next such lane to say so too instead of passing quietly.
    """
    def _keeps_a_book(cls) -> bool:
        """Does this lane ASSIGN `self._held` anywhere in its `__init__`?

        THE AST, NOT THE SOURCE TEXT. `"_held" in inspect.getsource(cls)` was the first version and
        it was satisfied by a COMMENT — penny_gap's own `NO_LANE_BOOK` docstring says `_held`, so
        deleting the declaration left the test green. A grep-the-source assertion is satisfied by
        writing a comment and broken by removing one, which is backwards.

        THE WHOLE MRO, because a lane can INHERIT its book: BCTROT assigns none and gets MOMENTUM's.
        """
        for k in cls.__mro__:
            if not k.__module__.startswith("kumo_strategies"):
                continue
            for node in ast.walk(ast.parse(inspect.getsource(k))):
                if (isinstance(node, ast.Attribute) and node.attr == "_held"
                        and isinstance(node.ctx, ast.Store)):
                    return True
        return False

    undeclared = [cls.__name__ for cls in LANES
                  if not _keeps_a_book(cls) and not getattr(cls, "NO_LANE_BOOK", None)]
    assert not undeclared, (
        f"these keep no `_held`, so protective_close is INERT on them while the coverage scan "
        f"reports them covered. Declare NO_LANE_BOOK with the reason: {undeclared}")


def test_the_no_mixin_lanes_are_actually_FOUND():
    """Guards the guard. If this set went empty the tests below would assert nothing, and the whole
    failure mode — a capability the host has and production does not — would be invisible again."""
    assert NO_MIXIN, [c.__name__ for c in LANES]


@pytest.mark.parametrize("cls", NO_MIXIN, ids=lambda c: c.__name__)
def test_a_lane_without_the_mixin_does_not_RAISE_into_nautilus(cls):
    """A protective close on these lanes reached `self.session_journal()` unguarded, and that method
    is the mixin's. The AttributeError would surface inside Nautilus's dispatch, on a live fill —
    the exact failure the module-level function was supposed to prevent, moved one call deeper."""
    lane = _lane(cls, held={"TOST"})
    protective_close(lane, _prot_fill("TOST", 97, closing_order_side(_side(cls))))


@pytest.mark.parametrize("cls", NO_MIXIN, ids=lambda c: c.__name__)
def test_a_lane_without_the_mixin_STILL_updates_its_book(cls):
    """The book is the part that must not depend on a journal being reachable. A lane that cannot
    write the row still must not go on believing it holds a position that is gone."""
    lane = _lane(cls, held={"TOST"})
    lane._book.take(97)                       # the venue filled it; the book is flat
    assert protective_close(lane, _prot_fill("TOST", 97, closing_order_side(_side(cls)))) is True
    assert "TOST" not in lane._held


@pytest.mark.parametrize("cls", NO_MIXIN, ids=lambda c: c.__name__)
def test_a_lane_without_the_mixin_SAYS_what_it_could_not_do(cls):
    """Silence here would be the failure this whole ticket is about: a position leaving the book
    with no durable trace. If the row cannot be written, the log has to name the reason."""
    lane = _lane(cls, held={"TOST"})
    lane._book.take(97)
    protective_close(lane, _prot_fill("TOST", 97, closing_order_side(_side(cls))))
    assert any("journal" in m.lower() or "session_journal" in m for m in lane._said), lane._said
