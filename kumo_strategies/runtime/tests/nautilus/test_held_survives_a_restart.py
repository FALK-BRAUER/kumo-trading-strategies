"""`_held` is built empty and nothing seeds it, so a restart blinds every lane (#194).

Verified across all eight adapters that carry a `_held`: `self._held: set[str] = set()` at
construction, and exactly ONE `.add` site each — `on_order_filled` with `intent == "enter"`. The set
therefore describes fills THIS PROCESS saw, not positions THIS LANE HOLDS.

`contract.protective_close` opens `if held is None or sym not in held: return False`, so every
position predating a boot is invisible: `_record_terminal` never fires, the venue's answer to a
protective stop-out goes unrecorded, and the claim outlives the position. Measured live: two
QC345-003 stop-outs at 13:36Z and 13:40Z on 2026-09-11, both after its 12:51Z recreate, both leaving
claims of 3.0 against a cache of 0.0, with no journal row of any kind.

TWO THINGS THIS FILE IS DELIBERATELY CAREFUL ABOUT.

THE FIXTURE MUST CONTAIN THE RESTART. A test that opens a position and stops it out inside one lane
lifetime passes TODAY and proves nothing, because the `.add` in `on_order_filled` populated `_held`
on the way through. The defect IS the restart, so every behavioural case below starts from a lane
that holds a position it never filled — and asserts `_held` is EMPTY first, so a fixture that
populated it through a helper or a shared setup cannot pass while testing nothing.

THE HOST IS NARROW, NOT FORGIVING. A Nautilus `Strategy` cannot be constructed here and cannot be
faked by assignment either — `id`, `cache` and `log` are read-only Cython attributes on `Component`,
so `cls.__new__(cls)` yields an object whose `id` is None and unwritable. The response is to bind
the REAL method to a minimal host rather than to loosen a double until it accepts anything: the
function under test is `HeldSeedMixin.seed_held_from_positions` itself, unmodified. What that leaves
unproven — that each lane actually INHERITS and CALLS it — is covered separately and structurally,
by MRO membership and by an AST read of the `on_start` each lane really runs.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
from nautilus_trader.model.identifiers import InstrumentId, StrategyId, Symbol, Venue
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.runtime.nautilus.held_seed import (
    HeldSeedMixin, position_is_unattributed, seeded_symbols, unattributed_positions)
from kumo_strategies.runtime.nautilus.sides import LONG, SHORT
from kumo_strategies.strategies import _layout

def _lane_classes():
    """Every `Strategy` subclass this package ships, discovered rather than listed.

    A hand-written list is how a ninth lane ships with the defect intact: the list is satisfied by
    the eight that were known when it was typed.
    """
    out = {}
    for mod in _layout.lane_modules():                       # ks#211: the ONE discovery point
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                out[obj.__name__] = obj
    return out


LANES = _lane_classes()
#: Lanes that keep a lane book — a lane may opt out only by declaring `NO_LANE_BOOK`.
BOOKED = {n: c for n, c in LANES.items() if not hasattr(c, "NO_LANE_BOOK")}


# -- the behaviour, bound to a narrow host -------------------------------------------------------

class _Host(HeldSeedMixin):
    """A lane reduced to exactly what the seed reads: an id, a cache, a log, a side, a `_held`."""

    def __init__(self, *, positions, side=LONG, everything=None, raises=False):
        self.id = "LANE-001"
        self.POSITION_SIDE = side
        self._held: set[str] = set()
        self.said: list[tuple[str, str]] = []
        self.asked: dict = {}

        def positions_open(**kw):
            if raises:
                raise RuntimeError("cache unavailable")
            if kw:
                self.asked.update(kw)
                return positions
            return positions if everything is None else everything

        self.cache = SimpleNamespace(positions_open=positions_open)
        self.log = SimpleNamespace(
            info=lambda m: self.said.append(("info", m)),
            warning=lambda m: self.said.append(("warning", m)),
            error=lambda m: self.said.append(("error", m)))


def _sid(value):
    """A REAL `StrategyId`, because a plain string is a double that cannot express the bug.

    `StrategyId.__eq__` is Cython-typed and RAISES on comparison to `str`; `Position.strategy_id` is
    always one of these and never None or "". Every fixture below carried plain strings, so both
    blockers the paper tenant found were invisible here while being certain in production.
    `None` stays representable — a non-Nautilus caller may pass a bare object.
    """
    return None if value is None else StrategyId(value)


def _pos_on(symbol, qty, *, venue, strategy_id="LANE-001"):
    """A position on a NAMED listing. IBKR returns one name on two venues routinely."""
    return SimpleNamespace(instrument_id=InstrumentId(Symbol(symbol), Venue(venue)),
                           signed_qty=qty, quantity=abs(qty), strategy_id=_sid(strategy_id))


def _pos(symbol, qty, *, strategy_id="LANE-001"):
    """A position carrying a REAL `InstrumentId`.

    Not a stand-in: the first version of this fixture hung `__str__` on a `SimpleNamespace`
    instance, where Python never consults it — dunders resolve on the type — so `str(symbol)`
    returned the namespace repr and the seed adopted garbage. A hand-rolled identifier is exactly
    the permissive double this repo has a standing rule against; the real one spells the symbol the
    way `contract.protective_close` will read it.
    """
    return SimpleNamespace(
        instrument_id=InstrumentId(Symbol(symbol), Venue("XNAS")),
        signed_qty=qty, quantity=abs(qty), strategy_id=_sid(strategy_id))


def test_a_RESTARTED_lane_adopts_a_position_it_never_filled():
    host = _Host(positions=[_pos("AAA", 100)])
    assert host._held == set(), "the fixture populated `_held`; this test would prove nothing"

    adopted = host.seed_held_from_positions()

    assert "AAA" in host._held, "a live position this lane owns is still invisible after a restart"
    assert adopted == {"AAA"}


def test_the_seed_asks_for_THIS_LANE_S_positions_not_the_account_s():
    """TECHIVOL-005 proposed exiting eight names belonging to two other strategies on its first live
    session by reading the account's positions. Adopting unattributed holdings is strictly worse
    than the blindness being fixed here."""
    host = _Host(positions=[_pos("AAA", 100)])
    host.seed_held_from_positions()
    assert host.asked.get("strategy_id") == host.id, (
        f"the seed asked the cache for {host.asked!r}; it must ask for THIS lane's positions")


def test_a_FLAT_position_is_not_adopted():
    """Zero is not held. Seeding it would make the lane assert a holding it does not have — the
    mirror of the defect being fixed."""
    host = _Host(positions=[_pos("AAA", 0)])
    host.seed_held_from_positions()
    assert host._held == set()


def test_a_position_on_the_WRONG_SIDE_is_not_adopted():
    """A SHORT lane adopting a LONG position hands `protective_close` a name whose closing side is
    inverted, so it would ignore every real protective close on it — #194 inverted."""
    short = _Host(positions=[_pos("AAA", 100), _pos("BBB", -50)], side=SHORT)
    short.seed_held_from_positions()
    assert short._held == {"BBB"}

    long_ = _Host(positions=[_pos("AAA", 100), _pos("BBB", -50)], side=LONG)
    long_.seed_held_from_positions()
    assert long_._held == {"AAA"}


def test_seeding_is_ADDITIVE_and_never_drops_a_name_held_from_this_session():
    """A name entered and exited this session is in neither the open positions nor correctly held.
    Assignment rather than union would silently discard names `on_order_filled` legitimately added."""
    host = _Host(positions=[_pos("AAA", 100)])
    host._held.add("FILLED_THIS_SESSION")
    host.seed_held_from_positions()
    assert host._held == {"AAA", "FILLED_THIS_SESSION"}


def test_seeding_NEVER_RAISES_because_it_runs_in_on_start():
    """A lane that dies in `on_start` never arms, never decides and reports nothing — strictly worse
    than the blindness this fixes. The same rule `_record_arm_state` learned when its
    `clock.timestamp_ns()` raised over the arming failure it was reporting."""
    host = _Host(positions=[], raises=True)
    assert host.seed_held_from_positions() == set()          # must not raise
    assert host._held == set()
    assert any(lvl == "error" for lvl, _ in host.said), (
        "a cache that cannot answer left no trace; silence is what #133 looked like")


def test_a_lane_with_no_POSITION_SIDE_is_REFUSED_and_says_so():
    host = _Host(positions=[_pos("AAA", 100)])
    del host.POSITION_SIDE
    assert host.seed_held_from_positions() == set()
    assert any(lvl == "error" and "POSITION_SIDE" in m for lvl, m in host.said)


def test_a_position_whose_SIDE_IS_UNRECOGNISED_is_refused_rather_than_assumed():
    """`qty > 0 if side == LONG else qty < 0` reads every non-LONG value as SHORT, so a lane whose
    POSITION_SIDE was mistyped would adopt short positions with no one having established its side."""
    assert seeded_symbols([_pos("AAA", -100)], "shrot") == set()
    assert seeded_symbols([_pos("AAA", 100)], None) == set()


def test_the_report_names_ONLY_unattributed_positions_not_other_lanes():
    """A warning that fires every start is one nobody reads. Another lane's attributed position is
    correctly not ours and must not be printed into this lane's log."""
    host = _Host(positions=[], everything=[_pos("ORPHAN", 100, strategy_id="EXTERNAL"),
                                           _pos("FOREIGN", 100, strategy_id="OTHER-002")])
    host.seed_held_from_positions()
    warned = [m for lvl, m in host.said if lvl == "warning"]
    assert warned, "the unattributed position went unreported"
    assert "ORPHAN" in warned[0]
    assert "FOREIGN" not in warned[0], (
        f"another lane's attributed position was reported as unattributed: {warned[0]!r}")


def test_an_UNATTRIBUTED_position_is_REPORTED_and_NOT_adopted():
    """The seed's known limit, made visible. A position reconciled after a cold start may carry no
    strategy id; it stays invisible to `protective_close`, which is deliberate — but the operator
    gets a NAME rather than silence."""
    host = _Host(positions=[], everything=[_pos("ORPHAN", 100, strategy_id="EXTERNAL")])
    host.seed_held_from_positions()
    assert host._held == set(), "an unattributed position was adopted; that is the TECHIVOL-005 defect"
    assert any(lvl == "warning" and "ORPHAN" in m for lvl, m in host.said), (
        f"the unattributed position went unreported: {host.said!r}")


def test_a_lane_that_keeps_NO_held_seeds_nothing_and_does_not_complain():
    host = _Host(positions=[_pos("AAA", 100)])
    del host._held
    assert host.seed_held_from_positions() == set()
    assert host.said == []


@pytest.mark.parametrize("qty,side,adopted", [
    (100, LONG, {"AAA"}), (-100, LONG, set()), (0, LONG, set()),
    (-100, SHORT, {"AAA"}), (100, SHORT, set()), (0, SHORT, set()),
])
def test_seeded_symbols_is_pure_and_total(qty, side, adopted):
    assert seeded_symbols([_pos("AAA", qty)], side) == adopted


# -- the wiring, structural rather than behavioural ----------------------------------------------

def test_every_lane_keeps_a_lane_book_or_says_so():
    """A lane without a `_held` must SAY so (`NO_LANE_BOOK`). Otherwise "this lane has no book" is
    indistinguishable from "this lane was forgotten", which is how a lane ships blind. Today every
    lane keeps one; a future exception is named here, not discovered."""
    excluded = set(LANES) - set(BOOKED)
    assert excluded == set(), f"lanes without a lane book: {excluded}"


@pytest.mark.parametrize("name", sorted(BOOKED), ids=str)
def test_every_booked_lane_INHERITS_the_seed(name):
    assert HeldSeedMixin in BOOKED[name].__mro__, (
        f"{name} cannot repopulate `_held`, so every position predating a boot stays invisible to "
        f"`protective_close` — the stop-out goes unrecorded and the claim is never retired")


@pytest.mark.parametrize("name", sorted(BOOKED), ids=str)
def test_every_booked_lane_ACTUALLY_CALLS_the_seed_in_on_start(name):
    """A method nobody calls is the shape this repo has filed four times this month (#26, #172,
    #174, #177). AST-bound on the `on_start` each lane really runs, found through the MRO so an
    inherited one counts — and `ast.Attribute` callees are read, which is the hole
    `test_foreign_order_events.py` shipped green with."""
    cls = BOOKED[name]
    owner = next(k for k in cls.__mro__ if "on_start" in k.__dict__)
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__dict__["on_start"])))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "seed_held_from_positions" in called, (
        f"{name} inherits the seed from {owner.__name__} but {owner.__name__}.on_start never calls "
        f"it, so the method exists and the lane is still blind after every restart")


# -- the out-of-band probe -----------------------------------------------------------------------

def test_unattributed_positions_is_a_PURE_function_over_a_supplied_list():
    """Readable BEFORE the pin that contains it. `_report_unattributed` runs inside `on_start`, so
    its count only exists after a lane has booted on a revision carrying this module — which is
    after a bump, not before one. The gate has to be answerable at the bump."""
    everything = [_pos("ORPHAN", 100, strategy_id="EXTERNAL"),
                  _pos("BLANK", 100, strategy_id=None),
                  _pos("MINE", 100, strategy_id="LANE-001"),
                  _pos("FOREIGN", 100, strategy_id="OTHER-002")]
    assert unattributed_positions(everything) == ["BLANK", "ORPHAN"]


def test_the_probe_excludes_positions_already_known_to_be_ours():
    mine = [_pos("MINE", 100)]
    assert unattributed_positions(mine + [_pos("ORPHAN", 100, strategy_id="EXTERNAL")], mine) == ["ORPHAN"]


def test_a_FAILED_enumeration_RAISES_rather_than_reading_as_zero():
    """`None` is not "no unattributed positions". A caller whose enumeration failed would otherwise
    be reassured by a broken probe — the false-14-row incident, inverted."""
    with pytest.raises(ValueError, match="must not read as zero"):
        unattributed_positions(None)
    assert unattributed_positions([]) == [], "an empty book is a real answer and must not raise"


def test_the_probe_returns_NAMES_not_a_count():
    """A name can be checked against the book; a zero cannot be checked against anything."""
    got = unattributed_positions([_pos("ORPHAN", 100, strategy_id="EXTERNAL")])
    assert got == ["ORPHAN"] and not isinstance(got, int)


def test_the_probe_reads_no_connector_and_cannot_become_a_broker_call():
    """The standing repo rule: never call a broker REST API from here. AST-bound on the function's
    own source, so an import added later fails this rather than shipping quietly."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap
    tree = _ast.parse(_textwrap.dedent(_inspect.getsource(unattributed_positions)))
    names = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
    attrs = {n.attr for n in _ast.walk(tree) if isinstance(n, _ast.Attribute)}
    banned = {"requests", "httpx", "urllib", "session", "client", "get", "post", "cache", "self"}
    assert not (names | attrs) & banned, f"the probe reached for {(names | attrs) & banned}"


# -- the two shapes ibkr-paper produces routinely and paper does not -------------------------------

def test_a_TWO_LISTING_name_seeds_under_the_venue_less_symbol_the_reader_uses():
    """IBKR routinely returns one name on two listings. The engine's own resolver already put a
    CGAU stop on `(CGAU.XNAS, SELL)` while the position was `(CGAU.XNYS, SELL)`, and
    `broker_protected` read False on two covered names.

    `_held` is keyed on the VENUE-LESS symbol because `contract.protective_close` reads
    `str(event.instrument_id.symbol)`, which drops the venue. So a protective fill arriving on the
    OTHER listing of a seeded name still matches. Keying the seed on the full instrument id would
    look more precise and would silently stop matching the reader — the seed and reader disagreeing
    is #194's own shape.
    """
    host = _Host(positions=[_pos_on("CGAU", 100, venue="XNYS")])
    host.seed_held_from_positions()

    assert host._held == {"CGAU"}, "the seed carried a venue into the key the reader has none of"
    # The reader's spelling, on the OTHER listing, against the same set.
    other_listing = InstrumentId(Symbol("CGAU"), Venue("XNAS"))
    assert str(other_listing.symbol) in host._held, (
        "a protective fill on the second listing would not match the seeded name")


def test_a_SHARED_symbol_seeds_into_BOTH_lanes_each_from_its_own_leg():
    """GLD is held by BCTROT and TECHIVOL at once on ibkr-paper — 23 and 39 shares under NETTING.
    Each lane must seed its own leg and neither may adopt the other's."""
    bctrot = _Host(positions=[_pos("GLD", 23)]); bctrot.id = "BCTROT-004"
    techivol = _Host(positions=[_pos("GLD", 39)]); techivol.id = "TECHIVOL-005"

    bctrot.seed_held_from_positions()
    techivol.seed_held_from_positions()

    assert bctrot._held == {"GLD"} and techivol._held == {"GLD"}
    assert bctrot.asked["strategy_id"] == "BCTROT-004"
    assert techivol.asked["strategy_id"] == "TECHIVOL-005", (
        "a shared symbol was seeded from someone else's leg")


# -- what production actually spells, found by driving it rather than reading it -----------------

def test_the_probe_does_not_RAISE_on_a_real_attributed_StrategyId():
    """`StrategyId.__eq__` is Cython-typed: `sid == ""` raises TypeError. The first version tested
    `strategy_id in (None, "")`, which reaches that arm on EVERY attributed position — so the probe
    died on the first real book on both tenants. It passed here only because the doubles held
    plain strings."""
    real = _pos("MINE", 100, strategy_id="QC345-003")
    assert position_is_unattributed(real) is False       # must not raise
    assert unattributed_positions([real]) == []


def test_EXTERNAL_is_what_an_unattributed_position_ACTUALLY_carries():
    """Nautilus spells this as a VALUE, not an absence: a position that did not originate from a
    managed strategy carries `StrategyId("EXTERNAL")`. Nothing ever carries None, so a probe
    testing for None returns zero on every tenant forever — and reads as the gate SATISFIED."""
    ext = _pos("AEM", 10, strategy_id="EXTERNAL")
    assert position_is_unattributed(ext) is True
    assert unattributed_positions([ext]) == ["AEM"]


def test_the_probe_uses_the_VENDOR_predicate_rather_than_a_hand_rolled_sentinel():
    """`is_external()` is asked for by behaviour. `EXTERNAL_STRATEGY_ID` exists in identifiers.pyx
    but is NOT exported from the compiled module, so importing it would break on the machine this
    runs on — verified, not assumed."""
    assert StrategyId("EXTERNAL").is_external() is True
    assert StrategyId("QC345-003").is_external() is False
    with pytest.raises(ImportError):
        from nautilus_trader.model.identifiers import EXTERNAL_STRATEGY_ID  # noqa: F401


def test_a_FRACTIONAL_position_is_not_read_as_FLAT():
    """`int(0.5) == 0`, so truncating reads a fractional holding as flat and leaves it out of
    `_held` — the #194 symptom surviving the #194 fix, for the positions nobody would check.
    Alpaca supports fractional shares."""
    host = _Host(positions=[_pos("FRAC", 0.5)])
    host.seed_held_from_positions()
    assert host._held == {"FRAC"}, "a fractional holding was truncated to flat and left unseeded"

    short = _Host(positions=[_pos("FRAC", -0.5)], side=SHORT)
    short.seed_held_from_positions()
    assert short._held == {"FRAC"}


def test_a_plain_string_attribution_still_works_for_non_nautilus_callers():
    """The string fallback is deliberate: doubles and non-Nautilus callers pass bare values, and the
    probe must not require a Cython type to answer."""
    assert position_is_unattributed(SimpleNamespace(strategy_id="EXTERNAL")) is True
    assert position_is_unattributed(SimpleNamespace(strategy_id="")) is True
    assert position_is_unattributed(SimpleNamespace(strategy_id="QC345-003")) is False
    assert position_is_unattributed(SimpleNamespace()) is True


def test_the_VENDOR_PREDICATE_is_consulted_not_merely_the_spelling():
    """The two paths must be separable, or one of them is unearned.

    `str(StrategyId("EXTERNAL")) == "EXTERNAL"`, so today the string fallback answers every real
    case correctly and removing `is_external()` changes no test — which is an insufficient bite
    hiding a real question: WHICH path is load-bearing? The vendor predicate is, because it follows
    Nautilus if the sentinel is ever respelled and the hardcoded string does not. An id that answers
    `is_external()` True while spelling itself something else separates them.
    """
    class _Respelled:
        def is_external(self): return True
        def __str__(self): return "NOT-THE-LITERAL-EXTERNAL"

    assert position_is_unattributed(SimpleNamespace(strategy_id=_Respelled())) is True, (
        "the probe read the spelling and ignored the vendor's own predicate")


def test_a_strategy_id_whose_predicate_RAISES_falls_back_rather_than_dying():
    """`is_external()` raising must not take the probe down — it runs inside `on_start` via
    `_report_unattributed`, and the whole module's rule is that the notice never costs the act."""
    class _Angry:
        def is_external(self): raise RuntimeError("boom")
        def __str__(self): return "EXTERNAL"

    assert position_is_unattributed(SimpleNamespace(strategy_id=_Angry())) is True
