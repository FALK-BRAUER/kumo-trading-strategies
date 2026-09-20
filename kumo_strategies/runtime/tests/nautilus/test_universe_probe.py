"""A lane trading a fraction of its pool must not report identical to a healthy one (#80).

THREE FACTS WERE RECORDED AND READ BY NOTHING IN-PROCESS:

    _unresolved_symbols     symbols with no instrument on this venue — dropped
    _ambiguous_symbols      symbols listed on SEVERAL venues — one identity picked (kumo-trading-platform issue 625)
    an empty pool           `symbols=[]` is accepted and then skipped as falsy

A lane that resolved 60 of 97 symbols reported `armed=True` with every other probe answering, and
only a log line said otherwise — and `self.log` is Nautilus's logger, which nothing in-process can
read back. 64 symbols carry two venue identities on staging and it took a screenshot to notice.

WHY COUNTS RATHER THAN A VERDICT. `Probe` deliberately carries no pass/fail: the strategy observes
and the platform judges. So this reports numbers a reader can check against each other —

    requested == resolved + len(unresolved)

— which is a detector rather than a description. Two derivations of one fact: if any path stops
recording, the arithmetic disagrees and says so. `ambiguous` is a SUBSET of resolved (those symbols
DID resolve, to an identity chosen among several) and is reported separately rather than folded into
either number.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Lane(RegistrationMixin):
    """Only the state the probe reads. An ENVIRONMENT double: it fakes what production does not own
    and runs `preflight` and `_universe_probe` for real.

    `is_armed` is a read-only property on the mixin, so it is overridden here rather than assigned —
    assigning raises, and a fixture that cannot construct the object proves nothing about it.
    """

    is_armed = True
    _armed_session = None

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def _ids(*specs):
    return [InstrumentId.from_str(s) for s in specs]


def _probe_of(lane):
    """The probe as `preflight` emits it, not as the helper returns it.

    Driving the real `preflight` is the point: a helper that returns the right dict while nothing
    calls it is the defect class this whole ticket is about.
    """
    broker = SimpleNamespace(
        equity=lambda: 100_000.0,
        strategy_positions=lambda: {},
        last_price=lambda s, **k: 1.0)
    probes = {p.name: p for p in lane.preflight(broker)}
    return probes


# ==================================================================================================
# IT IS EMITTED, ALWAYS
# ==================================================================================================

def test_the_probe_is_emitted_at_all():
    """The helper existing is not the same as the probe being emitted — that gap IS the ticket."""
    p = _probe_of(_Lane(_symbols=["AAA"], _iids=_ids("AAA.XNAS")))
    assert "universe" in p, "the universe probe is not emitted by preflight"


def test_it_is_emitted_even_when_there_is_NOTHING_to_report():
    """An omitted probe reads as "we never checked" and is indistinguishable from "not applicable".
    Absence reported as an all-clear is what this preflight exists to end."""
    p = _probe_of(_Lane(_iids=_ids("AAA.XNAS")))          # ids path: nothing was resolved
    assert "universe" in p
    assert p["universe"].error is None
    assert p["universe"].value["source"] == "instrument_ids"


def test_it_is_emitted_for_a_lane_that_has_resolved_NOTHING_yet():
    """Before `on_start`, `_iids` is legitimately empty on the symbols path. That is a real state and
    a useful one to see — requested 97, resolved 0 is a lane that has not started, not a healthy
    one."""
    p = _probe_of(_Lane(_symbols=["AAA", "BBB"], _iids=[]))
    assert p["universe"].value == {
        "source": "symbols", "requested": 2, "resolved": 0, "unresolved": [], "ambiguous": {},
        "with_bars": 0, "no_bars": [], "below_warmup": []}


# ==================================================================================================
# THE THREE FACTS
# ==================================================================================================

def test_a_partially_resolved_pool_is_VISIBLE():
    """THE HEADLINE. 60 of 97 and 97 of 97 must not read the same."""
    lane = _Lane(_symbols=["AAA", "BBB", "CCC"], _iids=_ids("AAA.XNAS", "BBB.XNAS"),
                 _unresolved_symbols=["CCC"])
    v = _probe_of(lane)["universe"].value
    assert v["requested"] == 3 and v["resolved"] == 2
    assert v["unresolved"] == ["CCC"]


def test_an_AMBIGUOUS_symbol_is_reported_even_though_it_RESOLVED():
    """It resolved — to one identity chosen among several. A correct pick and no pick are equally
    silent otherwise, and 64 symbols carrying two identities is a fact about the venue an operator
    should be able to read back."""
    lane = _Lane(_symbols=["SPY"], _iids=_ids("SPY.ARCX"),
                 _ambiguous_symbols={"SPY": ["SPY.ARCX", "SPY.XNAS"]})
    v = _probe_of(lane)["universe"].value
    assert v["resolved"] == 1, "an ambiguous symbol did resolve and must count as resolved"
    assert v["ambiguous"] == {"SPY": ["SPY.ARCX", "SPY.XNAS"]}


def test_an_EMPTY_pool_is_visible():
    """`symbols=[]` is accepted and then skipped as falsy, so the zero-resolved error never fires.
    The lane is a no-op that reports armed."""
    v = _probe_of(_Lane(_symbols=[], _iids=[]))["universe"].value
    assert v["requested"] == 0 and v["resolved"] == 0


# ==================================================================================================
# THE INVARIANT — this is the part that is a detector rather than a description
# ==================================================================================================

@pytest.mark.parametrize("lane", [
    _Lane(_symbols=["A", "B", "C"], _iids=_ids("A.XNAS", "B.XNAS"), _unresolved_symbols=["C"]),
    _Lane(_symbols=["A"], _iids=_ids("A.XNAS"), _ambiguous_symbols={"A": ["A.XNAS", "A.ARCX"]}),
    _Lane(_symbols=["A", "B"], _iids=[], _unresolved_symbols=["A", "B"]),
    _Lane(_iids=_ids("A.XNAS", "B.XNAS")),
], ids=["partial", "ambiguous", "none-resolved", "ids-path"])
def test_requested_equals_resolved_plus_unresolved(lane):
    """The relationship a reader can check. If any path stops recording, this disagrees.

    It holds on BOTH construction paths, which is why `source` exists: on the ids path `requested`
    counts the ids handed in, so `requested == resolved` and the arithmetic still closes. Without
    that, `requested=0, resolved=164` would look like a broken invariant rather than a different
    question.
    """
    v = _probe_of(lane)["universe"].value
    assert v["requested"] == v["resolved"] + len(v["unresolved"]), v


def test_ambiguous_is_a_SUBSET_of_resolved_not_a_third_bucket():
    """Folding ambiguity into `unresolved` would say the lane cannot trade those symbols, which is
    false — it can, on an identity we chose. Folding it into nothing would lose it entirely."""
    lane = _Lane(_symbols=["A", "B"], _iids=_ids("A.XNAS", "B.XNAS"),
                 _ambiguous_symbols={"B": ["B.XNAS", "B.ARCX"]})
    v = _probe_of(lane)["universe"].value
    assert v["requested"] == v["resolved"] + len(v["unresolved"])
    assert set(v["ambiguous"]) <= {"A", "B"}
    assert v["unresolved"] == []


# ==================================================================================================
# EVERY LANE, DISCOVERED
# ==================================================================================================

def test_the_probe_carries_no_verdict():
    """`Probe` has no ok/passed/status by design — the strategy observes, the platform judges. A
    probe that graded itself would report healthy right up until the thing deciding health is the
    thing that is broken."""
    v = _probe_of(_Lane(_symbols=["A"], _iids=_ids("A.XNAS")))["universe"].value
    for forbidden in ("ok", "passed", "status", "healthy", "degraded"):
        assert forbidden not in v, f"the probe grades itself via `{forbidden}`"


def test_every_lane_inherits_the_probe_by_existing():
    """Derived, not listed — the failure `test_contract` was written about, where three suites
    claimed "every strategy" and named their strategies by hand."""
    import inspect

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    lanes = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, Strategy)
                    and getattr(obj, "EXTERNAL_ID", None)):
                lanes.append(obj)
    assert lanes, "no lanes discovered — this assertion no longer describes the package"
    for cls in lanes:
        assert callable(getattr(cls, "_universe_probe", None)), (
            f"{cls.__name__} cannot report its universe, so a partially resolved pool on that lane "
            f"is invisible")


# ==================================================================================================
# DECLARED AND EMITTED MUST BE THE SAME SET — the gap that let `universe` ship undeclared
# ==================================================================================================

def test_every_DECLARED_probe_is_actually_EMITTED():
    """`PREFLIGHT_PROBES` is what cockpit's contract requires of us. Nothing bound it to what
    `preflight` actually returns, so the two could drift in either direction — and they did:
    `universe` was emitted for a whole release without being declared, so a lane that quietly
    stopped emitting it would still have satisfied the tuple.

    Emitted-but-undeclared and written-but-unread are the same family: a mechanism nothing is
    holding to account. Caught by cockpit's cross-repo check, which is the only reason it surfaced.
    """
    emitted = set(_probe_of(_Lane(_symbols=["AAA"], _iids=_ids("AAA.XNAS"))))
    declared = set(RegistrationMixin.PREFLIGHT_PROBES)
    assert declared - emitted == set(), (
        f"declared but never emitted: {sorted(declared - emitted)}. Cockpit requires these and no "
        f"lane produces them, so the requirement is enforced by nobody.")


def test_every_EMITTED_probe_is_DECLARED():
    """The other direction, and the one that actually happened. An undeclared probe is invisible to
    the contract: cockpit cannot require it, so nothing notices when a lane stops sending it."""
    emitted = set(_probe_of(_Lane(_symbols=["AAA"], _iids=_ids("AAA.XNAS"))))
    declared = set(RegistrationMixin.PREFLIGHT_PROBES)
    assert emitted - declared == set(), (
        f"emitted but never declared: {sorted(emitted - declared)}. Add them to PREFLIGHT_PROBES, "
        f"or a lane that stops emitting them still passes the contract.")


def test_the_two_sets_are_not_both_empty():
    """VACUITY GUARD. Both assertions above are satisfied by a lane that declares nothing and emits
    nothing — which is exactly the state a broken `preflight` would produce, and it would read as
    two passing contract tests."""
    assert RegistrationMixin.PREFLIGHT_PROBES, "nothing is declared; the checks above are vacuous"
    assert _probe_of(_Lane(_symbols=["AAA"], _iids=_ids("AAA.XNAS"))), "nothing is emitted"


# ==================================================================================================
# RESOLUTION IS NOT USABILITY — the gap the first version of this probe had
# ==================================================================================================

class _Bars(list):
    """A stand-in for the adapter's `deque` of bars. Length is the only property read."""


def _lane_with_bars(watched, bars_by_symbol, need=None):
    """`warmup_bars` is a read-only property on the real lanes, so it is overridden on a subclass
    rather than assigned — assigning raises, and a fixture that cannot be built proves nothing."""
    cls = _Lane if need is None else type("_LaneNeeding", (_Lane,), {"warmup_bars": need})
    return cls(_symbols=list(watched),
               _iids=_ids(*[f"{s}.XNAS" for s in watched]),
               _bars={k: _Bars(range(v)) for k, v in bars_by_symbol.items()})


def test_a_symbol_that_RESOLVED_but_got_NO_BARS_is_visible():
    """MEASURED, ibkr-paper-retired 2026-08-28 13:05 ET — BCTROT-004's first IBKR decision.

        this probe said        requested 109, resolved 109, unresolved []
        the journal said       bars missing for 24/109 — refusing to rank

    On the one night the distinction mattered, the detector written to make a degraded lane look
    degraded would have reported it perfectly healthy. A symbol can resolve, be subscribed, and
    never receive a single bar — FTNR sat in the pool a whole session exactly that way.
    """
    lane = _lane_with_bars(["AAA", "BBB", "CCC"], {"AAA": 300, "BBB": 300})
    v = _probe_of(lane)["universe"].value
    assert v["resolved"] == 3, "all three resolved — that is the point"
    assert v["unresolved"] == [], "none of them FAILED to resolve"
    assert v["with_bars"] == 2
    assert v["no_bars"] == ["CCC"], "the symbol with no data is invisible"


def test_the_probe_now_PREDICTS_a_coverage_refusal():
    """The property that makes this worth having: `min_bar_coverage` is enforced in `pgrunner` from
    the same dict, so the probe and the floor must agree about who has data.

    78% against an 80% floor is the exact staging case."""
    watched = [f"S{i}" for i in range(109)]
    have = {s: 300 for s in watched[:85]}          # 85/109 = 78%
    v = _probe_of(_lane_with_bars(watched, have))["universe"].value
    coverage = v["with_bars"] / v["resolved"]
    assert round(coverage, 2) == 0.78
    assert len(v["no_bars"]) == 24


def test_BELOW_WARMUP_is_separate_from_NO_BARS():
    """Different facts and different fixes: no bars at all is usually a venue that has never heard
    of the symbol, while thin bars is warmup still running. `_report_dataless` already draws this
    line; the probe was collapsing it."""
    lane = _lane_with_bars(["AAA", "BBB", "CCC"], {"AAA": 300, "BBB": 5}, need=200)
    v = _probe_of(lane)["universe"].value
    assert v["no_bars"] == ["CCC"]
    assert v["below_warmup"] == ["BBB"]
    assert v["with_bars"] == 2, "a thin symbol still HAS bars"


def test_a_lane_with_no_bars_dict_at_all_still_probes():
    """Research adapters and a lane before `on_start` have no `_bars`. Absent must not raise — a
    probe that throws is worse than one that under-reports."""
    v = _probe_of(_Lane(_symbols=["AAA"], _iids=_ids("AAA.XNAS")))["universe"].value
    assert v["no_bars"] == ["AAA"] and v["with_bars"] == 0


def test_the_resolution_invariant_still_holds_with_bars_added():
    """Adding fields must not break the relationship the probe is checked by."""
    lane = _lane_with_bars(["AAA", "BBB"], {"AAA": 300})
    v = _probe_of(lane)["universe"].value
    assert v["requested"] == v["resolved"] + len(v["unresolved"])
