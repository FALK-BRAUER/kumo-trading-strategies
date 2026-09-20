"""A symbol the venue lists TWICE must not resolve by cache iteration order (kumo-trading-platform issue 625).

MEASURED ON AN IBKR PAPER INSTANCE 2026-08-28, after #622 made `load_contracts` declare the full universe:
**64 symbols carry two instrument identities**, every pair a real venue plus a spurious XNAS.

    SPY   ARCX + XNAS        XLV   ARCX + XNAS
    RVTY  XNYS + XNAS        ARKK  BATS + XNAS
    UBS   XNYS + XNAS        EHC   XNYS + XNAS

IB is asked with `exchange="SMART"` and answers with contract details per listing, so Nautilus
creates an instrument per contract. That is IB being correct. The defect was entirely what we did
next — a dict comprehension:

    by_symbol = {i.symbol.value: i for i in cache.instrument_ids()}

which keeps whichever the cache iterated LAST. Arbitrary, and not stable across boots. It cost real
requests: BCTROT asked for `SPY.XNAS`, an instrument with no data, and the venue answered with an
empty array and no error.

THE TWIN IS SELF-IDENTIFYING, measured off staging's real cached instruments:

    SPY.ARCX    primaryExchange ARCA   venue ARCX    <- venue matches its own primary
    SPY.XNAS    primaryExchange ARCA   venue XNAS    <- does not
    RVTY.XNYS   primaryExchange NYSE   venue XNYS    <- matches
    RVTY.XNAS   primaryExchange NYSE   venue XNAS    <- does not

So the rule is "keep the candidate whose venue equals its own primaryExchange, mapped to a MIC".

WHY THAT RULE IS NOT IMPLEMENTED HERE. The mapping is `exchange_to_mic_venue`, in
`nautilus_trader.adapters.interactive_brokers`, which imports `ibapi` — not installed in this repo
and it must not be. `"ARCA"` is not `"ARCX"`, so a naive `primaryExchange == venue` compare matches
NOTHING on any real pair, falls through on every symbol, and looks exactly like a working preference
rule. And teaching this module IB's exchange names would re-create #622 inside the resolver written
to fix it.

The layer that knows the RULE is not the layer that knows the VENDOR. Cockpit injects `prefer`.
"""

from __future__ import annotations

from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin


class _Log:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, m):
        self.errors.append(str(m))

    def warning(self, m):
        self.warnings.append(str(m))

    def info(self, m):
        pass


class _Host(SymbolResolutionMixin):
    """The narrowest host carrying the REAL function.

    `log` is a read-only Cython attribute on Nautilus's Actor and cannot be spied on an instance, so
    the production function is bound to a plain object instead of loosening a fake.
    """

    id = "TESTLANE-999"

    def __init__(self, symbols):
        self.log = _Log()
        self._init_symbols(None, list(symbols))


class _Cache:
    """Returns ids in the order given, so iteration order is a variable the tests can move."""

    def __init__(self, ids):
        self._ids = list(ids)

    def instrument_ids(self):
        return list(self._ids)


def _ids(*specs):
    return [InstrumentId.from_str(s) for s in specs]


#: Real pairs off staging, with the primary each twin actually reports.
_PRIMARY = {
    "SPY.ARCX": "ARCA", "SPY.XNAS": "ARCA",
    "RVTY.XNYS": "NYSE", "RVTY.XNAS": "NYSE",
    "XLV.ARCX": "ARCA", "XLV.XNAS": "ARCA",
}
#: What cockpit's `exchange_to_mic_venue` does, restated here ONLY so a test can stand in for it.
#: Deliberately not imported: that module needs `ibapi`, which is the whole reason `prefer` is
#: injected rather than implemented here.
_MIC = {"ARCA": "ARCX", "NYSE": "XNYS", "NASDAQ": "XNAS", "BATS": "BATS"}


def _prefer_primary(sym, candidates):
    """Cockpit's rule, standing in: keep the identity whose venue IS its own primary listing."""
    for iid in candidates:
        primary = _PRIMARY.get(str(iid))
        if primary and _MIC.get(primary) == iid.venue.value:
            return iid
    return None


def test_the_fixture_can_tell_a_correct_pick_from_the_old_one():
    """FIXTURE PROPERTY FIRST, and it is load-bearing here.

    The old code kept whichever id came LAST. If every fixture listed the correct venue last, the
    broken implementation would pass every assertion below. So the spurious XNAS is listed LAST on
    purpose — the position the old bug would have chosen.
    """
    ids = _ids("SPY.ARCX", "SPY.XNAS")
    assert str(ids[-1]) == "SPY.XNAS", "the wrong id must be last, or the old bug survives this file"
    assert _prefer_primary("SPY", ids).venue.value == "ARCX"


def test_a_duplicate_no_longer_resolves_by_ITERATION_ORDER():
    """THE DEFECT. Same candidates, two cache orderings, one answer."""
    host_a, host_b = _Host(["SPY"]), _Host(["SPY"])
    host_a._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")))
    host_b._resolve_symbols(_Cache(_ids("SPY.XNAS", "SPY.ARCX")))
    assert host_a._iids == host_b._iids, (
        "the pick moved with cache order — a restart would silently re-point a held position, and "
        "instrument ids are what reconciliation keys on")


def test_the_INJECTED_preference_chooses_the_primary_listing():
    """THE RULE, supplied by cockpit because the MIC mapping lives in the IB adapter."""
    host = _Host(["SPY", "RVTY", "XLV"])
    host._resolve_symbols(
        _Cache(_ids("SPY.ARCX", "SPY.XNAS", "RVTY.XNYS", "RVTY.XNAS", "XLV.ARCX", "XLV.XNAS")),
        prefer=_prefer_primary)
    assert {str(i) for i in host._iids} == {"SPY.ARCX", "RVTY.XNYS", "XLV.ARCX"}


def test_the_preference_beats_the_fallback_where_they_DISAGREE():
    """Otherwise the preference could be ignored entirely and this file would still pass.

    `sorted()[0]` gives ARCX for SPY by luck and XNAS for RVTY by BAD luck — RVTY is the case that
    proves the preference is doing the work.
    """
    ids = _ids("RVTY.XNYS", "RVTY.XNAS")
    assert str(sorted(ids, key=str)[0]) == "RVTY.XNAS", "fallback and preference must differ here"

    host = _Host(["RVTY"])
    host._resolve_symbols(_Cache(ids), prefer=_prefer_primary)
    assert str(host._iids[0]) == "RVTY.XNYS"


def test_ambiguity_is_RECORDED_even_when_it_is_resolved_well():
    """A correct pick and no pick are equally silent otherwise. 64 symbols carrying two identities
    is a fact about the venue an operator should read back, not something absorbed quietly."""
    host = _Host(["SPY"])
    host._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")), prefer=_prefer_primary)
    assert host._ambiguous_symbols == {"SPY": ["SPY.ARCX", "SPY.XNAS"]}
    assert host.log.warnings, "an ambiguity resolved correctly still went unreported"


def test_an_UNAMBIGUOUS_symbol_records_nothing():
    """The record must mean something. If every symbol appeared in it, it would say nothing."""
    host = _Host(["AAA"])
    host._resolve_symbols(_Cache(_ids("AAA.XNAS")))
    assert host._ambiguous_symbols == {}
    assert not host.log.warnings


def test_no_preference_falls_back_DETERMINISTICALLY_and_says_so():
    """The fallback's only job is that the wrong answer is the SAME wrong answer every boot:
    stable-wrong is recoverable, unstable-right-looking is not."""
    host = _Host(["RVTY"])
    host._resolve_symbols(_Cache(_ids("RVTY.XNYS", "RVTY.XNAS")))
    assert str(host._iids[0]) == "RVTY.XNAS", "not the deterministic pick"
    assert host._ambiguous_symbols == {"RVTY": ["RVTY.XNAS", "RVTY.XNYS"]}


def test_a_preference_that_RAISES_is_reported_not_swallowed():
    """A preference caught quietly degrades to the arbitrary pick while looking principled — and the
    failure is then invisible precisely because a plausible answer still comes out."""
    def _boom(sym, candidates):
        raise KeyError("no contract details cached")

    host = _Host(["SPY"])
    host._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")), prefer=_boom)
    assert host._iids, "one bad preference dropped the symbol entirely"
    assert any("preference raised" in m for m in host.log.errors)


def test_a_preference_that_CANNOT_TELL_still_leaves_a_trace():
    """`None` is a real answer — "I cannot determine the primary" — and must not pass for a
    decision. This is the case cockpit hits when contract details are absent."""
    host = _Host(["SPY"])
    host._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")), prefer=lambda s, c: None)
    assert str(host._iids[0]) == "SPY.ARCX"                    # deterministic fallback
    assert any("did not choose" in m for m in host.log.errors)


def test_a_preference_returning_a_FOREIGN_id_is_refused():
    """A preference must choose among the candidates, not invent one. Returning an id the venue
    never listed would put an untradeable instrument on the order path."""
    host = _Host(["SPY"])
    host._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")),
                          prefer=lambda s, c: InstrumentId.from_str("SPY.XLON"))
    assert str(host._iids[0]) == "SPY.ARCX"
    assert any("did not choose" in m for m in host.log.errors)


def test_missing_and_ambiguous_are_DIFFERENT_states():
    """Both are degradation and they need different answers: one symbol cannot be traded at all, the
    other can but on an identity we picked. Collapsing them loses which."""
    host = _Host(["SPY", "NOSUCH"])
    host._resolve_symbols(_Cache(_ids("SPY.ARCX", "SPY.XNAS")), prefer=_prefer_primary)
    assert host._unresolved_symbols == ["NOSUCH"]
    assert list(host._ambiguous_symbols) == ["SPY"]
    assert len(host._iids) == 1
