"""The broker must not hold its own copy of the resolved ids (kumo-trading-platform issue 622).

`NautilusBroker(instrument_ids=...)` takes the SAME list the strategy takes, built by the same
`_instrument_ids(symbols)` call in cockpit. Two derivations of one fact, and the rule this repo keeps
relearning is that two derivations drift.

It also carries the same defect as the strategy's old signature: an id demands a venue at
CONSTRUCTION, which only a connected adapter can supply — so cockpit resolved it from Alpaca's asset
list and an IBKR-only instance could not build a lane without an Alpaca key.

The strategy resolves at `on_start` and holds `_iids`. The broker already holds `strategy`
(`momentum.py:724  broker.strategy = strategy`). So it should ASK, not copy.
"""

from __future__ import annotations

from types import SimpleNamespace

from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.nautilus.broker import NautilusBroker


def test_the_fixture_can_tell_the_two_sources_apart():
    """FIXTURE PROPERTY. If the broker's own list and the strategy's list were equal, a broker that
    ignored the strategy would pass — the classic agreement-is-not-connection blind spot."""
    assert InstrumentId.from_str("AAA.XNAS") != InstrumentId.from_str("BBB.XNYS")


def test_it_reads_the_STRATEGYS_resolved_ids_when_given_none():
    """THE FIX. Resolution happens once, at `on_start`, in the strategy. The broker asks."""
    iid = InstrumentId.from_str("RDN.XNYS")
    b = NautilusBroker(strategy=SimpleNamespace(_iids=[iid]), instrument_ids=None)
    assert b._iid("RDN") == iid


def test_a_symbol_the_strategy_could_not_resolve_is_REFUSED_not_guessed():
    """The drop must survive the hop. A symbol the venue has no instrument for must fail at submit
    with a clear message, not be guessed into an id that matches nothing in the cache — which is
    silent: no bars, no fills, and nothing that looks like an error."""
    b = NautilusBroker(strategy=SimpleNamespace(_iids=[]), instrument_ids=None)
    assert b._iid("JEPO") is None


def test_an_EXPLICIT_list_still_wins_so_the_pin_can_move_in_one_step():
    """ADDITIVE. Both tenants run this on a pinned ref; paper keeps using the id path until cockpit
    has moved, so the pin bump alone changes nothing on the tenant that is actually trading."""
    iid = InstrumentId.from_str("AAA.XNAS")
    other = InstrumentId.from_str("BBB.XNYS")
    b = NautilusBroker(strategy=SimpleNamespace(_iids=[other]), instrument_ids=[iid])
    assert b._iid("AAA") == iid
    assert b._iid("BBB") is None, "the explicit list must be authoritative when supplied"


def test_it_does_not_crash_before_the_strategy_is_attached():
    """`broker.strategy` is None between construction and `momentum.py:724`. Anything reaching the
    broker in that window must get a refusal, not an AttributeError that takes the build down."""
    b = NautilusBroker(strategy=None, instrument_ids=None)
    assert b._iid("AAA") is None
