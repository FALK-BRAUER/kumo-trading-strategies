"""A strategy takes SYMBOLS and resolves venues through Nautilus (kumo-trading-platform issue 622).

THE DEFECT THIS INTERFACE CAUSED, measured twice on ibkr-paper-retired (2026-08-27 18:38 SGT and
2026-08-28 06:20 SGT):

    RuntimeError: APCA_API_KEY_ID / APCA_API_SECRET_KEY not set
      kumo_strategies/runtime/executor/tradable.py:45   symbols()
      strategies/momentum.py:309                        _instrument_ids()
      api/engine_node.py:6657                           build_node()

13 restarts, `/positions` served [] against 22 non-flat positions. An IBKR-only instance could not
construct a single strategy without an Alpaca API key.

THE SIGNATURE IS WHY. `instrument_ids: list[InstrumentId]` demands *symbol + venue* at CONSTRUCTION,
and the venue half is a runtime fact only a connected adapter can produce. Cockpit met that
obligation the only way it could offline — by importing Alpaca's asset list into a strategy. That was
not carelessness; nothing else can satisfy the signature before a connection exists.

A SYMBOL is what this layer actually knows. The pool is a research artifact: "trade these ~97 names".
It has no opinion about XNAS vs XNYS and must not be forced to have one.

WHY on_start IS THE RIGHT PLACE, verified in the installed nautilus_trader rather than assumed:

    system/kernel.py:1022   self._connect_clients()
    system/kernel.py:1024   await self._await_engines_connected()   <- awaits connect
    system/kernel.py:1039   self._trader.start()                    <- on_start runs here

    adapters/interactive_brokers/data.py:147
        await self.instrument_provider.initialize()
        for instrument in self._instrument_provider.list_all():
            self._handle_data(instrument)                           <- into the Cache

So by `on_start` every declared instrument is in the Cache — on a COLD start, with no durable cache
and no vendor call. Resolving at construction cannot work; resolving here always can.
"""

from __future__ import annotations

import pytest

from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig


def _strategy(**kw) -> MomentumRotationStrategy:
    return MomentumRotationStrategy(
        cfg=kw.pop("cfg", MomentumRotationConfig()),
        source=kw.pop("source", StaticList(["AAA", "BBB"])),
        **kw,
    )


def _cache(mapping: dict[str, str]):
    """A Cache stand-in returning REAL `InstrumentId` objects, as the real one does.

    Not strings. A resolver that assumes strings must fail here rather than in production — this repo
    has shipped defects behind doubles that could not represent what production emits.
    """
    from nautilus_trader.model.identifiers import InstrumentId

    ids = [InstrumentId.from_str(f"{s}.{v}") for s, v in mapping.items()]

    class _C:
        def instrument_ids(self):
            return list(ids)

    return _C()


#: What IBKR actually resolved on staging, read off the durable cache. NYSE names included
#: deliberately: an all-NASDAQ fixture cannot tell a correct resolver from one hardcoding XNAS, which
#: is the original bug `_instrument_ids` was written to fix.
_VENUES = {"QQQ": "XNAS", "IWM": "XNAS", "SPY": "ARCX", "RDN": "XNYS", "BMO": "XNYS"}


def test_the_fixture_could_catch_a_hardcoded_venue():
    """FIXTURE PROPERTY FIRST. All-NASDAQ symbols would let a resolver that ignores the cache and
    hardcodes XNAS pass every assertion below."""
    assert len(set(_VENUES.values())) >= 3
    assert any(v != "XNAS" for v in _VENUES.values())


def test_a_strategy_can_be_constructed_from_SYMBOLS_ALONE():
    """THE HEADLINE. No venue, no adapter, no credential — construction must not require a runtime
    fact. This is what an IBKR-only instance needs in order to boot at all."""
    s = _strategy(symbols=["QQQ", "SPY"])
    assert s is not None


def test_construction_does_NOT_resolve_anything():
    """Resolution at construction is the bug, whatever its source. Nothing may be resolved before a
    connection exists — not from a vendor, not from a cache, not from a guess."""
    s = _strategy(symbols=["QQQ", "SPY"])
    assert not getattr(s, "_iids", []), (
        "instrument ids were resolved at construction, before any adapter connected (#622)"
    )


def test_on_start_resolves_the_venue_from_the_CACHE():
    """THE FIX. By `on_start` the adapter has loaded its instruments (kernel.py:1024 awaits connect
    before :1039 starts the trader), so the venue is a fact the Cache can answer for any broker."""
    s = _strategy(symbols=list(_VENUES))
    s._resolve_symbols(_cache(_VENUES))
    assert {str(i) for i in s._iids} == {f"{k}.{v}" for k, v in _VENUES.items()}


def test_a_NYSE_name_does_not_become_XNAS():
    """The original defect, restated against the new source. `BAC.XNAS` matches no instrument the
    provider loaded, so bars never arrive, warmup never completes, and any order is refused — while
    the strategy looks healthy."""
    s = _strategy(symbols=["RDN"])
    s._resolve_symbols(_cache(_VENUES))
    assert str(s._iids[0]) == "RDN.XNYS"


def test_an_UNRESOLVABLE_symbol_is_DROPPED_and_NAMED():
    """Behaviour that must survive, load-bearing in both directions. Defaulting was the original bug.
    Raising took the whole node down for one bad symbol: `JEPO`, a vision misread of JEPQ,
    crash-looped the engine and with it every other strategy and the entire UI feed. So: drop, and
    say which."""
    s = _strategy(symbols=["QQQ", "JEPO"])
    s._resolve_symbols(_cache(_VENUES))
    assert {str(i) for i in s._iids} == {"QQQ.XNAS"}
    # INSPECTABLE, which is the part a health frame or an operator can actually read back.
    #
    # Two earlier versions of this assertion were wrong and both are worth recording. `caplog.text`
    # captures nothing, because `self.log` is NAUTILUS's logger rather than Python's — it failed
    # against a working implementation. Spying on `s.log` then raised `attribute 'log' of Actor
    # objects is not writable`: it is a read-only Cython attribute. A message that can only be
    # observed by reading stdout is how #613's inert node stayed invisible for 26 minutes; state that
    # can be read back is the fix, and the log line is checked separately below.
    assert s._unresolved_symbols == ["JEPO"]


def test_resolving_NOTHING_does_not_take_the_node_down():
    """A pool that resolves to nothing is a broken LANE, not a broken node. Today the equivalent
    failure raises inside `build_node` and kills every other strategy and the whole UI feed with it —
    which is precisely what #622 is about."""
    s = _strategy(symbols=["AAA", "BBB"])
    s._resolve_symbols(_cache(_VENUES))          # must NOT raise — a broken lane is not a broken node
    assert s._iids == []
    assert s._unresolved_symbols == ["AAA", "BBB"]


def test_INSTRUMENT_IDS_still_work_so_the_pin_can_move_in_one_step():
    """ADDITIVE, deliberately. Both tenants run this package on a pinned ref; changing the signature
    outright would break the tenant that is actually trading. Ids keep working until cockpit has
    moved, then they go."""
    from nautilus_trader.model.identifiers import InstrumentId

    iid = InstrumentId.from_str("QQQ.XNAS")
    s = _strategy(instrument_ids=[iid])
    assert s._iids == [iid]


@pytest.mark.parametrize("kw", [{}, {"symbols": ["QQQ"], "instrument_ids": []}])
def test_supplying_BOTH_or_NEITHER_raises(kw):
    """A default that silently picks one is how a half-migrated caller looks healthy — the exact
    shape of every defect in this ticket's family. Exactly one must be given."""
    from nautilus_trader.model.identifiers import InstrumentId

    if "instrument_ids" in kw:
        kw["instrument_ids"] = [InstrumentId.from_str("QQQ.XNAS")]
    with pytest.raises((ValueError, TypeError)):
        _strategy(**kw)


def test_BOTH_failures_are_REPORTED_not_only_recorded():
    """The operator half. `_unresolved_symbols` is what a process can read back; a log line is what a
    human sees at 3am, and a lane that quietly trades a fraction of its pool is the failure mode this
    whole ticket exists to end.

    Asserted against the SOURCE with docstrings stripped, because `Actor.log` is a read-only Cython
    attribute and cannot be spied on — and because a docstring mentioning the message would otherwise
    satisfy a raw grep, a trap this codebase has already fallen into.
    """
    import ast
    import inspect
    import textwrap

    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

    tree = ast.parse(textwrap.dedent(inspect.getsource(MomentumRotationStrategy._resolve_symbols)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    src = ast.unparse(fn)
    assert src.count("self.log.error") >= 2, (
        "the dropped-symbols case and the nothing-resolved case are not both reported; a lane "
        "trading a fraction of its pool must not look identical to a healthy one"
    )
    assert "raise" not in src, (
        "resolution raises — one bad pool symbol would take the node down with every other strategy "
        "and the whole UI feed, which is the JEPO outage (#622)"
    )
