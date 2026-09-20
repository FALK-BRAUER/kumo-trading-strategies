"""Resolving a symbol is not enough — every VIEW of `_iids` must move with it (kumo-trading-platform issue 622).

`SymbolResolutionMixin` rebinds `self._iids` at `on_start`. Any view of that list built at
CONSTRUCTION was built from the empty list, and rebinding the source does not touch a copy taken from
it. The lane then resolves its whole pool correctly, logs nothing, reports `armed`, and can neither
enter nor exit a single position.

MEASURED ON THIS BRANCH, before the fix — QC27RotationStrategy, the shape that motivated the file:

    qc27_rotation.py:122   self._by_symbol = {i.symbol.value: i for i in self._iids}   # {} on the
                                                                                      # symbols path
    qc27_rotation.py:437   iid = self._by_symbol.get(symbol)
                           if iid is None ...: return          <- never enters
    qc27_rotation.py:455   iid = self._by_symbol.get(symbol)
                           if iid is None: return              <- never EXITS

Silent in both directions, which is worse than the boot crash #622 set out to fix: a node that will
not start gets looked at within minutes, and TECHIVOL-005 sat armed with zero allocation for days
without anyone noticing.

WHY THIS IS DISCOVERED, NOT LISTED. `_by_symbol` is one instance of "a second derivation of `_iids`
taken at the wrong time", and naming it would leave the next one uncovered — the failure this
package's own `test_contract` was written about, where three suites claimed "every strategy" and
named their strategies by hand. So the assertion walks instance state and finds every container of
`InstrumentId`, whatever it is called, and requires it to agree with `_iids`.

Two derivations of one fact are a detector: when they should match and do not, that is a live defect.
"""

from __future__ import annotations

from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig

#: `TemplateConfig.market_view` is REQUIRED (#212): a lane states its view or cannot construct.
#: The number here is a test's, measured for nothing; the template's docstring says what a real
#: lane owes.
_A_MEASURED_VIEW = MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                                    action=MarketAction.EXIT_ONLY)

import pytest

from nautilus_trader.model.identifiers import InstrumentId


#: Real venues, three of them, none inferable from the symbol. An all-XNAS fixture cannot tell a
#: resolver that reads the cache from one that hardcodes a venue.
_VENUES = {"QQQ": "XNAS", "SPY": "ARCX", "RDN": "XNYS"}


def _cache(mapping: dict[str, str] = _VENUES):
    """A Cache stand-in returning REAL `InstrumentId` objects, as the real one does."""
    ids = [InstrumentId.from_str(f"{s}.{v}") for s, v in mapping.items()]

    class _C:
        def instrument_ids(self):
            return list(ids)

    return _C()


def _lanes_with_symbol_resolution():
    """Every adapter in this package that resolves symbols, found by TYPE not by name."""
    import inspect

    from kumo_strategies.strategies import _layout
    from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin

    out = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, SymbolResolutionMixin)
                    and obj is not SymbolResolutionMixin):
                out.append(obj)
    return sorted(set(out), key=lambda c: c.__name__)


#: Resolved ONCE at import, and asserted non-empty here rather than inside each test. An empty
#: `parametrize` list generates zero cases and reports the same green as a passing suite — so a
#: rename or a moved module would silently delete this whole file's coverage. Four tests below
#: parametrize over it; this is the one place that can notice it went blank.
_LANES = _lanes_with_symbol_resolution()
assert _LANES, (
    "no lane in kumo_strategies.runtime.nautilus uses SymbolResolutionMixin. Either symbol "
    "resolution was removed — in which case delete this file — or discovery has broken and every "
    "test here is now generating zero cases while still reporting green."
)


def _build(cls, symbols, ids=None):
    """Construct `cls` from SYMBOLS ALONE — or from `ids`, the pre-#622 path used as the reference.

    Deliberately not a per-class table of every argument: this must keep working when a lane gains a
    parameter, or the test decays into a fixture nobody updates.
    """
    from kumo_strategies.strategies.momentum_rotation.candidates import StaticList

    import inspect

    # UNION OVER THE MRO, not the leaf signature. BCTROT takes `cfg` and `source` through `*args` /
    # `**kwargs` and names neither, so reading only its own parameters says it wants neither and the
    # call fails on the base class instead — a test failure that says nothing about the code.
    params, varkw = set(), False
    for base in cls.__mro__:
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, p in sig.parameters.items():
            if p.kind is p.VAR_KEYWORD:
                varkw = True
            elif p.kind is not p.VAR_POSITIONAL and name != "self":
                params.add(name)

    kw = {"instrument_ids": list(ids)} if ids is not None else {"symbols": list(symbols)}
    if "source" in params:
        kw["source"] = StaticList(list(symbols))
    if "cfg" in params:
        kw["cfg"] = _cfg_for(cls)
    if "order_id_tag" in params:
        kw["order_id_tag"] = "999"
    # CRSISHORT REFUSES A DEFAULT for each of these, deliberately, so the builder has to state them
    # the way a real caller would. They are not conveniences: an unstated price adjustment lets raw
    # prices through (a reverse split books as a -1000% short loss on an open position), an unstated
    # universe breadth lets the lane signal on a screen far narrower than the one measured, and a
    # borrow ceiling with no locate provider is an armed, inert gate.
    if "price_adjustment" in params:
        from kumo_strategies.strategies.crsi_short.engine import ADJUSTED

        kw["price_adjustment"] = ADJUSTED
    if "min_warm_symbols" in params:
        kw["min_warm_symbols"] = 1
    if "shadow_only" in params:
        # CRSISHORT refuses to construct while its order path is incomplete (#131). The discovery
        # suite builds every lane to CHECK it, not to trade it, so it states the shadow intent the
        # same way a cockpit shadow deployment would.
        kw["shadow_only"] = True
    if "borrow_rates" in params:
        kw["borrow_rates"] = lambda syms: {s: 0.0 for s in syms}
    assert varkw or {"symbols"} <= params, f"{cls.__name__} does not accept `symbols`"
    return cls(**kw)


def _cfg_for(cls):
    """The live-valid config for a lane, since several REFUSE a backtest default at construction."""
    name = cls.__name__
    if "QC27" in name:
        from kumo_strategies.strategies.qc27_tech_inverse_vol.config import QC27TechInverseVolConfig
        return QC27TechInverseVolConfig(momentum_price_field="close")
    if "Template" in name:
        from kumo_strategies.strategies.template import TemplateConfig
        # The template REFUSES `close_adj` at construction, exactly as the live lanes do — a price
        # field the live feed can never supply must fail at build, not at the first rebalance.
        return TemplateConfig(price_field="close", market_view=_A_MEASURED_VIEW)
    if "SmhGld" in name:
        from kumo_strategies.strategies.smhgld_sleeve import SmhGldSleeveConfig
        return SmhGldSleeveConfig()
    if "CrsiShort" in name:
        from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
        return CrsiShortConfig()
    if "QC345" in name:
        from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig
        from kumo_strategies.strategies.qc345_rotation.config import ExitConfig
        return QC345RotationConfig(momentum_price_field="close", corporate_action_window=252,
                                   exits=ExitConfig(stall_days=12))
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
    return MomentumRotationConfig()


def _submits_orders(cls) -> bool:
    """Does this lane build orders itself, or does its runner do it?

    The TEMPLATE subscribes bars and hands the panel to a session runner — the runner owns
    submission, which is the entire safety argument for having one. So the template legitimately has
    no symbol -> instrument lookup, and demanding one would put unused code in the file every new
    lane is copied from. Asked structurally rather than via `IS_TEMPLATE`, so a real lane that stops
    submitting is not quietly excused by a flag.
    """
    import ast
    import inspect
    import textwrap

    for base in cls.__mro__:
        try:
            src = textwrap.dedent(inspect.getsource(base))
        except (OSError, TypeError):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "submit_order" for n in ast.walk(ast.parse(src))):
            return True
    return False


def _lookups(obj):
    """Every way this lane answers "which instrument is `SYM`?", as {name: callable}.

    THREE FORMS, deliberately, because the fix for the original defect changed which one QC27 uses
    and a test that only knew the old form would have gone quiet the moment it passed:

      * an instance dict of symbol -> InstrumentId   (what `_by_symbol` WAS)
      * a class property returning one              (what `_by_symbol` IS)
      * a method like `_iid_for` that scans `_iids`  (Momentum, and the shape that never broke)

    Discovering all three is what keeps this file asserting something after the thing it caught is
    fixed. A regression test that stops exercising the repaired path is a monument, not a test.
    """
    out = {}
    for name in ("_iid_for", "_iid"):
        fn = getattr(obj, name, None)
        if callable(fn):
            out[name] = fn
    for name in dir(type(obj)):
        if name.startswith("__") or name == "_iids":
            continue
        if not isinstance(getattr(type(obj), name, None), property):
            continue
        # Deliberately unguarded: a lookup property that RAISES is a broken lookup. Skipping it
        # here would let another, working lookup satisfy the test on its behalf.
        val = getattr(obj, name)
        if isinstance(val, dict) and all(isinstance(v, InstrumentId) for v in val.values()):
            out[name] = (lambda n: lambda s: getattr(obj, n).get(s))(name)
    for name in _stored_lookup_names(type(obj)):
        out[name] = (lambda n: lambda s: getattr(obj, n).get(s))(name)
    return out


def _stored_lookup_names(cls):
    """Names of INSTANCE dicts that hold symbol -> InstrumentId, learned from a reference instance.

    Learned rather than read off the instance under test, because on the symbols path a broken one is
    EMPTY — and an empty dict is indistinguishable from `_pending` or `_bars` by inspection. Empty is
    the defect, so the defect cannot be what identifies it. The reference is built from
    `instrument_ids`, the pre-#622 path that works today, where the dict is populated and obvious.
    """
    ids = [InstrumentId.from_str(f"{s}.{v}") for s, v in _VENUES.items()]
    # NOT wrapped in try/except. This is the only detector for a stale EMPTY lookup, so swallowing a
    # construction failure here would disable it and leave the suite green — the precise shape of the
    # defect this file exists to catch, reproduced in the catcher.
    ref = _build(cls, list(_VENUES), ids=ids)
    return {n for n, v in vars(ref).items()
            if n != "_iids" and isinstance(v, dict) and v
            and all(isinstance(x, InstrumentId) for x in v.values())}


@pytest.mark.parametrize("cls", _LANES, ids=lambda c: c.__name__)
def test_a_lane_can_find_an_instrument_for_every_symbol_it_resolved(cls):
    """THE HEADLINE. Resolve from the cache, then require every lookup the lane owns to answer.

    MEASURED RED before the fix, on QC27RotationStrategy alone:

        QC27RotationStrategy resolved QQQ but `_by_symbol` cannot find it

    `_by_symbol` was a dict built in `__init__` from an `_iids` that `symbols` leaves empty until
    `on_start`. It froze at `{}`. Both `_open` and `_close` read it and return on `None`, so the lane
    could neither enter nor exit while resolving its whole pool correctly and reporting `armed`.
    """
    if not _submits_orders(cls):
        pytest.skip(f"{cls.__name__} submits no orders itself — its runner does, so it needs no "
                    f"symbol lookup and a forced one would be dead code in a template")
    s = _build(cls, list(_VENUES))
    s._resolve_symbols(_cache())
    assert s._iids, "fixture resolved nothing — every assertion below would be vacuous"

    lookups = _lookups(s)
    assert lookups, (
        f"{cls.__name__} exposes no symbol -> instrument lookup at all. Either it reads `_iids` "
        f"inline everywhere, or `_lookups` no longer recognises the form it uses — and this test "
        f"has quietly stopped checking it."
    )
    for name, fn in sorted(lookups.items()):
        for sym in _VENUES:
            assert fn(sym) is not None, (
                f"{cls.__name__} resolved {sym} but `{name}` cannot find it — orders for {sym} are "
                f"dropped by the `is None` guard at the top of the order path, in silence"
            )


@pytest.mark.parametrize("cls", _LANES, ids=lambda c: c.__name__)
def test_no_lookup_disagrees_with_the_ids_the_lane_resolved(cls):
    """Two derivations of one fact are a detector: differing when they should match is a live defect.

    The test above proves each lookup answers; this proves they all answer the SAME thing as `_iids`.
    A lookup populated from a different source — a stale copy, a second resolution pass — would pass
    the first test while routing orders to a venue the lane never resolved.
    """
    if not _submits_orders(cls):
        pytest.skip(f"{cls.__name__} submits no orders itself")
    s = _build(cls, list(_VENUES))
    s._resolve_symbols(_cache())
    resolved = {i.symbol.value: str(i) for i in s._iids}
    assert resolved

    lookups = _lookups(s)
    assert lookups, f"{cls.__name__} exposes no lookup — nothing to compare `_iids` against"

    for name, fn in sorted(lookups.items()):
        # WHOLE CONTENTS where the lookup is a container, not just the symbols `_iids` happens to
        # hold. Reading only the resolved keys makes the assertion one-directional: a lookup carrying
        # an EXTRA symbol the lane never resolved would pass, and that symbol is tradeable — it is a
        # venue guess reaching the order path by another name, which is the original defect.
        whole = getattr(s, name, None) if not callable(getattr(type(s), name, None)) else None
        if isinstance(whole, dict):
            got = {str(k): str(v) for k, v in whole.items()}
        else:
            got = {sym: str(fn(sym)) for sym in resolved if fn(sym) is not None}
        assert got == resolved, (
            f"{cls.__name__}.{name} disagrees with `_iids`: it holds {got} while the lane resolved "
            f"{resolved}. One of the two is what gets traded and it is not obvious which."
        )


def test_a_stored_copy_of_the_ids_cannot_come_back():
    """THE CLASS, not the instance. `_by_symbol` was one construction-time copy of `_iids`; the next
    one would be silent in exactly the same way.

    Asserted against instance state after construction from SYMBOLS, where `_iids` is legitimately
    empty. TWO shapes are offences and the second is the one that actually shipped:

      * an attribute already HOLDING instruments before any adapter connected — built from something
        other than what the lane will resolve;
      * an attribute that is a stored lookup on the ids path and is EMPTY here — frozen at nothing,
        which is exactly what `_by_symbol` did.

    The second cannot be recognised by inspecting the value, since `{}` is indistinguishable from
    `_pending` or `_bars`. Its NAME comes from the reference instance instead. An earlier version of
    this test checked only the first shape and therefore passed against the very defect it is named
    for — recorded here because that is the trap, not a footnote to it.

    A property or a method recomputes and is invisible to `vars()`, which is why those are correct.
    """
    offenders = {}
    for cls in _LANES:
        stored = _stored_lookup_names(cls)      # learned where they are populated: the ids path
        s = _build(cls, list(_VENUES))          # symbols path: `_iids` is empty until `on_start`
        for attr in stored & set(vars(s)):
            offenders[f"{cls.__name__}.{attr}"] = (
                "is a stored dict on the ids path, so on the symbols path it is frozen at whatever "
                "`_iids` held in __init__ — nothing")
        for attr, val in vars(s).items():
            if attr == "_iids":
                continue
            if isinstance(val, dict) and val and all(
                    isinstance(v, InstrumentId) for v in val.values()):
                offenders[f"{cls.__name__}.{attr}"] = "holds instruments before resolution ran"
            if isinstance(val, (list, tuple)) and val and all(
                    isinstance(v, InstrumentId) for v in val):
                offenders[f"{cls.__name__}.{attr}"] = "holds instruments before resolution ran"
    assert offenders == {}, (
        f"a lane stores instruments at construction, before any adapter has connected: {offenders}. "
        f"Derive them on read instead — `_iids` is not populated until `on_start` on the symbols "
        f"path, so a copy taken in `__init__` is frozen at whatever it was then (kumo-trading-platform issue 622)."
    )


@pytest.mark.parametrize("cls", _LANES, ids=lambda c: c.__name__)
def test_symbols_is_keyword_only_on_every_lane(cls):
    """A NEW PARAMETER MAY NOT BE POSITIONAL, because two repos share this signature across a pin.

    `symbols` was first added between `instrument_ids` and `bar_type_suffix`. That is a silent
    reassignment of every argument after it: `(cfg, source, ids, suffix)` binds the suffix string to
    `symbols`, `_init_symbols` then sees both an id list and a `symbols` and refuses — at BUILD, which
    is `Trader.add_strategy` raising and taking every other lane in the node down with it. That is
    the #377 shape, and it is the failure this whole ticket exists to stop causing.

    No caller in either repo binds positionally today (checked by AST, both trees), so this cost
    nothing to close and would have cost a node to leave open.
    """
    import inspect

    for base in cls.__mro__:
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):
            continue
        p = params.get("symbols")
        if p is None:
            continue
        assert p.kind is p.KEYWORD_ONLY, (
            f"{base.__name__}.__init__ takes `symbols` as {p.kind.name}. Anything positional here "
            f"shifts every parameter after it for a caller that binds by position, and this "
            f"signature is shared with cockpit across a version pin."
        )


def test_most_lanes_DO_submit_their_own_orders():
    """VACUITY GUARD for the skip above. If `_submits_orders` stopped matching how lanes submit,
    every parametrisation would skip and this file would report green while checking nothing."""
    submitting = [c.__name__ for c in _LANES if _submits_orders(c)]
    assert len(submitting) >= 3, (
        f"only {submitting} appear to submit orders; the skip has swallowed the suite")


def test_every_cockpit_LANE_resolves_symbols_at_all():
    """MEMBERSHIP, not discovery. Every test in this file parametrises over
    `_lanes_with_symbol_resolution()`, which finds lanes by TYPE — so a lane that stops using the
    mixin does not FAIL these checks, it vanishes from them.

    Measured: removing `SymbolResolutionMixin` from the template left the whole suite green. A lane
    copied from it would then take `instrument_ids` only, and demand a venue at construction — which
    is kumo-trading-platform issue 622, the defect that kept an IBKR-only node from booting at all.

    Discovery decides WHAT to check. It must not also decide WHETHER to check.
    """
    import inspect

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout
    from kumo_strategies.runtime.nautilus.symbol_resolution import SymbolResolutionMixin

    lanes, missing = [], []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, Strategy)
                    and getattr(obj, "EXTERNAL_ID", None)):
                lanes.append(obj.__name__)
                if not issubclass(obj, SymbolResolutionMixin):
                    missing.append(obj.__name__)
    assert lanes, "no lanes discovered — this assertion no longer describes the package"
    assert not missing, (
        f"{missing} declare EXTERNAL_ID but cannot resolve symbols, so they take instrument ids "
        f"only — a venue demanded at CONSTRUCTION, which is what stopped an IBKR-only node booting "
        f"(kumo-trading-platform issue 622). They are also invisible to every other test in this file.")
