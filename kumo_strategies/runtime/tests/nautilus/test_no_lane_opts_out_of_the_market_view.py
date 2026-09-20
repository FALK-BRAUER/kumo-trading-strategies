"""No lane opts out of the market view — refused at REGISTRATION, for the class (#212).

2026-09-13: "no strategy can opt out. Those are emergency rules." A lane whose config says
`signal=NONE`, or has no `market_view` at all, answers the three hooks with "declares no view" —
which the poller reads as `not_asked`, and `not_asked` is indistinguishable from "nothing is wrong".

MEASURED ON b575db1 by constructing each deployed lane's live config: 5 of 6 had no declared view.
    TECHIVOL-005    INDEX_VS_MA / 50 / EXIT_ONLY        the only one
    SMHGLD-007      signal=NONE                          #212, fixed here
    MOMENTUM-002    signal=NONE (config default)         #213, owed
    BCTROT-004      signal=NONE (inherits MOMENTUM's)    #214, owed
    QC345-003       signal=NONE (config default)         #215, owed
    CRSISHORT-006   no `market_view` field at all        #216, owed

THE RULE IS ENFORCED AT THE CLASS. `MarketAwareMixin.register` — the Nautilus registration seam
`Trader.add_strategy` drives — refuses a lane whose view is undeclared. A NEW lane cannot be
exempt: the exemption list is a module constant of exactly the four lanes above, each carrying the
ticket that owes its measured view, and this file asserts the list and its own xfails are the SAME
four names. Removing a lane's exemption before its view lands turns its xfail red-by-design; adding
a name to the list without an xfail here turns the equality red.

WHY NOT INVENT FOUR VIEWS TODAY: `market_view.py` — "MEASURED PER LANE, NEVER INHERITED"; TECHIVOL's
50 is TECHIVOL's number. Four lanes are owed a measurement, not a copied constant.
"""

from __future__ import annotations

import importlib
import re
from types import SimpleNamespace

import pytest
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout
from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig

#: Lane NAME (the part of the strategy id before the tag) -> the ticket that owes its measured view.
#: Typed HERE, independently of the module constant, so the two can be asserted equal.
OWED_XFAILS = {
    "MOMENTUM": "#213",
    "BCTROT": "#214",
    "QC345": "#215",
    "CRSISHORT": "#216",
}


def _adapters() -> tuple[dict[str, type], dict[str, str]]:
    found: dict[str, type] = {}
    broken: dict[str, str] = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for obj in vars(m).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                found[obj.__name__] = obj
    return found, broken


ADAPTERS, BROKEN = _adapters()


def _lane_name(cls) -> str:
    """The adapter module's `STRATEGY_NAME` — what `StrategyConfig(strategy_id=...)` is built from."""
    return getattr(importlib.import_module(cls.__module__), "STRATEGY_NAME")


#: The lane each adapter class registers as, and the config it registers with. `live_config` where
#: the package ships one; the config class's default where it does not (that IS the live shape:
#: cockpit's builders for MOMENTUM/BCTROT/QC345 construct the default and set no view).
def _live_cfg(cls):
    name = _lane_name(cls)
    by_name = {
        "MOMENTUM": ("kumo_strategies.strategies.momentum_rotation.config", "MomentumRotationConfig"),
        "BCTROT": ("kumo_strategies.strategies.momentum_rotation.config", "MomentumRotationConfig"),
        "TECHIVOL": ("kumo_strategies.strategies.qc27_tech_inverse_vol.live", "live_config"),
        "QC345": ("kumo_strategies.strategies.qc345_rotation.config", "QC345RotationConfig"),
        "CRSISHORT": ("kumo_strategies.strategies.crsi_short.config", "CrsiShortConfig"),
        "SMHGLD": ("kumo_strategies.strategies.smhgld_sleeve.live", "live_config"),
        "TEMPLATE": ("kumo_strategies.strategies.template.config", "TemplateConfig"),
    }
    if name not in by_name:
        return None
    mod, attr = by_name[name]
    ctor = getattr(importlib.import_module(mod), attr)
    if name == "TEMPLATE":
        # The template's `market_view` is REQUIRED (no default): a copier cannot construct a config
        # without stating a view. Pinned by `test_the_TEMPLATE_config_cannot_be_built_without_a_view`;
        # here it is built the way a copier must build it.
        # `price_field="close"` is the live shape ("`close` for anything live", the config's own
        # comment); the adapter refuses `close_adj` at construction because no live bar carries it.
        return ctor(price_field="close", market_view=MarketViewConfig(
            signal=MarketSignal.INDEX_VS_MA, window=50, action=MarketAction.EXIT_ONLY))
    return ctor()


AWARE = sorted((c for c in ADAPTERS.values() if issubclass(c, MarketAwareMixin)),
               key=lambda c: c.__name__)
NOT_AWARE = sorted((c for c in ADAPTERS.values() if not issubclass(c, MarketAwareMixin)),
                   key=lambda c: c.__name__)


def _owed_or_plain(cls):
    name = _lane_name(cls)
    if name in OWED_XFAILS:
        return pytest.param(cls, marks=pytest.mark.xfail(
            strict=True, reason=f"{OWED_XFAILS[name]}: {name} owes a measured market view"))
    return cls


# ------------------------------------------------------------------ discovery ------------------

def test_the_discovery_found_the_lanes_and_imported_every_module():
    assert BROKEN == {}, f"adapter modules this file cannot judge: {BROKEN}"
    names = {_lane_name(c) for c in AWARE}
    assert {"SMHGLD", "TECHIVOL", "MOMENTUM", "BCTROT", "QC345", "CRSISHORT"} <= names, sorted(names)


#: Parked research lanes cockpit does not build, keyed by CLASS (IntradayMomentumRotation shares
#: STRATEGY_NAME "MOMENTUM" with the live adapter, so a name key could read the wrong lane as parked).
PARKED_NOT_AWARE = {"IntradayMomentumRotation": "#28"}


@pytest.mark.parametrize("cls", NOT_AWARE, ids=lambda c: c.__name__)
def test_every_adapter_the_cockpit_could_register_is_MARKET_AWARE(cls):
    """An adapter without the mixin has no `register` hook to refuse it and no hooks to poll — it
    opts out by construction. The one here is a parked research lane cockpit does not build,
    named with its parking ticket so a second cannot appear quietly."""
    ticket = PARKED_NOT_AWARE.get(cls.__name__)
    assert ticket, f"{cls.__name__} carries no MarketAwareMixin and is not a ticketed parked lane"
    pytest.xfail(f"{ticket}: {cls.__name__} is a parked research lane without MarketAwareMixin — "
                 f"it cannot be registered under the rule")


# ------------------------------------------------------------------ the declaration, per lane --

@pytest.mark.parametrize("cls", [_owed_or_plain(c) for c in AWARE], ids=lambda c: c.__name__)
def test_every_market_aware_lane_DECLARES_a_view_on_its_live_config(cls):
    cfg = _live_cfg(cls)
    assert cfg is not None, f"{cls.__name__}: no live config known to this test — add it to `_live_cfg`"
    view = getattr(cfg, "market_view", None)
    assert isinstance(view, MarketViewConfig), (
        f"{_lane_name(cls)}: config carries no `market_view` — the lane opts out by omission (#212)")
    assert view.signal is not MarketSignal.NONE, (
        f"{_lane_name(cls)}: declares signal=NONE — an opt-out, not a measurement (#212)")
    assert isinstance(view.action, MarketAction)


def test_the_TEMPLATE_config_cannot_be_built_without_a_view():
    """The class rule at the TYPE level, one step before `register`: the file new lanes are copied
    from refuses to construct with no view, and refuses to supply a number (a window is measured
    per lane, never inherited — a copied 50 is a declared wrong number the register refusal cannot
    catch)."""
    import dataclasses

    from kumo_strategies.strategies.template.config import TemplateConfig

    f = {f.name: f for f in dataclasses.fields(TemplateConfig)}["market_view"]
    assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    with pytest.raises(TypeError):
        TemplateConfig()  # type: ignore[call-arg]


# ------------------------------------------------------------------ the bars, per lane ---------

#: Minimal construction kwargs per lane. `Strategy.__init__` runs without a kernel, so the deque
#: each production `__init__` builds is the one asserted on. A lane missing here fails loudly.
_CONSTRUCT = {
    "TECHIVOL": dict(symbols=["AAPL"], order_id_tag="005"),
    "SMHGLD": dict(symbols=["SMH", "GLD"], order_id_tag="007", shadow_only=True),
    "TEMPLATE": dict(symbols=["AAPL"], order_id_tag="099"),
}


@pytest.mark.parametrize("cls", AWARE, ids=lambda c: c.__name__)
def test_every_lane_with_a_DECLARED_view_retains_enough_bars_to_compute_it(cls):
    """`_view_prices` reads `_bars`, and every adapter caps `_bars` at `_need + 5`. A lane whose
    warmup is shorter than its view's window declares a view it can never compute — UNKNOWN
    forever, behind a surface that says "declared" (SMHGLD: 8 retained against 51 needed, found in
    the deployed package by the coverage review). Judged for every lane whose live config declares
    a view; an owed lane joins this test the day it declares, and `bars_to_keep` is how it passes."""
    cfg = _live_cfg(cls)
    view = getattr(cfg, "market_view", None)
    if view is None or view.signal is MarketSignal.NONE:
        pytest.skip(f"{_lane_name(cls)} declares no view yet ({OWED_XFAILS.get(_lane_name(cls), 'n/a')})")
    kwargs = _CONSTRUCT.get(_lane_name(cls))
    assert kwargs is not None, f"{cls.__name__}: no construction kwargs known — add it to `_CONSTRUCT`"
    lane = cls(cfg, **kwargs)
    kept = lane._bars[kwargs["symbols"][0]].maxlen
    assert kept is not None and kept >= view.window + view.dwell, (
        f"{_lane_name(cls)} retains {kept} bars per name; its declared {view.window}-session view "
        f"needs {view.window + view.dwell} — the view reads UNKNOWN forever (#212)")


# ------------------------------------------------------------------ the mechanism, per lane ----

def _init_source(cls) -> str:
    """`__init__` as THIS class defines it, or the nearest ancestor that does (BCTROT → MOMENTUM)."""
    import inspect
    import textwrap

    for klass in cls.__mro__:
        fn = klass.__dict__.get("__init__")
        if fn is not None and klass is not object:
            return textwrap.dedent(inspect.getsource(fn))
    return ""


#: Lanes whose `__init__` still sizes `_bars` as `need + 5` and are NOT owed a view: TECHIVOL computes
#: its 50-session view only because warmup 100 exceeds it. One line, byte-identical value today;
#: out of this PR by the lead's Monday-lane ruling. Keyed by lane name, ticketed.
CAP_MECHANISM_OWED = {"TECHIVOL": "#217"}


def _cap_owed_or_plain(cls):
    name = _lane_name(cls)
    ticket = OWED_XFAILS.get(name) or CAP_MECHANISM_OWED.get(name)
    if ticket:
        return pytest.param(cls, marks=pytest.mark.xfail(
            strict=True, reason=f"{ticket}: {name} does not size `_bars` through bars_to_keep yet"))
    return cls


@pytest.mark.parametrize("cls", [_cap_owed_or_plain(c) for c in AWARE], ids=lambda c: c.__name__)
def test_every_market_aware_lane_builds_its_bar_container_through_bars_to_keep(cls):
    """THE MECHANISM, not the value. The cap test above judges numbers and passes TECHIVOL by
    coincidence (warmup 100 > window 50); the day #213 declares MOMENTUM's view its 105-bar cap
    passes the same way and the `need + 5` pattern is never made explicit. AST: the `__init__`
    that builds `_bars` sizes its deque with `bars_to_keep(...)`. Owed lanes are strict xfails
    under their tickets, the same equality-to-`VIEW_OWED` shape as the declaration test."""
    import ast

    tree = ast.parse(_init_source(cls))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "bars_to_keep"]
    assert calls, (
        f"{_lane_name(cls)}: `__init__` builds `_bars` without `bars_to_keep(...)` — a declared view "
        f"wider than the warmup reads UNKNOWN forever behind `_need + 5` (#212)")


# ------------------------------------------------------------------ the refusal, at the class --

class _Base:
    """Stands in for Nautilus `Strategy` under the mixin: records that registration went through."""

    def register(self, *args, **kwargs):
        self.registered = (args, kwargs)


class _Lane(MarketAwareMixin, _Base):
    def __init__(self, strategy_id: str, cfg):
        self.id = SimpleNamespace(value=strategy_id, __str__=lambda s: strategy_id)
        self._cfg = cfg
        self._bars = {}
        self.registered = None


def _declared():
    return SimpleNamespace(market_view=MarketViewConfig(
        signal=MarketSignal.INDEX_VS_MA, window=50, action=MarketAction.EXIT_ONLY))


def _none():
    return SimpleNamespace(market_view=MarketViewConfig())


def test_FIXTURE_a_declared_view_registers_through_to_the_base():
    lane = _Lane("NEWLANE-099", _declared())
    lane.register("trader", "portfolio", msgbus="bus")
    assert lane.registered == (("trader", "portfolio"), {"msgbus": "bus"})


@pytest.mark.parametrize("cfg", [_none(), SimpleNamespace(), None], ids=["signal=NONE", "no field", "no cfg"])
def test_a_NEW_lane_with_an_undeclared_view_is_REFUSED_at_registration(cfg):
    """The class rule with no exemption possible: a name the exemption list does not carry is
    refused before Nautilus ever wires it, whichever way it fails to declare."""
    from kumo_strategies.runtime.nautilus.market_aware import MarketViewUndeclared

    lane = _Lane("NEWLANE-099", cfg)
    with pytest.raises(MarketViewUndeclared) as exc:
        lane.register("trader", "portfolio")
    assert "NEWLANE" in str(exc.value) and "#212" in str(exc.value)
    assert lane.registered is None, "refused and registered anyway"


def _undeclared_as_deployed(name: str):
    """Each owed lane's ACTUAL failure shape: CRSISHORT has no `market_view` field, the other three
    default to signal=NONE. Both must pass through the exemption; a refusal that only knew one
    shape would take CRSISHORT down on Monday."""
    return SimpleNamespace() if name == "CRSISHORT" else _none()


def test_MarketViewUndeclared_is_NOT_an_OSError_so_cockpit_cannot_absorb_it_as_transient():
    """Cockpit's `build_optional_strategy` (engine_node.py:10222) absorbs the OSError family as
    "transient, the node boots without the lane" and propagates everything else as
    misconfiguration. A refusal that inherited OSError would become a silent skip — the exact
    opposite of the rule (coverage review, #212)."""
    from kumo_strategies.runtime.nautilus.market_aware import MarketViewUndeclared

    assert not issubclass(MarketViewUndeclared, OSError)
    assert issubclass(MarketViewUndeclared, RuntimeError)


def test_the_base_registers_exception_is_NOT_swallowed_by_the_hook():
    """The hook wraps Nautilus' `register`; a base that raises must still raise through it."""
    class _Boom(_Base):
        def register(self, *a, **kw):
            raise ValueError("base refused")

    class _L(MarketAwareMixin, _Boom):
        def __init__(self):
            self.id = SimpleNamespace(value="NEWLANE-099")
            self._cfg = _declared()
            self._bars = {}

    with pytest.raises(ValueError, match="base refused"):
        _L().register("trader", "portfolio")


@pytest.mark.parametrize("name", sorted(OWED_XFAILS))
def test_an_OWED_lane_with_an_undeclared_view_registers_and_the_refusal_names_the_ticket(name, caplog):
    """The exemption is loud: the lane registers (it trades Monday) and the log carries the ticket
    that owes its view, so the exemption is a standing debt on the surface, not a silence.
    Parametrized over all four: a per-name defect (BCTROT's id parse, CRSISHORT's missing field
    versus NONE) is the class's to cover."""
    from kumo_strategies.runtime.nautilus.market_aware import VIEW_OWED

    ticket = VIEW_OWED[name]
    lane = _Lane(f"{name}-002", _undeclared_as_deployed(name))
    lane.log = SimpleNamespace(warning=lambda msg, *a, **k: caplog.records.append(msg),
                               error=lambda msg, *a, **k: caplog.records.append(msg),
                               info=lambda *a, **k: None)
    lane.register("trader", "portfolio")
    assert lane.registered is not None
    assert any(ticket in str(r) and name in str(r) for r in caplog.records), caplog.records


@pytest.mark.parametrize("name", sorted(OWED_XFAILS))
def test_an_OWED_lane_that_DECLARES_a_view_is_no_longer_exempt_in_effect(name):
    """Declaring the view is what closes the ticket; the exemption must be inert the moment it is
    declared, not a permanent pass for the name."""
    lane = _Lane(f"{name}-002", _declared())
    lane.log = SimpleNamespace(warning=lambda *a, **k: pytest.fail("warned about a declared view"),
                               error=lambda *a, **k: None, info=lambda *a, **k: None)
    lane.register("trader", "portfolio")
    assert lane.registered is not None


def test_the_exemption_list_and_this_files_xfails_are_the_SAME_four_names_with_tickets():
    """Two derivations of one fact, asserted equal: the module constant the refusal reads and the
    xfails this file carries. Neither can grow or shrink without the other."""
    from kumo_strategies.runtime.nautilus.market_aware import VIEW_OWED

    assert dict(VIEW_OWED) == OWED_XFAILS
    for name, ticket in VIEW_OWED.items():
        assert re.fullmatch(r"#\d+", ticket), f"{name}: exemption without a ticket number ({ticket!r})"
    assert len(VIEW_OWED) == 4


@pytest.mark.parametrize("name", sorted(OWED_XFAILS))
def test_an_OWED_lanes_entries_blocked_names_its_TICKET_on_every_poll(name):
    """The debt is visible per poll, not only in the boot WARNING: the poller journals this reason
    on /health.market_aware. On b575db1 the reason called NONE "a measurement result, not a gap" —
    false prose beside a value that is now owed."""
    lane = _Lane(f"{name}-002", _undeclared_as_deployed(name))
    verdict = lane.entries_blocked()
    assert verdict.state == "no"
    assert any(OWED_XFAILS[name] in r and "#212" in r for r in verdict.reasons), verdict.reasons
    assert not any("measurement result" in r for r in verdict.reasons), verdict.reasons


def test_a_NON_owed_undeclared_lane_polled_on_a_host_answers_with_a_refusal_shaped_reason():
    """Unreachable on the node (register refuses it), pollable on a host: the reason must say so
    and name the rule, never read as a lane that chose not to look."""
    verdict = _Lane("NEWLANE-099", _none()).entries_blocked()
    assert verdict.state == "no"
    assert any("#212" in r and "refused at registration" in r for r in verdict.reasons), verdict.reasons


def test_the_lane_name_is_derived_from_the_strategy_id_not_from_the_class():
    """`self.id` is `NAME-tag`; the exemption is keyed by NAME. A subclass (BCTROT under MOMENTUM)
    registers under its OWN name, so the class is the wrong key."""
    from kumo_strategies.runtime.nautilus.market_aware import lane_name_of

    assert lane_name_of(SimpleNamespace(id="BCTROT-004")) == "BCTROT"
    assert lane_name_of(SimpleNamespace(id="CRSISHORT-006")) == "CRSISHORT"
    assert lane_name_of(SimpleNamespace(id="NEWLANE-099")) == "NEWLANE"


def test_the_register_hook_is_the_mixins_own_and_sits_FIRST_in_every_lanes_MRO():
    """The seam is `Strategy.register`, which `Trader.add_strategy` calls from Python. The mixin's
    override is reached only if the mixin precedes `Strategy` in the MRO — it does, on every lane."""
    assert "register" in vars(MarketAwareMixin), "MarketAwareMixin defines no register hook"
    for cls in AWARE:
        mro = cls.__mro__
        assert mro.index(MarketAwareMixin) < mro.index(Strategy), cls.__name__
        owner = next(k for k in mro if "register" in vars(k))
        assert owner is MarketAwareMixin, (
            f"{cls.__name__}.register resolves to {owner.__name__}, not the mixin — the refusal is "
            f"shadowed")
