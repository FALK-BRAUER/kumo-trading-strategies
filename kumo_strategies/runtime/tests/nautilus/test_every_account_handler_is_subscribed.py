"""A lane that DEFINES `_on_broker_account` must SUBSCRIBE it (#174, #208).

CRSISHORT defined the handler at `crsi_short.py` and subscribed nothing: the class had no
`account_topic` argument, no `_account_topic`, and no `msgbus.subscribe` call. `broker_equity()`
therefore returned None on every call, forever, and there was no error anywhere — the handler was
simply never invoked (#174, fixed in c4aed2c).

Measured on ibkr-paper at e600383 by the cockpit session: BCTROT-004 on the SAME node raised "no
broker account snapshot on 'broker.account' yet" at 05:45:37Z and passed equity from 05:45:39Z,
while CRSISHORT-006 still returned None at 05:49:43Z. Two lanes, one node, one clock — so this is
wiring, not warmup.

THEN IT HAPPENED AGAIN (#208). `SmhGldSleeveStrategy` defines the same handler at
`smhgld_sleeve.py:615`, takes no `account_topic`, subscribes nothing — and this file, whose docstring
said "the test is written over every lane, not over CRSISHORT", did not see it. The lane list was a
HAND-KEPT tuple written before SMHGLD existed. Measured on both tenants at cockpit 04036f6 /
strategies 7fae46b: `PREFLIGHT DEGRADED SMHGLD-007: equity: returned None` x11 in one hour on
paper, x2 per boot on ibkr-paper (platform issue 1022). Not a sizing blocker — SMHGLD sizes from the sleeve —
but an ERROR line per tick and a probe that can never pass, which is a probe nobody reads.

SO THE LANES ARE DISCOVERED, NOT LISTED. A hand-kept tuple is a test that covers the adapters its
author knew about; the whole point of a class-level test is the adapter its author did not.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout
from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy
from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig

#: A bus topic no default, no settings file and no fixture in this package produces — so a
#: subscription that lands on it came from the attribute the host set, not from a literal.
_PROBE_TOPIC = "test.account.12345"


def _adapters() -> tuple[dict[str, type], dict[str, str]]:
    """Every Nautilus `Strategy` subclass the package ships, by walking the package — the same
    discovery `test_a_lane_that_subscribes_must_also_request.py` uses, for the same reason.

    IMPORT FAILURES ARE RETURNED, NOT SWALLOWED. A module that fails to import leaves the sweep
    silently otherwise, and the lane it defines is then exactly as unjudged as SMHGLD was under the
    hand-kept list — with a vacuity guard that only knows the names its author knew."""
    found: dict[str, type] = {}
    broken: dict[str, str] = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for obj in vars(m).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                found[obj.__name__] = obj
    return found, broken


ADAPTERS, BROKEN_MODULES = _adapters()


def _defines_handler(cls) -> bool:
    return any(k.__dict__.get("_on_broker_account") for k in cls.__mro__)


#: The lanes this file judges: every adapter that defines the handler. Derived, never typed.
LANES = sorted((c for c in ADAPTERS.values() if _defines_handler(c)), key=lambda c: c.__name__)


def _named_or_xfail(cls):
    """BCTROT subscribes (it inherits MOMENTUM's `on_start`) but does not NAME `account_topic` — it
    forwards through `**kwargs`, which is not a public seam (#514). Found by this file's derived
    list; NOT fixed here by the lead's ruling of 2026-09-12 (a touched constructor on a lane that
    trades Monday stays out of the Sunday bundle). `strict=True`: the moment #209 names the kwarg
    this turns red-by-design and the mark comes off."""
    if cls.__name__ == "BCTRotationStrategy":
        return pytest.param(cls, marks=pytest.mark.xfail(
            strict=True, reason="#209: BCTROT forwards account_topic through **kwargs, unnamed"))
    return cls


LANES_NAMED = [_named_or_xfail(c) for c in LANES]


def test_the_discovery_found_the_lanes_this_file_exists_for():
    """Vacuity guard. An empty sweep parametrizes nothing and the file passes asserting nothing —
    and the two lanes that were each, in turn, the one nobody listed must both be in it."""
    assert BROKEN_MODULES == {}, (
        f"adapter modules that failed to import are adapters this file cannot judge: {BROKEN_MODULES}")
    names = {c.__name__ for c in LANES}
    assert {"CrsiShortStrategy", "SmhGldSleeveStrategy"} <= names, sorted(names)


def _on_start_source(cls):
    """`on_start` as THIS class defines it, or the nearest ancestor that does.

    BCTROT inherits MOMENTUM's, which is a real subscription and must count as one.
    """
    for klass in cls.__mro__:
        fn = klass.__dict__.get("on_start")
        if fn is not None:
            return inspect.getsource(fn), klass
    return None, None


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_lane_that_defines_an_account_handler_SUBSCRIBES_it(cls):
    """AST-bound, never a substring match on the source text.

    A grep for "subscribe" is satisfied by `subscribe_bars` — which every lane calls — so the
    check has to find a `self.msgbus.subscribe(...)` whose second argument is this very handler,
    AND whose first argument is the ATTRIBUTE `self._account_topic`. A literal `"broker.account"`
    there passes every other test in this file with the constructor argument dead (coverage
    review, #208) — it is the #174 silence one rename away.

    `_on_start_source` walks the lane's OWN MRO first, so a subclass that overrides `on_start`
    without the subscription is judged on its override, not on its parent's — BCTROT is green
    today because it does NOT override MOMENTUM's.
    """
    src, owner = _on_start_source(cls)
    assert src is not None, f"{cls.__name__} has no on_start at all"

    # `textwrap.dedent`, NOT `inspect.cleandoc`: cleandoc strips the first line's
    # indentation and then dedents the REST against it, which flattens the body and
    # raises IndentationError. It cost this test a run where all five lanes failed
    # identically — the signature of a broken detector, not of five broken lanes.
    tree = ast.parse(textwrap.dedent(src))
    found, literal = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # self.msgbus.subscribe(topic, handler)
        if (isinstance(fn, ast.Attribute) and fn.attr == "subscribe"
                and isinstance(fn.value, ast.Attribute) and fn.value.attr == "msgbus"):
            args = list(node.args) + [k.value for k in node.keywords]
            if not any(isinstance(a, ast.Attribute) and a.attr == "_on_broker_account" for a in args):
                continue
            topic = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "topic"), None)
            if isinstance(topic, ast.Attribute) and topic.attr == "_account_topic":
                found.append(node)
            else:
                literal.append(ast.unparse(node))

    assert not literal, (
        f"{cls.__name__} subscribes `_on_broker_account` on something other than "
        f"`self._account_topic`: {literal}. The constructor argument is dead and the topic is a "
        f"literal that fails silently the day cockpit renames it (#208 coverage review).")
    assert found, (
        f"{cls.__name__} defines `_on_broker_account` and never subscribes it in "
        f"{owner.__name__}.on_start — the handler can never fire, so `broker_equity()` returns "
        f"None forever. At the flip that refuses every entry while the lane reads armed and warm "
        f"(#174); before it, an ERROR line per preflight tick on every tenant (#208).")


@pytest.mark.parametrize("cls", LANES_NAMED, ids=lambda c: c.__name__)
def test_the_topic_a_lane_subscribes_is_a_CONSTRUCTOR_argument(cls):
    """The bus topic is cockpit's to name, so it is an argument rather than an import — this package
    keeps no dependency on cockpit. A lane that hardcoded it would keep working until cockpit renamed
    the topic and then fail exactly the way this ticket describes: silently."""
    params = inspect.signature(cls.__init__).parameters
    assert "account_topic" in params, (
        f"{cls.__name__}.__init__ takes no `account_topic`: the bus topic is cockpit's to name and "
        f"a hardcoded one fails silently when it is renamed")


def _init_that_names(cls, param: str):
    """The `__init__` in the lane's MRO that NAMES `param` in its own signature, or None.

    Resolved through the MRO because that is where the argument is honoured: BCTROT's own
    `__init__` does not name `account_topic` (#209), MOMENTUM's does, and MOMENTUM's is the one
    that must assign it."""
    for klass in cls.__mro__:
        fn = klass.__dict__.get("__init__")
        if fn is None:
            continue
        try:
            if param in inspect.signature(fn).parameters:
                return fn, klass
        except (TypeError, ValueError):
            continue
    return None, None


# `LANES`, not `LANES_NAMED`: this test resolves the naming `__init__` through the MRO, so BCTROT is
# judged on MOMENTUM's, which assigns — an honest pass, not the #209 gap (that is the NAMING test).
@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_the_CONSTRUCTOR_ASSIGNS_the_topic_it_names_to_the_attribute_on_start_reads(cls):
    """`__init__` cannot be driven (it needs a running kernel), and the effect hosts hand
    `_account_topic` to the lane themselves — so a signature that names `account_topic` and never
    assigns `self._account_topic = account_topic` is green on every other test here with the
    argument dead (coverage review, #208). Pinned by AST on the `__init__` that names it."""
    fn, owner = _init_that_names(cls, "account_topic")
    if fn is None:
        pytest.fail(f"{cls.__name__}: no __init__ in the MRO names `account_topic`")
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    assigned = [
        n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_account_topic" for t in n.targets)
        and isinstance(n.value, ast.Name) and n.value.id == "account_topic"
    ]
    assert assigned, (
        f"{owner.__name__}.__init__ names `account_topic` and never assigns it to "
        f"`self._account_topic` — the attribute `on_start` subscribes on is not the one the caller "
        f"set (#208)")


@pytest.mark.parametrize("cls", LANES_NAMED, ids=lambda c: c.__name__)
def test_every_lane_defaults_to_the_SAME_topic_cockpit_publishes_on(cls):
    """Cockpit passes no `account_topic` to any builder (read: kumo-trading-platform backend/strategies/*.py)
    and publishes on `broker.account`. So the DEFAULT is what runs, on every lane, and a lane whose
    default differs from its siblings' subscribes a topic nobody publishes on — #174's silence with
    a different cause."""
    default = inspect.signature(cls.__init__).parameters["account_topic"].default
    assert default == "broker.account", (
        f"{cls.__name__} defaults account_topic to {default!r}; its siblings and cockpit's "
        f"publisher use 'broker.account'")


# ==================================================================================================
# THE EFFECT, per lane that has been the one nobody listed: a frame published on the subscribed
# topic reaches `broker_equity()`. A subscription to the wrong topic, or one whose handler is not
# the one `broker_equity` reads, passes every AST test above and still returns None forever.
# ==================================================================================================

def test_CRSISHORT_equity_ARRIVES_through_the_bus_it_subscribed():
    """12345.0 rather than a round number: a value no default, no settings file and no fixture in
    this package can produce, so reading it back proves it came off the frame that was published.
    """
    from kumo_strategies.strategies.crsi_short.tests.test_crsi_short_adapter import _lane

    lane = _lane()
    # A TOPIC THE DEFAULT CANNOT PRODUCE. With "broker.account" here, a fix that subscribes the
    # literal passes this test with the constructor argument dead (coverage review, #208).
    lane._account_topic = _PROBE_TOPIC
    lane._history_days = 400
    lane._iids = []
    lane._need = CrsiShortConfig().warmup_sessions
    lane._resolve_symbols_if_needed = lambda: None
    lane.begin_arming = lambda: None
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [], orders_inflight=lambda **kw: [],
                                 positions_open=lambda **kw: [])
    bus = {}
    lane.msgbus = SimpleNamespace(subscribe=lambda topic, handler: bus.__setitem__(topic, handler))

    CrsiShortStrategy.on_start(lane)

    assert lane._account_topic in bus, (
        f"on_start subscribed nothing on {lane._account_topic!r}; broker_equity() returns None "
        f"forever and no error is raised anywhere (#174)")
    assert lane.broker_equity() is None, "equity read as present before any frame arrived"

    bus[lane._account_topic]({"equity": 12345.0})
    assert lane.broker_equity() == 12345.0, (
        "the frame published on the subscribed topic did not reach broker_equity(): the handler "
        "that was subscribed is not the one the reader consults")


class _SmhGldHost(HeldSeedMixin):
    """A narrow host carrying the REAL functions off `SmhGldSleeveStrategy`.

    `Strategy.__init__` needs a running kernel and `Actor.log` / `Actor.cache` are read-only Cython
    attributes, so the class cannot be `__new__`'d and patched. The three methods under test are the
    attributes themselves — deleting or renaming one fails here rather than exercising a copy.
    """

    POSITION_SIDE = SmhGldSleeveStrategy.POSITION_SIDE
    on_start = SmhGldSleeveStrategy.on_start
    _on_broker_account = SmhGldSleeveStrategy._on_broker_account
    broker_equity = SmhGldSleeveStrategy.broker_equity

    id = "SMHGLD-007"


def _smhgld_lane():
    lane = _SmhGldHost()
    lane._held = set()
    lane._pending = {}
    lane._broker_account = None
    lane._account_topic = _PROBE_TOPIC          # see _PROBE_TOPIC: the default cannot produce it
    # Everything `on_start` touches that is not the subject: no legs, history requested (so the
    # no-history ERROR branch is not entered), symbols resolved, arming a no-op.
    lane._iids = []
    lane._history_days = 30
    lane._resolve_symbols_if_needed = lambda: None
    lane.begin_arming = lambda: None
    lane._loop = None
    lane.cache = SimpleNamespace(positions_open=lambda **kw: [])
    lane.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                               error=lambda *a, **k: None)
    lane.subscribed = []
    lane.msgbus = SimpleNamespace(
        subscribe=lambda topic, handler: lane.subscribed.append((topic, handler)))
    return lane


def test_FIXTURE_the_smhgld_host_reaches_the_end_of_on_start():
    """The host must survive `on_start` with the subject ABSENT, or the test below is measuring a
    crash in the fixture rather than a missing subscription. Seen: on 7fae46b this passes and the
    effect test fails — the pair that proves the fixture, not the fix."""
    lane = _smhgld_lane()
    lane.begin_arming = lambda: lane.__setattr__("armed", True)

    _SmhGldHost.on_start(lane)

    assert getattr(lane, "armed", False), "on_start did not reach begin_arming — the host is broken"


def test_SMHGLD_equity_ARRIVES_through_the_bus_it_subscribed():
    """The measured defect (#208): x11/h `PREFLIGHT DEGRADED SMHGLD-007: equity: returned None` on
    paper, because nothing was subscribed. Same shape and same probe value as the CRSISHORT test."""
    lane = _smhgld_lane()

    _SmhGldHost.on_start(lane)

    topics = {t: h for t, h in lane.subscribed}
    assert lane._account_topic in topics, (
        f"on_start subscribed {sorted(topics) or 'nothing'}, not {lane._account_topic!r}; "
        f"broker_equity() returns None forever and the preflight probe fails every tick (#208)")
    assert lane.broker_equity() is None, "equity read as present before any frame arrived"

    topics[lane._account_topic]({"equity": 12345.0})
    assert lane.broker_equity() == 12345.0, (
        "the frame published on the subscribed topic did not reach broker_equity(): the handler "
        "that was subscribed is not the one the reader consults")


def _crsi_started():
    from kumo_strategies.strategies.crsi_short.tests.test_crsi_short_adapter import _lane

    lane = _lane()
    lane._account_topic = _PROBE_TOPIC
    lane._history_days = 400
    lane._iids = []
    lane._need = CrsiShortConfig().warmup_sessions
    lane._resolve_symbols_if_needed = lambda: None
    lane.begin_arming = lambda: None
    lane.cache = SimpleNamespace(orders_open=lambda **kw: [], orders_inflight=lambda **kw: [],
                                 positions_open=lambda **kw: [])
    bus = {}
    lane.msgbus = SimpleNamespace(subscribe=lambda topic, handler: bus.__setitem__(topic, handler))
    CrsiShortStrategy.on_start(lane)
    return lane, bus[_PROBE_TOPIC]


def _smhgld_started():
    lane = _smhgld_lane()
    _SmhGldHost.on_start(lane)
    subscribed = dict(lane.subscribed)
    if _PROBE_TOPIC not in subscribed:
        pytest.fail(f"SMHGLD subscribed nothing on {_PROBE_TOPIC!r} — the reader cannot be probed "
                    f"until #208's subscription exists (see the effect test)")
    return lane, subscribed[_PROBE_TOPIC]


@pytest.mark.parametrize("started", [_crsi_started, _smhgld_started], ids=["CRSISHORT", "SMHGLD"])
@pytest.mark.parametrize("frame", [{"equity": float("nan")}, {"equity": None}, {}],
                         ids=["nan", "none", "missing-key"])
def test_a_frame_that_carries_NO_NUMBER_reads_as_None_not_as_a_value_or_a_raise(started, frame):
    """Both readers carry `math.isfinite` today; a copy that drops it is exactly the sibling drift
    this file exists for. NaN passes `is not None` and `float()` and then disarms every comparison
    downstream (the daily-loss halt included — it fails by never halting); a missing key must be
    None, not KeyError inside a bus callback."""
    lane, handler = started()
    handler(frame)
    assert lane.broker_equity() is None, f"{frame!r} read as {lane.broker_equity()!r}"


def test_SMHGLD_has_no_OTHER_reader_of_broker_equity_than_the_cockpit_probe():
    """States, on the record, what the lead's grep found: inside this package nothing in the SMHGLD
    adapter reads `broker_equity()` — the lane sizes from its sleeve (platform issue 986). So #208 is an
    error storm and a masked probe, not a sizing defect. If a reader appears, this fails and the
    ticket's impact line must be rewritten, not the assertion."""
    src = inspect.getsource(SmhGldSleeveStrategy)
    tree = ast.parse(textwrap.dedent(src))
    readers = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "broker_equity"
    ]
    assert readers == [], (
        f"SmhGldSleeveStrategy now reads broker_equity() at {len(readers)} site(s) — #208's impact "
        f"('not a sizing blocker') no longer holds; re-derive it")
