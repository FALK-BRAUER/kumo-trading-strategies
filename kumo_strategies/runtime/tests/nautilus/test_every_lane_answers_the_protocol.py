"""Every registered lane implements all three `MarketAware` hooks, honestly (#873, #147).

Cockpit's poller is LIVE on ibkr-paper, asking every lane every 60 seconds, and reading `not_asked`
for all of them — because nothing implements them. The consumer is deployed; the producers are the
gap.

`not_asked` is the unknown-read-as-pass class at the top of the protocol: cockpit cannot tell "no
lane implements this" from "every lane says nothing is wrong". The moment the lanes answer, the
poller's three states mean what they say.

WHAT HONEST MEANS HERE, and why a stub would fail these tests:

  - A lane with no view answers RISK_ON **with the reason** ("no market view configured"), which is
    a real answer about a real declaration, not a placeholder.
  - A lane with no emergency trigger answers NO **with the reason that no trigger is wired** — not
    silence, not None.
  - A lane with no envelope answers UNKNOWN **with the reason**, which is the third state doing its
    job rather than a gap.

THE TEST IS OVER EVERY LANE, NOT PER LANE. An hour ago a per-lane test would have passed for the
three adapters that already subscribed their account handler and would never have been written for
the two that did not — the template being one of them, and the template being where the defect was
copied from. Same argument, same day, so the class test comes first here.

TWO CONSTRAINTS THAT ARE NOT STYLE. A hook must never RAISE: cockpit counts faults and pages on
them, so a raising hook is a lane reporting itself broken. A hook must never BLOCK: there is no
timeout on the poller's side, so a hook that reaches for data stalls neighbouring lanes' timers.
Both are asserted below rather than documented.
"""

from __future__ import annotations

import inspect

import pytest

from kumo_strategies.strategies.bct.nautilus import BCTRotationStrategy
from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy
from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy
from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import QC27RotationStrategy
from kumo_strategies.strategies.qc345_rotation.nautilus import QC345RotationStrategy
from kumo_strategies.strategies.template.nautilus import TemplateRotationStrategy
from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, NO, OUT_OF_ENVELOPE, RISK_OFF, RISK_ON, UNKNOWN, YES, Assessment, SelfAction,
    Verdict)

#: Every lane the platform can register. The template is included deliberately: it is what new lanes
#: are copied from, and it is where two defects were copied from today.
LANES = [MomentumRotationStrategy, BCTRotationStrategy, QC27RotationStrategy,
         QC345RotationStrategy, CrsiShortStrategy, TemplateRotationStrategy]

HOOKS = ["entries_blocked", "emergency_exit", "self_assessment"]



def _bare(cls):
    """A lane instance with NOTHING initialised — no `_cfg`, no `_bars`, no Nautilus `id`.

    `cls.__new__(cls)` rather than `object.__new__(cls)`: these classes are backed by Nautilus'
    Cython `Strategy`, which refuses the generic allocator outright.

    This is the most hostile shape a hook can meet and it is not contrived — cockpit's poller can
    reach a lane in the seconds between registration and `on_start`, and every hook must ANSWER
    there rather than raise. A fixture that constructed a working lane would test the happy path
    and miss precisely the state the poller sees first.
    """
    return cls.__new__(cls)


def _ids(c):
    return c.__name__


@pytest.mark.parametrize("cls", LANES, ids=_ids)
@pytest.mark.parametrize("hook", HOOKS)
def test_the_lane_IMPLEMENTS_the_hook(cls, hook):
    """Present and callable. `MarketAware` is a Protocol, so nothing enforces this structurally —
    a lane that implements none of it typechecks perfectly and the poller reads `not_asked`."""
    fn = getattr(cls, hook, None)
    assert fn is not None, (
        f"{cls.__name__} does not implement {hook}(): cockpit's poller reads `not_asked` for this "
        f"lane, which is indistinguishable from the lane saying nothing is wrong")
    assert callable(fn)


@pytest.mark.parametrize("cls", LANES, ids=_ids)
@pytest.mark.parametrize("hook", HOOKS)
def test_the_hook_TAKES_NOTHING_the_poller_cannot_supply(cls, hook):
    """`MarketAware` declares `hook(self)`. A required argument makes the lane uncallable by the
    poller and the failure arrives as a fault, which cockpit pages on."""
    fn = getattr(cls, hook)
    params = [p for name, p in inspect.signature(fn).parameters.items() if name != "self"]
    required = [p for p in params if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert not required, (
        f"{cls.__name__}.{hook} requires {[p.name for p in required]}; the poller calls it with "
        f"no arguments")


@pytest.mark.parametrize("cls", LANES, ids=_ids)
@pytest.mark.parametrize("hook", HOOKS)
def test_the_hook_ANSWERS_on_a_bare_lane_and_never_raises(cls, hook):
    """THE CONSTRAINT THAT BITES. A lane polled seconds after registration holds no bars, no config
    beyond its defaults and no live history — and that is the NORMAL case for weeks, not an error.

    `object.__new__` rather than a constructed strategy on purpose: it is the most hostile shape the
    hook can meet, with every attribute absent. A hook that reads `self._cfg` without guarding it
    raises `AttributeError` here, and a raising hook is a lane reporting itself broken to a surface
    that pages on faults.
    """
    lane = _bare(cls)
    try:
        got = getattr(cls, hook)(lane)
    except Exception as exc:                                            # noqa: BLE001
        pytest.fail(f"{cls.__name__}.{hook} raised {type(exc).__name__}: {exc}. A hook must never "
                    f"raise — cockpit counts faults and pages on them.")
    want = Assessment if hook == "self_assessment" else Verdict
    assert isinstance(got, want), f"{cls.__name__}.{hook} returned {type(got).__name__}, not {want.__name__}"


@pytest.mark.parametrize("cls", LANES, ids=_ids)
@pytest.mark.parametrize("hook", HOOKS)
def test_the_answer_CARRIES_ITS_REASON(cls, hook):
    """UNKNOWN is a first-class answer WITH a reason. An empty `reasons` standing in for "fine" is
    the bare-bool failure this protocol's three states exist to prevent, reintroduced as a tuple."""
    lane = _bare(cls)
    got = getattr(cls, hook)(lane)
    assert got.reasons, (
        f"{cls.__name__}.{hook} answered {got.state!r} with no reason. An operator being de-risked "
        f"deserves to know why, and an unexplained answer is what makes a valve unauditable")
    assert all(isinstance(r, str) and r.strip() for r in got.reasons)


@pytest.mark.parametrize("cls", LANES, ids=_ids)
@pytest.mark.parametrize("hook", HOOKS)
def test_the_hook_DOES_NO_IO(cls, hook):
    """No timeout exists on the poller's side, so a hook that reaches for anything stalls the timers
    of every lane behind it. AST-bound: find any call to a name that fetches, sleeps or awaits."""
    import ast, textwrap
    fn = None
    for klass in cls.__mro__:
        fn = klass.__dict__.get(hook)
        if fn is not None:
            break
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    forbidden = {"sleep", "request_bars", "request_instrument", "urlopen", "get", "post",
                 "run_until_complete", "run_coroutine_threadsafe"}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            bad.append(type(node).__name__)
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in forbidden:
                bad.append(name)
    assert not bad, f"{cls.__name__}.{hook} blocks or does I/O: {sorted(set(bad))}"


# -- the answers themselves, per declaration -------------------------------------------------------

def test_a_lane_with_NO_VIEW_answers_RISK_ON_and_says_why():
    """MOMENTUM, BCTROT and QC345 declare `signal=NONE`. That is a REAL answer — the lane has no
    window it could measure on its own pool's usable history — not an absence of one."""
    for cls in (MomentumRotationStrategy, BCTRotationStrategy, QC345RotationStrategy):
        v = cls.entries_blocked(_bare(cls))
        assert v.state == NO, f"{cls.__name__} blocks entries without a declared view"
        assert not v.acts
        assert any("view" in r.lower() or "signal" in r.lower() for r in v.reasons), v.reasons


def test_no_lane_has_an_EMERGENCY_TRIGGER_yet_and_every_lane_SAYS_SO():
    """No lane has a trigger wired. Every lane answers NO with THAT as the reason — not silence.

    This is the test that must be deleted the day a trigger lands, and it names what to change.
    """
    for cls in LANES:
        v = cls.emergency_exit(_bare(cls))
        assert v.state == NO and not v.acts, f"{cls.__name__} declares an emergency with no trigger"
        assert any("trigger" in r.lower() for r in v.reasons), (
            f"{cls.__name__} answered NO without saying that no trigger is wired: {v.reasons}")


def test_a_lane_with_NO_ENVELOPE_answers_UNKNOWN_rather_than_healthy():
    """The third state doing its job. A lane with no registered envelope cannot place itself in a
    distribution, and answering IN_ENVELOPE would be a health claim nothing supports."""
    for cls in (QC345RotationStrategy, CrsiShortStrategy, TemplateRotationStrategy):
        a = cls.self_assessment(_bare(cls))
        assert a.state == UNKNOWN, f"{cls.__name__} claims {a.state} with no envelope registered"
        assert not a.acts and a.action is None
        assert not a.evidence_sufficient


def test_an_UNKNOWN_assessment_NEVER_stands_a_lane_down():
    """Rule 4. `action` is None unless the lane is out of its envelope AND the evidence is
    sufficient — so a lane that cannot assess itself cannot quarantine itself."""
    for cls in LANES:
        a = cls.self_assessment(_bare(cls))
        if a.state != OUT_OF_ENVELOPE or not a.evidence_sufficient:
            assert a.action is None, f"{cls.__name__} requested stand-down on {a.state!r}"
        else:
            assert a.action is SelfAction.STAND_DOWN


# -- TECHIVOL actually computes its declared view --------------------------------------------------

def _qc27_with_bars(closes):
    """A TECHIVOL lane holding `closes` for two tickers and nothing else.

    Bars only: the hook must compute from what the lane already accumulated, so a fixture that
    supplied a panel through some other door would prove nothing about the live path.
    """
    import pandas as pd

    lane = _bare(QC27RotationStrategy)
    # THE LIVE CONFIG, not a hand-built one: the point is that the lane's OWN declared
    # 50-session INDEX_VS_MA / EXIT_ONLY view is what answers.
    from kumo_strategies.strategies.qc27_tech_inverse_vol.live import live_config
    lane._cfg = live_config()
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    lane._bars = {t: [dict(ticker=t, date=d, open=c, high=c, low=c, close=c, volume=1e6)
                      for d, c in zip(dates, closes)] for t in ("AAA", "BBB")}
    return lane


def test_TECHIVOL_answers_its_DECLARED_view_and_the_two_directions_DISAGREE():
    """The lane declares INDEX_VS_MA over 50 sessions with EXIT_ONLY. Both directions are asserted
    in one test on purpose: a hook that returned NO unconditionally passes a rising-market test
    perfectly, and identical answers where they must differ is a dead mechanism.
    """
    rising = _qc27_with_bars([100.0 + i for i in range(120)])
    up = QC27RotationStrategy.entries_blocked(rising)
    assert up.state == NO, f"a rising index blocked entries: {up}"

    # A run-up, then a decline far enough below the 50-session average to be unambiguous.
    falling = _qc27_with_bars([100.0 + i for i in range(90)] + [190.0 - 4 * i for i in range(40)])
    down = QC27RotationStrategy.entries_blocked(falling)
    assert down.state == YES, f"a collapsed index did not block entries: {down}"
    assert down.acts
    assert up.state != down.state, "the view returns the same answer in both regimes — it is inert"
    assert any("average" in r for r in down.reasons), down.reasons


def test_TECHIVOL_does_NOT_declare_an_emergency_on_a_bear_signal():
    """A 50-day moving average is a BEAR DETECTOR, and TECHIVOL wires it to EXIT_ONLY.

    "Today is not the day to buy" and "get out now" are different claims. Collapsing them is #144 —
    a flag named for blocking that performed a liquidation — so the same risk-off state that blocks
    entries above must NOT raise an emergency here.
    """
    falling = _qc27_with_bars([100.0 + i for i in range(90)] + [190.0 - 4 * i for i in range(40)])
    assert QC27RotationStrategy.entries_blocked(falling).state == YES
    v = QC27RotationStrategy.emergency_exit(falling)
    assert v.state == NO and not v.acts, f"a bear signal was escalated to an emergency: {v}"
    assert any("liquidat" in r.lower() or "flatten" in r.lower() for r in v.reasons), v.reasons


def test_a_lane_declaring_LIQUIDATE_DOES_raise_the_emergency_on_the_same_bars():
    """The control for the test above. If EXIT_ONLY answered NO because the hook is inert rather
    than because the lane declared EXIT_ONLY, this fails — same bars, same signal, different
    declaration, different answer."""
    from dataclasses import replace

    from kumo_strategies.strategies.market_view import MarketAction

    falling = _qc27_with_bars([100.0 + i for i in range(90)] + [190.0 - 4 * i for i in range(40)])
    falling._cfg = replace(falling._cfg, market_view=replace(
        falling._cfg.market_view, action=MarketAction.LIQUIDATE))
    v = QC27RotationStrategy.emergency_exit(falling)
    assert v.state == YES and v.acts, (
        f"a lane declaring LIQUIDATE on a risk-off view did not ask to be flattened: {v}")


# -- the envelope branch is REACHED, not merely written --------------------------------------------

def _rotation_lane(cls, tag, *, gap=None):
    """A lane carrying a real fitted config and the order-id tag cockpit allocates it."""
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExecutionConfig, ExitConfig, MomentumRotationConfig, PortfolioConfig)

    lane = _bare(cls)
    lane._cfg = MomentumRotationConfig(
        portfolio=PortfolioConfig(n_hold=8, buffer=5),
        exits=ExitConfig(give_back_frac=0.5),
        execution=ExecutionConfig(decision_slots=("open+5m",), min_abs_gap_pct=gap))
    lane._order_id_tag = tag
    return lane


def test_MOMENTUM_and_BCTROT_RESOLVE_a_registered_envelope():
    """The branch must be REACHED. `self_assessment` answers UNKNOWN either way today — no live
    windows have accumulated — so a registry that never resolved would produce an identical-looking
    answer with a different and wrong reason, and nothing would ever notice.

    The two lanes are asymmetric by deployment: MOMENTUM runs one slot with NO gap, BCTROT three
    slots with a 1.5% gap. Resolving BCTROT to the no-gap distribution would read its percentile off
    a strategy it is not running.
    """
    mom = _rotation_lane(MomentumRotationStrategy, "002", gap=None)
    assert mom.envelope_key() == "MOMENTUM-002", mom.envelope_key()
    assert mom._registered_envelope() is not None

    bct = _rotation_lane(BCTRotationStrategy, "004", gap=0.015)
    assert bct.envelope_key() == "BCTROT-004+gap", bct.envelope_key()
    assert bct._registered_envelope() is not None

    assert mom._registered_envelope() is not bct._registered_envelope(), (
        "both lanes resolved to the SAME envelope — the key is not reading the configuration")


def test_the_two_UNKNOWNS_are_DISTINGUISHABLE_in_their_reasons():
    """"No envelope was ever fitted for me" and "I have one and no live evidence yet" are different
    facts on an operator's surface, and both are UNKNOWN. If the reasons did not differ, the
    registry would be decorative."""
    with_env = MomentumRotationStrategy.self_assessment(
        _rotation_lane(MomentumRotationStrategy, "002"))
    without = QC345RotationStrategy.self_assessment(_bare(QC345RotationStrategy))

    assert with_env.state == without.state == UNKNOWN
    assert with_env.reasons != without.reasons, (
        "a lane WITH a registered envelope gave the same reason as one without — the envelope "
        "lookup is inert")
    assert any("no registered envelope" in r for r in without.reasons), without.reasons
    assert any("has a registered envelope" in r for r in with_env.reasons), with_env.reasons


def test_DWELL_changes_the_live_answer_so_it_is_read_rather_than_merely_present():
    """`dwell` reaches the live path only THROUGH `market_state`, so "it is read" has to be shown
    rather than asserted.

    `test_every_backtest_enforced_field_is_read_live_or_DECLARED_dead` warns in its own message that
    "reclassification is not implementation". This is the evidence for removing `dwell` from that
    list: the same bars, the same lane, two dwell settings, two different answers.

    The bars cross below the average exactly ONCE at the end. dwell=1 acts on that first close;
    dwell=3 requires three consecutive and must not.
    """
    from dataclasses import replace

    # A long rise, so the 50-session average sits well below the recent closes, then ONE session
    # that plunges through it. tail(1) is all-below; tail(3) is not.
    closes = [100.0 + i for i in range(90)] + [185.0, 188.0, 100.0]
    quick = _qc27_with_bars(closes)
    quick._cfg = replace(quick._cfg, market_view=replace(quick._cfg.market_view, dwell=1))
    patient = _qc27_with_bars(closes)
    patient._cfg = replace(patient._cfg, market_view=replace(patient._cfg.market_view, dwell=3))

    a = QC27RotationStrategy.entries_blocked(quick)
    b = QC27RotationStrategy.entries_blocked(patient)
    assert a.state != b.state, (
        f"dwell 1 and dwell 3 gave the SAME answer ({a.state}) on bars that cross once — dwell is "
        f"not reaching the live decision")


def test_WINDOW_changes_the_live_answer_too():
    """Same argument for `window`: a 50-session average and a 5-session average must disagree on a
    series that has recently turned, or the window is decorative on the live path."""
    from dataclasses import replace

    # A long decline that has just bounced. The last close sits BELOW the 50-session average — the
    # decline dominates it — and ABOVE the 5-session average, which only sees the bounce. The two
    # windows must therefore disagree; if they do not, neither is being read.
    closes = [200.0 - 1.6 * i for i in range(60)] + [103.0, 106.0, 109.0, 112.0, 115.0]
    slow = _qc27_with_bars(closes)
    fast = _qc27_with_bars(closes)
    fast._cfg = replace(fast._cfg, market_view=replace(fast._cfg.market_view, window=5))

    a = QC27RotationStrategy.entries_blocked(slow)
    b = QC27RotationStrategy.entries_blocked(fast)
    assert a.state != b.state, (
        f"a 50-session and a 5-session average gave the same answer ({a.state}) on a series that "
        f"has just turned — window is not reaching the live decision")
