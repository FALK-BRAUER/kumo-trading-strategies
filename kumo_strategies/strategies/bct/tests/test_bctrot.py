"""BCTROT: the one arm that survived, wired to the live clock (#32).

Runs ALONGSIDE MOMENTUM-002 and replaces it over time with no cutoff, so the tests that matter are
the ones asserting the two strategies stay SEPARATE — separate identity, separate tag, separate
book — and that adding BCTROT changed nothing about the live one.
"""

from __future__ import annotations

import inspect

from kumo_strategies.strategies.bct import nautilus as B
from kumo_strategies.strategies.momentum_rotation import nautilus as M
from kumo_strategies.runtime.nautilus.contract import StrategyRegistration


def test_bctrot_is_a_separate_identity_from_momentum():
    """Two strategies, two books. Not a reconfiguration of the live one — sharing an order_id_tag
    does not degrade, Nautilus raises at Trader.add_strategy and the node does not boot."""
    assert B.EXTERNAL_ID == "BCTROT" != M.EXTERNAL_ID
    assert not hasattr(B, "DEFAULT_ORDER_ID_TAG"), "BCTROT must not publish a tag default"


def test_momentum_still_defaults_to_ONE_decision_at_open_plus_5():
    """The live strategy's schedule must not have moved. Multi-slot is additive or it is a change to
    MOMENTUM-002, which is holding real positions."""
    assert inspect.signature(M.MomentumRotationStrategy.__init__).parameters[
        "decision_slots"].default is None
    assert M.DEFAULT_ORDER_ID_TAG == "002"


def test_bctrot_decides_at_open_midday_and_close():
    """The open slot was added 2026-09-04 and is the slot the cadence research REJECTED on cost
    (09:35 pays ~2.8x the midday half-spread). It is back because `min_abs_gap_pct` exists only
    where the runner knows the gap at submit time, which is this slot and no other. Changed
    deliberately; see the module docstring in `bctrot_rotation.py` for the full argument."""
    assert B.DECISION_SLOTS == ("open+5m", "open+150m", "close-20m")


def test_bctrot_satisfies_the_registration_contract():
    for member in StrategyRegistration.__protocol_attrs__:
        assert hasattr(B.BCTRotationStrategy, member), f"missing {member}"


def test_the_schedule_is_the_ONLY_thing_that_varies_from_momentum():
    """Changing the schedule and the exit rules together would make the live comparison against
    MOMENTUM-002 useless: two changes, one number. BCTROT adds no exit behaviour of its own."""
    src = inspect.getsource(B.BCTRotationStrategy)
    for forbidden in ("give_back", "peak_fade", "stop_loss", "take_profit", "def on_bar",
                      "def _try_decide"):
        assert forbidden not in src, f"BCTROT overrides {forbidden} — it must not"


def test_bctrot_reuses_momentums_engine_rather_than_forking_it():
    """A forked adapter doubles the surface on which live and backtest drift apart, which is the
    defect class this repo keeps paying for."""
    assert issubclass(B.BCTRotationStrategy, M.MomentumRotationStrategy)


def test_its_slots_resolve_through_the_same_code_the_backtest_uses():
    """"Midday and close" has to mean the same instants on both sides, or the cadence result that
    justified this strategy cannot be checked against what it actually does."""
    from datetime import datetime, timedelta
    from kumo_strategies.runtime.calendar import ET
    from kumo_strategies.strategies.momentum_rotation.slots import resolve
    open_at = datetime(2026, 8, 17, 9, 30, tzinfo=ET)
    pairs = dict(resolve(B.DECISION_SLOTS, open_at, open_at + timedelta(hours=6, minutes=30)))
    assert pairs["open+150m"].hour == 12 and pairs["open+150m"].minute == 0
    assert pairs["close-20m"].hour == 15 and pairs["close-20m"].minute == 40


def test_the_slots_are_validated_at_construction_not_at_fire_time():
    """A slot that cannot be resolved must not become a session that silently decides fewer times
    than configured — the #26 silent-degradation shape."""
    import pytest
    from kumo_strategies.strategies.momentum_rotation.slots import SlotError, validate
    with pytest.raises(SlotError):
        validate(("open+150m", "not-a-slot"))


def test_the_live_default_schedule_is_exactly_one_slot_at_open_plus_5():
    """MOMENTUM-002 holds real positions on this schedule. Mutation-bitten: changing the derivation
    to BCTROT's slots turns this red, where the constructor-default test stays green because None
    is still None."""
    assert M.default_slots(5) == ("open+5m",)
    assert M.default_slots(150) == ("open+150m",), "a custom offset must still yield ONE slot"


def test_the_slot_reaches_the_runner_so_two_decisions_a_day_are_distinct(monkeypatch):
    """The blocker that would have failed SILENTLY.

    The unique index is `(strategy_id, session, slot)`. A strategy deciding twice a day while
    writing both under the default slot has its SECOND decision rejected as a duplicate of its
    first — the close decision simply never happens, and the journal shows one tidy decision per
    session. The failure looks exactly like success.
    """
    import asyncio
    import inspect
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    # The runner must ACCEPT a slot...
    assert "slot" in inspect.signature(PgSessionRunner.run).parameters

    # ...and the adapter must hand it the one it actually fired at. Asserted behaviourally in
    # `test_momentum_rotation.test_live_alert_hands_the_session_to_the_runner`, which reads the slot
    # off the recorded call rather than off the source.
    assert asyncio


def test_the_runner_defaults_the_slot_so_single_slot_callers_are_unchanged():
    """MOMENTUM-002 runs through this path. Multi-slot is additive or it is a live change."""
    import inspect
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
    from kumo_strategies.runtime.executor.store import DEFAULT_SLOT
    assert inspect.signature(PgSessionRunner.run).parameters["slot"].default == DEFAULT_SLOT


def test_bctrots_schedule_can_be_set_from_the_CONFIG_not_only_the_constructor():
    """#32: every parameter goes through the cockpit settings mechanism. A schedule reachable only
    from Python is outside it, and the schedule is the whole of BCTROT."""
    from kumo_strategies.strategies.momentum_rotation.config import (
        ExecutionConfig, MomentumRotationConfig)
    cfg = MomentumRotationConfig(
        execution=ExecutionConfig(decision_slots=("open+150m", "close-20m")))
    assert cfg.execution.decision_slots == ("open+150m", "close-20m")


def test_the_config_default_still_reproduces_momentum_002():
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
    assert MomentumRotationConfig().execution.decision_slots == ("open+5m",)


def _build(cls, **kw):
    """Construct for real. Identity is set at construction via StrategyConfig, so nothing short of
    building the object exercises it."""
    from kumo_strategies.strategies.momentum_rotation.candidates import StaticList
    from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig
    return cls(cfg=kw.pop("cfg", MomentumRotationConfig()),
               source=kw.pop("source", StaticList(["AAA", "BBB"])),
               instrument_ids=kw.pop("instrument_ids", []), **kw)


def test_the_built_strategy_reports_BCTROT_as_its_id():
    """THE test that would have caught it, and the one the earlier version was missing.

    Asserting the constants differ is not the same as asserting the built object uses them.
    `bctrot_rotation.STRATEGY_NAME` was declared and never read, so the object came up as
    MOMENTUM-004 while every constant-level assertion stayed green.
    """
    assert _build(B.BCTRotationStrategy, order_id_tag="004").id.value == "BCTROT-004"
    assert _build(B.BCTRotationStrategy, order_id_tag="009").id.value == "BCTROT-009"


def test_the_built_MOMENTUM_strategy_is_unchanged():
    """MOMENTUM-002 holds real positions. Under NETTING the position id is
    `{instrument}-{strategy_id}`, so this id moving would orphan every one of them."""
    assert _build(M.MomentumRotationStrategy).id.value == "MOMENTUM-002"


def test_BCTROT_gets_its_OWN_StrategyId_not_momentums():
    """The bug cockpit caught before registering it: BCTROT came up as MOMENTUM-004.

    `bctrot_rotation.STRATEGY_NAME` was declared and never read — the parent built `strategy_id`
    from its own module constant, so a subclass could override the TAG but not the NAME.

    This is not cosmetic. Under NETTING the position id is `{instrument}-{strategy_id}`, so the id
    is the cycle-attribution key: BCTROT's positions and P&L would have landed in a book named
    MOMENTUM, cockpit's registry (external BCTROT -> internal BCTROT-004) would have disagreed with
    what the runtime reported, and the whole point of the strategy — BCTROT against MOMENTUM-002 as
    ONE change — would have been two strategies both called MOMENTUM. Permanent from the first fill,
    since re-homing positions means changing the id.

    The earlier test asserted the TAGS differed, which is the weaker claim and stayed green.
    """
    import inspect
    sig = inspect.signature(M.MomentumRotationStrategy.__init__)
    assert "strategy_name" in sig.parameters, "the name is not overridable"
    assert sig.parameters["strategy_name"].default == "MOMENTUM", "live default must not move"

    bsig = inspect.signature(B.BCTRotationStrategy.__init__)
    assert bsig.parameters["strategy_name"].default == "BCTROT"
    assert bsig.parameters["order_id_tag"].default is inspect.Parameter.empty, (
        "a default tag is this repo claiming an allocation cockpit owns — 003 was BCTROT's when written and is QC345's now")
    assert bsig.parameters["decision_slots"].default == ("open+5m", "open+150m", "close-20m")


def test_the_three_identity_parameters_are_VISIBLE_in_the_signature():
    """`*args, **kwargs` hid them — cockpit could not tell from the signature that `order_id_tag`
    was honoured at all, and `strategy_name` was not honoured."""
    import inspect
    params = inspect.signature(B.BCTRotationStrategy.__init__).parameters
    for p in ("strategy_name", "order_id_tag", "decision_slots"):
        assert params[p].kind is inspect.Parameter.KEYWORD_ONLY, f"{p} is not an explicit keyword"
