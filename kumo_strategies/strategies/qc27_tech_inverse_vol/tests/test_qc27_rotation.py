"""Tests for the QC27 Nautilus binding (#33).

Same standard as `test_qc345_rotation.py`: no `BacktestEngine` (a second engine per interpreter
aborts natively), and the clock is a REAL Nautilus `TestClock` rather than a stub — the whole reason
the trigger lives on Nautilus's clock is that a hand-rolled one could not be exercised by a backtest,
and testing against a hand-rolled stub would give that back.

What is pinned: identity and the tag collision that would stop the node booting, warmup arithmetic,
the monthly cadence coming from the backtest's own function, and the rule that book state moves only
on order EVENTS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.strategies.qc27_tech_inverse_vol.nautilus import (
    EXTERNAL_ID, STRATEGY_NAME, QC27RotationStrategy, warmup_bars_needed,
)
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    QC27TechInverseVolConfig, rebalance_dates,
)


LIVE_CFG = QC27TechInverseVolConfig(momentum_price_field="close")


def _strategy(**kw) -> QC27RotationStrategy:
    return QC27RotationStrategy(
        cfg=kw.pop("cfg", LIVE_CFG),
        instrument_ids=kw.pop("instrument_ids", []),
        order_id_tag=kw.pop("order_id_tag", "005"),
        **kw,
    )


def _event(sym: str, side: str = "BUY", coid: str = "O-20260831-000001"):
    """A REAL `InstrumentId`, not a SimpleNamespace. Production reads
    `str(event.instrument_id.symbol)`, and a Nautilus `Symbol` stringifies to its value while a
    namespace double stringifies to "namespace(value='AAA')" — so every lookup would miss and the
    tests would fail for a reason production never hits."""
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import InstrumentId

    # A REAL `OrderSide` too, for the same reason (kumo-trading-platform issue 748). Production's `OrderFilled`
    # carries a side and a client order id, and the handlers now read both: a SELL cannot complete an
    # entry, and an order another component minted is ignored. A double without them was testing
    # handlers that could not tell their own fills from a protective stop's.
    return SimpleNamespace(
        instrument_id=InstrumentId.from_str(f"{sym}.XNAS"),
        order_side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
        client_order_id=coid,
    )


# -- identity ------------------------------------------------------------------------------------
def test_order_id_tag_is_required_and_has_no_default():
    """`research/qc27/ACTIVATION.md` proposed TECH_IVOL-003. 003 is QC345's. Two strategies sharing
    an order_id_tag does not degrade — Nautilus raises at `Trader.add_strategy` and the node does
    not boot. BCTROT already paid for this once (003 -> 004). Cockpit allocates the tag, so this
    constructor must refuse to guess one."""
    with pytest.raises(TypeError, match="order_id_tag"):
        QC27RotationStrategy(cfg=LIVE_CFG, instrument_ids=[])


def test_identity_is_set_through_the_config_not_change_id():
    """Set via StrategyConfig so the tag binds at registration; `change_id()` would leave the tag at
    its default and desync client_order_id generation from the position id."""
    s = _strategy(order_id_tag="005")
    assert s.config.strategy_id == STRATEGY_NAME
    assert s.config.order_id_tag == "005"
    assert str(s.id) == "TECHIVOL-005"
    assert EXTERNAL_ID == "QC27", "QC27 stays provenance; the wire id is the strategy name"


def test_the_tag_is_not_one_already_allocated():
    """MANUAL 001, MOMENTUM 002, QC345 003, BCTROT 004 are live. Pinning that the tested default is
    none of them, so a copy-paste of this test file cannot reintroduce the collision."""
    taken = {"001", "002", "003", "004"}
    assert _strategy().config.order_id_tag not in taken


# -- warmup --------------------------------------------------------------------------------------
def test_warmup_takes_the_max_not_the_sum():
    """Every window is a TRAILING window ending at the same bar, so they overlap. Summing them
    overstates the requirement — QC345 shipped that bug and reported 295 bars where 254 was right."""
    cfg = QC27TechInverseVolConfig()
    need = warmup_bars_needed(cfg)
    assert need == max(cfg.lookback_sessions + 2, cfg.warmup_sessions,
                       cfg.liquidity_window + 1, cfg.realized_vol_window + 1)
    assert need < (cfg.lookback_sessions + 2) + cfg.warmup_sessions, "summed instead of maxed"


def test_warmup_honours_qc27s_own_stated_100_session_warmup():
    """The strategy page specifies a 100-session warmup and momentum only needs 65, so the stated
    figure is what binds. A config change that raised lookback past 100 must move this."""
    assert warmup_bars_needed(QC27TechInverseVolConfig()) == 100
    assert warmup_bars_needed(QC27TechInverseVolConfig(lookback_sessions=200)) == 202


def test_refuses_to_rank_before_the_universe_is_warm():
    """A cross-sectional strategy warms as a UNIVERSE. Ranking three warm names out of fifty is not
    an early answer, it is a different and wrong strategy."""
    s = _strategy()
    assert s.warm is False
    for sym in [f"T{i}" for i in range(s._cfg.portfolio_size)]:
        s._bars[sym].extend([{"date": pd.Timestamp("2025-01-02")}] * s._need)
    assert s.warm is True


# -- monthly cadence -----------------------------------------------------------------------------
def test_the_cadence_rule_is_the_backtests_own_function():
    """One derivation, not two. A second adapter-local notion of 'first session of the month' is a
    second derivation of a fact the backtest already derives, and two derivations disagree."""
    s = _strategy()
    sessions = pd.date_range("2025-01-01", "2025-03-31", freq="B")
    panel = pd.DataFrame({"date": sessions})
    expected = set(rebalance_dates(panel["date"]))
    for day in sessions:
        assert s._is_rebalance(day, panel) is (day in expected)


def test_only_the_first_session_of_a_month_is_a_rebalance():
    """The rule stated independently of the function, so breaking it silently requires changing
    BOTH. January 2025 opens on the 1st; February on the 3rd (the 1st is a Saturday)."""
    s = _strategy()
    panel = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-03-31", freq="B")})
    assert s._is_rebalance(pd.Timestamp("2025-01-01"), panel) is True
    assert s._is_rebalance(pd.Timestamp("2025-01-02"), panel) is False
    assert s._is_rebalance(pd.Timestamp("2025-02-03"), panel) is True
    assert s._is_rebalance(pd.Timestamp("2025-02-04"), panel) is False


def test_the_adapter_reads_the_cadence_from_the_config():
    """THE SECOND HARDWIRED CALL SITE. `_is_rebalance` called `rebalance_dates()` with no period, so
    the adapter rebalanced monthly whatever the config said -- while the backtest took a
    `rebalance_period` argument and could report weekly. Research and production disagreed about the
    cadence with nothing failing to say so.

    Asserted on weekly dates that are NOT month starts, because those are exactly the sessions a
    hardwired-monthly adapter refuses. Found by mutation bite: reverting the fix left every other
    test in this file green."""
    weekly = QC27TechInverseVolConfig(momentum_price_field="close", rebalance_period="W")
    s = _strategy(cfg=weekly)
    panel = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-03-31", freq="B")})
    monthly = set(rebalance_dates(panel["date"], "M"))
    weekly_only = [d for d in rebalance_dates(panel["date"], "W") if d not in monthly]
    assert weekly_only, "fixture cannot distinguish the two cadences"
    for day in weekly_only:
        assert s._is_rebalance(day, panel) is True, f"weekly config refused {day.date()}"


def test_the_default_cadence_is_still_monthly():
    """The fix must not silently change what an unconfigured strategy does."""
    assert QC27TechInverseVolConfig().rebalance_period == "M"


def test_an_empty_panel_is_never_a_rebalance():
    assert _strategy()._is_rebalance(pd.Timestamp("2025-01-01"), pd.DataFrame({"date": []})) is False


# -- book state ----------------------------------------------------------------------------------
def test_book_state_moves_only_on_fills():
    """Submitting is not holding. Marking held at submit is how a rejected entry becomes a phantom
    position the strategy then refuses to re-enter."""
    s = _strategy()
    s._pending["AAA"] = "enter"
    assert "AAA" not in s._held
    s.on_order_filled(_event("AAA"))
    assert "AAA" in s._held


def test_a_rejected_entry_does_not_enter_the_book():
    s = _strategy()
    s._pending["AAA"] = "enter"
    s.on_order_rejected(_event("AAA"))
    assert "AAA" not in s._held
    assert "AAA" not in s._pending


def test_a_denied_exit_leaves_the_position_held():
    """The mirror case. Forgetting the pending intent must not also forget that we still own it —
    a position with no claim reads as foreign and never gets exited."""
    s = _strategy()
    s._held.add("AAA")
    s._pending["AAA"] = "exit"
    s.on_order_denied(_event("AAA"))
    assert "AAA" in s._held


# -- bar ingestion -------------------------------------------------------------------------------
def test_republished_daily_bar_replaces_rather_than_appends():
    """Alpaca republishes the CURRENT session's daily bar as it updates. Appending every arrival put
    ~103 copies of one date per symbol after thirteen hours of uptime, which gave the trailing
    window zero variance and a NaN score, and the strategy sat inert."""
    s = _strategy()

    # REAL Nautilus identifiers. A SimpleNamespace double stringifies to
    # "namespace(value='AAA')" rather than "AAA", so `_ingest`'s
    # `str(bar.bar_type.instrument_id.symbol)` keys under the wrong symbol and the deque stays
    # empty -- the test then fails for a reason production never hits. This double was written that
    # way first and caught by exactly that symptom; the fix is to use the type production uses.
    from nautilus_trader.model.data import BarType

    bt = BarType.from_str("AAA.XNAS-1-DAY-LAST-EXTERNAL")

    class _B:
        def __init__(self, ts, close):
            self.ts_event, self.open, self.high, self.low, self.close, self.volume = (
                ts, close, close, close, close, 1_000)
            self.bar_type = bt

    day = pd.Timestamp("2025-01-02", tz="UTC").value
    s._ingest(_B(day, 10.0))
    s._ingest(_B(day, 11.0))              # same session, republished
    assert len(s._bars["AAA"]) == 1, "republished bar appended instead of replacing"
    assert s._bars["AAA"][-1]["close"] == 11.0, "kept the stale copy instead of the latest"


# -- the live path cannot supply adjusted prices ---------------------------------------------------
def test_a_price_field_a_live_bar_cannot_supply_is_refused_at_construction():
    """THE DEFECT THE NAUTILUS BUILD EXISTS TO CATCH.

    `build_feature_panel` requires `cfg.momentum_price_field` as a COLUMN and defaults to
    `close_adj`. A Nautilus `Bar` carries OHLCV and nothing else, so the live feed can never produce
    it. Unguarded, that does not fail at startup — it fails at the FIRST REBALANCE, weeks later,
    with `bars missing momentum price field 'close_adj'`. Identical shape to QC345's
    `KeyError: 'eligible'`, which sat latent until the strategy was enabled and then booked nothing
    on its first live session.

    Refusing at construction turns a silent latent crash into a node that will not boot, in front of
    whoever is deploying.
    """
    with pytest.raises(ValueError, match="close_adj"):
        QC27RotationStrategy(cfg=QC27TechInverseVolConfig(),      # default IS close_adj
                             instrument_ids=[], order_id_tag="005")


def test_the_error_names_the_fix_and_its_cost():
    """An error that says only 'invalid' makes the next person rediscover the analysis. This one has
    to name the working setting AND the exposure that setting carries, because QC27 — unlike QC345 —
    has no corporate_action_window to neutralise splits."""
    with pytest.raises(ValueError) as e:
        QC27RotationStrategy(cfg=QC27TechInverseVolConfig(), instrument_ids=[], order_id_tag="005")
    msg = str(e.value)
    assert "momentum_price_field='close'" in msg, "does not name the fix"
    assert "corporate_action_window" in msg, "does not name the cost of the fix"


def test_a_field_a_bar_does_supply_constructs_fine():
    s = QC27RotationStrategy(cfg=QC27TechInverseVolConfig(momentum_price_field="close"),
                             instrument_ids=[], order_id_tag="005")
    assert str(s.id) == "TECHIVOL-005"


# -- the venue's actual answer must reach the journal -----------------------------------------------
def _recording_strategy():
    """A strategy with a runner that records terminals, and a loop to schedule onto."""
    import asyncio

    class Runner:
        def __init__(self):
            self.terminal = []

        async def record_terminal(self, session, symbol, ok, detail):
            self.terminal.append((symbol, ok, detail))

    # A RUNNING loop in a thread, mirroring production: the adapter schedules onto the node's loop
    # from Nautilus's dispatch thread. An idle `new_event_loop()` accepts the schedule and never runs
    # it, so the test would see nothing recorded whether or not the handlers were wired.
    import threading

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    s = _strategy(session_runner=Runner())
    s._loop = loop
    return s


@pytest.mark.parametrize("handler,ok", [
    ("on_order_filled", True), ("on_order_rejected", False),
    ("on_order_denied", False), ("on_order_canceled", False),
])
def test_every_order_event_records_the_venues_answer(handler, ok):
    """kumo-trading-platform issue 383, IN THE MIRROR — and I wrote the lesson into the method I then failed to call.

    MOMENTUM had the CALLER wired for weeks with no gateway method, so every call no-opped through a
    getattr: 22 fills, zero terminal rows. QC27 is the reverse — `QC27SessionRunner.record_terminal`
    exists, with a docstring saying exactly why ("its ABSENCE is what left MOMENTUM with 22 fills and
    zero terminal rows") — and the adapter wires NONE of its four order handlers to it.

    So TECHIVOL-005 writes no terminal rows at all. The submit-time row records Nautilus ACCEPTING the
    order, never what the venue said, and the retry counter that reads `phase == "terminal" and not
    ok` can never advance — a symbol the venue refused is silently treated as done.

    QC345 wires all four. Asserted per handler rather than once, because wiring three of four is the
    version of this that looks fixed.
    """
    import time

    s = _recording_strategy()
    getattr(s, handler)(_event("AAA"))
    for _ in range(200):                       # the record lands on the loop thread
        if s._runner.terminal:
            break
        time.sleep(0.01)
    assert s._runner.terminal, f"{handler} recorded nothing — the venue's answer never reaches the journal"
    sym, got_ok, _ = s._runner.terminal[-1]
    assert (sym, got_ok) == ("AAA", ok)


# -- the slot a session files under is the slot it FIRED at, not the one armed next ---------------
def _fired_slot_for(cls, *, offset_at_fire: int, offset_after_rearm: int):
    """Drive `_on_session_alert` with an `_arm` that MOVES the offset, and report what the session
    would file under. `__new__` rather than the full constructor: this exercises the alert path
    only, and a real Nautilus Strategy needs a node to construct."""
    # `clock` is a Cython attribute on Nautilus's `Actor` and is not writable on an instance, so
    # the probe SUBCLASSES the strategy and shadows it with a plain property. The alert path under
    # test is inherited unchanged.
    class _Probe(cls):
        clock = property(lambda self: SimpleNamespace(
            utc_now=lambda: pd.Timestamp("2026-08-24T16:00:00Z")))
        log = property(lambda self: SimpleNamespace(
            warning=lambda *a, **k: None, info=lambda *a, **k: None))

    s = _Probe.__new__(_Probe)
    s._open_offset = offset_at_fire
    s._armed_session = pd.Timestamp("2026-08-24")
    # `_due = None` skips the missed-rebalance block entirely, so the read-only `warm`
    # property is never reached. Nothing here needs to fake it.
    s._due = None

    seen = {}
    # `_arm` re-reads the settings and moves the offset — which is exactly what #514's fix makes it
    # do. Before that fix this could not happen, so this test could not fail.
    def _arm(_after):
        s._open_offset = offset_after_rearm
    s._arm = _arm
    # `_session_coro` is where the real slot is read; capture what it WOULD be handed.
    s._try_decide = lambda: seen.setdefault("slot", s._fired_slot)

    s._on_session_alert()
    return seen.get("slot")


def test_the_session_files_under_the_slot_it_FIRED_at_not_the_next_one():
    """ba37ef9 REINTRODUCED BY #514's FIX, unless the fired slot is snapshotted before re-arming.

    `_on_session_alert` re-arms FIRST — deliberately, because a lane that stopped scheduling looks
    exactly like one that decided to hold. `_slot_name` is a LIVE property over `self._open_offset`,
    read later at decide time. That is safe only while the offset can never change.

    #514 makes it change: cockpit edits `*_SLOTS`, `_arm` re-reads, and the session that just fired
    at the OLD offset gets journalled under the NEW one. Same defect ba37ef9 fixed — settings said
    `open+315m`, the lane fired at open+315m, the journal said `open+150m` — but WORSE, because it
    now appears only on the session after an operator edits a setting. It correlates with the edit,
    so it looks like the edit working.

    The fix is a snapshot taken before `_arm`, which is what `momentum_rotation` already does with
    `_armed_slot` -> `_due_slot`.
    """
    assert _fired_slot_for(QC27RotationStrategy, offset_at_fire=150, offset_after_rearm=315) \
        == "open+150m", "the session filed under the slot armed NEXT, not the one it fired at"


def test_the_snapshot_is_not_a_frozen_constant():
    """The control. Returning a hardcoded "open+150m" would satisfy the test above. A session that
    fires at 315 must file under 315."""
    assert _fired_slot_for(QC27RotationStrategy, offset_at_fire=315, offset_after_rearm=150) \
        == "open+315m"


# -- the slot knob must be LIVE, not captured at build (kumo-trading-platform issue 514) --------------------------
def _armed_offsets(reader, *, start=150):
    """Arm twice with a settings reader in between, and report the offset used each time."""
    # ONE clock object, not a fresh namespace per access — a property returning a new object each
    # time silently discarded `set_time_alert` and every arm blew up on the throwaway.
    _clock = SimpleNamespace(utc_now=lambda: pd.Timestamp("2026-08-24T16:00:00Z"),
                             set_time_alert=lambda *a, **k: None)
    _log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)

    class _Probe(QC27RotationStrategy):
        clock = property(lambda self: _clock)
        log = property(lambda self: _log)
        id = property(lambda self: "TECHIVOL-005")

    s = _Probe.__new__(_Probe)
    s._open_offset = start
    s._read_open_offset = reader
    s._armed_session = None
    used = []
    s._calendar = SimpleNamespace(next_fire=lambda after, off: (
        used.append(off) or (pd.Timestamp("2026-08-25"), pd.Timestamp("2026-08-25T16:00:00Z"))))
    s._arm(pd.Timestamp("2026-08-24T16:00:00Z"))
    s._arm(pd.Timestamp("2026-08-25T16:00:00Z"))
    return used


def test_a_settings_change_moves_the_next_alert_WITHOUT_a_redeploy():
    """kumo-trading-platform issue 514. `_slot_and_offset_from_settings()` was read once at build and captured, and
    `set_time_alert(..., override=True)` re-armed from the boot-time copy forever — so an operator
    could edit `*_SLOTS`, see it accepted, and watch the lane keep firing at the old time until the
    engine was recreated. Verified live 2026-08-24: nothing changed until the recreate.

    Same dead-knob class as `qc27.ALLOCATED_EQUITY`, which went unnoticed for the lane's whole life
    because the constant happened to equal the setting.
    """
    moved = iter([150, 315])
    assert _armed_offsets(lambda: next(moved)) == [150, 315], \
        "the second arm reused the boot-time offset — the knob is still dead"


def test_no_reader_is_TODAYS_BEHAVIOUR_byte_for_byte():
    """Every backtest and test passes no reader. If this moves, the fix has changed a live schedule
    rather than making it editable."""
    assert _armed_offsets(None) == [150, 150]


def test_a_RAISING_reader_keeps_the_lane_SCHEDULING():
    """A settings read that fails must not stop the lane arming. `_on_session_alert` re-arms first
    precisely because a strategy that stopped scheduling is indistinguishable from one that decided
    to hold — silent, and wrong until somebody looks. So the old offset stands."""
    def boom():
        raise RuntimeError("settings unavailable")
    assert _armed_offsets(boom) == [150, 150]


def test_a_JUNK_offset_is_refused_rather_than_adopted():
    """A slot that cannot be resolved must not become a session that silently decides fewer times
    than configured (the #26 shape). None, non-numeric and negative all keep the current value."""
    for bad in (None, "not-a-number", -30):
        assert _armed_offsets(lambda: bad) == [150, 150], f"adopted {bad!r}"
