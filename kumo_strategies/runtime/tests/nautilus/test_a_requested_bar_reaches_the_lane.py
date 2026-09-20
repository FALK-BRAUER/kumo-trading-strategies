"""A lane that REQUESTS bars must be able to RECEIVE them (#202).

`request_bars` and `subscribe_bars` deliver on DIFFERENT handlers: a live bar arrives on `on_bar`,
a requested batch on `on_historical_data`. ks#200 wired the request and not the receipt, so on
ibkr-paper the batch arrived, the strategy logged receiving it, and `_bars` stayed empty:

    18:24:24  SMHGLD: [REQ]--> RequestBars(SMH.XNAS-1-DAY-LAST-EXTERNAL, start 2026-08-12)
    18:24:24  DataClient: SMH.XNAS: Number of bars retrieved in batch: 21
    18:24:24  SMHGLD: Received <Bar[20]> data for SMH.XNAS-1-DAY
    18:40:00  SMHGLD-007: rung fired and this lane CANNOT DECIDE —
              bars GLD 0/3 SHORT, SMH 0/3 SHORT

Twenty bars per leg delivered, zero counted. The four lanes that warm implement the handler; the two
that could not warm did not.

THE PAIR THIS FILE BINDS IS REQUEST -> RECEIPT, and it is the third such pair to fail today: a slot
argument with no method to read it (#197), a warmth requirement with no request to satisfy it
(#200), and now a request with no handler to receive it. Each one was one end of a seam landing
without the other, and each was silent.

WHY THE LOG LINE DID NOT HELP. The strategy itself printed "Received <Bar[20]> data" — Nautilus logs
the delivery, not the handling. A lane can be told it received something and do nothing with it, and
those two facts sit adjacent in the same log looking like agreement.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import defaultdict, deque

import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout


def _adapters():
    found = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for name, obj in vars(m).items():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                found[obj.__name__] = obj
    return found


ADAPTERS = _adapters()


def _calls_in(cls, method):
    for klass in cls.__mro__:
        fn = klass.__dict__.get(method)
        if fn is None:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except (OSError, SyntaxError, TypeError):       # Cython base: unreadable
            return set()
        return {n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    return set()


def test_the_discovery_found_the_adapters():
    assert len(ADAPTERS) >= 7, sorted(ADAPTERS)
    assert {"SmhGldSleeveStrategy", "TemplateRotationStrategy"} <= set(ADAPTERS)


def test_the_two_handlers_are_genuinely_DIFFERENT_on_this_Nautilus():
    """The premise, re-derived from the installed package rather than quoted. If a future version
    routed requested bars to `on_bar`, this rule would be obsolete and should say so rather than
    quietly becoming folklore."""
    from nautilus_trader.common.actor import Actor

    assert hasattr(Actor, "on_historical_data")
    assert hasattr(Actor, "on_bar")
    assert Actor.on_historical_data is not Actor.on_bar


@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_a_lane_that_REQUESTS_bars_can_RECEIVE_them(name):
    cls = ADAPTERS[name]
    if "request_bars" not in _calls_in(cls, "on_start"):
        return
    owner = next((k for k in cls.__mro__ if "on_historical_data" in k.__dict__), None)
    assert owner is not None and not owner.__module__.startswith("nautilus_trader"), (
        f"{name}.on_start calls request_bars and the class has no on_historical_data. The batch "
        f"arrives on a DIFFERENT handler from live bars, so it is delivered, logged, and dropped — "
        f"the lane never warms and every decision slot returns silently (#202, SMHGLD-007 "
        f"2026-09-11 18:40Z)")


@pytest.mark.parametrize("name", sorted(ADAPTERS), ids=str)
def test_the_receipt_actually_INGESTS_rather_than_merely_existing(name):
    """A handler that exists and stores nothing is the same defect wearing a method name. AST-bound
    on whatever the class really resolves."""
    cls = ADAPTERS[name]
    # Only a handler THIS PACKAGE defines. Every Nautilus Strategy inherits a Cython no-op of this
    # name, so `hasattr` is satisfied by the base — which is exactly the "present but does nothing"
    # state the defect consisted of, and why the check is on the DEFINITION rather than the name.
    owner = next((k for k in cls.__mro__ if "on_historical_data" in k.__dict__), None)
    if owner is None or owner.__module__.startswith("nautilus_trader"):
        return
    calls = _calls_in(cls, "on_historical_data")
    assert calls & {"_ingest", "append", "add"}, (
        f"{name}.on_historical_data does not pass the data to an ingest path; a requested batch "
        f"would be received and discarded. Calls: {sorted(calls)}")


def _bar(symbol, close, ts):
    bt = BarType.from_str(f"{symbol}.XNAS-1-DAY-LAST-EXTERNAL")
    return Bar(bar_type=bt, open=Price.from_str(f"{close:.2f}"),
               high=Price.from_str(f"{close:.2f}"), low=Price.from_str(f"{close:.2f}"),
               close=Price.from_str(f"{close:.2f}"), volume=Quantity.from_int(100),
               ts_event=ts, ts_init=ts)


def test_a_REQUESTED_BATCH_lands_in_bars_on_SMHGLD_the_lane_that_could_not_warm():
    """The effect, on the real method, with real Nautilus Bars — not the wiring.

    A batch of three sessions per leg is exactly `_need`, so this is the difference between the lane
    deciding tonight and returning silently.
    """
    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
    from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig

    lane = SmhGldSleeveStrategy.__new__(SmhGldSleeveStrategy)
    lane._cfg = SmhGldSleeveConfig()
    lane._need = 3
    lane._bars = defaultdict(lambda: deque(maxlen=64))

    assert SmhGldSleeveStrategy.warm.fget(lane) is False, "the fixture started warm; this proves nothing"

    day = 86_400_000_000_000
    for leg in ("SMH", "GLD"):
        for i in range(3):
            SmhGldSleeveStrategy.on_historical_data(lane, _bar(leg, 100 + i, (i + 1) * day))

    assert SmhGldSleeveStrategy.warm.fget(lane) is True, (
        f"a requested batch did not make the lane warm: {SmhGldSleeveStrategy._warmth(lane)}")


def test_a_NON_BAR_arriving_on_the_receipt_is_ignored_rather_than_raising():
    """The handler receives every requested data TYPE. An instrument definition or a tick reaching
    `_ingest` would raise into Nautilus's dispatch — and a lane that raises in a data handler is a
    lane reporting itself broken over something it was never asked to handle."""
    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
    from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig

    lane = SmhGldSleeveStrategy.__new__(SmhGldSleeveStrategy)
    lane._cfg = SmhGldSleeveConfig()
    lane._need = 3
    lane._bars = defaultdict(lambda: deque(maxlen=64))

    SmhGldSleeveStrategy.on_historical_data(lane, object())          # must not raise
    SmhGldSleeveStrategy.on_historical_data(lane, None)
    assert not lane._bars
