"""`shadow_only` was a DEAD KNOB: assigned at construction, read nowhere (#209).

    crsi_short.py:376   self._shadow_only = bool(shadow_only)   <- the ONLY occurrence

It gated a build-time raise and NOTHING AT RUNTIME, while its own docstring promised "register a
lane that computes and publishes without trading". That was invisible for as long as the gateway had
no submission path — A KNOB WIRED TO NOTHING AND A KNOB WIRED TO SOMETHING THAT NEVER FIRES LOOK
IDENTICAL. The moment cockpit wired submission (#1026), the lane would have traded while cockpit's
flag, this lane's docstring and the operator all believed that flag stopped it.

IT IS A DIFFERENT QUESTION FROM `ORDER_PATH_COMPLETE`, and both are needed:

    ORDER_PATH_COMPLETE   is the path BUILT?            a property of the CODE
    shadow_only           does THIS DEPLOYMENT trade?   a property of the CALLER

Collapsing them into one flag would cost the compute-only registration that is platform issue 853's first
phase — and that is the safest way to validate any new lane, so it is worth more than the tidiness
of having one flag instead of two.

THE TEST THAT MATTERS IS A SHADOW LANE WITH A RUNNER ATTACHED. A shadow lane with no runner proves
nothing: it sends nothing because there is nothing to send through, which is the state that hid this
defect in the first place.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy

SESSION = pd.Timestamp("2026-09-14")


class _Runner:
    """A runner that RECORDS being used. If the lane reaches it, the lane traded."""

    def __init__(self):
        self.calls = []

    async def run(self, panel, session, slot="", jobs=None, opens=None):
        # `jobs` and `opens` because `momentum_rotation.py:717` passes them, and
        # `test_every_runner_DOUBLE_accepts_what_an_adapter_can_pass` enforces it. A double NARROWER
        # than the adapter dies with TypeError inside the adapter's `except` — and this test would
        # then read "the runner was never called", which is EXACTLY what it asserts on success. It
        # would have passed for the opposite of its stated reason.
        self.calls.append((session, slot))
        return SimpleNamespace(submitted=3, state="TRADING", decided=True, blocked=None)


class _Host:
    """The REAL `_session_coro` on a host that can hold an id and a log.

    `CrsiShortStrategy.__new__` will not do: `id` and `log` are read-only Cython attributes on
    Nautilus's `Component`, so an uninitialised instance cannot be given either. The method under
    test is the attribute itself, so renaming or deleting it fails this rather than exercising a
    copy.
    """

    _session_coro = CrsiShortStrategy._session_coro


def _lane(*, shadow_only, runner):
    lane = _Host()
    lane.id = "CRSISHORT-006"
    lane._shadow_only = shadow_only
    lane._runner = runner
    lane.said: list[tuple[str, str]] = []
    # `*a, **k` because this fake should not make the test about logger-call minutiae. A double
    # narrower than production turns a handled error into a TypeError inside the handler, which is
    # the mirror of a permissive double: this one manufactures a failure rather than hiding one, and
    # either way the test stops being about the thing under test.
    lane.log = SimpleNamespace(
        info=lambda m, *a, **k: lane.said.append(("info", m)),
        warning=lambda m, *a, **k: lane.said.append(("warning", m)),
        error=lambda m, *a, **k: lane.said.append(("error", m)))
    lane.decided = []
    lane._decide_for = lambda session, panel: lane.decided.append(session)
    return lane


def _run(lane, panel=None):
    # `report_wrong_sided_positions` reads the cache; a narrow host has none, and this test is about
    # the handover rather than that report.
    import kumo_strategies.strategies.crsi_short.nautilus as mod

    original = mod.report_wrong_sided_positions
    mod.report_wrong_sided_positions = lambda *a, **k: None
    try:
        asyncio.run(CrsiShortStrategy._session_coro(
            lane, SESSION, panel if panel is not None else pd.DataFrame(), "open-10m"))
    finally:
        mod.report_wrong_sided_positions = original


def test_a_SHADOW_LANE_WITH_A_RUNNER_sends_nothing():
    """The case that matters. A runner is attached and submittable; the lane must not reach it."""
    runner = _Runner()
    lane = _lane(shadow_only=True, runner=runner)

    _run(lane)

    assert runner.calls == [], "a shadow lane reached its session runner and submitted"
    assert lane.decided == [SESSION], "a shadow lane must still DECIDE and publish"


def test_a_TRADING_LANE_WITH_A_RUNNER_does_reach_it():
    """The guard must not become a lane that never trades — otherwise it is the #26 shape, a flag
    that silently disables the thing it names."""
    runner = _Runner()
    lane = _lane(shadow_only=False, runner=runner)

    _run(lane)

    assert runner.calls == [(str(SESSION.date()), "open-10m")], lane.said


def test_the_TWO_REASONS_for_not_submitting_are_reported_DIFFERENTLY():
    """"Asked not to trade" and "nothing to trade through" are different facts. Reporting them
    identically is how a lane that lost its runner reads as a deliberate shadow deployment for a
    week — and it is the same ambiguity that let this knob sit dead."""
    shadow = _lane(shadow_only=True, runner=_Runner())
    _run(shadow)
    runnerless = _lane(shadow_only=False, runner=None)
    _run(runnerless)

    said_shadow = " ".join(m for _, m in shadow.said)
    said_none = " ".join(m for _, m in runnerless.said)

    assert "shadow_only=True" in said_shadow, shadow.said
    assert "no session runner" in said_none, runnerless.said
    assert said_shadow != said_none


def test_a_shadow_lane_with_NO_runner_also_sends_nothing():
    """True, and on its own it proves nothing — this is the state that HID the defect, because a
    lane with no runner sends nothing whatever the flag says. Kept as the control."""
    lane = _lane(shadow_only=True, runner=None)
    _run(lane)
    assert lane.decided == [SESSION]


def test_the_flag_is_READ_on_the_submission_path_not_merely_assigned():
    """The defect, as a structural assertion. `_shadow_only` was assigned at `__init__` and appeared
    nowhere else in the module; an attribute that is only ever written is a knob wired to nothing."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(CrsiShortStrategy._session_coro)))
    reads = {n.attr for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}
    assert "_shadow_only" in reads, (
        "the submission path does not read `_shadow_only`; the flag gates construction only and a "
        "wired gateway would trade a lane the caller asked not to")
