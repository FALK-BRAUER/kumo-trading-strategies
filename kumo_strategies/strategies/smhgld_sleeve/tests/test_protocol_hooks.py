"""The three MarketAware hooks, answered honestly rather than stubbed.

A stub returning NO turns every reading green and nothing is true. `not_asked` on the operator's
surface is indistinguishable from "every lane says nothing is wrong", which is the unknown-read-as-
pass class at the top of the safety protocol. These pin that this lane answers, and what from.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.nautilus.market_aware import MarketAwareMixin
from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
from kumo_strategies.strategies.market_view import MarketSignal, NO, UNKNOWN
from kumo_strategies.strategies.smhgld_sleeve import live_config


class _Bare(MarketAwareMixin):
    """The shape the poller can meet seconds after registration: no bars, maybe no config."""

    EXTERNAL_ID = "SMHGLD"

    def __init__(self, cfg=None):
        self._cfg = cfg
        self._bars = {}


def test_the_mixin_is_first_in_the_mro():
    """Mixed in FIRST so Nautilus' `Strategy` cannot shadow a hook, and so a lane overriding one
    does it deliberately rather than by resolution order."""
    mro = SmhGldSleeveStrategy.__mro__
    assert mro[1] is MarketAwareMixin, f"MarketAwareMixin is not first: {[c.__name__ for c in mro[:4]]}"


def test_entries_blocked_with_no_bars_is_UNKNOWN_not_an_opt_out():
    """This asserted NO naming signal=NONE and called it "a MEASUREMENT — eleven mechanisms tested,
    none separated from chance". That conflated a timing signal INSIDE the lane (no information,
    #177) with the platform's governance view (#212): a lane answering NONE is invisible to the
    poller. With a declared view and no bars yet the honest answer is UNKNOWN, naming the window
    — a lane that is watching and cannot yet see. The full set of answers lives in
    `test_the_sleeve_declares_a_market_view.py`."""
    verdict = _Bare(live_config()).entries_blocked()

    assert verdict.state == UNKNOWN, verdict
    assert not any("NONE" in reason for reason in verdict.reasons), verdict.reasons


def test_emergency_exit_answers_no_naming_that_nothing_is_wired():
    """Silence and 'no emergency' are different facts. Only one of them is true here."""
    verdict = _Bare(live_config()).emergency_exit()

    assert verdict.state == NO
    assert any("trigger" in reason for reason in verdict.reasons), verdict.reasons


def test_self_assessment_is_unknown_and_cannot_quarantine_the_lane():
    """UNKNOWN is the third state doing its job. `evidence_sufficient=False` is what stops a lane
    that cannot assess itself from acting on an assessment it did not make."""
    assessment = _Bare(live_config()).self_assessment()

    assert assessment.state == UNKNOWN
    assert assessment.evidence_sufficient is False
    assert assessment.action is None


def test_no_hook_raises_when_the_lane_holds_nothing_at_all():
    """A hook that raises is a lane reporting itself broken, and cockpit pages on faults. A lane
    polled seconds after registration may hold no config — that is normal, not an error."""
    bare = _Bare(cfg=None)

    for hook in (bare.entries_blocked, bare.emergency_exit, bare.self_assessment):
        hook()  # must not raise


def test_the_declared_view_is_the_labs_measured_one_not_NONE():
    """The opposite of what this test said on b575db1. `signal=NONE` was an opt-out wearing a
    measurement's clothes (#212); the measured view is INDEX_VS_MA / 50 / EXIT_ONLY, and EXIT_ONLY
    costs this lane nothing because it never opens a name it does not hold."""
    assert live_config().market_view.signal is MarketSignal.INDEX_VS_MA
