"""Every ExitConfig rule must either be implemented live or be declared as not implemented.

The failure this exists to prevent: `ExitConfig` is shared between the backtest harness and the live
runner, but the rules are written twice and only `give_back_frac` was ever written on the live side.
A research config setting `stall_days` typechecks, backtests, deploys — and is then silently ignored
by the runner holding real positions. Nothing raises. The exit simply never fires, and the book is
managed by a smaller ruleset than the one that justified entering it.

SCOPE — read before trusting this file. These are CLASSIFICATION tests, not behavioural ones. They
prove that every `ExitConfig` field has been consciously classified as live-supported or not, and that
`unsupported_live_exits` reports the unsupported ones correctly. They do NOT prove that a rule listed
in `LIVE_SUPPORTED_EXITS` is implemented *correctly* — adding a name to that set is taken at its word
here. Someone shipping a broken `stall_days` and listing it would pass.

That gap closes in #197 P3, which adds a fixture per rule driving `PgSessionRunner` and asserting the
exact exit symbol and reason. Until then the behavioural cover for the one supported rule
(`give_back_frac`) lives in tests/runtime/executor/test_executor.py.

What this file DOES catch, and what motivated it: a new rule added to `ExitConfig` with no live
support, which is how the current three got in.

The gate that binds the deployed config is in the other repo —
kumo-trading-platform `backend/strategies/test_live_config.py` — because it imports the real `live_config()`.
A hardcoded copy of the deployed values here would prove nothing: the copy and the original drift the
moment anyone edits the cockpit.

Tracked as kumo-trading-platform issue 197 B12. This file retires when the rules move into one shared evaluator
(#197 P2) and LIVE_SUPPORTED_EXITS covers every field.
"""

from dataclasses import fields

import pytest

from kumo_strategies.strategies.momentum_rotation.config import (
    ExitConfig,
    LIVE_SUPPORTED_EXITS,
    unsupported_live_exits,
)

# A value that counts as "the operator asked for this rule", per field. Any non-None value does; these
# are realistic ones so a failure message reads like a config someone might actually write.
ASKED_FOR = {
    "stall_days": 10,
    "max_hold_days": 30,
    "off_peak_pct": 0.15,
    "give_back_frac": 0.5,
    "give_back_min_peak_atr": 1.0,
    "give_back_confirm_sessions": 2,
    "take_profit_atr": 3.0,
    "stop_loss_atr": 2.0,
    "peak_fade_off_pct": 0.08,
    "peak_fade_confirm": 2,
    "peak_fade_lower_highs": 3,
}


def test_every_exit_field_is_classified():
    """A new ExitConfig field must be added to ASKED_FOR here, which forces a decision about whether
    the live runner honours it. Without this, a new rule slips in unclassified and untested."""
    declared = {f.name for f in fields(ExitConfig)}
    assert declared == set(ASKED_FOR), (
        f"ExitConfig fields changed: {declared ^ set(ASKED_FOR)}. Add the field to ASKED_FOR and "
        f"either implement it in PgSessionRunner and add it to LIVE_SUPPORTED_EXITS, or leave it out "
        f"of that set so live refuses entries when it is configured."
    )


def test_supported_exits_are_a_subset_of_the_real_fields():
    """Guards the reverse drift: a name in LIVE_SUPPORTED_EXITS that no longer exists on ExitConfig
    would silently stop protecting anything."""
    declared = {f.name for f in fields(ExitConfig)}
    assert LIVE_SUPPORTED_EXITS <= declared, f"stale names: {LIVE_SUPPORTED_EXITS - declared}"


def test_an_empty_config_asks_for_nothing():
    assert unsupported_live_exits(ExitConfig()) == []


@pytest.mark.parametrize("name", sorted(LIVE_SUPPORTED_EXITS))
def test_a_supported_rule_is_not_flagged(name):
    assert unsupported_live_exits(ExitConfig(**{name: ASKED_FOR[name]})) == []


@pytest.mark.parametrize("name", sorted(set(ASKED_FOR) - LIVE_SUPPORTED_EXITS))
def test_an_unsupported_rule_is_flagged_by_name(name):
    """Flagged individually and by name — the operator has to be told which rule is being ignored,
    not merely that something is."""
    assert unsupported_live_exits(ExitConfig(**{name: ASKED_FOR[name]})) == [name]


def test_all_unsupported_rules_are_reported_together():
    """Reporting one at a time would mean fixing a config, redeploying, and discovering the next."""
    unsupported = sorted(set(ASKED_FOR) - LIVE_SUPPORTED_EXITS)
    cfg = ExitConfig(**{n: ASKED_FOR[n] for n in unsupported})
    assert unsupported_live_exits(cfg) == unsupported


def test_every_exit_rule_is_now_supported_live():
    """Since #197 P2 both drivers call the same `exits.evaluate_exits`, so there is no longer a
    second copy that can fall behind — every ExitConfig field is honoured live.

    This set survives as a TRIPWIRE rather than a description: a new field added to `ExitConfig` and
    handled by the evaluator has to be added here deliberately, and this test fails until it is.
    """
    assert set(LIVE_SUPPORTED_EXITS) == {f.name for f in fields(ExitConfig)}, (
        "ExitConfig and LIVE_SUPPORTED_EXITS have diverged — either the evaluator gained a rule that "
        "was not declared, or a field was added that evaluate_exits does not handle."
    )
