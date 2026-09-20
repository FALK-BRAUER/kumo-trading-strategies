"""The settings schema must not be able to drift from the dataclasses (#32).

A settings form maintained by hand in another repository is the failure this schema exists to
prevent, so the tests that matter are the ones asserting COVERAGE — that every field is present,
every field is documented, and every exit field carries an honest `live_supported` flag. A schema
that is merely well-formed is not useful; one that is complete is.

`MomentumRotationConfig` gained nine fields and lost two in a single day's work, which is the rate
these have to survive.
"""

from __future__ import annotations

import json
import typing
from dataclasses import fields, is_dataclass

import pytest

from kumo_strategies.strategies.momentum_rotation.config import (
    LIVE_SUPPORTED_EXITS, ExitConfig, MomentumRotationConfig)
from kumo_strategies.strategies.momentum_rotation.settings import schema


@pytest.fixture(scope="module")
def sch() -> dict:
    return schema()


def _nested() -> list[str]:
    """EVERY nested dataclass reaches the schema. Not-operator-settable groups are OFFERED
    READ-ONLY rather than omitted — an upstream field the operator UI cannot SHOW is the
    DTO-drops-fields shape, and omitting them tripped cockpit's canary."""
    hints = typing.get_type_hints(MomentumRotationConfig)
    return [f.name for f in fields(MomentumRotationConfig) if is_dataclass(hints[f.name])]


def test_WITHHELD_GROUPS_ARE_OFFERED_READ_ONLY_NOT_OMITTED():
    """A group missing from the schema must be missing ON PURPOSE.

    The guard above exists because a group silently vanishing from the schema is the #26 shape —
    a knob that exists in code, renders nowhere, and is therefore unsettable and unnoticed. So an
    exclusion cannot be an omission: it is declared in `settings._not_operator_settable()` and
    named here, and adding a nested config without deciding either way still fails.

    `MarketViewConfig` is withheld under rule 2 of the market-view protocol — the view is DECLARED
    ON THE STRATEGY, not supplied by whoever runs it. `rebalance_period` is the precedent and it
    cost a published number. `window` in particular is a MEASURED quantity per lane, and a
    dropdown invites picking one; turning a lane off is what the lifecycle states are for.
    """
    from kumo_strategies.strategies.market_view import MarketViewConfig
    from kumo_strategies.strategies.momentum_rotation.settings import _not_operator_settable

    withheld = _not_operator_settable()
    assert MarketViewConfig in withheld, (
        "the market view is settable from the operator UI — rule 2 says the declaration travels "
        "with the strategy")

    # PRESENT, and flagged. Settable and DESCRIBABLE are two different guards: omitting the group
    # made the field invisible, which is the DTO-drops-fields shape and tripped cockpit's canary.
    # The operator approved a 50-day view and had no surface on which to see whether it was running.
    group = schema()["groups"]["market_view"]
    assert group["x-operator-settable"] is False
    assert "not editable" in group["x-not-settable-reason"]
    assert set(group["fields"]) == {"signal", "window", "dwell", "action"}, (
        "the withheld group is offered but not described — read-only must still mean VISIBLE")


def test_every_config_group_is_present(sch):
    assert set(sch["groups"]) == set(_nested()), "a config group is missing from the schema"


@pytest.mark.parametrize("group", _nested())
def test_every_field_of_every_group_is_present(sch, group):
    """The drift guard. Parametrised per group so a failure names which one fell behind."""
    hints = typing.get_type_hints(MomentumRotationConfig)
    cls = hints[group]
    expected = {f.name for f in fields(cls) if not is_dataclass(typing.get_type_hints(cls)[f.name])}
    assert set(sch["groups"][group]["fields"]) == expected


def test_every_field_is_documented(sch):
    """A new parameter cannot reach cockpit as a bare name with no explanation.

    Field docstrings are where the reasoning for each parameter already lives. Requiring one here
    means documenting a parameter is a condition of adding it, rather than a separate task that
    never happens.
    """
    undocumented = [f"{g}.{name}"
                    for g, spec in sch["groups"].items()
                    for name, f in spec["fields"].items() if not f["description"]]
    assert not undocumented, f"fields with no docstring: {undocumented}"


def test_defaults_match_the_dataclasses(sch):
    """The schema's defaults are the real ones — a UI seeded from them shows what actually runs."""
    cfg = MomentumRotationConfig()
    for group, spec in sch["groups"].items():
        actual = getattr(cfg, group)
        for name, f in spec["fields"].items():
            want = getattr(actual, name)
            want = list(want) if isinstance(want, tuple) else want
            assert f["default"] == want, f"{group}.{name} default disagrees with the dataclass"


def test_exit_fields_carry_live_supported_and_it_is_honest(sch):
    """A UI that offers a rule live cannot honour is worse than one that omits it.

    Such a rule typechecks, backtests, deploys, and is then silently ignored by the thing holding
    real positions — nothing fails, the exit simply never fires.
    """
    exits = sch["groups"]["exits"]["fields"]
    assert all("live_supported" in f for f in exits.values())
    flagged = {name for name, f in exits.items() if f["live_supported"]}
    assert flagged == {f.name for f in fields(ExitConfig)} & set(LIVE_SUPPORTED_EXITS)


def test_only_exits_carry_the_live_flag(sch):
    """The flag means something specific. Putting it on groups with no such distinction would
    imply the others had been checked, which they have not."""
    for group, spec in sch["groups"].items():
        if group == "exits":
            continue
        assert not any("live_supported" in f for f in spec["fields"].values()), group


def test_optional_fields_report_nullable_rather_than_hiding_it(sch):
    """None is a value here, not an absence: every exit rule defaults to None so that "unset" and
    "asked for" stay distinguishable. A UI unable to express "off" would force every rule on."""
    exits = sch["groups"]["exits"]["fields"]
    assert exits["give_back_frac"]["nullable"] is True
    assert exits["peak_fade_confirm"]["nullable"] is False, "an int with a real default is not optional"


def test_the_schema_is_json_serialisable(sch):
    """It crosses a process boundary to reach cockpit, so tuples and frozensets must already be gone."""
    assert json.loads(json.dumps(sch)) == sch


def test_a_tuple_field_is_described_as_an_array(sch):
    t = sch["groups"]["gates"]["fields"]["exclude_instrument_types"]
    assert t["type"] == "array" and t["items"]["type"] == "string"
    assert isinstance(t["default"], list)


def test_an_unmappable_type_raises_rather_than_defaulting_to_string(sch):
    """The mapping is deliberately not a catch-all. A parameter rendered as the wrong control is a
    parameter that gets set wrongly, so an unknown type must stop the build rather than degrade."""
    from kumo_strategies.strategies.momentum_rotation.settings import _describe_type
    with pytest.raises(TypeError, match="no JSON mapping"):
        _describe_type(complex)


# --- schema_for: the generator has to describe MORE than one strategy (#32, #37) ----------------

def test_schema_for_describes_qc345_including_its_Literals_and_nesting():
    """The second strategy is the real test of a generator. Without this it raised on four fields,
    and the alternative — hand-writing a qc345 schema in cockpit — is the drift #32 exists to stop."""
    from kumo_strategies.strategies.momentum_rotation.settings import schema_for
    from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig
    s = schema_for(QC345RotationConfig)
    declared = {f.name for f in fields(QC345RotationConfig)}
    assert set(s["fields"]) | set(s["groups"]) == declared, "a qc345 field is undescribed"
    assert s["groups"]["market_view"]["x-operator-settable"] is False, (
        "qc345's market view is settable from the operator UI — rule 2 has been lost on this lane")


def test_a_Literal_becomes_an_enum_rather_than_a_bare_string():
    """The whole value of a Literal reaching the UI. Free text on `momentum_price_field` is how a
    value typechecks, deploys, and silently changes what the strategy ranks on."""
    from kumo_strategies.strategies.momentum_rotation.settings import schema_for
    from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig
    f = schema_for(QC345RotationConfig)["fields"]["momentum_price_field"]
    assert f["type"] == "string"
    assert f["enum"] == ["close", "close_adj", "close_split", "close_split_dividend"]


def test_a_Literal_with_mixed_member_types_raises():
    from kumo_strategies.strategies.momentum_rotation.settings import _describe_type
    with pytest.raises(TypeError, match="mixed or unmapped"):
        _describe_type(typing.Literal["a", 1])


def test_nested_ExitConfig_carries_live_supported_keyed_on_the_CLASS(sch):
    """Not on a field named "exits". Both strategies name it that today and the flag must not depend
    on them continuing to — a renamed field would silently drop the flag, and a UI with no flag
    offers rules live cannot honour."""
    from kumo_strategies.strategies.momentum_rotation.settings import schema_for
    from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig
    e = schema_for(QC345RotationConfig)["groups"]["exits"]["fields"]
    assert all("live_supported" in f for f in e.values())
    assert e["stall_days"]["live_supported"] is True


def test_schema_still_produces_the_momentum_shape_cockpit_already_reads(sch):
    """`schema()` is now built on `schema_for`. Its published shape must not have moved."""
    assert set(sch) == {"strategy", "source", "top_level", "groups"}
    assert sch["strategy"] == "momentum_rotation"
    assert set(sch["groups"]) == set(_nested())
    assert "give_back_frac" in sch["groups"]["exits"]["fields"]


def test_every_group_carries_its_own_default_object(sch):
    """A consumer filling in defaults usually descends only into an object already PRESENT, so a
    nested group without a default of its own renders and then reaches nothing. Cockpit hit that
    with `exits` and found it by reading the deployed response, not by a test. Emitting the default
    here means the next consumer cannot fall in."""
    for group, spec in sch["groups"].items():
        assert "default" in spec, f"{group} has no group-level default"
        assert set(spec["default"]) >= set(spec["fields"]), f"{group} default is missing fields"


def test_group_defaults_equal_the_real_dataclass_defaults(sch):
    """The default object has to be usable as-is, not merely present."""
    cfg = MomentumRotationConfig()
    for group, spec in sch["groups"].items():
        actual = getattr(cfg, group)
        for name, want in spec["default"].items():
            got = getattr(actual, name)
            got = list(got) if isinstance(got, tuple) else got
            assert want == got, f"{group}.{name} group-default disagrees with the dataclass"


def test_nested_group_defaults_recurse_for_other_strategies():
    from kumo_strategies.strategies.momentum_rotation.settings import schema_for
    from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig
    d = schema_for(QC345RotationConfig)["groups"]["exits"]["default"]
    assert set(d) == {f.name for f in fields(ExitConfig)}
    assert d["give_back_frac"] is None, "an unset rule must default to off, not to a value"


def test_the_decision_SCHEDULE_is_an_operator_visible_setting(sch):
    """BCTROT's defining parameter. It was a constructor argument — the single setting an operator
    most needs to see and the only one they could not, which fails #32's requirement that every
    parameter go through the cockpit settings mechanism."""
    f = sch["groups"]["execution"]["fields"]["decision_slots"]
    assert f["type"] == "array" and f["items"]["type"] == "string"
    assert f["default"] == ["open+5m"], "the default must reproduce MOMENTUM-002 exactly"
    assert f["description"], "an operator setting a schedule needs to know it is anchors not times"


@pytest.mark.parametrize("name,default", [
    ("decision_slots", ["open+5m"]),
    ("max_stale_days", 4),
    ("equity_per_position", 1_000.0),
])
def test_live_behaviour_parameters_reach_the_schema_with_their_live_defaults(sch, name, default):
    """These ran the live strategy from constructor defaults, invisible to the operator. Moving them
    is only safe if the defaults are unchanged — MOMENTUM-002 is holding positions on them."""
    assert sch["groups"]["execution"]["fields"][name]["default"] == default
