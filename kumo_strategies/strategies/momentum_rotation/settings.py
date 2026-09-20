"""Machine-readable settings schema, generated FROM the config dataclasses (#32).

WHY GENERATED AND NOT WRITTEN
-----------------------------
2026-08-15: "All params should use the cockpit settings mechanism." The obvious way to do that
is to add the fields to a settings form in cockpit. That would be wrong within a week.

`MomentumRotationConfig` gained nine fields in a single day's work — `give_back_min_peak_atr`,
`give_back_confirm_sessions`, `take_profit_atr`, `peak_fade_off_pct`, `peak_fade_confirm`,
`peak_fade_lower_highs`, `cluster_exit_score`, `cluster_corr`, `inverse_vol_entry_relative` — and
two were deleted. A hand-maintained list in another repository does not survive that, and its
failure mode is the #26 shape one repo over: cockpit shows a parameter that no longer exists, or
silently omits one that does, and the operator's mental model of what is configured stops matching
what is running.

So the dataclasses are the single source of truth and this walks them. A new field appears in
cockpit because it was added here, or it does not appear at all — there is no third state where the
two disagree.

WHAT IT CARRIES BEYOND TYPES AND DEFAULTS
-----------------------------------------
`live_supported`, per exit field, from `LIVE_SUPPORTED_EXITS`. A settings UI that lets an operator
set a rule the live runner cannot honour is worse than one that omits it: the rule typechecks,
backtests, deploys, and is then silently ignored by the thing holding real positions. Cockpit needs
to be able to grey those out, so the flag travels with the schema rather than being re-derived.

Descriptions come from the field docstrings, read out of the source with `ast`. Python discards
attribute docstrings at runtime, so there is no reflective way to get them — but they are where the
reasoning for every parameter already lives, and duplicating that into a UI string table is the same
drift problem in miniature. `test_settings_schema.py` fails on any field without one, which makes
documenting a new parameter a condition of adding it.
"""

from __future__ import annotations

import ast
import enum
import inspect
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from typing import Any

from kumo_strategies.strategies.momentum_rotation import config as config_module
from kumo_strategies.strategies.momentum_rotation.config import (
    LIVE_SUPPORTED_EXITS, ExitConfig, MomentumRotationConfig)

#: JSON type for each Python scalar we actually use. Deliberately not a catch-all: an unmapped type
#: raises rather than degrading to "string", because a parameter rendered as the wrong control is a
#: parameter that will be set wrongly.
_JSON_TYPES: dict[Any, str] = {int: "integer", float: "number", bool: "boolean", str: "string"}


def _field_docs(cls: type) -> dict[str, str]:
    """Field docstrings, read from source. Python drops these at runtime, so `ast` is the only way.

    A docstring counts only when it is the statement immediately following the annotated assignment,
    which is the convention the config module already follows throughout.
    """
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):       # pragma: no cover - only when source is unavailable
        return {}
    tree = ast.parse(inspect.cleandoc(src))
    body = tree.body[0].body if isinstance(tree.body[0], ast.ClassDef) else []
    out: dict[str, str] = {}
    for prev, node in zip(body, body[1:]):
        if (isinstance(prev, ast.AnnAssign) and isinstance(prev.target, ast.Name)
                and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            out[prev.target.id] = inspect.cleandoc(node.value.value)
    return out


#: Nested configs that are DECLARATIONS, not operator knobs, and are therefore kept out of the
#: settings schema entirely rather than rendered as controls.
#:
#: `MarketViewConfig` is here because of rule 2 of the market-view protocol: the view is DECLARED
#: ON THE STRATEGY, not supplied by whoever is running it. `rebalance_period` is the precedent and
#: it cost this repo a published number — a research-only knob where a sweep reported weekly while
#: both production call sites stayed hardwired monthly, and nothing failed to say so. A view an
#: operator can set from a UI is that shape with a bigger blast radius, because `window` is a
#: MEASURED quantity per lane and a dropdown invites picking one.
#:
#: Turning a lane off is what the lifecycle states are for. This is not the control for that.
#:
#: Excluded rather than given a JSON mapping DELIBERATELY: adding one for `MarketSignal` would make
#: the field render as a tidy enum dropdown and become settable by accident, which is the opposite
#: of the decision. The schema builder's refusal to guess was right and is what surfaced this.
NOT_OPERATOR_SETTABLE: frozenset = frozenset()   # populated below, after the import cycle


def _describe_type(ann: Any) -> dict[str, Any]:
    """One field's type, as JSON Schema. Optionality is reported, not flattened away.

    `None` is a meaningful value across this config rather than an absence: every exit rule defaults
    to None precisely so "unset" and "asked for" stay distinguishable (see `unsupported_live_exits`).
    A UI that cannot express "off" would force every rule on.
    """
    origin = typing.get_origin(ann)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(f"unions of more than one non-None type are not supported: {ann!r}")
        return {**_describe_type(args[0]), "nullable": True}
    if origin in (tuple, list):
        (item, *_), = (typing.get_args(ann),)
        return {"type": "array", "items": _describe_type(item)}
    if origin is typing.Literal:
        vals = list(typing.get_args(ann))
        kinds = {type(v) for v in vals}
        if len(kinds) != 1 or kinds.pop() not in _JSON_TYPES:
            raise TypeError(f"Literal with mixed or unmapped member types: {ann!r}")
        # An enum rather than a bare string is the whole value of a Literal reaching the UI. A
        # free-text `momentum_price_field` is how a value typechecks, deploys, and silently changes
        # what the strategy ranks on.
        return {"type": _JSON_TYPES[type(vals[0])], "nullable": False, "enum": vals}
    if origin is dict or ann is dict:
        return {"type": "object"}
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        # DELIBERATE, which is what the refusal below was asking for. An enum reaching the UI as a
        # closed set is the whole value of declaring it — the same argument the `Literal` branch
        # above makes: free text on a field like `momentum_price_field` is how a value typechecks,
        # deploys, and silently changes what the strategy ranks on.
        #
        # ADDED BECAUSE EXCLUDING THE CLASS MASKED ITS ABSENCE. `MarketViewConfig` was withheld
        # from the schema, so `schema_for` never reached this branch and nobody could tell whether
        # the class was unwalkable or merely withheld — two different facts. Found by the paper
        # session calling `schema_for(MarketViewConfig)` directly. Walkability and settability are
        # now decided separately, on their own merits.
        members = [m.value for m in ann]
        kinds = {type(v) for v in members}
        if len(kinds) != 1 or kinds.pop() not in _JSON_TYPES:
            raise TypeError(f"enum with mixed or unmapped member types: {ann!r}")
        return {"type": _JSON_TYPES[type(members[0])], "nullable": False, "enum": members}
    if ann in _JSON_TYPES:
        return {"type": _JSON_TYPES[ann], "nullable": False}
    raise TypeError(
        f"no JSON mapping for {ann!r}. Add one deliberately rather than letting it fall back to a "
        "string — a parameter rendered as the wrong control is a parameter that gets set wrongly.")


def _default_of(f) -> Any:
    if f.default is not MISSING:
        # AN ENUM MEMBER DOES NOT SURVIVE THE WIRE. `json.dumps` raises on it, so a default that
        # reached cockpit as `MarketSignal.NONE` would fail at the boundary rather than in a test
        # — the shape `market_events.payload()` has a round-trip test for, for the same reason.
        return f.default.value if isinstance(f.default, enum.Enum) else f.default
    if f.default_factory is not MISSING:            # type: ignore[misc]
        return f.default_factory()                  # type: ignore[misc]
    return None


def _jsonable(v: Any) -> Any:
    """Tuples become lists and frozensets become sorted lists, so the schema survives `json.dumps`."""
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    return v


def _group(cls: type, *, live_supported: frozenset[str] | None = None) -> dict[str, Any]:
    """Scalar fields of one dataclass. Nested dataclasses are groups and are handled by `schema_for`."""
    docs = _field_docs(cls)
    hints = typing.get_type_hints(cls)
    out: dict[str, Any] = {}
    for f in fields(cls):
        if is_dataclass(hints[f.name]):
            continue
        entry = {**_describe_type(hints[f.name]),
                 "default": _jsonable(_default_of(f)),
                 "description": docs.get(f.name, "")}
        if live_supported is not None:
            entry["live_supported"] = f.name in live_supported
        out[f.name] = entry
    return out


def schema_for(cls: type, *, live_supported: frozenset[str] | None = None) -> dict[str, Any]:
    """Describe ANY strategy config dataclass, recursing into nested ones.

    Public because it is the only non-private way in, and the alternative is a second strategy's
    settings being hand-written in cockpit — the exact drift #32 exists to prevent. QC345 needs it:
    it nests `ExitConfig` and carries three `Literal` unions.

    `live_supported` applies to this class's own fields. Nested `ExitConfig` picks up
    `LIVE_SUPPORTED_EXITS` automatically, keyed on the CLASS rather than on a field called "exits" —
    both strategies happen to name it that today, and the flag must not depend on them continuing to.
    """
    hints = typing.get_type_hints(cls)
    groups = {}
    for f in fields(cls):
        sub = hints[f.name]
        if not is_dataclass(sub):
            continue
        groups[f.name] = schema_for(
            sub, live_supported=LIVE_SUPPORTED_EXITS if sub is ExitConfig else None)
        if sub in _not_operator_settable():
            # OFFERED READ-ONLY, NOT OMITTED. Omitting it was my first answer and it is wrong: an
            # upstream field the operator UI cannot SHOW is the DTO-drops-fields shape, and it
            # tripped cockpit's canary (every declared field must be offered). Raised by the paper
            # session; their resolution is better than mine and this is it.
            #
            # Not settable, for rule 2 — the view is DECLARED ON THE STRATEGY, `window` is a
            # MEASURED quantity per lane, and a dropdown invites picking one. Visible, so nobody
            # has to read the source to know what a lane declares.
            groups[f.name]["x-operator-settable"] = False
            groups[f.name]["x-not-settable-reason"] = (
                "Declared by the strategy in code — shown, not editable. The view travels with "
                "the strategy (market-view protocol rule 2); `window` is measured per lane and "
                "must not be chosen from a list. To stop a lane trading, use its lifecycle state.")
    return {"title": cls.__name__,
            "description": inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0],
            "fields": _group(cls, live_supported=live_supported),
            "groups": groups,
            # A GROUP-LEVEL DEFAULT, not only per-field ones. A consumer that fills in defaults
            # typically descends only into an object already PRESENT, so a nested group with no
            # default of its own is in the schema, renders, and reaches nothing — cockpit hit
            # exactly that with `exits` and caught it by reading the deployed response rather than
            # by any test. Emitting it here means the next consumer cannot fall in.
            "default": _group_default(cls)}


def _group_default(cls: type) -> dict[str, Any]:
    """Every field of `cls` at its default, nested groups included. Recursive by construction."""
    hints = typing.get_type_hints(cls)
    out: dict[str, Any] = {}
    for f in fields(cls):
        sub = hints[f.name]
        out[f.name] = _group_default(sub) if is_dataclass(sub) else _jsonable(_default_of(f))
    return out


def _not_operator_settable() -> frozenset:
    from kumo_strategies.strategies.market_view import MarketViewConfig
    return frozenset({MarketViewConfig})


def schema() -> dict[str, Any]:
    """The full settings schema: one group per nested config dataclass, plus the top-level fields.

    Shaped as groups rather than a flat field list because that is how the parameters actually
    divide — scoring, portfolio construction, exits, execution, gates — and a flat list of ~40
    settings gives an operator no way to tell which ones interact.
    """
    built = schema_for(MomentumRotationConfig)
    return {"strategy": "momentum_rotation",
            "source": f"{config_module.__name__}.MomentumRotationConfig",
            "top_level": built["fields"],
            "groups": built["groups"]}


def main() -> None:                                  # pragma: no cover - thin CLI
    """`python -m kumo_strategies.strategies.momentum_rotation.settings` — dump the schema."""
    import json
    print(json.dumps(schema(), indent=2, sort_keys=True))


if __name__ == "__main__":                           # pragma: no cover
    main()
