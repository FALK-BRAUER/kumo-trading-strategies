"""A strategy config as `parameters.json` and back — so a variation holds EVERY parameter the run used
and `run.py` holds none (ks#240).

`to_parameters(cfg)` writes a frozen dataclass as plain JSON: nested dataclasses become objects,
enums become their `.value`. `from_parameters(cls, d)` rebuilds it, resolving nested dataclasses and
enums from the class's type hints, and refusing a key the class does not declare — a typo in
`parameters.json` fails by name instead of silently running the default.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from enum import Enum
from typing import Any


def to_parameters(cfg: Any) -> dict:
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"not a dataclass: {type(cfg).__name__}")
    out: dict = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if dataclasses.is_dataclass(v):
            out[f.name] = to_parameters(v)
        elif isinstance(v, Enum):
            out[f.name] = v.value
        elif isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    return out


def _unwrap_optional(hint: Any) -> Any:
    """`X | None` / `Optional[X]` → X; anything else unchanged."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def from_parameters(cls: type, d: dict) -> Any:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"not a dataclass: {cls.__name__}")
    hints = typing.get_type_hints(cls)
    declared = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - declared)
    if unknown:
        raise KeyError(f"{cls.__name__} does not declare {unknown}; declared: {sorted(declared)}")
    kw: dict = {}
    for name, v in d.items():
        hint = _unwrap_optional(hints[name])
        if v is None:
            kw[name] = None
        elif dataclasses.is_dataclass(hint) and isinstance(v, dict):
            kw[name] = from_parameters(hint, v)
        elif isinstance(hint, type) and issubclass(hint, Enum):
            kw[name] = hint(v)
        elif typing.get_origin(hint) is tuple and isinstance(v, list):
            kw[name] = tuple(v)
        else:
            kw[name] = v
    return cls(**kw)
