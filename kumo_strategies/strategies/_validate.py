"""Field-type validation shared by every strategy config.

WHY THE CONFIG REFUSES RATHER THAN THE CALLER COERCES. Cockpit builds a live config from its settings
domain and hands the values over UNCOERCED, so `QC27_PORTFOLIO_SIZE="10"` arrives as a string. The
mechanism is not a missing cast -- it walks `dataclasses.fields()` and tests `is_dataclass(f.type)`,
and because every config here uses `from __future__ import annotations`, **`f.type` is a STRING on
every field**. That test is therefore unconditionally False: nested dataclasses are never constructed
and raw values pass straight through. It degrades silently to "hand the dict over".

`portfolio_size="10"` then reaches slot arithmetic. `equity / "10"` raises somewhere unrelated;
`"10" * n` concatenates and does not raise at all. Neither failure names the config.

So the callee refuses what it cannot use. A config that cannot be trusted to be the type it declares
is not a config, and the boundary holds regardless of which caller is on the other side of it.

`typing.get_type_hints` RESOLVES the string annotations -- the same resolution needed to find the
price-field settings for the deployment conformance suite.
"""

from __future__ import annotations

import dataclasses
import typing


def check_field_types(instance) -> None:
    """Raise `TypeError` naming the field if a value does not match its declared type.

    Deliberately narrow: it checks the simple scalars a settings domain can get wrong -- int, float,
    str, bool -- and leaves anything generic, optional or nested alone. A validator that tried to
    police every annotation would either reject legitimate values or need a type system, and the
    defect this exists for is a JSON string arriving where an int was declared.

    `bool` is checked BEFORE `int` because `bool` is a subclass of `int` in Python, so a `True` handed
    to an int field would otherwise pass and a `1` handed to a bool field would too.
    """
    hints = typing.get_type_hints(type(instance))
    for f in dataclasses.fields(instance):
        declared = hints.get(f.name)
        if declared not in (int, float, str, bool):
            continue                      # generics, optionals, literals, nested -- not ours to police
        value = getattr(instance, f.name)
        if declared is bool:
            ok = isinstance(value, bool)
        elif declared is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif declared is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str)
        if not ok:
            raise TypeError(
                f"{type(instance).__name__}.{f.name} declares {declared.__name__} but got "
                f"{type(value).__name__} {value!r}. Settings overrides arrive uncoerced, and a value "
                f"of the wrong type here fails later somewhere that does not name the config.")
