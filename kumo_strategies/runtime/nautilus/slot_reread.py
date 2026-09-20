"""Re-read decision slots from settings before arming — the method two adapters CALL and never had.

THE LIVE FAILURE. ibkr-paper booted at 16:07:28Z on `strategies ea03cfb`; five lanes armed and the
sixth did not:

    SMHGLD-007: UNARMED — the calendar answered but arming FAILED:
    AttributeError("'SmhGldSleeveStrategy' object has no attribute '_reread_slots'").
    This is not a network problem and will not be retried.

`smhgld_sleeve.py` accepted `read_slots` at :111, stored it at :153, and CALLED `self._reread_slots()`
at :217 inside `_arm` — a method it never defined. The call is UNCONDITIONAL and runs before
`next_slot_fire`, so there is no caller-side workaround: passing `read_slots=None` does not help,
because the `if self._read_slots is None: return` guard lives inside the method that is missing.

IT IS TWO LANES, NOT ONE. `template_rotation` has the identical defect — accepts the argument,
calls the method, defines nothing — and nobody has hit it only because nothing has armed it yet.
Reported as SMHGLD alone; found by asking every adapter the same question instead of the one that
failed. The first death hides identical siblings.

WHY A MIXIN RATHER THAN THE THIRD AND FOURTH COPY. The obvious repair is to paste `crsi_short`'s
copy onto both, and it is the wrong one, because THE TWO EXISTING COPIES ARE NOT THE SAME METHOD:

    momentum_rotation   validates the new slots and REFUSES them if unusable, keeping the old ones
    crsi_short          adopts whatever settings returned, unvalidated

So pasting the nearer copy would have propagated the WEAKER of the two into two more lanes, and the
divergence would have been four-way instead of two-way. `validate` is the behaviour that belongs
here: an unusable slot tuple adopted silently is a session that decides fewer times than configured,
which is the #26 shape this package has already paid for.

THE VALIDATION IS SAFE FOR EVERY LANE THAT TAKES THIS ARGUMENT, checked rather than assumed: all
four spell slots in one grammar (`open+5m`, `close-20m`) and all four resolve them through the same
`next_slot_fire`, so a tuple that fails `validate` here would have failed in `resolve` anyway — the
difference is whether the lane keeps its working schedule or arms on a broken one.

NEVER LETS A SETTINGS READ STOP THE LANE SCHEDULING. A lane that stopped arming looks exactly like
one that decided to hold — which is precisely how this defect presented, as a lane that simply was
not there. A raising reader, a None, an empty tuple, or one that fails validation all leave the
current schedule in place and say so.
"""

from __future__ import annotations

from kumo_strategies.strategies.momentum_rotation.slots import validate as validate_slots

__all__ = ["SlotRereadMixin"]


class SlotRereadMixin:
    """`_reread_slots`, once, for every adapter whose constructor accepts `read_slots`.

    `test_the_slot_reread_argument_and_its_method_travel_together` binds the constructor argument to
    the method across every adapter discovered at runtime, so a lane that accepts one without the
    other fails on the day it is written rather than at 16:07Z on a Friday.
    """

    def _reread_slots(self) -> None:
        """Refresh `_slots` from settings, or keep what we have."""
        read = getattr(self, "_read_slots", None)
        if read is None:
            return
        try:
            got = read()
        except Exception as exc:                                       # noqa: BLE001
            self._slot_log("warning", f"could not re-read decision slots ({exc}) — "
                                      f"keeping {list(self._slots)}")
            return
        if not got:
            return
        slots = tuple(got)
        if slots == self._slots:
            return
        try:
            validate_slots(slots)
        except Exception as exc:                                       # noqa: BLE001
            self._slot_log("warning", f"settings gave unusable decision slots {list(slots)} "
                                      f"({exc}) — keeping {list(self._slots)}")
            return
        self._slot_log("info", f"decision slots moved {list(self._slots)} -> {list(slots)} "
                               f"by settings")
        self._slots = slots

    def _slot_log(self, level: str, msg: str) -> None:
        """Log without ever becoming the reason a lane failed to arm — which is this defect's own
        shape, and the rule `_record_arm_state` learned when its clock read raised over the arming
        failure it was reporting."""
        try:
            getattr(self.log, level)(f"{getattr(self, 'id', type(self).__name__)}: {msg}")
        except Exception:                                              # noqa: BLE001
            pass
