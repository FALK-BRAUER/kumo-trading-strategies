"""Strategy lifecycle. Spec: issue #187.

    DISABLED --enable--> WARMUP --history ready--> SHADOW --operator--> TRADING
       ^                                             ^                     |
       |                                             +--pause--------------+
       +-- LIQUIDATING <--flatten-- HALTED <--risk breach / disconnect -----+

Two properties matter more than the diagram:

  SHADOW IS THE DRY-RUN STATE. It runs the full decision path and publishes what it WOULD do without
  submitting anything. It is how a strategy earns trust before it trades and how a config change is
  validated against live data rather than a backtest.

  ONLY AN OPERATOR MAY ENTER TRADING. Everything can stop it automatically; nothing can start it.
  That asymmetry is the whole safety model, so it is enforced here rather than by convention.

The dry-run state is called SHADOW, not ARMED, deliberately. kumo-trading-platform's managers.py already runs
an ATTACHED->ARMED->APPLYING->APPLIED machine for per-symbol managers (#55), where ARMED means
"attached and waiting to trigger". Here it would mean "decides and publishes but submits nothing" —
a genuinely different thing. Two ARMEDs meaning different things in one engine is a bug waiting to
happen at merge, so this one yields the name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    DISABLED = "DISABLED"
    WARMUP = "WARMUP"
    SHADOW = "SHADOW"
    TRADING = "TRADING"
    HALTED = "HALTED"
    LIQUIDATING = "LIQUIDATING"

    @property
    def may_submit_entries(self) -> bool:
        return self is State.TRADING

    @property
    def may_submit_exits(self) -> bool:
        return self in (State.TRADING, State.LIQUIDATING)

    @property
    def decides(self) -> bool:
        """SHADOW decides and publishes; it just does not act."""
        return self in (State.SHADOW, State.TRADING, State.LIQUIDATING)


#: (from, to) -> operator_only. Anything absent is rejected.
_ALLOWED: dict[tuple[State, State], bool] = {
    (State.DISABLED, State.WARMUP): True,
    (State.WARMUP, State.SHADOW): False,          # automatic once history is sufficient
    (State.WARMUP, State.DISABLED): True,
    (State.SHADOW, State.TRADING): True,          # the one transition a machine may never make
    (State.SHADOW, State.DISABLED): True,
    (State.SHADOW, State.HALTED): False,
    (State.TRADING, State.SHADOW): True,          # 'pause': keep positions, stop acting
    (State.TRADING, State.HALTED): False,        # risk breach / disconnect
    (State.TRADING, State.LIQUIDATING): True,
    (State.HALTED, State.SHADOW): True,           # resuming is always deliberate
    (State.HALTED, State.LIQUIDATING): True,
    (State.HALTED, State.DISABLED): True,
    (State.LIQUIDATING, State.DISABLED): False,  # automatic once flat
    (State.LIQUIDATING, State.HALTED): False,
}


class TransitionRejected(RuntimeError):
    pass


@dataclass
class Lifecycle:
    state: State = State.DISABLED
    reason: str = "initial"

    def can(self, to: State, *, by_operator: bool) -> tuple[bool, str]:
        if to is self.state:
            return False, f"already {to.value}"
        key = (self.state, to)
        if key not in _ALLOWED:
            return False, f"{self.state.value} -> {to.value} is not a legal transition"
        if _ALLOWED[key] and not by_operator:
            return False, (f"{self.state.value} -> {to.value} is operator-only; "
                           "an automatic path may not make this transition")
        return True, ""

    def to(self, target: State, *, reason: str, by_operator: bool = False) -> State:
        ok, why = self.can(target, by_operator=by_operator)
        if not ok:
            raise TransitionRejected(why)
        self.state, self.reason = target, reason
        return self.state

    def halt(self, reason: str) -> State:
        """Risk breach or lost feed. Always permitted from any acting state, never operator-gated."""
        if self.state in (State.SHADOW, State.TRADING):
            return self.to(State.HALTED, reason=reason, by_operator=False)
        return self.state
