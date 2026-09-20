"""The acceptance gate for porting a lane onto `run_sessions` (#120 phase 2).

Phase 1 unified MOMENTUM and BCTROT. Phase 2 brings TECHIVOL-005 (QC27) and QC345-003 onto the same
loop. The whole value of that is one engine; the whole RISK is that "unified" quietly becomes
"changed the historical numbers", so the port is only as good as the gate that says it reproduced.

This module is the gate, and it is written BEFORE the configs are frozen deliberately: it needs the
config sheet at RUN time, not at WRITE time, so it is progress that cannot be invalidated by what
the sheet turns out to say.

SIX RULES, EACH ENFORCED RATHER THAN ASKED FOR
-----------------------------------------------

DECLARED FIRST   Tolerances are frozen before any delta is seen. "A tolerance chosen after seeing
                 the delta is not a gate" — and the way that happens is never a decision, it is a
                 number nudged while staring at a near-miss. `declare()` then `evaluate()`, and
                 declaring after evaluating RAISES.

PROVENANCE       Every config field carries how it is known. A gate run against a field nobody can
                 confirm is CONDITIONAL, never PASS — because a delta on an unconfirmed field
                 cannot be attributed, and the cheap resolution is to tune until it matches, which
                 ports the wrong strategy behind a green gate.

STAMP != CONFIG  A journal stamp proves what the runner USED at that instant, not what the config
                 SAYS. A delayed fire, a retry, or a re-decision after a restart all stamp
                 something true about the event and false about the setting. BCTROT's
                 "open+215m" was exactly this — one row, 2026-08-24, a timestamp read as a
                 property. So STAMP_DERIVED is its own level and carries how many sessions it held.

THREE GATES      PORT ("does the new engine reproduce the old numbers"), DEPLOYMENT ("does it
                 reproduce what LIVE did") and ABLATION ("does this mechanism change anything")
                 answer different questions, and the first two treat IDENTICAL as the pass while
                 the third treats it as the failure. A passing port gate must never be reported as
                 the deployment answer.

NO INHERITED     A view's `window` has NO DEFAULT. 50 is TECHIVOL's fitted answer on TECHIVOL's
WINDOW           data; a lane that inherits it has inherited a number nobody measured for it. That
                 is the same error as inheriting `min_abs_gap_pct` from a sibling, which is the one
                 that flipped BCTROT's sign.

THREE STATES     `sessions_in_panel` / `evaluated` / `decided` / `exit_only`. A session with no
                 decision row is "evaluated, nothing to do" — QC345 has 20 of 21 such a month — and
                 is NOT "not evaluated" (warm-up, `trade_from`). An exit-only month must never
                 render as empty; cockpit spent #859 and #884 enforcing that on the display plane
                 and the backtest surface must not reintroduce it.

WHY BCTROT IS THE WORKED EXAMPLE OF WHY THIS IS NEEDED
-------------------------------------------------------
BCTROT-004 is not "MOMENTUM-002 plus two changes". It is plus two changes AND a different cadence —
three decisions a session against MOMENTUM's one. And the cadence and `min_abs_gap_pct=0.015`
landed in the SAME cockpit commit (4e5778a, 2026-09-04, first boot 16:07Z): one change event with
two behavioural halves. Any window including 09-04 sees both; before it, deployed BCTROT was two
slots with no gap filter. A gate window that straddles a change event is measuring two
configurations and reporting one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AGREES_UNPROVEN",
    "CONDITIONAL",
    "FAIL",
    "INERT",
    "PASS",
    "PROVEN",
    "STAMP_DERIVED",
    "UNKNOWN",
    "ConfigField",
    "ConfigSheet",
    "GateResult",
    "PortGate",
    "Provenance",
    "SessionCounts",
    "Tolerance",
    "ViewArm",
    "straddles_change",
]

#: How a config field is known. FOUR levels, for the reason every multi-state in this package
#: exists: "we read it off a journal row" is neither "the config says so" nor "nobody knows", and
#: "two sources agree" is neither proof nor ignorance. Collapsing any of them loses a distinction
#: that has already cost a published number — `open+215m` was a stamp read as config, and
#: `min_abs_gap_pct` was an agreement inherited from a sibling.
PROVEN = "proven"
#: Two sources agree but neither is authoritative — e.g. this repo's default equals what the stack
#: appears to use. Agreement is evidence and it is NOT proof: the sibling that "obviously" had the
#: same value is how `min_abs_gap_pct` was inherited, and that flipped BCTROT's sign.
AGREES_UNPROVEN = "agrees_unproven"
STAMP_DERIVED = "stamp_derived"
UNKNOWN = "unknown"

Provenance = str

PASS = "pass"
FAIL = "fail"
CONDITIONAL = "conditional"
#: An ABLATION whose two arms produced identical results. A verdict of its own, because for that
#: gate identical is not a pass — it means the mechanism never bound.
INERT = "inert"


@dataclass(frozen=True)
class ConfigField:
    """One behaviour-bearing setting, and HOW it is known.

    `sessions_consistent` is not decoration. A STAMP_DERIVED value that held for 3 of 3 sessions is
    a different claim from one seen in a single row — and the single row is what `open+215m` was.
    """

    name: str
    value: object
    provenance: Provenance
    source: str = ""
    sessions_consistent: int = 0

    def __post_init__(self) -> None:
        if self.provenance not in (PROVEN, AGREES_UNPROVEN, STAMP_DERIVED, UNKNOWN):
            raise ValueError(
                f"{self.name}: provenance {self.provenance!r} is not one of "
                f"{PROVEN}/{AGREES_UNPROVEN}/{STAMP_DERIVED}/{UNKNOWN}. An unrecognised level would be "
                f"treated as trustworthy by anything testing equality against {PROVEN!r}, so it is "
                f"refused rather than defaulted.")
        if self.provenance is not UNKNOWN and not self.source:
            raise ValueError(
                f"{self.name}: a known value must name where it came from. 'proven' with no source "
                f"is an assertion, and the whole point of this sheet is that assertions about live "
                f"config are what moved three published numbers in one day.")

    @property
    def trustworthy(self) -> bool:
        """PROVEN, or STAMP_DERIVED that held across more than one session.

        A single stamp is one event. `open+215m` was one row from 2026-08-24 — a delayed firing
        recorded faithfully and read as a setting.
        """
        if self.provenance == PROVEN:
            return True
        # THE COUNT, NOT THE CATEGORY NAME. A stamp that held 3 of 3 sessions and a stamp from one
        # row are the same category and different evidence; `open+215m` was the second.
        return self.provenance == STAMP_DERIVED and self.sessions_consistent >= 2


@dataclass(frozen=True)
class ConfigSheet:
    """Every behaviour-bearing field for one lane, with its provenance."""

    lane: str
    fields: tuple[ConfigField, ...] = ()

    @property
    def untrustworthy(self) -> tuple[str, ...]:
        """Fields a gate result must be reported as CONDITIONAL on."""
        return tuple(f.name for f in self.fields if not f.trustworthy)

    @property
    def complete(self) -> bool:
        return not self.untrustworthy


@dataclass(frozen=True)
class Tolerance:
    """A limit, and the reason it is that number.

    `reason` is required. A tolerance without one cannot be argued with later, and "0.1 pp" with no
    stated basis is indistinguishable from a number chosen to fit a delta somebody already saw.
    """

    metric: str
    limit: float
    reason: str

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError(f"{self.metric}: a tolerance cannot be negative ({self.limit})")
        if not self.reason.strip():
            raise ValueError(
                f"{self.metric}: a tolerance needs a stated reason. Without one it cannot be "
                f"argued with, and it is indistinguishable from a number chosen to fit a delta "
                f"somebody had already seen.")


@dataclass(frozen=True)
class SessionCounts:
    """What the run actually did, in the three states that are not each other.

    An EXIT-ONLY month is "evaluated, nothing to do" and must never render as an empty result.
    QC345 has 20 of 21 such sessions in a month; a surface that shows those as empty reads as a
    broken lane.
    """

    in_panel: int
    evaluated: int
    decided: int
    exit_only: int = 0

    def __post_init__(self) -> None:
        if self.evaluated > self.in_panel:
            raise ValueError(f"evaluated {self.evaluated} > in_panel {self.in_panel}")
        if self.decided + self.exit_only > self.evaluated:
            raise ValueError(
                f"decided {self.decided} + exit_only {self.exit_only} > evaluated "
                f"{self.evaluated}")

    def render(self) -> str:
        """Never an empty string, and never a bare zero. `decided 0 of 21 evaluated` is a result;
        an empty cell is a question."""
        return (f"decided {self.decided} of {self.evaluated} evaluated "
                f"({self.exit_only} exit-only, {self.in_panel} in panel)")


@dataclass(frozen=True)
class GateResult:
    lane: str
    kind: str
    verdict: str
    deltas: dict[str, float] = field(default_factory=dict)
    conditional_on: tuple[str, ...] = ()
    counts: SessionCounts | None = None
    breaches: tuple[str, ...] = ()

    def summary(self) -> str:
        head = f"{self.lane} {self.kind} gate: {self.verdict.upper()}"
        if self.conditional_on:
            head += f" — CONDITIONAL on {', '.join(self.conditional_on)}"
        # THE DELTA IS THE NUMBER, WITH ITS SIGN. "within tolerance" hides whether the port made
        # the lane look better or worse, and only one of those is a reproduction.
        body = "  ".join(f"{m} {d:+.4f}" for m, d in sorted(self.deltas.items()))
        return f"{head}\n  {body}" + (f"\n  {self.counts.render()}" if self.counts else "")


class AlreadyEvaluated(RuntimeError):
    """Raised when tolerances are declared after a delta has been seen."""


class PortGate:
    """Declare tolerances, THEN evaluate. Not the other way round, and it is enforced.

    THREE KINDS, AND THEY DISAGREE ABOUT WHAT IDENTICAL MEANS — which is the whole reason they are
    one class with a declared direction rather than three tools:

      port         does the new engine reproduce the OLD ENGINE?   IDENTICAL IS THE PASS.
      deployment   does it reproduce WHAT LIVE DID?                identical is the pass.
      ablation     does adding a mechanism change anything?        IDENTICAL IS THE FAILURE.

    That is CLAUDE.md's rule stated as an API: "identical when they should differ means a dead
    mechanism; differing when they should match means a live defect." A market view that is
    declared but never binds produces byte-identical arms, and a gate that reports that as "no harm
    done" has inverted its own finding — nothing happened is not the same as nothing broke.
    `backtesting/ablation.py` already raises `InertArmError` on this; the verdict here is `INERT`
    so the same report can carry it.

    A passing PORT gate answers nothing about deployment, and reporting it as though it did is the
    specific substitution this class exists to prevent.
    """

    KINDS = ("port", "deployment", "ablation")

    def __init__(self, sheet: ConfigSheet, *, kind: str) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, not {kind!r}")
        self.sheet, self.kind = sheet, kind
        self._tolerances: dict[str, Tolerance] = {}
        self._evaluated = False

    def declare(self, *tolerances: Tolerance) -> PortGate:
        if self._evaluated:
            raise AlreadyEvaluated(
                f"{self.sheet.lane}: tolerances declared AFTER a delta was seen. A tolerance "
                f"chosen once the answer is visible is not a gate — and it never happens as a "
                f"decision, it happens as a number nudged while staring at a near-miss.")
        for t in tolerances:
            self._tolerances[t.metric] = t
        return self

    def evaluate(self, deltas: dict[str, float], *, counts: SessionCounts | None = None
                 ) -> GateResult:
        if not self._tolerances:
            raise ValueError(
                f"{self.sheet.lane}: no tolerances declared. An unbounded gate passes everything, "
                f"which is worse than no gate because it reports a verdict.")
        self._evaluated = True

        undeclared = sorted(set(deltas) - set(self._tolerances))
        if undeclared:
            raise ValueError(
                f"{self.sheet.lane}: deltas supplied for {undeclared} with no tolerance declared "
                f"for them. Judging a metric whose limit was never stated is the same failure as "
                f"choosing the limit afterwards.")

        if self.kind == "ablation" and deltas and all(d == 0.0 for d in deltas.values()):
            # IDENTICAL ARMS ARE THE FINDING, NOT THE PASS. The mechanism was declared and never
            # bound: a view computed and discarded, a flag read and ignored. Reporting "within
            # tolerance" here would record a dead mechanism as a safe one.
            return GateResult(lane=self.sheet.lane, kind=self.kind, verdict=INERT,
                              deltas=dict(deltas), conditional_on=self.sheet.untrustworthy,
                              counts=counts,
                              breaches=("every metric is identical across the two arms — the "
                                        "mechanism under test never bound. Nothing happened is not "
                                        "the same as nothing broke.",))

        breaches = tuple(
            f"{m}: |{deltas[m]:+.4f}| exceeds {self._tolerances[m].limit} "
            f"({self._tolerances[m].reason})"
            for m in sorted(deltas) if abs(deltas[m]) > self._tolerances[m].limit)

        untrustworthy = self.sheet.untrustworthy
        if breaches:
            verdict = FAIL          # a breach is a breach whatever the provenance
        elif untrustworthy:
            # CONDITIONAL, NOT PASS. A delta inside tolerance on a field nobody can confirm has not
            # been shown to be a reproduction — it has been shown to be small, which is what a
            # wrong config also looks like when the two happen to be close.
            verdict = CONDITIONAL
        else:
            verdict = PASS
        return GateResult(lane=self.sheet.lane, kind=self.kind, verdict=verdict,
                          deltas=dict(deltas), conditional_on=untrustworthy,
                          counts=counts, breaches=breaches)


@dataclass(frozen=True)
class ViewArm:
    """One arm of a market-view ablation: which view, how long it must persist, and whether it ACTS.

    `window` HAS NO DEFAULT, deliberately. 50 sessions is TECHIVOL's fitted answer on TECHIVOL's own
    data, and a lane that inherits it has inherited a number nobody measured for that lane. It is
    the same error as inheriting `min_abs_gap_pct` from a sibling — the error that flipped BCTROT's
    sign — so the type refuses to supply one.

    `dwell` IS SWEPT, NOT TAKEN AS 1. dwell=1 means a single session flips the state, and a
    whipsaw month is exactly the scenario that turns a protective mechanism into a fee generator.
    The 20-day arm sat in cash 30% of sessions on average and 41-68% in some periods, missing
    recoveries as well as falls.

    `acts=False` IS THE SHADOW ARM: the view is evaluated, the event fires, the label renders, and
    NOTHING IS GATED. It is the observe-only wiring that must precede any lane arming, and it is
    also a legitimate backtest arm — so one comparison machinery serves both. CRSISHORT's
    `shadow_only` registration refusal is the in-repo precedent.
    """

    lane: str
    signal: str | None
    window: int
    action: str | None
    dwell: int = 1
    acts: bool = True

    def __post_init__(self) -> None:
        if self.signal is not None and self.window < 2:
            raise ValueError(
                f"{self.lane}: window {self.window} is not a window. It must be measured on THIS "
                f"lane's own panel — inheriting 50 because it worked for TECHIVOL is how a fitted "
                f"constant becomes a platform assumption.")
        if self.signal is not None and self.action is None and self.acts:
            raise ValueError(
                f"{self.lane}: a view that ACTS must say what it does — blocking entries and "
                f"liquidating are different actions and every published figure measures LIQUIDATE. "
                f"Pass acts=False for a shadow arm, which needs no action.")
        if self.dwell < 1:
            raise ValueError(f"{self.lane}: dwell must be >= 1, got {self.dwell}")

    @property
    def label(self) -> str:
        if self.signal is None:
            return "no view"
        act = "shadow" if not self.acts else self.action
        return f"{self.signal}/{self.window}d dwell={self.dwell} {act}"


def split_view_effect(deltas: dict[str, float]) -> dict[str, dict[str, float]]:
    """Separate the BLOCKED-ENTRY effect from the LIQUIDATION effect. They are different bets.

    A net figure hides one behind the other, and they can cancel. Blocked entries during a drawdown
    cost the RECOVERY — which for momentum is where most of the return lives. A liquidation
    realises losses and then pays to re-enter. Reporting one number for both is how a mechanism
    that helps on one axis and hurts on the other is recorded as neutral.

    Keys are matched by prefix so a caller cannot silently omit one half: a result with only
    `blocked_*` metrics reports an empty `liquidated` group rather than folding it into the total.
    """
    out: dict[str, dict[str, float]] = {"blocked": {}, "liquidated": {}, "other": {}}
    for metric, value in deltas.items():
        if metric.startswith("blocked_"):
            out["blocked"][metric] = value
        elif metric.startswith("liquidated_"):
            out["liquidated"][metric] = value
        else:
            out["other"][metric] = value
    return out


def straddles_change(window_start: str, window_end: str, change_dates: tuple[str, ...]
                     ) -> tuple[str, ...]:
    """Change events inside the gate window — each one means TWO configurations, one number.

    BCTROT is the worked example: cockpit 4e5778a (2026-09-04) added the third decision slot AND
    `min_abs_gap_pct=0.015` in ONE commit. Before it the deployed lane was two slots with no gap
    filter. A window spanning that date measures both and reports one figure, and the two halves
    are not separable after the fact because they never varied independently.
    """
    return tuple(d for d in change_dates if window_start <= d <= window_end)
