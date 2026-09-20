"""A lane's PRE-REGISTERED envelope: what it said it would do, so "not doing it" is checkable.

THIS ANALYSIS WAS DONE BY HAND ONCE AND THROWN AWAY. From the TECHIVOL paper post-mortem:

    TECHIVOL's paper window sits at the 13th percentile of its own backtest's 13-session
    distribution, where 38% of all such windows are negative and p10 is -6.40% — live delivered
    exactly -6.4%. Its picks are fine. Its turnover is not.

That is a self-assessment. It stopped a lane being blamed for a selection failure it did not have,
and the next incident would have had to recompute it under pressure. So the lane exposes it instead.

**THE HOOK ANSWERS "WHERE DOES LIVE SIT IN MY OWN DISTRIBUTION?", NOT "AM I DOWN?"** — and the
consequence that matters is that being DOWN is not out-of-envelope when the lane's own history says
38% of windows are negative. A lane that cannot say that gets de-risked for behaving as measured.

FOUR RULES, EACH ENFORCED HERE RATHER THAN ASKED FOR:

  PRE-REGISTERED   The envelope is frozen. A band refitted to include whatever just happened cannot
                   fail, which makes it worthless as a check.
  VERSIONED        It carries the fingerprint of the config it was fitted on, and assessing a
                   different config REFUSES rather than warns. A lane whose sizing, universe,
                   cadence or exit rule changed has new behaviour and an old distribution, and an
                   assessment against the wrong envelope is worse than none because it carries the
                   authority of a number.
  ITS OWN          About the LANE'S behaviour, not the market's. "The market fell" is
                   `entries_blocked` / `emergency_exit`. "I am not doing what I said" is this.
                   Collapsing them liquidates a working lane in a bad month.
  UNKNOWN          Too little live evidence answers UNKNOWN, and UNKNOWN never acts. For a newly
                   registered lane this is the common case for WEEKS, not an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kumo_strategies.strategies.market_view import (
    IN_ENVELOPE, OUT_OF_ENVELOPE, UNKNOWN, Assessment)

__all__ = ["Envelope", "StaleEnvelope", "assess", "IN_ENVELOPE", "OUT_OF_ENVELOPE", "UNKNOWN"]


class StaleEnvelope(RuntimeError):
    """Raised when an envelope is asked about a config it was not fitted on.

    NOT A WARNING. A warning would leave the caller holding a number that looks like an assessment
    and describes a strategy that is no longer running — the shape this package has shipped
    repeatedly, where a knob agrees with its configured value because nothing recomputed it.
    """


@dataclass(frozen=True)
class Envelope:
    """What a lane pre-registers about its own behaviour, fitted on its own backtest.

    FROZEN, deliberately: a caller that wants a wider band must REGISTER a new envelope, which
    changes the fingerprint and is therefore visible. Widening one in place is how a check comes to
    contain everything it has ever seen.
    """

    config_fingerprint: str
    """What this was fitted on. Assessing any other config refuses."""

    horizon_sessions: int
    """The window length the distribution describes. A percentile over 13 sessions says nothing
    about a 60-session one."""

    samples: int
    """How many windows the distribution was built from. A percentile from 4 windows and one from
    64 are different claims wearing the same word."""

    distribution: dict[str, dict[int, float]] = field(default_factory=dict)
    """metric -> {percentile: value}, from the lane's OWN backtest at its OWN fill hour."""

    bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    """metric -> (low percentile, high percentile) that count as in-envelope."""

    min_live_sessions: int = 0
    """Below this the lane answers UNKNOWN. It cannot place itself in a distribution of N-session
    windows without N sessions."""

    min_windows_to_act: int = 1
    """How many independent live windows must agree before `acts` is allowed to be True.

    THE TWO LATENCIES ARE DIFFERENT AND BOTH WERE MISSTATED AS "MONTHS". At the registered values:

        min_live_sessions 13          -> 2.6 weeks before a lane can ASSESS at all
        horizon 13 x min_windows 3    -> 39 sessions, ~7.8 weeks before it can ACT

    So a lane is unassessable for under three weeks and unable to stand down for about eight. The
    docstrings said "months" for the first, which the value does not support — the operator caught it. The
    second is roughly two months and that is the number that matters operationally, because it is
    how long a lane whose edge HAS gone keeps trading before it can say so.

    WHETHER 3 WINDOWS IS RIGHT AT A 13-SESSION HORIZON IS AN OPEN MEASUREMENT, not a preference:
    it trades false stand-downs against eight weeks of a dead lane, and nobody has measured the
    exchange rate. On #147's thread.

    n=1 against n=many is a percentile, not a verdict. The 13th percentile of a distribution where
    38% of windows are negative says nothing after one window — and the first bad month is precisely
    when a lane most needs to be left alone. PRE-REGISTERED here rather than passed to `assess`, or
    it would be chosen after seeing the answer.
    """

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError(
                f"an envelope needs samples > 0, got {self.samples!r}: a percentile with no "
                f"distribution behind it is a number wearing a claim it cannot support")
        if self.horizon_sessions < 1:
            raise ValueError(f"horizon_sessions must be >= 1, got {self.horizon_sessions!r}")


def _percentile_of(dist: dict[int, float], value: float) -> tuple[float, bool]:
    """(percentile, within_registered_range), interpolated between the points the lane registered.

    THE SECOND RETURN IS THE LOAD-BEARING ONE. An earlier version clamped to the nearest edge and
    returned only the number — so a return of -30% against a distribution whose 5th percentile is
    -9.1% came back as "the 5th percentile", which is exactly the band's floor, and read as
    IN-ENVELOPE. The worst possible observation was reported as the edge of normal.

    A distribution cannot extrapolate beyond what it was fitted on, so the honest answer outside the
    registered range is "beyond it", not a number.
    """
    points = sorted(dist.items())
    if value < points[0][1]:
        return float(points[0][0]), False
    if value > points[-1][1]:
        return float(points[-1][0]), False
    for (p_lo, v_lo), (p_hi, v_hi) in zip(points, points[1:]):
        if v_lo <= value <= v_hi:
            if v_hi == v_lo:
                return float(p_lo), True
            return p_lo + (p_hi - p_lo) * (value - v_lo) / (v_hi - v_lo), True
    return float(points[-1][0]), True


def assess(envelope: Envelope, live: dict[str, float], *, config_fingerprint: str,
           live_sessions: int, live_windows: int, consecutive_breaches: int = 0) -> Assessment:
    """Where does `live` sit in the lane's own pre-registered distribution?

    TWO COUNTERS FOR TWO QUESTIONS, and sharing one is how the defect existed (#147, #171):

      `live_windows`          WARMUP — how many independent windows the lane has produced. Below
                              `min_live_sessions` worth, it cannot place itself at all and answers
                              UNKNOWN. Being OLD is not evidence of anything.
      `consecutive_breaches`  CONFIRMATION — how many windows IN A ROW have been outside. This is
                              what `min_windows_to_act` was always documented to mean.

    Until this change `min_windows_to_act` was compared against `live_windows`, so a lane that had
    produced three windows and breached ONCE was quarantined. Measured: `live_windows=3` with a
    single -30% window returned `acts=True`, `action=STAND_DOWN`. "Three windows must agree" was
    never implemented — the knob was a warmup counter wearing a confirmation rule's name.

    `consecutive_breaches` DEFAULTS TO 0 rather than to 1, so a caller that has not been updated
    gets "no confirmed evidence" and the lane is never quarantined by omission. The failure
    direction of a forgotten argument must be the safe one.
    """
    if config_fingerprint != envelope.config_fingerprint:
        raise StaleEnvelope(
            f"this envelope was fitted on config fingerprint {envelope.config_fingerprint!r} and "
            f"the lane is running {config_fingerprint!r}. Its behaviour changed and its "
            f"distribution did not, so any percentile from it describes a strategy that is no "
            f"longer running. Refit and re-register, or revert the config.")

    if live_sessions < envelope.min_live_sessions:
        return Assessment(UNKNOWN, (
            f"{live_sessions} live sessions against a distribution of "
            f"{envelope.horizon_sessions}-session windows; {envelope.min_live_sessions} needed "
            f"before this lane can place itself in it",))

    reasons: list[str] = []
    outside: list[str] = []
    # WHICH TAIL, tracked as it is decided rather than re-derived afterwards. A lane BELOW its band
    # has lost its edge; a lane ABOVE it is doing something nobody predicted. Opposite responses.
    below = above = False
    for metric, value in sorted(live.items()):
        dist = envelope.distribution.get(metric)
        band = envelope.bands.get(metric)
        if not dist or not band:
            reasons.append(f"{metric}: no band registered, not assessed")
            continue
        pct, within = _percentile_of(dist, value)
        lo, hi = band
        if not within:
            reasons.append(
                f"{metric} {value:+.2f} is BEYOND the registered range of {envelope.samples} "
                f"windows (which spans {min(dist.values()):+.2f} to {max(dist.values()):+.2f}) — "
                f"the lane has never behaved like this in its own backtest")
            outside.append(metric)
            if value < min(dist.values()):
                below = True
            else:
                above = True
            continue
        reasons.append(
            f"{metric} {value:+.2f} sits at the {pct:.0f}th percentile of {envelope.samples} "
            f"windows (band {lo:.0f}-{hi:.0f})")
        if pct < lo:
            outside.append(metric)
            below = True
        elif pct > hi:
            outside.append(metric)
            above = True

    if not outside:
        return Assessment(IN_ENVELOPE, tuple(reasons))

    # OUT is an OBSERVATION; acting on it needs repeated evidence. The state says what was seen and
    # `acts` says whether it is enough — kept separate so a single bad window is still visible on a
    # surface without being a trigger.
    # BELOW WINS WHEN BOTH TAILS ARE TRIPPED. Two metrics can disagree — return under its p10 while
    # turnover sits over its p90 — and that combination is a lane whose edge has gone while it
    # trades more, which is the quarantine case, not the investigate one.
    tail = "below" if below else ("above" if above else "")

    if consecutive_breaches < envelope.min_windows_to_act:
        reasons.append(
            f"outside on {', '.join(outside)}, but {consecutive_breaches} consecutive breach(es) "
            f"against {envelope.min_windows_to_act} needed before acting — one window against a "
            f"distribution is a percentile, not a verdict")
        return Assessment(OUT_OF_ENVELOPE, tuple(reasons), evidence_sufficient=False, breach=tail)

    return Assessment(OUT_OF_ENVELOPE, tuple(reasons), breach=tail)
