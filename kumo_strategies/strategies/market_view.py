"""The lane's own view of its market — ONE implementation, for every driver (kumo-trading-platform issue 873).

2026-09-11: *"a market view is separate per strategy. that is why I suggested strategy to
define and cockpit to run it."* So this is not a platform-wide breadth number handed down. Each
lane declares the market it cares about — TECHIVOL's is its 138-name tech pool, QC345's is its own
164 — and the platform polls, acts and journals. This module is the declaring half.

It lives here rather than in a runner for the same reason `exits.py` does: the rule had no single
home, so each driver grew its own and one grew incompletely. A backtest that measures one rule while
the live adapter runs another is not a backtest of anything.

WHY A LEVEL AND NOT A COUNT. QC27 already ships a de-risking valve — residual cash to `GLD` when
fewer than `portfolio_size` names carry positive momentum — and it is inverted in production:
measured mean cash weight 0.01 while the book is >30% below its high, against 0.44 while within 5%
of it. It goes to cash near the top and rides the bottom fully invested. The defect is not speed.
An arm at the SAME 63-day horizon, asking whether the index is below its own average, cut the worst
period's drawdown from -41.3% to -18.4%. A COUNT of names still carrying positive momentum stays
high through a decline, because after a run-up ten still qualify; a LEVEL does not.

    A market view built from the lane's own selection scores cannot fire when it matters,
    because it is made of the things that are falling.

WHY NOT FASTER. Measured over 8 independent periods, summed return against no filter:

    63d MA  -53.5pp     50d MA  -33.3pp     20d MA  -95.9pp     20d breadth  -103.3pp

Monotone in the wrong direction. Fast views whipsaw — the 20-day arm sat in cash 30% of sessions on
average and 41-68% in some periods, so it missed recoveries as well as falls. `window` defaults to
50 because that is where the measurement put it, not because it is a round number.

THREE STATES, NOT TWO (#873). `UNKNOWN` is not `RISK_OFF`. A view that cannot be computed — too
little history, an empty panel — must say so, because "not asked" is not "not protected" and a
caller that liquidates on an unreadable signal has turned a data outage into a realised loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd

RISK_ON = "risk_on"
RISK_OFF = "risk_off"
UNKNOWN = "unknown"

#: A `Verdict`'s three answers. NOT a bool, for the same reason `MarketState` has three states: a
#: hook that could not compute must be able to SAY so. A lane newly registered answers UNKNOWN for
#: months, and that is the normal case rather than an error.
YES = "yes"
NO = "no"

#: An `Assessment`'s three. Same rule.
IN_ENVELOPE = "in_envelope"
OUT_OF_ENVELOPE = "out_of_envelope"


class MarketSignal(str, Enum):
    """How the lane reads its market. Deliberately small — every member here is measured."""

    NONE = "none"
    """No view. The lane never blocks entries or asks to be flattened."""

    INDEX_VS_MA = "index_vs_ma"
    """Equal-weight index of the lane's own universe against its own moving average.

    A LEVEL. This is the arm that worked; see the module docstring for why that matters."""


class MarketAction(str, Enum):
    """What a lane DOES when its view says risk-off. Blocking and liquidating are different acts.

    EVERY PUBLISHED FIGURE FOR THIS VIEW MEASURES `LIQUIDATE`. `runner_qc27_verified` empties the
    target weights, so every held name is targeted to zero — that is what "−42.8pp at 50d" describes.
    `EXIT_ONLY` is UNMEASURED and plausibly much cheaper: it neither crystallises losses nor pays a
    round trip per flip (#147).
    """

    EXIT_ONLY = "exit_only"
    """Open nothing new; keep what you hold and let the lane's own exits work."""

    LIQUIDATE = "liquidate"
    """Flatten the book. A superset — a lane being flattened is not also opening positions."""


class SelfAction(str, Enum):
    """What a lane does when it concludes something is wrong WITH ITSELF.

    A SEPARATE TYPE FROM `MarketAction`, DELIBERATELY, and this was asked for as a third member of
    that enum. It should not be one. `MarketAction` answers "what do I do when MY MARKET is bad";
    this answers "what do I do when I AM bad". Different subject, different trigger, different
    clearing rule — and one enum spanning both makes `MarketViewConfig(action=STAND_DOWN)`
    constructible, a market view that quarantines, which means nothing. That would then need a
    runtime guard for something a type can prevent outright.

    Rule 3 of this protocol is that an action belongs to its trigger. Two triggers, two types.
    """

    INVESTIGATE = "investigate"
    """SOMETHING IS ODD AND NOTHING SHOULD BE REDUCED. A lane sitting ABOVE its own envelope.

    A SEPARATE MEMBER BECAUSE THE OTHER TAIL IS NOT THE SAME CLAIM. The band is two-sided and the
    detection should be: a lane far above its own 90th percentile is genuinely not doing what it
    said, and the live hypotheses are a sizing bug, a data error or a config change. Every one of
    those is a reason to LOOK. None is a reason to stop trading.

    Falk's definition of quarantine is A STRATEGY THAT LOST ITS EDGE. A lane up 30% in 13 sessions
    has not lost its edge, and `assess` stood it down anyway until #147 — one action, named for one
    tail, serving both. That is the #144 shape exactly: a flag named for blocking that liquidated.

    AN ACTION THAT REDUCES RISK ON A LANE THAT IS WINNING IS THE MOST EXPENSIVE THING THIS MECHANISM
    COULD DO, because it looks correct in every log line while it costs return.
    """

    STAND_DOWN = "stand_down"
    """Stop opening, keep managing what is held, and RAISE A HAND — a human decides.

    IT IS A REQUEST, NOT A STATE THE LANE ENTERS. kumo-trading-platform renders the badge as "requesting
    stand-down" rather than "standing down", and that is the sharper reading: the lane is making a
    CLAIM that a human resolves, not occupying a condition it can later leave. Both repos say the
    same thing about it deliberately — a lane that could put itself into and out of a stand-down
    would be certifying its own recovery, which is precisely what it cannot do.

    THREE MECHANISMS THAT ALL STOP A LANE BUYING, AND THEY ARE NOT THE SAME CLAIM:

      EXIT_ONLY   a MARKET call. "Today is not the day." Self-clearing: when the market turns, the
                  lane resumes on its own, and nobody is asked anything.
      HALT        a RISK breach (`daily_loss` -> `Action.HALT`). Durable, and an operator clears
                  it. The lane may be working perfectly; it has simply lost too much today.
      STAND_DOWN  the lane's own verdict that ITS EDGE IS GONE — it is outside its own
                  pre-registered envelope with enough independent windows to be sure. Neither the
                  market's fault nor a loss limit. It self-clears NEVER, because a lane cannot
                  certify its own recovery with the same evidence that condemned it.

    Exits keep working. A lane that believes it has no edge should stop ACQUIRING, not abandon
    positions into whatever liquidity exists — that would be LIQUIDATE, which is the emergency
    action and answers a different question.

    NAMED `STAND_DOWN` AND NOT `QUARANTINE`, ON PURPOSE. kumo-trading-platform already uses "quarantine" for
    the #79 quarantine PLANE — broker activity cockpit did not originate, rendered as
    "UNCLAIMED · foreign strategy". That is about POSITIONS WHOSE OWNER IS UNKNOWN. This is a lane
    reporting on ITSELF, with a known owner. One word for two unrelated things across two repos is
    a collision that costs somebody a day, and the repo that has not shipped the word yet is the
    one that should move.
    """

    @property
    def reduces(self) -> bool:
        """Does acting on this make the book SMALLER?

        A PROPERTY ON THE ENUM, not a rule each caller remembers. `if assessment.action:` is the
        call site that makes this dangerous — a truthy action read as "reduce something" is how
        INVESTIGATE would come to flatten a winning lane. The type answers it once, and a member
        added later cannot avoid the question.
        """
        return self is SelfAction.STAND_DOWN


#: Whether each signal reads a LEVEL, a STATE, or something else. THE RULE THIS MAP ENFORCES is that
#: a view may not be built from a COUNT of the lane's own selection scores: after a run-up ten names
#: still rank well while the index is falling, which is exactly how QC27's shipped valve came to sit
#: in cash near the high and fully invested at the low. Declared per member so a new signal cannot be
#: added without answering the question.
SIGNAL_KIND: "dict[MarketSignal, str]" = {
    MarketSignal.INDEX_VS_MA: "level",
}



@dataclass(frozen=True)
class MarketViewConfig:
    """All new automation is opt-in, so `signal` defaults to NONE and nothing changes until a lane
    asks for a view. `window` is only read by `INDEX_VS_MA`."""

    signal: MarketSignal = MarketSignal.NONE
    """WHAT the lane reads. `NONE` is the inert default and it is LOUD: it yields RISK_ON with the
    reason "no market view configured", which travels all the way to the notification payload. A
    lane that has not measured a view declares NONE rather than a window nobody fitted for it."""

    window: int = 50
    """Sessions in the moving average. MEASURED PER LANE, NEVER INHERITED.

    50 is TECHIVOL's fitted answer on TECHIVOL's own data — summed return against no view: 63d
    -78.5pp, 50d -42.8pp, 20d -71.7pp, so faster is worse and the horizon was never the defect.
    It is NOT a platform default: a lane adopting it has adopted a number nobody measured for it,
    which is the error that made one fitted constant look like a house setting.

    The default of 50 exists so `MarketViewConfig()` is constructible; it is inert while
    `signal=NONE`, and a lane declaring a signal must have measured its own."""
    action: "MarketAction | None" = None
    """REQUIRED whenever a signal is declared, and deliberately without a default.

    Defaulting to EXIT_ONLY would ship the arm nobody has measured; defaulting to LIQUIDATE would
    make the more destructive act the one you get by not thinking. This package's answer to that
    choice is the same as for `POSITION_SIDE`, `price_adjustment` and `strategy_id`: refuse to make
    it, and require the lane to say."""
    dwell: int = 1
    """Sessions the state must hold before it is reported. 1 = act on the first close that crosses.

    A DOCSTRING RATHER THAN A `#:` COMMENT, because the settings-schema extractor reads docstrings
    and this field was reaching cockpit undescribed. The `#:` form renders in some doc tools and
    is invisible to the walker — a distinction nobody would notice until an operator saw a blank
    description on a field that decides how fast a lane de-risks.

    DWELL BELONGS TO THE POLLER for `emergency_exit` (#873 counts consecutive `yes` polls); this
    is the signal's own, so a backtest and a live poller can express the same rule.

    MEASURED, not assumed: on TECHIVOL, dwell 1/2/3 spans 220-252% compounded and all three save
    ~14.6pp of drawdown, while dwell=5 is worse on both axes. No dwell setting approaches the
    ACTION choice, which moves the result by 172pp — dwell is not the lever."""

    def __post_init__(self) -> None:
        # THE TYPE SEPARATION, ENFORCED. `SelfAction.STAND_DOWN` answers "what do I do when I AM
        # bad"; this config answers "what do I do when MY MARKET is bad". A market signal
        # delivering a verdict about the lane's own edge is meaningless, and accepting one here
        # would let it be configured and then silently ignored by `blocks_entries`/`liquidates`,
        # which only know `MarketAction`.
        if self.action is not None and not isinstance(self.action, MarketAction):
            raise ValueError(
                f"action must be a MarketAction, not {type(self.action).__name__}"
                f"({self.action!r}). A market view decides what to do about the MARKET; a lane's "
                f"verdict on its own edge is a SelfAction and reaches the platform through "
                f"`self_assessment`, not through this config.")
        if self.signal is not MarketSignal.NONE and self.action is None:
            raise ValueError(
                f"a lane declaring {self.signal.name} must also declare its action — "
                f"MarketAction.EXIT_ONLY (open nothing, keep what you hold) or "
                f"MarketAction.LIQUIDATE (flatten). There is no default: every published figure for "
                f"this view measures LIQUIDATE, so a default would either ship an unmeasured arm or "
                f"make flattening the thing you get by not thinking (#147).")


@dataclass(frozen=True)
class MarketState:
    """What the lane says about its market, and why. `reasons` exists for the UI (#873)."""

    state: str
    reasons: tuple[str, ...] = ()
    action: "MarketAction | None" = None

    @property
    def blocks_entries(self) -> bool:
        """May the lane OPEN anything? False under either action once the view is risk-off.

        Separate from `liquidates` because #873 wants both and they are not the same act — and
        because a single flag that meant one and did the other is the defect this package already
        shipped once (#144).
        """
        return self.state == RISK_OFF and self.action is not None

    @property
    def liquidates(self) -> bool:
        """`UNKNOWN` does NOT block. It raises its own alarm through the caller's journal instead —
        a view we could not compute is not evidence the market is bad, and treating it as such makes
        every data gap a de-risking event."""
        return self.state == RISK_OFF and self.action is MarketAction.LIQUIDATE



@dataclass(frozen=True)
class Verdict:
    """An answer WITH ITS REASON, in THREE states.

    #873 needs the reason for the UI, and an operator being de-risked deserves to know why — a bare
    bool is what makes a valve unexplainable after the fact.

    THREE, NOT TWO, and this shipped wrong once: `answer: bool` cannot express "could not compute",
    so a hook with too little data would have to answer False and be read as a definite no. The
    market view was built with three states deliberately and these two hooks were given a bool; the
    inconsistency was the defect, found by the cockpit session building against it.
    """

    state: str
    reasons: tuple[str, ...] = ()

    @property
    def acts(self) -> bool:
        """NOTHING IN THIS PROTOCOL ACTS ON UNKNOWN. Rule 4, generalised — the poller relies on this
        rather than special-casing each type."""
        return self.state == YES


@dataclass(frozen=True)
class Assessment:
    """What a lane says about ITSELF, as opposed to about its market.

    Separate from the two market hooks because "the market is falling" and "I am not working" want
    different responses, and a lane that cannot tell them apart de-risks for the wrong reason.

    THREE STATES, for the reason in `Verdict`: a lane with too few live sessions to place itself in
    its own distribution must be able to say UNKNOWN, which is the common case for WEEKS after a
    lane is registered.
    """

    state: str
    reasons: tuple[str, ...] = ()
    evidence_sufficient: bool = True
    """Whether enough independent live windows have accumulated to ACT on an out-of-envelope
    observation. Separate from `state` on purpose: a single bad window should be VISIBLE on a
    surface without being a trigger, and folding the two would make the lane either blind or
    trigger-happy. The threshold itself is pre-registered on the `Envelope`."""

    @property
    def acts(self) -> bool:
        """One derivation. `state` says what was seen, `evidence_sufficient` whether it is enough;
        `acts` is the only thing a caller should branch on.

        THE HIGH TAIL IS NOT A QUARANTINE. `acts` is what fires the stand-down, so a lane above its
        own band must not set it — it is OUT_OF_ENVELOPE, visible, and answers INVESTIGATE."""
        return (self.state == OUT_OF_ENVELOPE and self.evidence_sufficient
                and self.breach != "above")

    breach: str = ""
    """WHICH tail the lane sits outside — "below", "above", or "" for neither.

    CARRIED, NOT DERIVED, because the percentile that decided it is not on this object and a caller
    re-deriving it would be a second derivation of one fact. It is what separates "lost its edge"
    from "doing something nobody predicted", and those want opposite responses.
    """

    @property
    def action(self) -> "SelfAction | None":
        """WHAT TO DO, not merely whether to. `acts` returned a bool into the void — the detector
        (#147's envelopes) and the evidence discipline were both built and there was nothing a lane
        could SAY once it concluded it was out of its own envelope.

        `None` when it does not act, so a caller cannot accidentally stand a lane down on an
        UNKNOWN or on a single bad window.

        TWO TAILS, TWO ANSWERS (#147). The band is two-sided and the DETECTION should be — a lane
        far above its own 90th percentile is genuinely not doing what it said. But STAND_DOWN means
        "this lane lost its edge", and a lane beating its envelope has not. It answers INVESTIGATE,
        which does not reduce: the live hypotheses are a sizing bug, a data error or a config
        change, all reasons to look and none to stop trading.

        Until #147 both tails returned STAND_DOWN. Driven: a lane at +30% return_pct, against a 90th
        percentile of +6.38, was told to stand down.
        """
        if self.state != OUT_OF_ENVELOPE or not self.evidence_sufficient:
            return None
        if self.breach == "above":
            return SelfAction.INVESTIGATE
        return SelfAction.STAND_DOWN if self.acts else None


class MarketAware(Protocol):
    """The three hooks the platform calls (kumo-trading-platform issue 873).

    THREE, NOT TWO, AND THEY ARE NOT INTERCHANGEABLE:

      entries_blocked    may I OPEN anything? EXIT_ONLY — the book stands and the lane's own exits
                         keep working.
      emergency_exit     must I CLOSE what I hold? LIQUIDATE — the destructive act, and the one
                         every published measurement of this view actually describes.
      self_assessment    how am I doing? A lane's own health, which is a different question from
                         its market's.

    Collapsing any two is not hypothetical: a property named `blocks_entries` returned True while
    the runner emptied the target weights and sold the entire book (#144). The names and the acts
    had drifted apart and every figure published under the first name measured the second.
    """

    def entries_blocked(self) -> Verdict: ...

    def emergency_exit(self) -> Verdict: ...

    def self_assessment(self) -> Assessment: ...
    """"I am not doing what I said." Carries `action` — `SelfAction.STAND_DOWN` or None — so the
    platform has something to act on rather than a bool."""


def equal_weight_index(prices: pd.DataFrame) -> pd.Series:
    """The lane's own market: equal-weight cumulative return of every name it ranks.

    Equal weight rather than cap weight on purpose. The view has to describe the pool the lane
    actually picks from, and a cap-weighted tech index is four names — which would have this lane
    de-risking on the mega-caps it mostly does not hold.
    """
    return (1 + prices.pct_change().mean(axis=1)).cumprod()


def target_weights_under(view, weights: dict, held) -> dict:
    """The target book once the view has had its say — RULE 3 AS ONE FUNCTION (#170).

    Blocking entries and liquidating are different actions, and the QC27 runner implemented only
    one of them: it read `.liquidates` and never `.blocks_entries`, so an EXIT_ONLY view changed
    nothing. Both EXIT_ONLY arms then reproduced the no-view control BYTE FOR BYTE, down to 29,285
    identical fills — a dead mechanism that measures as a clean zero and reads like an answer.

    Pure and extracted rather than inline, for two reasons. It is the one place the two actions
    diverge, so a driver cannot implement half of it by accident; and inline it could only be
    tested through a panel coaxed into producing a risk-off entry, which took three fixtures and
    still did not separate the arms. The rule is small enough to state directly, so it is.

      LIQUIDATE    empty. Sell everything.
      EXIT_ONLY    keep every name ALREADY HELD that is still wanted; open nothing new. Exits are
                   not blocked — that is what "exit only" means — so a held name that has left the
                   ranking entirely is absent here and gets sold by the ordinary rule.
      otherwise    the lane's own weights, untouched.
    """
    if view is None:
        return dict(weights)
    if getattr(view, "liquidates", False):
        return {}
    if getattr(view, "blocks_entries", False):
        return {s: w for s, w in weights.items() if s in held}
    return dict(weights)


def market_state(prices: pd.DataFrame, cfg: MarketViewConfig, *, asof: pd.Timestamp) -> MarketState:
    """The lane's view as at `asof`, computed from data STRICTLY BEFORE it.

    ENTRY-COMPUTABLE BY CONSTRUCTION, not by the caller remembering to lag. A view that reads
    `asof`'s own close is a view no live lane can act on at `asof`'s open, and the resulting backtest
    is a measurement of hindsight. The cut happens here so no driver can get it wrong.
    """
    if cfg.signal is MarketSignal.NONE:
        return MarketState(RISK_ON, action=cfg.action, reasons=("no market view configured",))
    if prices is None or prices.empty:
        return MarketState(UNKNOWN, action=cfg.action, reasons=("no price panel",))

    history = prices[prices.index < asof]
    need = cfg.window + cfg.dwell
    if len(history) < need:
        return MarketState(UNKNOWN, action=cfg.action, reasons=(
            f"{len(history)} sessions of history, {need} needed for a {cfg.window}-session "
            f"average with dwell {cfg.dwell}",))

    index = equal_weight_index(history)
    ma = index.rolling(cfg.window).mean()
    recent = (index <= ma).tail(cfg.dwell)
    if recent.isna().any():
        return MarketState(UNKNOWN, action=cfg.action, reasons=("moving average is not yet defined",))
    if bool(recent.all()):
        last, avg = float(index.iloc[-1]), float(ma.iloc[-1])
        return MarketState(RISK_OFF, action=cfg.action, reasons=(
            f"equal-weight index {last:.4f} at or below its {cfg.window}-session average {avg:.4f}"
            + (f" for {cfg.dwell} consecutive sessions" if cfg.dwell > 1 else ""),))
    return MarketState(RISK_ON, action=cfg.action, reasons=(
        f"equal-weight index above its {cfg.window}-session average",))
