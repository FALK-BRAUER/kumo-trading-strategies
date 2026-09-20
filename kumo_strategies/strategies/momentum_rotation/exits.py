"""The capital-efficiency exits — ONE implementation, for every driver. (kumo-trading-platform issue 197 P2)

These rules used to live twice: `backtesting/runner_verified.py` implemented all four, and
`runner.py` (the session runner beside this file) implemented one. `ExitConfig` is shared, so a research config setting
`stall_days` typechecked, backtested, deployed, and was then silently ignored by the runner holding
real positions. Nothing raised. The exit simply never fired.

That is why this module exists rather than a tidier version of either copy: the rules had no single
home, so each driver grew its own and one grew incompletely.

PURE. No I/O, no Nautilus, no SQLAlchemy, same discipline as `score.py` and `engine.py` — because the
two callers differ in everything except the rule. The backtest holds state in a dict for one process
lifetime; live reloads it from Postgres on every session. Those are both legitimate ways to *supply*
`TrailState`; neither is a legitimate reason to have a second copy of the arithmetic.

STATE IS RETURNED, NOT MUTATED. `peak` and `sessions_since_high` advance as a position is evaluated,
and the caller has to persist that. Returning it keeps this pure and makes the live path's durability
requirement explicit instead of incidental — the give-back trail being non-durable is precisely how
it went dead in production (#197 B1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from kumo_strategies.strategies.give_back import LONG, gave_back
from kumo_strategies.strategies.momentum_rotation.config import ExitConfig

#: `quality` values for TrailState.
LIVE = "live"
"""Entry and peak are both observed: the position was opened while this runner was watching."""
RECONSTRUCTED = "reconstructed"
"""Entry from the actual fill, peak rebuilt from bars covering the whole holding period."""
ADOPTED = "adopted"
"""The position predates any usable record. Entry may be right, PEAK IS NOT KNOWN.

Peak-relative rules are SKIPPED for these until the position makes a new observed high, at which
point the peak becomes real and the caller promotes it to `live`. This is the fix for #197 B1: the
old code seeded `entry, peak = (px, px)` for anything without a row, which quietly asserted a peak
that had never happened — leaving give-back unarmed on seven positions and firing it on a 0.06%
artefact for the eighth."""


@dataclass(frozen=True)
class TrailState:
    """What the exits need to know about one open position."""

    entry_px: float
    peak_px: float
    opened_session: date | None = None
    sessions_held: int = 0
    sessions_since_high: int = 0
    peak_high_px: float | None = None
    """Highest HIGH since entry — what PEAK measures against.

    `peak_px` tracks the highest CLOSE, which is right for give-back: that is profit you could
    actually have closed at. It is WRONG for the fade threshold. PEAK compares against the high of
    day, and a session's high routinely exceeds every close in the holding period, so measuring
    "5% below the peak" off closes puts the line too low and fires late — sometimes never.

    None until the first session with a usable high, after which it only ever rises."""

    last_high: float | None = None
    """Previous session's HIGH. PEAK's lower-high branch compares highs, not closes — a name can
    close green while printing a lower high, and that sequence is the fade it detects."""

    lower_high_run: int = 0
    """Consecutive sessions each printing a lower high than the one before."""

    off_peak_run: int = 0
    """Consecutive sessions closing at or below the off-peak threshold."""

    sessions_in_give_back: int = 0
    """Consecutive sessions the give-back condition has held. PEAK's anti-noise principle
    (kumo-trading-platform #46): never fire on a single observation. Reset the moment the condition lifts."""
    quality: str = LIVE

    @property
    def peak_is_trustworthy(self) -> bool:
        return self.quality != ADOPTED


@dataclass(frozen=True)
class ExitPlan:
    """`exits` is symbol -> human reason; `state` is the advanced trail to persist."""

    exits: dict[str, str]
    state: dict[str, TrailState]


def peak_fade(cfg: ExitConfig, st: TrailState) -> str | None:
    """kumo-trading-platform's PEAK gate (#46), ported to a session clock.

    `_sustained_fade` fires on EITHER of two conditions, and NEITHER is a single-observation check:

        `lower_high_bars` consecutive bars each making a lower high than the one before, OR
        price sustained `off_hod_pct`+ below the high for 2+ consecutive bars

    The ticket records why: "a single red bar off a fresh HoD is NOT an exit — the -1.5% wiggle
    resumed". `off_peak_pct` in this module is the first half of that idea with the second half
    missing — it fires the instant one close crosses the line, which is exactly what PEAK refuses.

    The two branches are OR, not AND, and they catch different shapes. A name can grind down on
    steadily lower highs while never closing far below its peak (branch 1), or gap down and sit
    there without making a sequence (branch 2). Requiring both would miss both.

    This is a SESSION-level analogue: PEAK reads intraday bars and a high of day, these exits read
    one observation per decision. Same logic, coarser clock, and a fade that resolves within a
    session is invisible here by construction.
    """
    if cfg.peak_fade_lower_highs and st.lower_high_run >= cfg.peak_fade_lower_highs:
        return f"faded on {st.lower_high_run} consecutive lower highs"
    if cfg.peak_fade_off_pct is not None and st.off_peak_run >= (cfg.peak_fade_confirm or 2):
        ref = st.peak_high_px if st.peak_high_px is not None else st.peak_px
        return (f"held {100*cfg.peak_fade_off_pct:.0f}%+ below its {ref:.2f} peak for "
                f"{st.off_peak_run} sessions")
    return None


def needs_highs(cfg: ExitConfig) -> bool:
    """Does this config need per-symbol session HIGHS?

    BOTH PEAK branches do. This returned true only for the lower-high branch, and the threshold
    branch therefore ran against closes without ever asking for a high — the precise reference the operator
    corrected it away from. Highs are weakly greater than closes, so the real peak sits at or above
    the close peak and the real threshold sits at or above the one that was being tested: the rule
    was firing strictly less often than configured, everywhere, silently.

    It cost a sweep to find. `peak_fade_off_pct` at 8% and 10% produced byte-identical fills to
    their controls across three quarters, with give-back switched off entirely, while 6% moved
    trades. That cliff was the missing highs, not a property of the market.

    Same shape as `needs_atr` before `take_profit_atr` was added to it: a predicate that names one
    field instead of the question it answers. Callers must consult this rather than testing fields.
    """
    return cfg.peak_fade_lower_highs is not None or cfg.peak_fade_off_pct is not None


def needs_atr(cfg: ExitConfig) -> bool:
    """Does this exit config require per-symbol ATR?

    Callers must consult this rather than testing individual fields. A runner that checked only
    `give_back_min_peak_atr` computed no ATR when `take_profit_atr` was set alone, and
    `evaluate_exits` refused the run — correctly, but the caller should never have got there. One
    predicate means adding a third ATR-scaled rule cannot leave a driver silently behind, which is
    the #26 shape at one remove.
    """
    return (cfg.give_back_min_peak_atr is not None or cfg.take_profit_atr is not None
            or cfg.stop_loss_atr is not None)


def _peak_is_big_enough(cfg: ExitConfig, st: TrailState,
                        atr: dict[str, float] | None, sym: str) -> bool:
    """Has this position run far enough for give-back to be protecting a gain rather than noise?

    Returns True when the guard is off, which is today's behaviour.

    A symbol with no usable ATR does NOT arm. That is deliberate and matches how an ADOPTED peak is
    handled: when the scale of a move cannot be established, the honest response is to leave the
    position alone rather than to act on a number whose meaning is unknown. Arming instead would
    make the guard silently optional per symbol, which is how a rule ends up looking armed while
    being inert.
    """
    floor = cfg.give_back_min_peak_atr
    if floor is None:
        return True
    a = (atr or {}).get(sym)
    if a is None or not _finite(a) or a <= 0:
        return False
    return (st.peak_px - st.entry_px) >= floor * a


def evaluate_exits(cfg: ExitConfig, prices: dict[str, float],
                   state: dict[str, TrailState],
                   atr: dict[str, float] | None = None,
                   highs: dict[str, float] | None = None) -> ExitPlan:
    """Apply every configured exit rule to every position with a usable price.

    `atr` is per-symbol average true range in PRICE units, required whenever `needs_atr(cfg)` is
    true. Passing such a config without the input RAISES rather than silently arming as before — that combination is the #26 failure mode exactly, and this module
    already lost a rule to it once.

    Rule order matters and is deliberate: the cheapest and least reversible checks come first, so a
    position that has simply run out of time is not also reported as a give-back. First match wins.

    A symbol missing from `prices`, or carrying a non-finite one, is left alone entirely — its state
    is not advanced either. Advancing `sessions_held` on a day we could not see the price would let a
    data outage age a position out of the book.
    """
    # Asks the predicate rather than naming fields. Naming them is the exact anti-pattern
    # `needs_atr` exists to prevent, and this guard — in the function that documents the rule —
    # listed two of the three. `stop_loss_atr` passed straight through it and ran inert.
    if needs_atr(cfg) and atr is None:
        raise ValueError(
            "an ATR-scaled exit rule is configured but no `atr` was passed. Proceeding would "
            "silently ignore it and reproduce the behaviour these rules exist to prevent (#26). "
            "Consult `needs_atr(cfg)` in the driver and supply `trailing_atr(...)`.")

    exits: dict[str, str] = {}
    out: dict[str, TrailState] = {}

    for sym, st in state.items():
        px = prices.get(sym)
        if px is None or not _finite(px) or px <= 0:
            out[sym] = st
            continue

        # Both PEAK branches want the session HIGH; `needs_highs` is how a driver knows to supply
        # it. The fallback to the close is for the symbol that is simply missing from an otherwise
        # supplied map, and it degrades safely in the same direction for both branches — a close is
        # weakly below the high, so the peak is understated and the rule fires LESS often rather
        # than firing on something it should not.
        hi = (highs or {}).get(sym, px)
        lower_high = st.last_high is not None and hi < st.last_high
        # Measured against the highest HIGH since entry, not the highest close. A session high
        # routinely exceeds every close in the holding period, so a close-based peak puts the
        # threshold too low and the rule fires late or never.
        # Falls back to the CLOSE peak when no high peak has been recorded yet — a position
        # reconstructed from bars, or one carried from before this field existed, has a real
        # close-peak and no high-peak. Starting from zero would silently reset its reference to
        # today's high and put the threshold far too low.
        peak_hi = max(st.peak_high_px if st.peak_high_px is not None else st.peak_px, hi)
        off_peak_now = (cfg.peak_fade_off_pct is not None and peak_hi is not None
                        and px <= peak_hi * (1 - cfg.peak_fade_off_pct))
        made_high = px > st.peak_px
        nxt = replace(
            st,
            peak_px=max(st.peak_px, px),
            sessions_held=st.sessions_held + 1,
            sessions_since_high=0 if made_high else st.sessions_since_high + 1,
            peak_high_px=peak_hi,
            last_high=hi,
            lower_high_run=(st.lower_high_run + 1) if lower_high else 0,
            off_peak_run=(st.off_peak_run + 1) if off_peak_now else 0,
            # A new observed high makes the peak real, so an adopted position becomes trustworthy
            # from here — this is how a position recovers full exit coverage without guessing.
            quality=LIVE if (made_high and st.quality == ADOPTED) else st.quality,
        )
        out[sym] = nxt

        # THE FLOOR, ahead of everything — including the trustworthiness gate below. Every other
        # rule here is peak-relative and therefore needs the position to have gone up first; this
        # one measures from entry, so it is the only rule that covers a position that fell straight
        # from entry, and the only one that can act on an ADOPTED position whose peak is unknown.
        if cfg.stop_loss_atr is not None:
            a = (atr or {}).get(sym)
            if a is not None and _finite(a) and a > 0:
                loss = nxt.entry_px - px
                if loss >= cfg.stop_loss_atr * a:
                    exits[sym] = (f"stopped out {loss / a:.1f} ATR below entry "
                                  f"(stop {cfg.stop_loss_atr:.1f})")
                    continue

        # INTO STRENGTH, before the weakness rules. Ordered first among the peak rules deliberately: a position that has
        # hit its target is being sold for a reason that has nothing to do with fading, and letting
        # a give-back or stall rule claim it would mislabel the exit in the journal. First match
        # wins, so order IS the semantics here.
        if cfg.take_profit_atr is not None:
            a = (atr or {}).get(sym)
            if a is not None and _finite(a) and a > 0:
                gain = px - nxt.entry_px
                if gain >= cfg.take_profit_atr * a:
                    exits[sym] = (f"took profit at {gain / a:.1f} ATR "
                                  f"(target {cfg.take_profit_atr:.1f})")
                    continue
        # PEAK before the single-observation rules. It is the confirmed fade; letting a bare
        # threshold claim the same exit first would report the weaker reason in the journal.
        if nxt.peak_is_trustworthy:
            why = peak_fade(cfg, nxt)
            if why:
                exits[sym] = why
                continue

        if cfg.max_hold_days and nxt.sessions_held >= cfg.max_hold_days:
            exits[sym] = f"held {nxt.sessions_held} sessions (cap {cfg.max_hold_days})"
            continue
        if cfg.stall_days and nxt.sessions_since_high >= cfg.stall_days:
            exits[sym] = f"no new high for {nxt.sessions_since_high} sessions (cap {cfg.stall_days})"
            continue

        # Peak-relative rules below. An ADOPTED position has no trustworthy peak, so applying them
        # would either fire on a fabricated peak or, worse, look armed while being inert.
        if not nxt.peak_is_trustworthy:
            continue

        if cfg.off_peak_pct and px < nxt.peak_px * (1 - cfg.off_peak_pct):
            off = 1 - px / nxt.peak_px
            exits[sym] = f"{100*off:.1f}% below its peak (cap {100*cfg.off_peak_pct:.0f}%)"
            continue
        if cfg.give_back_frac:
            # THE ARITHMETIC IS SHARED WITH THE SHORT SIDE (`strategies/give_back.py`, #110/#123),
            # not copied. It used to live here as three lines, which is how the short book would
            # have acquired a second copy that agrees today and drifts on the first fix.
            #
            # The shared form breaches on `<=` where this read `<`. For `frac < 1.0` they differ
            # only at exact float equality; at `frac = 1.0` the strict form demands giving back MORE
            # than the whole peak, so a flat exit could fire only once the trade was already a loss.
            breached, peak_gain, now_gain = gave_back(
                entry_px=nxt.entry_px, extreme_px=nxt.peak_px, price=px,
                frac=cfg.give_back_frac, side=LONG)
            # PEAK's lesson, ported (kumo-trading-platform #46): "a single red bar off a fresh HoD is NOT an
            # exit — the -1.5% wiggle resumed". Every rule in this module fires on ONE observation
            # crossing a line, which is exactly what that ticket rejects. The counter advances only
            # while the condition holds and resets the moment it lifts, so a wiggle cannot
            # accumulate toward an exit across unrelated sessions.
            #
            # It should matter MORE the tighter the trail: at give_back=0.15 a single noisy print is
            # enough to breach, which is precisely where confirmation earns its keep.
            nxt = replace(nxt, sessions_in_give_back=(nxt.sessions_in_give_back + 1) if breached else 0)
            out[sym] = nxt
            need = cfg.give_back_confirm_sessions or 1
            if not _peak_is_big_enough(cfg, nxt, atr, sym):
                continue
            if breached and nxt.sessions_in_give_back >= need:
                # A position BELOW entry has given back its whole run and then some. Reporting
                # `1 - now/peak` there yields nonsense — the live journal recorded "gave back 30825%
                # of a 0.1% peak" for SU on 6 Aug, which told the operator nothing except that
                # something was wrong. Past 100% the useful fact is the loss, not the ratio.
                # THE TRAIL IS NAMED, like take-profit names its target (#221). The journal carries
                # this string verbatim, and "gave back 41% of a 48.7% peak" cannot tell an operator
                # whether the lane was on 0.35 or 0.50 — the read-back that ks#221's acceptance asks
                # for ("exits fire at 35%, not 50%") was an inference from 41% sitting between the
                # two. With the trail on the row it is a grep.
                trail = f"(trail {100*cfg.give_back_frac:.0f}%)"
                if now_gain <= 0:
                    exits[sym] = (f"gave back all of a {100*peak_gain:.1f}% peak and is "
                                  f"{100*abs(now_gain):.1f}% below entry {trail}")
                else:
                    exits[sym] = (f"gave back {100*(1 - now_gain/peak_gain):.0f}% "
                                  f"of a {100*peak_gain:.1f}% peak {trail}")
    return ExitPlan(exits=exits, state=out)


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def reconstruct_trail(entry_px: float, closes: list[float], *, covers_entry: bool) -> TrailState:
    """Rebuild a position's trail from bars instead of remembering it.

    This is what lets a driver with NO persistent store honour `ExitConfig` safely. The Nautilus
    adapters keep a bar deque per symbol and Nautilus itself knows each position's real entry price
    and open time, so the trail can be recomputed on every decision rather than carried across
    restarts in memory.

    That distinction is the whole point. #197 B1 was a trail that did not survive a restart:
    give-back went dead in production because the peak was remembered rather than derived. An
    in-memory trail in these adapters would reproduce that bug behind a green test suite. A
    reconstructed one cannot, because there is nothing to lose — a fresh process rebuilds the same
    trail from the same bars.

    `closes` are the closes over the holding period, oldest first. `covers_entry` says whether that
    history actually reaches back to the entry; when it does not, the peak is NOT known and the
    state is marked ADOPTED so peak-relative rules skip it rather than trusting a peak that may
    never have happened. That is the same fix as #197 B1: seeding `peak = entry` for a position
    whose history is missing asserts a peak nobody observed.
    """
    if not closes:
        # No bars at all: entry is known, peak is not. Never claim peak == entry.
        return TrailState(entry_px=entry_px, peak_px=entry_px,
                          quality=LIVE if covers_entry else ADOPTED)
    peak = max(closes)
    since_high = len(closes) - 1 - max(range(len(closes)), key=closes.__getitem__)
    return TrailState(
        entry_px=entry_px,
        peak_px=max(peak, entry_px) if covers_entry else peak,
        sessions_held=len(closes),
        sessions_since_high=since_high,
        quality=RECONSTRUCTED if covers_entry else ADOPTED,
    )
