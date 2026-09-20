"""Configuration for momentum rotation. Every knob lives here; nothing downstream hard-codes a number.

Defaults are the values that survived measurement (research/residual-gate, issue #15), not guesses:
the pool matters more than the rule — the identical rotation on random liquidity-matched universes
lost money (median -6.5%) while BCT's pool returned +53%, 0 of 40 seeds beating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from kumo_strategies.strategies.market_view import MarketViewConfig


@dataclass(frozen=True)
class ScoreConfig:
    """How strength is measured. All measures are scale-free — standardised against each name's own
    history — because a fixed threshold cannot span a universe whose volatility varies ~5x."""

    lookback: int = 20
    """Sessions of momentum. 20 beat 10 and 5 in testing, and beat them monotonically."""

    vol_window: int = 60
    """Sessions used to standardise. Momentum is divided by this name's own return SD, so a quiet
    name and a wild one compare fairly."""

    min_history: int = 25
    """Below this, the name has no usable distribution and is not rankable."""

    def __post_init__(self) -> None:
        """Refuse a value that is not the type this config declares.

        Settings overrides arrive UNCOERCED — cockpit tests `is_dataclass(f.type)`, and
        `from __future__ import annotations` makes `f.type` a STRING on every field, so
        that test is unconditionally False and raw values pass through. A string "10"
        for an int then fails later, somewhere that does not name the config.
        """
        from kumo_strategies.strategies._validate import check_field_types

        check_field_types(self)


@dataclass(frozen=True)
class IntradayConfig:
    """Intra-session timing overlay. OFF by default (weight 0.0), per the opt-in rule.

    ScoreConfig measures 20 SESSIONS over a 60-session vol window. Re-ranking that inside a session
    cannot move the top 5, which is why decision cadence measured identical at 30/60/120/390 minutes
    (+25% at every one). If intraday work is to pay, it has to time entries, not re-evaluate a daily
    number faster: prefer a name that is strong on the daily horizon AND not selling off right now.
    """

    lookback_bars: int = 12
    """Bars of intra-session momentum. 12 x 5min = one hour."""

    vol_window_bars: int = 78
    """Bars used to standardise — 78 x 5min = one full RTH session."""

    min_history: int = 20
    """Below this the bar-level distribution is not usable."""

    weight: float = 0.0
    """Blend weight on the intraday rank, added to the daily rank. 0.0 reproduces the daily-only
    strategy exactly.

    MEASURED: gross return peaks at exactly 0.0 and decays symmetrically either side (+29.8% at 0,
    +24.5% at -0.25, +23.4% at +0.25, +14.2% at +1.0). Turnover is also minimised at 0. Perturbing
    the daily ranking in ANY direction displaces better picks with worse ones."""

    entry_only: bool = False
    """Apply the overlay only to entry ORDERING, leaving eligibility and exits on the daily score.
    Isolates 'time the entry' from 'churn the book', which plain `weight` conflates."""


@dataclass(frozen=True)
class PortfolioConfig:
    n_hold: int = 5
    """Positions held. Fewer concentrates the edge — 3 beat 5 beat 10 — at the cost of variance."""

    buffer: int = 3
    """A held name is only sold once it falls out of the top (n_hold + buffer). Without this the
    book thrashes on rank noise around the boundary."""

    max_weight: float = 1.0
    """Cap on any single position as a fraction of equity. 1/n_hold is equal weight."""

    max_correlation: float | None = None
    """Reject a candidate whose trailing correlation to a name already in the book exceeds this.

    Measured motivation: with n_hold=5 the book sat >=60% in one sector on 63% of sessions and
    100% in one sector on 19% — five tickers holding one bet (APA/PARR/EQNR/CVE/VIST is a single
    oil trade). Correlation rather than sector because sector is only a proxy for co-movement, and
    labels cover just 56% of the book while returns cover all of it.

    None disables the cap. Costs almost no turnover: it changes WHICH name fills a slot, not how
    many trades happen."""

    corr_window: int = 60
    """Sessions of daily returns used for the correlation estimate. Point-in-time, ending at t."""

    size_to_book: bool = False
    """Spread the deployed gross over the names the lane ACTUALLY holds, not over `n_hold`.

    Sizing divides by a PLANNED book. Any rule that declines an entry — a gate, `max_correlation`,
    `max_positions`, a thin pool, `min_abs_gap_pct` — therefore leaves that slot's capital in cash,
    and the lane earns its return on less money than it was given.

    Measured at +2.72pp on one panel, and THAT NUMBER SHOULD NOT BE TRUSTED as it stands. Three
    problems, all found in review: the interval came from a percentile block bootstrap, which is
    biased low when the effect is carried by a handful of sessions — so "excludes zero" is the
    method reporting its own bias; the contrast compares arms with roughly twice the per-name
    exposure, so it measures return rather than edge; and the backtest arm does not model what live
    would do (see the denominator note below). The honest version of this reasoning is the comment
    in kumo-trading-platform `strategies/momentum.py`, which is why the flag is not enabled there.

    THE TWO DRIVERS DISAGREE ON THE BOOK. `runner_verified` divides by `len(dec.hold)` computed
    BEFORE the gap filter declines anything; `pgrunner` counts the book AFTER. Live would be far
    more concentrated than the arm that was measured. `pgrunner` also has no cash check and there is
    no aggregate `max_deployed_frac` check anywhere in `_submit`, so 150% of gross across two
    sessions is reachable. Fix both before this is ever enabled, then re-measure.

    WHAT THAT EVIDENCE IS AND IS NOT. 129 tradeable sessions, 2025-12-01 to 2026-06-05 — the panel
    spans 160 but `LedgerBook` has no eligible names before December, so the first 31 have no pool.
    84 round trips. Same sign in both halves (+0.16pp and +1.35pp) but the magnitude sits in the
    second. There is NO out-of-sample window for it: the 140-session window that confirmed
    `lookback=40` is an IC study over names and needs no candidate ledger, while a traded backtest
    needs the pool, and the pool does not exist that far back. Six months, one regime, and small.

    IT CONCENTRATES. The same gross over ~3.4 names is roughly double the per-name notional. The
    measured drawdown barely moved, but that is one regime, and concentration is precisely what
    behaves differently when the tail stops paying.

    Interacts with `inverse_vol_entry_relative`, which normalises WITHIN an entry batch. This changes
    the DENOMINATOR the batch is scaled against. Both may be on; they compose, and neither
    double-counts, because the weight path is expressed as a multiple of the equal-weight slot.

    Off by default. Every result recorded before 2026-09-04 was measured with it off."""

    rebalance_band: float | None = None
    """Trade every position to its TARGET each session, not just the ones entering or leaving.

    2026-09-04: *"There is a sizing. The sizing needs to be executed. That can mean buys or
    sells of certain quantities of a symbol. Simple."*

    The runner models trading as two event kinds — an ENTRY sized once at arrival, and an EXIT that
    sells everything — and nothing ever revisits a position's size. That is what makes weights depend
    on when a name arrived: three names entered while the book was thin keep a third of the gross
    each, and a name entering later gets whatever denominator applied that day.

    Target-based sizing removes the special cases rather than adding one:

        target_qty = target_notional / price
        delta      = target_qty - held_qty
        delta > 0  -> BUY  delta        (an entry is simply a delta from zero)
        delta < 0  -> SELL -delta       (an exit is simply target_qty = 0)

    `0.25` means "only trade when actual is more than 25% away from target". A band, because
    converging exactly churns on mark-to-market drift and every correction pays the spread.

    SUPERSEDES `trim_band`, which only ever sold the overweight names and never topped up the
    underweight ones — half of the same idea. Both are backtest-only; `trim_band`'s docstring lists
    the four live invariants a partial sell disturbs, and they apply here identically.

    None disables it, which is today's behaviour and how every recorded result was measured."""

    trim_band: float | None = None
    """Trim a position back toward its target slot once it exceeds it by this fraction.

    THE OTHER HALF OF `size_to_book`. That flag divides the gross by the book at the moment of
    ENTRY and nothing rebalances afterwards, so a name entered while the book was thin keeps a share
    the book can no longer afford: three names entered at 1/3 of gross take 80% of equity, and when
    five more qualify the next session there is cash for two of them. The resulting weights are set
    by ARRIVAL ORDER, which no signal chose.

    `0.25` means "sell back to target once a position is 25% above it". A band rather than an exact
    rebalance because trimming to the millimetre churns on mark-to-market drift, and every trim pays
    the spread twice over the position's life.

    None disables it, which is today's behaviour and what every recorded result was measured with.

    BACKTEST ONLY FOR NOW, and deliberately so. A partial sell breaks four invariants the live
    runner holds and the harness does not:

      trail state     a trimmed position stays OPEN, so give-back peak tracking must survive the
                      partial sale or the remaining shares get a fresh and wrong trail
      order identity  `client_order_id` is keyed (session, symbol, side, attempt), so a trim and a
                      later full exit of the same symbol in one session collide and Nautilus denies
                      the second locally — #51's replay guard firing on a legitimate order
      round trips     FIFO round-trip accounting assumes whole exits; every KPI reads those
      claims ledger   reconciliation tracks whole positions

    So this measures whether trimming pays BEFORE anything touches the order path. If it does not,
    none of those four invariants had to be disturbed to find that out.
    """

    inverse_vol_sizing: bool = False
    """Size positions by 1/sigma instead of equal dollar, normalised to the same gross exposure.
    A wild name and a quiet one then contribute comparable risk rather than comparable notional."""

    cluster_exit_score: float | None = None
    """Exit a held name when its CORRELATION CLUSTER is weakening, not when the cluster is large.

    That distinction is the whole design and it is easy to get backwards. #25 measured where the
    money is: the wildest volatility quartile carried +90.4% of P&L, so concentration is the edge.
    A rule that fires on cluster SIZE caps the thing that pays. This fires on the cluster's mean
    SCORE falling below the threshold — the shared bet is dying, which is different from the book
    holding one bet four times on purpose.

    Live evidence for the mechanism, from MOMENTUM-002's own journal:

      2026-08-13  AEM/CGAU/FSM/WPM exited together, four precious-metals miners gapping as one
      2026-08-14  all four bought back the next session, paying spread twice
      2026-08-04..06  SU/DINO/PAA, an energy cluster, -$1,192 — most of the realised loss

    The ranking exits these names one at a time as each individually falls out of the top
    (n_hold + buffer). The buffer sweep showed rotating FASTER is strictly worse, so the answer is
    not a shorter buffer; it is noticing that the four names are one position.

    Cluster membership comes from measured covariance, never from sector labels. #24 framed those as
    alternatives (B vs F3); the miners correlate regardless of how GICS files them, and `panel_stats`
    already returns the matrix so this needs no new data.

    None disables it, which is today's behaviour."""

    cluster_corr: float = 0.7
    """Trailing |correlation| above which two held names count as the same bet, for
    `cluster_exit_score`. Not a diversification cap — it defines the cluster, it does not limit it."""

    inverse_vol_entry_relative: bool = False
    """Normalise the inverse-vol weights across the ENTERING names rather than the whole book.

    Only meaningful with `inverse_vol_sizing`. It exists because sizing happens at entry and is never
    rebalanced, which makes the two normalisations measure different things:

      book-relative (False)  each name enters at its target share of a full inverse-vol book. But a
                             momentum book enters its most VOLATILE names, so their weights sit below
                             the book average, every entry batch buys less than a full slot, and the
                             book runs persistently under-deployed. Measured: mean deployment fell
                             63.3% -> 59.4% with the traded name set completely unchanged, so the
                             shortfall is pure cash drag, not selection.
      entry-relative (True)  the batch deploys the same gross as equal weight would, and 1/sigma only
                             decides how that gross is SPLIT between the names entering together.

    False conflates the risk tilt with that cash drag; a return difference then has two causes and
    the ablation cannot say which. True isolates the tilt. Run both — the gap between them IS the
    cost of never rebalancing."""


@dataclass(frozen=True)
class ExitConfig:
    """Rules that free capital early, for when the constraint is capital rather than ideas.

    The rotation already exits a name when it leaves the ranking, but that can take weeks while the
    position does nothing. Measured on the traded book: 82 of 164 round trips finished within +/-2%
    while consuming 35.7% of all capital-time, and the top 10 trades produced 95% of P&L from 17.6%
    of it. Cutting stalled positions frees slots for names that are actually moving.

    All default OFF — a stall rule that fires too early sells the pause before the second leg, which
    is exactly how the 19-day pool expiry was costing ~50 days of every winner.
    """

    stall_days: int | None = None
    """Exit if the position has gone this many sessions without a new closing high since entry."""

    max_hold_days: int | None = None
    """Hard time stop, in sessions held."""

    off_peak_pct: float | None = None
    """Exit when price falls this fraction below its peak CLOSE since entry — a PRICE-based trail,
    independent of entry. This is the measure kumo-trading-platform's shipped PEAK algo (#46) uses
    (`_sustained_fade`: off-high %, lower-highs), so if it performs comparably the systematic
    strategy can adopt PEAK unchanged rather than asking a reviewed, paper-deployed exit gate to
    change its trigger math."""

    peak_fade_off_pct: float | None = None
    """PEAK's off-high threshold — but CONFIRMED, unlike `off_peak_pct` (#30, kumo-trading-platform #46).

    Fires only once price has held this far below the peak for `peak_fade_confirm` consecutive
    sessions. `off_peak_pct` fires on the first close that crosses the line, which is precisely what
    PEAK's ticket rejects: "a single red bar off a fresh HoD is NOT an exit — the -1.5% wiggle
    resumed"."""

    peak_fade_confirm: int = 2
    """Consecutive sessions the off-peak condition must hold. 2 matches PEAK's own default and is
    the minimum that is not a single-observation check."""

    peak_fade_lower_highs: int | None = None
    """Consecutive lower HIGHS that also trigger the fade, independently of the threshold.

    PEAK's two branches are OR because they catch different shapes: a name can grind down on
    steadily lower highs while never closing far below its peak, or gap down and sit there without
    making a sequence. Requiring both would miss both."""

    stop_loss_atr: float | None = None
    """Hard floor: exit if price falls this many ATR BELOW THE ENTRY. None disables it, which is
    today's behaviour — the strategy has never had a stop (#30).

    Measured from entry rather than from the peak, and that is the whole point rather than a
    simplification. Every other exit here is peak-relative: `give_back_frac` trails a winner,
    `peak_fade_*` measures off the high, `off_peak_pct` and `stall_days` need a peak to exist. All
    of them require the position to have gone UP first. A name that falls from the moment it is
    entered arms none of them, and until this field existed the only thing between such a position
    and an unbounded loss was rotation dropping it out of the ranking.

    It is also the only rule that can protect an ADOPTED position. The peak-relative rules are
    skipped when `peak_is_trustworthy` is false, because acting on a fabricated peak is worse than
    not acting — so an adopted position currently runs with almost no exit coverage. Entry price is
    known for adopted positions (#197 B1 takes the broker's real entry), so this rule works where
    the others must decline to.

    ATR-scaled for the reason in #14: a fixed percentage cannot span a universe whose volatility
    varies ~5x. 2.0 means "two average daily ranges against us from entry"."""

    take_profit_atr: float | None = None
    """Exit INTO STRENGTH when the position is this many ATR in profit (#30 item 3).

    Every other rule in this class fires on WEAKNESS — give-back, off-peak, stall, time — and so
    does the ranking exit. Nothing sells while a position is still rising. That gap is the whole
    reason this exists; the operator raised it on 2026-08-15 and it had never been considered.

    Measured cost of not having it, from the 2026-08-13 journal. Give-back sells AFTER surrendering
    the run, by definition:

        AEM   gave back 51% of a 5.3% peak   ->  2.7% left behind
        CGAU  gave back 75% of a 4.6% peak   ->  3.5% left behind
        FSM   gave back all  of a 4.1% peak  ->  4.1% left behind
        WPM   gave back 73% of a 2.8% peak   ->  2.0% left behind

    ~$1,192 in one session on ~$9,700 positions.

    THE ARGUMENT AGAINST IS STRONG AND MUST BE MEASURED, NOT ASSUMED. This strategy is
    tail-driven — the wildest volatility quartile carries +90.4% of P&L (#25) — and a profit target
    caps the right tail by construction. The buffer sweep and the cluster exit both showed that
    cutting sooner loses. This is a third variation on "exit earlier", and the prior from those two
    is bad.

    In ATR rather than percent for #14's reason: a fixed 10% target is a different instrument on a
    1.5% ATR name than on a 7% one, and this book spans both.

    A FULL exit, not a partial. Scaling out is the version that would keep the tail while banking
    some of the run, and it is unreachable today — `ExitPlan.exits` is symbol -> reason with no
    quantity, so partial exits cannot be expressed by any driver. Test the full exit first; if it is
    not clearly negative, the partial is worth the plumbing.

    None disables it, which is today's behaviour."""

    give_back_confirm_sessions: int | None = None
    """Consecutive sessions the give-back condition must hold before it fires (#30).

    Ported from kumo-trading-platform's PEAK gate (#46), which fires only on `lower_high_bars` consecutive
    lower highs OR the threshold sustained for 2+ consecutive bars — "never a single-bar/single-tick
    check". Its own ticket records why: "a single red bar off a fresh HoD is NOT an exit — the -1.5%
    wiggle resumed".

    EVERY rule in this module violates that principle today; each fires the instant one observation
    crosses a line. The live PRU exit fired on a 0.7% peak from a single crossing.

    Expected to matter most at TIGHT give-back settings: at 0.15 a single noisy print is enough to
    breach, so confirmation is worth more there than at 0.5. That interaction is the thing worth
    measuring, not the field in isolation.

    None or 1 reproduces today's single-observation behaviour exactly.

    NOTE this is a SESSION-level analogue. PEAK is intraday (bars within a session, high of day);
    these exits evaluate once per decision on the peak close since entry. Same principle, coarser
    clock."""

    give_back_min_peak_atr: float | None = None
    """Minimum peak profit, in ATR, before `give_back_frac` is allowed to arm (#24 D, #30).

    `give_back_frac` is scale-free: it treats a 0.7% peak and a 28% peak identically. Live evidence
    that this is wrong — every give-back exit MOMENTUM-002 has fired:

        08-06  SU    0.1% peak   -$554   (compounded by #197 B1)
        08-10  PRU   0.7% peak   -$126   NOT a bug — the rule working as written
        08-13  AEM   5.3% peak           legitimate
        08-13  FSM   4.1% peak           legitimate
        08-13  WPM   2.8% peak           legitimate
        08-13  CGAU  4.6% peak           legitimate

    Two of six armed on a peak smaller than one normal session's range, and both were losses. A
    "peak" that size is noise, so give-back is selling a fluctuation rather than protecting a gain.

    Expressed in ATR rather than as a percentage for the reason in #14: a fixed threshold cannot
    span a universe whose volatility varies ~5x. 1.0 means "the position must have run at least one
    average day's range before give-back can arm".

    None disables the guard, which is today's behaviour and keeps every recorded result reproducible."""

    give_back_frac: float | None = None
    """Exit if the position surrenders this fraction of its peak OPEN profit (0.5 = half the run).
    Only arms once the trade is in profit, so it trails a winner rather than tightening on a loser."""


LIVE_SUPPORTED_EXITS: frozenset[str] = frozenset(
    {"give_back_frac", "give_back_min_peak_atr", "give_back_confirm_sessions",
     "take_profit_atr", "stop_loss_atr", "peak_fade_off_pct", "peak_fade_confirm",
     "peak_fade_lower_highs", "off_peak_pct", "stall_days", "max_hold_days"})
"""Which ExitConfig rules the LIVE runner actually implements.

The rules are currently written twice — once in the backtest harness and once in the live runner —
and the live copy only ever grew one of them. `ExitConfig` is shared, so a research config setting
`stall_days` typechecks, backtests, deploys, and is then silently ignored by the thing holding real
positions. Nothing failed; the exit simply never fired.

This set is the honest statement of what live can honour. Since #197 P2 it covers EVERY field,
because both drivers now call the same `exits.evaluate_exits` — there is no longer a second copy that
can fall behind. It stays as a tripwire: a new `ExitConfig` field that the evaluator does not handle
must be added here deliberately, and `test_exit_coverage.py` fails until someone does."""


def unsupported_live_exits(exits: ExitConfig) -> list[str]:
    """Configured exit rules the live runner cannot honour — empty when live can do everything asked.

    Pure and dependency-free so both the CI gate and the runtime degrade path can call it. A rule
    counts as configured when it is set to anything other than None; `ExitConfig` defaults every rule
    off precisely so "unset" and "asked for" are distinguishable.
    """
    return sorted(
        f.name
        for f in fields(exits)
        if f.name not in LIVE_SUPPORTED_EXITS and getattr(exits, f.name) is not None
    )


@dataclass(frozen=True)
class ExecutionConfig:
    """How a decision becomes a fill.

    Every field here was unread — `grep` for `.execution` across the repo returned nothing but this
    class and the line that instantiates it. Two of the three have been deleted rather than wired,
    because they described things that are not true (#26):

      `cost_bps = 10.0`   A flat round-trip rate, superseded by the MEASURED per-symbol half-spreads
                          in `backtesting/costs.py`. Its docstring claimed "the measured result
                          survived 40bps", which reads as a live sensitivity result for a number
                          connected to nothing. Keeping a second, cruder cost assumption next to the
                          real one invites someone to tune it.
      `gap_through_stop`  Described what happens when a bar opens beyond a stop. There is no stop
                          mechanism in the backtest at all. A config field is a claim about
                          behaviour, and this one would have told a reader the opposite of the truth.

    If either becomes real, add it back with the code that honours it.
    """

    decision_slots: tuple[str, ...] = ("open+5m",)
    """WHEN in the session the strategy decides, as session-relative anchor names (#29, #32).

    BCTROT's defining parameter, and the reason it lives here rather than staying a constructor
    argument: it was the one setting an operator most needs to see and the only one they could not.
    The operator's requirement on #32 is that every parameter goes through the cockpit settings mechanism,
    and a schedule reachable only from Python is outside it.

    ANCHORS, NOT TIMES. `open+150m` and `close-20m` resolve against each session's real open and
    close, so a half day still fires correctly, and the NAME is the idempotency key — the decision
    store's unique index is `(strategy_id, session, slot)`. A wall-clock timestamp could not tell a
    retry of the 12:00 decision from a new one at 12:03.

    The default is the single 09:35 slot, which reproduces MOMENTUM-002 exactly. BCTROT sets
    `("open+150m", "close-20m")`.

    Validated at construction, not at fire time: a slot that cannot be resolved must not become a
    session that silently decides fewer times than configured."""

    min_abs_gap_pct: float | None = None
    """Decline an entry whose overnight gap sits inside +/- this, as a fraction (0.015 = 1.5%).

    A DEAD BAND, not a direction. `research/residual-gate/FINDINGS-gap-entry.md` tested the operator's
    hypothesis that small gap-ups keep running and small gap-downs keep losing; the bucket study said
    the shape is MAGNITUDE. Big gaps pay in both directions and the flat middle — 45% of candidates —
    is where the return disappears. So this skips the middle and takes both tails.

    Evidence, and it is unusually good for this repo: the threshold curve is smooth rather than a
    spike (sharpe 1.54 / 1.95 / 2.19 / 2.30 / 2.13 / 2.02 at 0.0 / 1.0 / 1.25 / 1.5 / 1.75 / 2.0%),
    and it survives every other config tested — n_hold 5, 8 and 10, give-back 0.35, lookback 40, and
    exits switched off entirely (+0.43 to +1.26 sharpe). A rule that only works at one setting is
    fitted to that setting; this one is not.

    THE GAP IS A PROPERTY OF THE DAY. `today's open / yesterday's close` is fixed the moment the
    market opens, so a lane deciding at 09:35, 12:00 and 15:40 reaches the same verdict at all three
    — there is no slot list, and there is nothing to remember between slots.

    TWO EARLIER VERSIONS GOT THE INPUT WRONG, and both looked right. The first took the FILL price
    and carried a `min_abs_gap_slots` companion — `fill / prior close` is the overnight gap only at
    the open, and by midday it is "the day's move so far", so the slot list existed to hide a
    disagreement the rule had created. The second read `day["open"]` from the runner's panel, which
    is trimmed to sessions strictly BEFORE the one being decided, so it computed the PREVIOUS
    session's gap — live, at all three slots, journalled as healthy. Today's open now comes from the
    caller, which is the only place it exists.

    An unavailable open ADMITS the entry, deliberately and tested. That matches every other
    missing-data path here — never let a feed gap shrink the book — and the alternative, declining,
    turns a data outage into a full entry stop. It is journalled as a RISK line naming the symbols,
    because "inert this slot" and "nothing to decline" are otherwise the same empty result.

    None disables it."""

    max_stale_days: int = 4
    """Sessions of missing data before the strategy refuses to decide rather than deciding on an old
    panel. A stale decision is not a cautious one — it trades yesterday's ranking at today's prices."""

    equity_per_position: float = 1_000.0
    """Notional per position when the runner has no better figure. The live runner sizes from broker
    equity; this is the fallback, and it is a real number rather than a placeholder because a wrong
    fallback silently changes position size."""

    fill: str = "next_open"
    """Signals are computed on the close of session t; the earliest honest fill is the next open.
    Using the signal bar's own high/low is look-ahead and manufactures edge.

    Now actually consulted. `runner_verified.run` had its own `entry` argument defaulting to the same
    string, so this field was a second copy of the decision that nothing read — setting it to
    `"close"` changed nothing and reported no error. That is a look-ahead trap of exactly the shape
    #10 warns about: the knob that appears to control fill timing did not."""


@dataclass(frozen=True)
class GateConfig:
    """Data-integrity gates. Every one of these exists because its absence produced a false result
    during research — see issue #13 and the commit history."""

    corporate_action_move: float = 0.40
    """A single-session move beyond this is treated as a split, not a trade. The data is raw and
    unadjusted, so a 5:1 split reads as -80%. Four separate results were invalidated by this."""

    corporate_action_window: int = 41
    """Sessions around a flagged action that are untradeable. Must exceed the holding period, or an
    action landing late in a hold slips through."""

    exclude_instrument_types: tuple[str, ...] = ("etf_or_product", "adr_or_international")
    """Leveraged and inverse products reverse-split constantly and dominated a result until removed.
    Classifier: research-lab ledger-analysis/build_ledger_symbol_profile_features.py."""

    min_dollar_volume: float = 5e6
    """Median daily dollar volume. Deliberately low — the edge is not confined to large caps."""

    dollar_volume_window: int = 60
    """Trailing window for the liquidity median. It exists because the median used to be taken over
    the WHOLE panel handed in, which in a backtest is the entire history including the future: a
    name that became liquid later passed the gate on dates when it was not. Live never had the bug
    -- the runner passes trailing bars only -- so this is what makes the two agree."""

    min_atr_pct: float = 0.01
    """Names that barely move get an enormous share count under risk-based sizing, so one tick
    becomes multiple R. Require the name to actually move."""


@dataclass(frozen=True)
class MomentumRotationConfig:
    """The whole strategy. Swap `candidate_source` to change what it trades without touching logic."""

    candidate_source: str = "bct_positions"
    """Name in the candidate-source registry. `bct_positions` is a start, not the design — see
    candidates.py. The rotation rule is worthless without a good pool, so this is the load-bearing
    choice, and it is deliberately pluggable."""

    candidate_params: dict = field(default_factory=dict)

    score: ScoreConfig = field(default_factory=ScoreConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    market_view: "MarketViewConfig" = field(default_factory=lambda: MarketViewConfig())
    """NO VIEW, DELIBERATELY — and "no view" rather than "a view whose window is unmeasured".

    MOMENTUM-002 and BCTROT-004 rank the ledger-provider book, whose membership is trustworthy only from
    2025-12-01 (`LedgerBook.COVERAGE_START`, a deliberate refusal: earlier community coverage
    recovered partially and answering anyway would invent membership). Over that usable history
    the deepest drawdown is **14.0% for MOMENTUM and 9.2% for BCTROT**.

    A BEAR-DETECTION WINDOW FITTED THERE IS FITTED ON A PERIOD WITH ALMOST NO BEAR IN IT. That is
    exactly how TECHIVOL's 50 became a number nobody had measured for any other lane, and doing it
    twice more would turn one fitted constant into a platform assumption.

    WHY `signal=NONE` AND NOT A VIEW MARKED UNMEASURED. A lane declaring a view with an unmeasured
    window READS AS A LANE WITH A VIEW to everything downstream — the label plane renders it, the
    poller reports it, and the honest content lives in a comment nobody sees. `signal=NONE` is the
    inert state and it is LOUD BY CONSTRUCTION: it yields RISK_ON with the reason "no market view
    configured", and that reason travels in `reasons` all the way to the notification payload. A
    lane that could not measure and says so beats a lane carrying a number that looks measured.

    REVISIT WHEN EITHER HOLDS — stated so this does not become permanent by default, which is how
    a gap becomes furniture:

      1. the pool's usable history extends back far enough to contain a real drawdown, or
      2. a live bear occurs and the window becomes measurable forward.

    Measured 2026-09-11, `research/market-view/FINDINGS-all-lanes.md`: over 2025-12-01 to
    2026-09-10 the 50-day view costs MOMENTUM 1.84pp and BCTROT 4.71pp and saves 0.05pp and
    -0.02pp of drawdown respectively. Inheriting TECHIVOL's window would have shipped that.
    """



def with_decision_slots(cfg: MomentumRotationConfig, slots: tuple[str, ...]) -> MomentumRotationConfig:
    """The same strategy on another schedule — the ONLY way a backtest changes its slots (#270).

    `runner_cadence` took `slots=` at the call site, so a sweep could measure a schedule nothing
    deploys and the deployed schedule could drift from the measured one without a diff anywhere.
    `run_sessions` reads `cfg.execution.decision_slots` and nothing else; a schedule arm is a CONFIG
    arm, spelled here so every sweep says the same thing the deployment says.
    """
    return replace(cfg, execution=replace(cfg.execution, decision_slots=tuple(slots)))
